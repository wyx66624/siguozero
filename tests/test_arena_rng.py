from __future__ import annotations

import random
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    import torch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch training extra is not installed")
class ArenaRngTests(unittest.TestCase):
    def engines(self):
        from junqi.training.inference import InferenceEngine
        from junqi.training.models import (
            GamePolicyTransformer, ModelConfig, PieceConditionedLayoutPointerDecoder,
        )

        torch.set_num_threads(1)
        config = ModelConfig.tiny()
        candidate, opponent = [InferenceEngine(
            "two_player", GamePolicyTransformer(config),
            PieceConditionedLayoutPointerDecoder(config),
        ) for _ in range(2)]
        candidate.checkpoint_update, opponent.checkpoint_update = 35, 30
        return candidate, opponent

    def test_cpu_seed_reproduces_layout_and_action_randomness(self):
        from junqi.training.arena import seed_inference
        from junqi.training.checkpoint import capture_rng_state, restore_rng_state

        engine = SimpleNamespace(policy=SimpleNamespace(device=torch.device("cpu")))
        original = capture_rng_state()
        try:
            with patch("junqi.training.arena.manual_seed_all") as accelerator_seed:
                seed_inference(12345, engine)
                first = (random.random(), torch.rand(8))
                seed_inference(12345, engine)
                second = (random.random(), torch.rand(8))
                self.assertEqual(first[0], second[0])
                torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
                accelerator_seed.assert_not_called()
        finally:
            restore_rng_state(original)

    def test_cuda_and_npu_seed_dispatch_uses_backend_before_seed(self):
        from junqi.training.arena import seed_inference

        for backend in ("cuda", "npu"):
            with self.subTest(backend=backend):
                # NPU is deliberately represented without registering torch_npu,
                # so both accelerator contracts are exercised on CPU-only CI.
                device = SimpleNamespace(type=backend)
                engine = SimpleNamespace(policy=SimpleNamespace(device=device))
                with patch("junqi.training.arena.is_accelerator", return_value=True), \
                        patch("junqi.training.arena.random.seed"), \
                        patch("junqi.training.arena.torch.manual_seed"), \
                        patch("junqi.training.arena.manual_seed_all") as accelerator_seed:
                    seed_inference(98765, engine)
                accelerator_seed.assert_called_once_with(backend, 98765)

    def test_match_reuses_frozen_candidate_and_loads_only_opponent(self):
        from junqi.training.arena import MatchSettings, run_match, sha256_file
        from junqi.training.distributed import DistributedContext
        from junqi.training.inference import InferenceEngine

        candidate, opponent = self.engines()
        context = DistributedContext.initialize("cpu")
        settings = MatchSettings(pairs=1, max_plies=2, smoke_test=True)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            candidate_path, opponent_path = root / "candidate.pt", root / "opponent.pt"
            candidate_path.write_bytes(b"pinned candidate")
            opponent_path.write_bytes(b"pinned opponent")
            with patch.object(InferenceEngine, "from_checkpoint", return_value=opponent) as load, \
                    patch.object(candidate.actor, "sample", wraps=candidate.actor.sample) as sample:
                summary = run_match(
                    candidate_path, opponent_path, settings, context, root / "match",
                    candidate_sha256=sha256_file(candidate_path),
                    opponent_sha256=sha256_file(opponent_path),
                    candidate_engine=candidate,
                )
            load.assert_called_once_with(
                opponent_path, device="cpu", mode="two_player", temporal_cache_entries=8,
            )
            self.assertTrue(sample.called)
            self.assertEqual(summary["games"], 2)
            self.assertEqual(summary["candidate_update"], 35)
            self.assertEqual(summary["opponent_update"], 30)
            self.assertFalse(candidate.policy.training)
            self.assertTrue(all(not p.requires_grad for p in candidate.policy.parameters()))

            with patch.object(InferenceEngine, "from_checkpoint") as load:
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    run_match(
                        candidate_path, opponent_path, settings, context, root / "bad_hash",
                        candidate_sha256="wrong", candidate_engine=candidate,
                    )
                load.assert_not_called()

    def test_borrowed_candidate_sets_opponent_precision(self):
        from junqi.training.arena import MatchSettings, run_match
        from junqi.training.distributed import DistributedContext
        from junqi.training.inference import InferenceEngine

        candidate, opponent = self.engines()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            for dtype in (None, torch.float16, torch.bfloat16):
                with self.subTest(dtype=dtype):
                    candidate.actor.amp_dtype = dtype
                    opponent.actor.amp_dtype = torch.float32
                    with patch.object(InferenceEngine, "from_checkpoint", return_value=opponent):
                        run_match(
                            root / "candidate.pt", root / "opponent.pt",
                            MatchSettings(pairs=1, max_plies=2, smoke_test=True),
                            DistributedContext.initialize("cpu"), root / "precision",
                            candidate_engine=candidate,
                        )
                    self.assertIs(opponent.actor.amp_dtype, dtype)

    def test_cancel_parallel_stream_does_not_accept_partial_match(self):
        from junqi.training import arena_two_player
        from junqi.training.arena import MatchSettings, run_match
        from junqi.training.distributed import DistributedContext
        from junqi.training.inference import InferenceEngine

        candidate, opponent = self.engines()
        original_configs = [engine.policy.config for engine in (candidate, opponent)]
        original_batch_sizes = [engine.actor.max_batch_size for engine in (candidate, opponent)]
        stop = Mock(side_effect=[False, True])
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            with patch.object(InferenceEngine, "from_checkpoint", return_value=opponent), \
                    patch.object(arena_two_player, "prepare_group", wraps=arena_two_player.prepare_group) as prepare, \
                    patch.object(candidate.actor, "sample", wraps=candidate.actor.sample) as sample, \
                    patch("junqi.training.arena.summarize_games") as summarize:
                with self.assertRaisesRegex(InterruptedError, "model selection interrupted"):
                    run_match(
                        root / "candidate.pt", root / "opponent.pt",
                        MatchSettings(pairs=2, max_plies=64, smoke_test=True,
                                      parallel_games=2, inference_batch_size=2,
                                      environment_workers=2),
                        DistributedContext.initialize("cpu"), root / "cancel",
                        candidate_engine=candidate, stop_requested=stop,
                    )
            self.assertEqual(stop.call_count, 2)
            self.assertTrue(prepare.called)
            self.assertTrue(sample.called)
            summarize.assert_not_called()
            self.assertFalse(list(root.glob("cancel.*.jsonl")))
            self.assertFalse(list(root.glob("cancel*.json")))
            for engine, config, batch_size in zip(
                (candidate, opponent), original_configs, original_batch_sizes, strict=True
            ):
                self.assertIs(engine.policy.config, config)
                self.assertEqual(engine.actor.max_batch_size, batch_size)
                self.assertFalse(engine.policy._inference_board_cache)
                self.assertFalse(engine.policy._inference_temporal_cache)

    def test_borrowed_candidate_without_update_is_rejected(self):
        from junqi.training.arena import MatchSettings, run_match
        from junqi.training.distributed import DistributedContext
        from junqi.training.inference import InferenceEngine

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(InferenceEngine, "from_checkpoint") as load:
                with self.assertRaisesRegex(RuntimeError, "checkpoint update"):
                    run_match(
                        root / "candidate.pt", root / "opponent.pt", MatchSettings(pairs=1),
                        DistributedContext.initialize("cpu"), root / "match",
                        candidate_engine=SimpleNamespace(),
                    )
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
