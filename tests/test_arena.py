from __future__ import annotations

from dataclasses import asdict, replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr, redirect_stdout

try:
    import torch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


class ArenaFixtures:
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def engines(self, mode="two_player"):
        from junqi.training.inference import InferenceEngine
        from junqi.training.models import (
            GamePolicyTransformer, ModelConfig, PieceConditionedLayoutPointerDecoder,
        )
        from junqi.training.modes import TrainingMode
        config = ModelConfig.tiny()
        return tuple(InferenceEngine(
            TrainingMode(mode),
            GamePolicyTransformer(config), PieceConditionedLayoutPointerDecoder(config),
        ) for _ in range(2))

    def checkpoint(self, directory: Path, update: int, *, latest: bool = False,
                   nonfinite: bool = False, mode="two_player") -> Path:
        from junqi.training.arena import atomic_json
        from junqi.training.checkpoint import CHECKPOINT_FORMAT_VERSION
        from junqi.training.models import (
            GamePolicyTransformer, ModelConfig, PieceConditionedLayoutPointerDecoder,
        )
        config = ModelConfig.tiny()
        directory.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(update)
        policy = GamePolicyTransformer(config).state_dict()
        if nonfinite:
            next(iter(policy.values())).flatten()[0] = float("nan")
        metadata = {
            "format_version": CHECKPOINT_FORMAT_VERSION, "update": update,
            "mode": mode, "dead_rules_enabled": True,
        }
        path = directory / ("latest.pt" if latest else f"update_{update:09d}.pt")
        # Publish using replace just like the trainer; never mutate mmap data.
        temporary = path.with_suffix(".tmp")
        torch.save({
            **metadata, "config": {"model": asdict(config)}, "policy": policy,
            "layout": PieceConditionedLayoutPointerDecoder(config).state_dict(),
            "reference_policy": {"unused": torch.ones(2)},
            "policy_optimizer": {"unused": torch.ones(2)},
            "trainer_state": {"unused": 1},
        }, temporary)
        temporary.replace(path)
        if latest:
            atomic_json(directory / "manifest.json", {**metadata, "latest": "latest.pt", "reason": "test"})
        return path

    def config(self, pairs=1, recent=2, mode="two_player"):
        from junqi.training.arena import ARENA_VERSION, MatchSettings
        return {
            "arena_version": ARENA_VERSION,
            "match": asdict(MatchSettings(pairs=pairs, max_plies=4, smoke_test=True, mode=mode)),
            "every_updates": 5, "recent_opponents": recent,
        }


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch training extra is not installed")
class ArenaTests(ArenaFixtures, unittest.TestCase):
    def test_small_all_win_sample_does_not_prove_improvement(self):
        from junqi.training.arena import paired_statistics
        result = paired_statistics([1.0])
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertLessEqual(result["score_ci"][0], 0.5)
        self.assertEqual(result["bootstrap_ci95_descriptive"], [1.0, 1.0])

    def test_statistics_detect_leading_lagging_and_balanced_samples(self):
        from junqi.training.arena import paired_statistics
        for value, verdict in ((1, "ahead_of_opponent"), (0, "behind_opponent"), (0.5, "inconclusive")):
            result = paired_statistics([value] * 200)
            self.assertEqual(result["score"], value)
            self.assertEqual(result["verdict"], verdict)
        self.assertEqual(paired_statistics([0.25, 0.75]), paired_statistics([0.25, 0.75]))

    def test_invalid_samples_and_temperatures_rejected(self):
        from junqi.training.arena import MatchSettings, paired_statistics
        for values in ([], [float("nan")], [float("inf")], [-1], [1.1]):
            with self.assertRaises(ValueError):
                paired_statistics(values)
        for alpha in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                paired_statistics([0.5], alpha=alpha)
        for kwargs in ({"pairs": 0}, {"temperature": float("nan")}, {"max_plies": -1},
                       {"layout_temperature": 0}, {"seed": -1}):
            with self.assertRaises(ValueError):
                MatchSettings(**kwargs)

    def test_pair_is_reproducible_balanced_frozen_and_finishes(self):
        from junqi.training.arena import MatchSettings, play_pair
        candidate, opponent = self.engines()
        settings = MatchSettings(pairs=1, max_plies=6, smoke_test=True)
        original = [p.clone() for p in candidate.policy.parameters()]
        first = play_pair(candidate, opponent, 0, settings)
        second = play_pair(candidate, opponent, 0, settings)
        self.assertEqual(first, second)
        self.assertEqual([r["candidate_seat"] for r in first], [0, 1])
        self.assertEqual([r["plies"] for r in first], [6, 6])
        self.assertTrue(all(r["terminal_reason"] == "max_plies_draw" for r in first))
        for before, after in zip(original, candidate.policy.parameters()):
            torch.testing.assert_close(before, after)
            self.assertIsNone(after.grad)
            self.assertFalse(after.requires_grad)

    def test_models_keep_their_layouts_when_swapping_seats(self):
        from junqi.training.arena import MatchSettings, new_game, play_pair
        candidate, opponent = self.engines()
        with patch("junqi.training.arena.new_game", wraps=new_game) as create:
            play_pair(candidate, opponent, 0, MatchSettings(pairs=1, max_plies=2, smoke_test=True))
        a = create.call_args_list[0].kwargs["setups"]
        b = create.call_args_list[1].kwargs["setups"]
        self.assertIs(a[0], b[1])
        self.assertIs(a[1], b[0])

    def test_each_seat_dispatches_to_its_model_and_only_player_state(self):
        from junqi.training.arena import MatchSettings, play_pair
        from junqi.training.encoding import PolicyState
        candidate, opponent = self.engines()
        calls = []

        def sample(label, states, **kwargs):
            self.assertEqual(len(states), 1)
            self.assertIsInstance(states[0], PolicyState)
            calls.append(label)
            return [[states[0].legal_actions[0]]], []

        with patch.object(candidate.actor, "sample", side_effect=lambda states, **kw: sample("new", states, **kw)), \
                patch.object(opponent.actor, "sample", side_effect=lambda states, **kw: sample("old", states, **kw)):
            play_pair(candidate, opponent, 0, MatchSettings(pairs=1, max_plies=4, smoke_test=True))
        self.assertEqual(calls, ["new", "old", "new", "old", "old", "new", "old", "new"])

    def test_illegal_action_fails_instead_of_becoming_a_loss(self):
        from junqi.training.arena import MatchSettings, play_pair
        candidate, opponent = self.engines()
        with patch.object(candidate.actor, "sample", return_value=([[(999, 999)]], [])):
            with self.assertRaises(ValueError):
                play_pair(candidate, opponent, 0, MatchSettings(pairs=1, max_plies=2))

    def test_real_flag_capture_scores_follow_candidate_seat(self):
        from junqi import ArmPoint, GameConfig, GameVariant, InformationMode, JunqiGame, Piece, PieceType
        from junqi.training.arena import MatchSettings, play_pair, summarize_games
        candidate, opponent = self.engines()

        def position(_mode, **kwargs):
            pieces = {
                ArmPoint(0, 6, 2): Piece(0, PieceType.FLAG),
                ArmPoint(1, 6, 2): Piece(1, PieceType.FLAG),
                ArmPoint(1, 5, 2): Piece(0, PieceType.ENGINEER),
                ArmPoint(1, 2, 3): Piece(1, PieceType.COMMANDER),
            }
            return JunqiGame.from_position(GameConfig(
                variant=GameVariant.TWO_PLAYER, information_mode=InformationMode.DARK,
                max_plies=4, dead_rules_enabled=True,
            ), pieces)

        with patch("junqi.training.arena.new_game", side_effect=position), \
                patch.object(candidate.actor, "sample", return_value=([[(51, 56)]], [])), \
                patch.object(opponent.actor, "sample", return_value=([[(51, 56)]], [])):
            records = play_pair(candidate, opponent, 0, MatchSettings(pairs=1, max_plies=4))
        self.assertEqual([r["candidate_reward"] for r in records], [1, -1])
        self.assertEqual([r["winner_team"] for r in records], [0, 0])
        self.assertTrue(all(r["terminal_reason"] == "team_eliminated" for r in records))
        stats = summarize_games(records, MatchSettings(pairs=1), alpha=0.05)
        self.assertEqual((stats["wins"], stats["draws"], stats["losses"]), (1, 0, 1))

    def test_mode_and_rule_mismatch_are_rejected(self):
        from junqi.training.arena import validate_engines
        from junqi.training.modes import TrainingMode
        candidate, opponent = self.engines()
        opponent.mode = TrainingMode.FOUR_DARK
        with self.assertRaisesRegex(ValueError, "two_player"):
            validate_engines(candidate, opponent)
        opponent.mode = TrainingMode.TWO_PLAYER
        opponent.policy.config = replace(opponent.policy.config, dead_rules_enabled=False)
        with self.assertRaisesRegex(ValueError, "dead rules"):
            validate_engines(candidate, opponent)

    def test_score_and_ci_use_complete_pairs_not_individual_games(self):
        from junqi.training.arena import MatchSettings, summarize_games
        settings = MatchSettings(pairs=3)
        rewards = [1, -1, 0, 0, -1, 1]
        records = [{
            "pair_index": i // 2, "candidate_seat": i % 2,
            "candidate_reward": reward, "candidate_score": (reward + 1) / 2,
            "plies": 100, "terminal_reason": "no_interaction_draw" if reward == 0 else "flag_captured",
        } for i, reward in enumerate(rewards)]
        result = summarize_games(records, settings, alpha=0.05)
        self.assertEqual((result["pairs"], result["games"]), (3, 6))
        self.assertEqual((result["wins"], result["draws"], result["losses"]), (2, 2, 2))
        self.assertEqual(result["bootstrap_ci95_descriptive"], [0.5, 0.5])
        with self.assertRaisesRegex(ValueError, "unpaired"):
            summarize_games(records[:-1], settings, alpha=0.05)
        with self.assertRaisesRegex(ValueError, "unpaired"):
            summarize_games(records[:-1] + [records[0]], settings, alpha=0.05)

    def test_snapshot_is_finite_compact_immutable_and_loadable(self):
        from junqi.training.evaluate_history import pin_checkpoint, sha256_file
        from junqi.training.inference import InferenceEngine
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.checkpoint(root / "checkpoints", 10)
            source_hash = sha256_file(source)
            snapshot = pin_checkpoint(source, root / "eval")
            path = root / "eval" / snapshot["path"]
            payload = torch.load(path, mmap=True, map_location="cpu", weights_only=False)
            self.assertNotIn("reference_policy", payload)
            self.assertNotIn("policy_optimizer", payload)
            self.assertNotIn("trainer_state", payload)
            engine = InferenceEngine.from_checkpoint(path, device="cpu", temporal_cache_entries=8)
            self.assertEqual(engine.checkpoint_update, 10)
            self.assertEqual(engine.policy.config.inference_temporal_cache_entries, 8)
            self.assertFalse(engine.policy.training)
            self.assertEqual(sha256_file(source), source_hash)
            self.checkpoint(root / "checkpoints", 10)
            self.assertEqual(sha256_file(path), snapshot["sha256"])

    def test_nonfinite_and_racing_checkpoints_are_not_accepted(self):
        from junqi.training.evaluate_history import (
            PublicationInProgress, pin_checkpoint, read_manifest,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.checkpoint(root / "checkpoints", 10, latest=True, nonfinite=True)
            with self.assertRaisesRegex(ValueError, "NaN/Inf"):
                pin_checkpoint(source, root / "eval")
            source = self.checkpoint(root / "checkpoints", 10, latest=True)
            manifest = read_manifest(source.parent)
            with self.assertRaises(PublicationInProgress):
                pin_checkpoint(source, root / "eval", manifest={**manifest, "update": 9})
            with patch("junqi.training.evaluate_history.read_manifest", return_value={**manifest, "update": 11}):
                with self.assertRaises(PublicationInProgress):
                    pin_checkpoint(source, root / "eval", manifest=manifest)

    def test_manifest_validation_does_not_accept_non_two_player(self):
        from junqi.training.arena import atomic_json
        from junqi.training.evaluate_history import read_manifest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.checkpoint(root, 1, latest=True)
            data = read_manifest(root)
            for change in ({"mode": "four_dark"}, {"latest": "../latest.pt"}, {"update": True}):
                atomic_json(root / "manifest.json", {**data, **change})
                with self.assertRaises(ValueError):
                    read_manifest(root)

    def test_lock_released_on_close_and_duplicate_watcher_rejected(self):
        from junqi.training.evaluate_history import OutputLock
        with tempfile.TemporaryDirectory() as tmp:
            lock = OutputLock(Path(tmp))
            try:
                with self.assertRaisesRegex(RuntimeError, "another evaluator"):
                    OutputLock(Path(tmp))
            finally:
                lock.close()
            replacement = OutputLock(Path(tmp))
            replacement.close()

    def test_cadence_pending_resume_settings_and_snapshot_tamper(self):
        from junqi.training.evaluate_history import ArenaStore
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.checkpoint(root / "checkpoints", 10)
            store = ArenaStore(root / "eval", self.config(), baseline)
            self.checkpoint(root / "checkpoints", 14, latest=True)
            self.assertIn("idle", store.prepare(root / "checkpoints"))
            self.checkpoint(root / "checkpoints", 15, latest=True)
            job = store.prepare(root / "checkpoints")["job"]
            self.checkpoint(root / "checkpoints", 25, latest=True)
            reloaded = ArenaStore(root / "eval", self.config(), None)
            self.assertEqual(reloaded.prepare(root / "checkpoints")["job"], job)
            with self.assertRaisesRegex(ValueError, "settings changed"):
                ArenaStore(root / "eval", self.config(pairs=2), None)
            with self.assertRaisesRegex(ValueError, "unsafe snapshot"):
                reloaded.model_path({"path": "../checkpoints/latest.pt"})
            model = reloaded.model_path(job["candidate"])
            with model.open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                reloaded.verify(job["candidate"])

    def test_watch_waits_without_repeating_or_resuming_training(self):
        from junqi.training.evaluate_history import evaluate_history
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            baseline = self.checkpoint(root / "checkpoints", 10)
            latest = self.checkpoint(root / "checkpoints", 10, latest=True)
            with patch("junqi.training.evaluate_history.time.sleep", side_effect=KeyboardInterrupt), \
                    patch("junqi.training.evaluate_history.run_match") as match:
                with self.assertRaises(KeyboardInterrupt):
                    evaluate_history([
                        "--checkpoint-dir", str(latest.parent), "--output-dir", str(root / "eval"),
                        "--baseline", str(baseline), "--watch", "--device", "cpu",
                    ])
                match.assert_not_called()
            self.assertEqual(json.loads((root / "eval/status.json").read_text())["state"], "stopped")
            self.assertEqual(json.loads((root / "eval/state.json").read_text())["rounds"], [])

    def test_failed_evaluation_keeps_pending_job_and_never_publishes_a_score(self):
        from junqi.training.evaluate_history import evaluate_history
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            baseline = self.checkpoint(root / "checkpoints", 10)
            latest = self.checkpoint(root / "checkpoints", 20, latest=True)
            with patch("junqi.training.evaluate_history.run_match", side_effect=RuntimeError("illegal action")):
                with self.assertRaisesRegex(RuntimeError, "illegal action"):
                    evaluate_history([
                        "--checkpoint-dir", str(latest.parent), "--output-dir", str(root / "eval"),
                        "--baseline", str(baseline), "--every-updates", "5", "--device", "cpu",
                    ])
            self.assertEqual(json.loads((root / "eval/status.json").read_text())["state"], "failed")
            self.assertEqual(json.loads((root / "eval/state.json").read_text())["rounds"][0]["status"], "pending")
            self.assertFalse((root / "eval/latest.json").exists())
            self.assertTrue(latest.exists())

    def test_cli_one_round_idle_resume_and_second_opponent(self):
        from junqi.training.evaluate_history import sha256_file
        from junqi.training.evaluate_two_player import evaluate as evaluate_history
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            baseline = self.checkpoint(root / "checkpoints", 10)
            latest = self.checkpoint(root / "checkpoints", 20, latest=True)
            latest_hash = sha256_file(latest)
            arguments = [
                "--checkpoint-dir", str(latest.parent), "--output-dir", str(root / "eval"),
                "--baseline", str(baseline), "--device", "cpu", "--pairs", "1",
                "--max-plies", "4", "--smoke-test", "--every-updates", "5", "--once",
            ]
            first = evaluate_history(arguments)
            self.assertEqual(first["verdict"], "smoke_test_not_strength_evidence")
            self.assertEqual(first["matches"][0]["games"], 2)
            self.assertEqual(first["matches"][0]["candidate_update"], 20)
            self.assertEqual(sha256_file(latest), latest_hash)
            with patch("junqi.training.evaluate_history.run_match") as run:
                idle = evaluate_history(arguments)
                run.assert_not_called()
                self.assertEqual(idle["idle"], "waiting_for_new_checkpoint")
            self.checkpoint(root / "checkpoints", 25, latest=True)
            second = evaluate_history(arguments)
            self.assertEqual(len(second["matches"]), 2)
            self.assertEqual([o["update"] for o in second["opponents"]], [10, 20])
            self.assertNotEqual(first["matches"][0]["settings"]["seed"], second["matches"][0]["settings"]["seed"])
            self.assertLess(second["matches"][0]["alpha"], first["matches"][0]["alpha"])
            self.assertEqual(len(json.loads((root / "eval/history.json").read_text())), 2)

    def test_finished_matches_survive_interrupted_round_without_replay(self):
        from junqi.training.distributed import DistributedContext
        from junqi.training.evaluate_history import ArenaStore, execute_round
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            baseline = self.checkpoint(root / "checkpoints", 10)
            self.checkpoint(root / "checkpoints", 20, latest=True)
            store = ArenaStore(root / "eval", self.config(), baseline)
            job = store.prepare(root / "checkpoints")["job"]
            context = DistributedContext.initialize("cpu")
            execute_round(store, root / "eval", job, context)
            # Simulate crash after the per-opponent report was committed but
            # before round completion. Replaying must reuse that same report.
            store.state["rounds"][0]["status"] = "pending"
            store.save()
            reloaded = ArenaStore(root / "eval", self.config(), None)
            with patch("junqi.training.evaluate_history.run_match") as run:
                execute_round(reloaded, root / "eval", reloaded.prepare(root / "checkpoints")["job"], context)
                run.assert_not_called()
            self.assertEqual(len(json.loads((root / "eval/history.json").read_text())), 1)

    def test_snapshot_retention_keeps_baseline_and_never_touches_training(self):
        from junqi.training.evaluate_history import ArenaStore
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.checkpoint(root / "checkpoints", 10)
            store = ArenaStore(root / "eval", self.config(recent=1), baseline)
            pinned = []
            foreign = root / "eval/models/user_file.txt"
            foreign.write_text("keep")
            for update in (20, 30, 40):
                self.checkpoint(root / "checkpoints", update, latest=True)
                job = store.prepare(root / "checkpoints")["job"]
                pinned.append(store.model_path(job["candidate"]))
                store.complete(job, [{
                    "verdict": "inconclusive", "candidate_update": update,
                    "opponent_update": opponent["update"], "score_ci": [0, 1],
                    "games": 2, "wins": 0, "draws": 2, "losses": 0, "score": 0.5,
                } for opponent in job["opponents"]])
            self.assertTrue(store.model_path(store.state["baseline"]).exists())
            self.assertFalse(pinned[0].exists())
            self.assertFalse(pinned[1].exists())
            self.assertTrue(pinned[2].exists())
            self.assertTrue(foreign.exists())
            self.assertTrue(baseline.exists())
            self.assertTrue((root / "checkpoints/latest.pt").exists())

    def test_cli_rejects_overlapping_output_and_checkpoint_directories(self):
        from junqi.training.evaluate_history import evaluate_history
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(tmp)
            for output in (root, root / "eval", root.parent):
                with self.assertRaises(SystemExit):
                    evaluate_history(["--checkpoint-dir", str(root), "--output-dir", str(output)])

    @unittest.skipIf(os.name == "nt", "distributed CPU integration is exercised under WSL/Linux")
    def test_two_rank_gloo_matches_single_process_with_uneven_pairs(self):
        from junqi.training.evaluate_two_player import evaluate as evaluate_history
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            baseline = self.checkpoint(root / "checkpoints", 10)
            latest = self.checkpoint(root / "checkpoints", 20, latest=True)
            common = [
                "--checkpoint-dir", str(latest.parent), "--baseline", str(baseline),
                "--device", "cpu", "--pairs", "3", "--max-plies", "4",
                "--smoke-test", "--every-updates", "5", "--once",
            ]
            single = evaluate_history(common + ["--output-dir", str(root / "single")])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
            result = subprocess.run([
                sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2",
                "--module", "junqi.training.evaluate_two_player", *common,
                "--output-dir", str(root / "distributed"),
            ], env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            distributed = json.loads((root / "distributed/latest.json").read_text())
            self.assertEqual(distributed["matches"][0]["world_size"], 2)
            self.assertEqual(distributed["matches"][0]["games"], 6)
            for key in ("score", "score_ci", "bootstrap_ci95_descriptive", "wins", "draws", "losses"):
                self.assertEqual(single["matches"][0][key], distributed["matches"][0][key])
            def read_games(directory):
                return sorted([
                    json.loads(line) for path in directory.glob("matches/*.jsonl")
                    for line in path.read_text().splitlines()
                ], key=lambda record: (record["pair_index"], record["candidate_seat"]))
            self.assertEqual(read_games(root / "single"), read_games(root / "distributed"))

    @unittest.skipIf(os.name == "nt", "Bash launcher is exercised under WSL/Linux")
    def test_npu_launcher_selects_history_or_legacy_evaluator_without_starting_npu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cann = root / "set_env.sh"
            cann.write_text(":\n")
            capture = root / "capture.sh"
            capture.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
            capture.chmod(0o700)
            script = Path(__file__).resolve().parents[1] / "scripts/evaluate_npu_cluster.sh"
            env = dict(os.environ, CANN_ENV_FILE=str(cann), PYTHON_BIN=str(capture),
                       NNODES="2", NODE_RANK="1", MASTER_ADDR="10.1.2.3", MASTER_PORT="29511",
                       NPROC_PER_NODE="2", EVAL_OUTPUT_DIR=str(root / "eval"),
                       EVAL_BASELINE=str(root / "old.pt"))
            for mode, module in (("history", "evaluate_two_player"), ("self_play", "evaluate")):
                result = subprocess.run([
                    "bash", str(script), str(root / "checkpoints/latest.pt"),
                ], env={**env, "EVAL_MODE": mode}, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                arguments = result.stdout.splitlines()
                self.assertEqual(arguments[arguments.index("--module") + 1], f"junqi.training.{module}")
                self.assertEqual(arguments[arguments.index("--nproc-per-node") + 1], "2")
                if mode == "history":
                    self.assertIn("--checkpoint-dir", arguments)
                    self.assertIn("--pairs", arguments)
                    self.assertNotIn("--games", arguments)
                else:
                    self.assertIn("--checkpoint", arguments)
                    self.assertIn("--games", arguments)


if __name__ == "__main__":
    unittest.main()
