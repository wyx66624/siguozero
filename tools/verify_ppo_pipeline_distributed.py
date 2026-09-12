"""Bounded two-rank CPU PPO, scheduled arena saves, and checkpoint resume.

Run with torch.distributed.run --standalone --nproc-per-node=2. This writes
tiny-model evaluation checkpoints only inside the explicitly supplied new run.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from junqi.training.distributed import DistributedContext
from junqi.training.metrics import MetricLogger
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--environment-workers", type=int, default=2)
    parser.add_argument('--optimized', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    context = DistributedContext.initialize("cpu")
    assert context.world_size == 2
    root = Path(__file__).resolve().parents[1]
    settings = TrainingSettings.from_yaml(root / "configs/bootstrap.yaml", "four_dark", tiny=True, overrides={
        "device": "cpu", "total_updates": 2, "target_environment_plies": 8,
        "anchor_batch": 4, "base_game_pool_size": 4, "actor_inference_batch": 2,
        "policy_microbatch": 2, "ppo_minibatch_samples": 2, "max_game_plies": 64,
        "rollout_environment_workers": args.environment_workers,
        "arena_enabled": True, "arena_games": 4, "arena_max_plies": 4,
        "arena_parallel_games": 2, "arena_inference_batch_size": 2,
        "arena_environment_workers": 1, "arena_interval_environment_plies": 4,
        "checkpoint_policy": "evaluation",
    })
    if args.optimized:
        settings = replace(settings, ppo_deferred_values=True, ppo_pipeline_groups=2,
            ppo_fused_optimizer=True, layout_prefetch_games=4,
            model=replace(settings.model, ppo_tensor_learner=True, ppo_varlen_attention=True))
        settings.validate()
    save_reasons = []
    try:
        with patch.object(MetricLogger, "start_resource_monitor"):
            trainer = SelfPlayTrainer(settings, run_directory=args.run_dir, distributed=context, auto_resume=False)
        context.barrier()
        assert not list(trainer.run_directory.rglob("*.pt"))
        evaluate = trainer._maybe_evaluate_model

        def stop_after_first():
            result = evaluate()
            if trainer.update == 1:
                trainer.stop_requested = True
            return result

        with patch.object(trainer, "_maybe_evaluate_model", side_effect=stop_after_first), \
                patch.object(trainer, "save_checkpoint", wraps=trainer.save_checkpoint) as saves:
            trainer.train()
            save_reasons.extend(call.kwargs["reason"] for call in saves.call_args_list)
        assert trainer.update == 1 and trainer.cumulative["environment_plies"] == 4
        checkpoint = trainer.checkpoints.latest_path
        with patch.object(MetricLogger, "start_resource_monitor"):
            resumed = SelfPlayTrainer(settings, run_directory=args.run_dir, distributed=context)
        assert resumed.update == 1 and resumed.cumulative["environment_plies"] == 4
        assert resumed.pool.state_dict() == trainer.pool.state_dict()
        for name in ("policy", "critic", "layout"):
            for key, expected in getattr(trainer, name).state_dict().items():
                torch.testing.assert_close(getattr(resumed, name).state_dict()[key], expected, rtol=0, atol=0)
        with patch.object(resumed, "save_checkpoint", wraps=resumed.save_checkpoint) as saves:
            resumed.train()
            save_reasons.extend(call.kwargs["reason"] for call in saves.call_args_list)
        assert resumed.update == 2 and resumed.cumulative["environment_plies"] == 8
        assert save_reasons == ["before_model_selection", "after_model_selection"] * 2
        digest = hashlib.sha256()
        for name in ("policy", "critic"):
            for value in getattr(resumed, name).state_dict().values():
                digest.update(value.detach().cpu().numpy().tobytes())
        hashes = context.gather_object(digest.hexdigest())
        if context.primary:
            assert len(set(hashes)) == 1
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert len(payload["trainer_state"]["distributed"]["rank_states"]) == 2
            result = dict(world_size=2, backend=context.backend, environment_steps=8,
                          optimized=args.optimized, deferred_values=settings.ppo_deferred_values,
                          pipeline_groups=settings.ppo_pipeline_groups,
                          environment_workers_per_rank=args.environment_workers,
                          completed_update=2, resumed_update=1, saves=save_reasons,
                          checkpoints_only_at_evaluation=True, ranks_have_identical_parameters=True,
                          pool_and_model_resume_exact=True, scheduled_evaluations=2,
                          checkpoint=str(checkpoint), complete=True)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result), flush=True)
        context.barrier()
    finally:
        context.close()


if __name__ == "__main__":
    main()
