"""Bounded PPO timing: real updates plus independent-history cost estimates.

Uses fresh random weights in an isolated run directory. Automatic checkpoints
are disabled for this probe; --measure-checkpoint times one real save and then
removes that disposable checkpoint. Context sweeps are synthetic workload
measurements, not legal trajectories or evidence of playing strength.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch

from benchmark_cuda import initialize_optimizer_state
from benchmark_ppo import diverse_states
from junqi.training.accelerator import (
    empty_cache, get_device_name, max_memory_allocated, reset_peak_memory_stats,
    synchronize,
)
from junqi.training.metrics import MetricLogger
from junqi.training.ppo import PPOSample
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer


class TimingTrainer(SelfPlayTrainer):
    def save_checkpoint(self, *, reason, archive):
        # Never create a resumable production run from this timing experiment.
        return self.checkpoints.latest_path

    def _normalize_ppo_batch(self, samples):
        lengths = [len(sample.state.records) for sample in samples]
        self.timing_histories.append({
            "update": self.update + 1, "mean_tokens": statistics.mean(lengths),
            "min_tokens": min(lengths), "max_tokens": max(lengths),
        })
        print(json.dumps({"real_batch": self.timing_histories[-1]}), flush=True)
        return super()._normalize_ppo_batch(samples)


def estimate_budgets(result):
    """Extrapolate measured backward work, keeping unmeasured overhead explicit."""
    phases = {item["phase"]: item for item in result["phases"]}
    tokens = sorted(item["context_tokens"] for item in result["phases"]
                    if item["phase"].startswith("policy_accumulated_backward_"))
    # Collection, checkpoint I/O and amortized layout/logging. Opening updates
    # measured about 9-10 s collection plus 11 s saving; long games are not
    # measured here, so this is an explicit planning allowance, not a bound.
    overhead = [20, 45]
    availability = .8
    estimates = []
    for context in tokens:
        learner_seconds = 0.0
        for name in ("policy", "critic"):
            phase = phases[f"{name}_accumulated_backward_{context}_tokens"]
            step = phases[f"{name}_optimizer_step_{context}_tokens"]["seconds"]
            learner_seconds += result[f"{name}_epochs"] * (
                phase["seconds"] * result["transitions_per_update"] / phase["samples"] + step
            )
        seconds = [learner_seconds + value for value in overhead]
        budgets = {}
        for moves in (100_000, 1_000_000, 5_000_000, 10_000_000, result["nominal_total_real_moves"]):
            updates = math.ceil(moves / result["transitions_per_update"])
            budgets[str(moves)] = [updates * value / (86400 * availability) for value in seconds]
        estimates.append({
            "context_tokens": context, "learner_seconds_per_update": learner_seconds,
            "estimated_seconds_per_update_range": seconds,
            "estimated_moves_per_calendar_day_range": [
                result["transitions_per_update"] * 86400 * availability / value
                for value in reversed(seconds)
            ],
            "estimated_calendar_days_by_real_move_budget": budgets,
        })
    return {"training_fraction_of_calendar_time": availability,
            "assumed_overhead_seconds_per_update_range": overhead,
            "limitations": "Synthetic context sweep extrapolation; long-run game lengths, late-game collection and strength convergence were not measured. Ranges are scenarios, not confidence intervals.",
            "by_context": estimates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("four_dark", "double_open"), default="four_dark")
    parser.add_argument("--real-updates", type=int, default=2)
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 256, 512, 1001])
    parser.add_argument("--probe-microbatches", type=int, default=4)
    parser.add_argument("--measure-checkpoint", action="store_true")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.real_updates < 1 or args.probe_microbatches < 1:
        parser.error("update and probe counts must be positive")
    torch.set_num_threads(4)
    settings = TrainingSettings.from_yaml(
        Path(__file__).parents[1] / "configs/bootstrap.yaml", args.mode,
        model_scale="main", dead_rules_enabled=True,
        overrides={"device": "cuda", "anchor_batch": 512,
                   "policy_microbatch": 8, "base_game_pool_size": 8,
                   "actor_inference_batch": 8, "checkpoint_every_updates": 1,
                   "ppo_sequence_training": False, "arena_enabled": False},
    )
    if any(not 1 <= size <= settings.model.max_sequence_tokens for size in args.contexts):
        parser.error("contexts must fit the configured history window")
    with patch.object(MetricLogger, "start_resource_monitor"):
        trainer = TimingTrainer(settings, run_directory=args.run_dir, auto_resume=False)
    trainer.timing_histories = []
    result = {
        "device": get_device_name(trainer.device), "torch": torch.__version__,
        "mode": args.mode, "model_scale": "main", "dead_rules_enabled": True,
        "microbatch": 8, "games": 8, "actor_batch": 8, "transitions_per_update": 512,
        "policy_epochs": settings.policy_epochs, "critic_epochs": settings.critic_epochs,
        "nominal_total_updates": settings.total_updates,
        "nominal_total_real_moves": settings.total_updates * settings.anchor_batch,
        "weights": "fresh random; no existing run loaded or changed",
        "context_sweep_kind": "synthetic independent histories; one accumulated epoch",
        "phases": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def measure(name, action):
        empty_cache(trainer.device)
        reset_peak_memory_stats(trainer.device)
        synchronize(trainer.device)
        started = time.perf_counter()
        extra = action() or {}
        synchronize(trainer.device)
        phase = {"phase": name, "seconds": time.perf_counter() - started,
                 "peak_allocated_gib": max_memory_allocated(trainer.device) / 2**30,
                 **extra}
        result["phases"].append(phase)
        print(json.dumps(phase), flush=True)
        persist()
        return phase

    try:
        for module, optimizer in (
            (trainer.policy, trainer.policy_optimizer),
            (trainer.critic, trainer.critic_optimizer),
            (trainer.layout, trainer.layout_optimizer),
        ):
            initialize_optimizer_state(module, optimizer)
        # Retain the production LR schedule and stop at a bounded move budget.
        trainer.settings = replace(settings, target_environment_plies=args.real_updates * 512)
        measure("real_training_loop_without_checkpoint", lambda: (trainer.train() or {}))
        result["real_history_batches"] = trainer.timing_histories
        result["real_updates"] = [json.loads(line) for line in
                                  (trainer.run_directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        if args.measure_checkpoint:
            def checkpoint():
                path = SelfPlayTrainer.save_checkpoint(trainer, reason="timing_probe", archive=False)
                size = path.stat().st_size
                # The exact file returned by this isolated probe is disposable.
                path.unlink()
                return {"bytes": size, "gib": size / 2**30,
                        "removed_disposable_checkpoint": True}
            measure("checkpoint_to_workspace_drive", checkpoint)

        trainer.settings = settings
        for tokens in args.contexts:
            count = settings.policy_microbatch * args.probe_microbatches
            states = diverse_states(settings.mode, settings.model, tokens, count)
            trainer.policy.eval()
            old_logs = []
            with torch.no_grad(), trainer._autocast():
                for start in range(0, count, settings.policy_microbatch):
                    batch = states[start:start + settings.policy_microbatch]
                    logs = trainer.policy(batch, [state.legal_actions for state in batch])
                    old_logs.extend(float(row[0]) for row in logs)
            samples = [PPOSample(
                state=state, action=state.legal_actions[0], old_log_prob=log,
                old_value=0, advantage=1 if index % 2 else -1,
                value_target=1 if index % 2 else -1, behavior_version=trainer.update,
            ) for index, (state, log) in enumerate(zip(states, old_logs, strict=True))]
            for name, module, optimizer, backward in (
                ("policy", trainer.policy, trainer.policy_optimizer, trainer._backward_policy_epoch),
                ("critic", trainer.critic, trainer.critic_optimizer, trainer._backward_critic_epoch),
            ):
                module.train()
                measure(f"{name}_accumulated_backward_{tokens}_tokens", lambda backward=backward: {
                    "metrics": backward(samples), "samples": count,
                    "microbatches": args.probe_microbatches, "context_tokens": tokens,
                })
                def step(module=module, optimizer=optimizer):
                    torch.nn.utils.clip_grad_norm_(module.parameters(), 1, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                measure(f"{name}_optimizer_step_{tokens}_tokens", step)
        result["estimates"] = estimate_budgets(result)
        persist()
    finally:
        trainer.logger.close()


if __name__ == "__main__":
    main()
