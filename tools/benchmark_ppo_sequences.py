"""Bounded real PPO timing and exact-prefix learner A/B on the local GPU."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import os
import tempfile
import time
from unittest.mock import patch

import torch

from benchmark_cuda import initialize_optimizer_state
from benchmark_ppo import diverse_states
from benchmark_ppo_timing import TimingTrainer
from junqi.training.accelerator import (
    empty_cache, get_device_name, max_memory_allocated, max_memory_reserved,
    reset_peak_memory_stats, synchronize,
)
from junqi.training.metrics import MetricLogger
from junqi.training.ppo import FrozenValueActor, PPOSample
from junqi.training.rollout import FrozenPolicyActor
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer


class ProbeTrainer(TimingTrainer):
    def _normalize_ppo_batch(self, samples):
        result = super()._normalize_ppo_batch(samples)
        self.last_samples = result
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("four_dark", "double_open"), default="four_dark")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--transitions", type=int, default=4096)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument("--skip-learner-comparison", action="store_true")
    parser.add_argument("--keep-checkpoint", action="store_true")
    parser.add_argument("--context-tokens", type=int, default=1001)
    parser.add_argument("--capacity", action="store_true")
    parser.add_argument("--capacity-only", action="store_true",
                        help="Run long-history capacity checks without a training loop")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.games, args.transitions, args.updates, args.microbatch) <= 0:
        parser.error("counts must be positive")
    if not 0 < args.memory_fraction <= 1:
        parser.error("memory fraction must be in (0, 1]")
    if (args.capacity or args.capacity_only) and not 253 <= args.context_tokens <= 1001:
        parser.error("capacity checks require 253..1001 context tokens")
    # Concurrent probes invalidate wall-clock measurements and can exhaust VRAM.
    # Keep the descriptor alive until this process exits; no stale lock on failure.
    probe_lock = open(Path(tempfile.gettempdir()) / "siguozero-cuda-probe.lock", "a+b")
    if os.name == "posix":
        import fcntl
        try:
            fcntl.flock(probe_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another PPO GPU probe holds the exclusive benchmark lock")
    else:
        import msvcrt
        probe_lock.seek(0)
        probe_lock.write(b"0")
        probe_lock.flush()
        probe_lock.seek(0)
        try:
            msvcrt.locking(probe_lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            parser.error("another PPO GPU probe holds the exclusive benchmark lock")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    settings = TrainingSettings.from_yaml(
        Path(__file__).parents[1] / "configs/bootstrap.yaml", args.mode, model_scale="main",
        overrides={"device": "cuda", "base_game_pool_size": args.games,
                   "actor_inference_batch": args.games, "anchor_batch": args.transitions,
                   "policy_microbatch": args.microbatch, "ppo_minibatch_samples": 512,
                   "ppo_sequence_training": True, "ppo_max_samples_per_sequence": 64,
                   "arena_enabled": False},
    )
    settings = replace(settings, model=replace(settings.model, inference_temporal_cache_entries=12 * args.games))
    with patch.object(MetricLogger, "start_resource_monitor"):
        trainer = ProbeTrainer(settings, run_directory=args.run_dir, auto_resume=False)
    trainer.timing_histories = []
    result = {"device": get_device_name(trainer.device), "torch": torch.__version__,
              "config_revision": settings.raw_config["config_revision"],
              "model_config": asdict(settings.model), "amp": settings.amp,
              "requested_updates": args.updates,
              "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                 for name in ("src/junqi/training/models.py", "src/junqi/training/encoding.py",
                                              "src/junqi/training/trainer.py", "configs/bootstrap.yaml",
                                              "tools/benchmark_ppo_sequences.py")},
              "mode": args.mode, "model_scale": "main", "sequence_microbatch": args.microbatch,
              "memory_fraction": args.memory_fraction,
              "process_id": os.getpid(), "exclusive_probe_lock": str(probe_lock.name),
              "samples_per_sequence_limit": 64, "optimizer_minibatch_samples": 512,
              "transitions_per_update": args.transitions, "games": args.games,
              "actor_batch": args.games, "cache_entries_per_model": 12 * args.games,
              "weights": "isolated fresh random weights; not a strength benchmark", "phases": []}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def measure(name, action):
        empty_cache(trainer.device)
        reset_peak_memory_stats(trainer.device)
        synchronize(trainer.device)
        started = time.perf_counter()
        try:
            extra = action() or {}
        except Exception as error:
            result["failed_phase"] = name
            result["error"] = f"{type(error).__name__}: {error}"
            persist()
            raise
        synchronize(trainer.device)
        phase = {"phase": name, "seconds": time.perf_counter() - started,
                 "peak_allocated_gib": max_memory_allocated(trainer.device) / 2**30,
                 "peak_reserved_gib": max_memory_reserved(trainer.device) / 2**30, **extra}
        result["phases"].append(phase)
        print(json.dumps(phase), flush=True)
        persist()

    def measure_capacity():
        trainer.settings = settings
        longest = diverse_states(settings.mode, settings.model, args.context_tokens, 8)
        # 64 decision positions in each of 8 independent player histories.
        capacity_samples = [
            PPOSample(replace(state, records=state.records[:length]), state.legal_actions[0],
                      -4., 0., .5 if index % 2 else -.5, .5 if index % 2 else -.5, 0)
            for state in longest
            for index, length in enumerate(range(args.context_tokens - 4 * 63, args.context_tokens + 1, 4))
        ]
        for name, module, optimizer, backward in (
            ("policy", trainer.policy, trainer.policy_optimizer, trainer._backward_policy_epoch),
            ("critic", trainer.critic, trainer.critic_optimizer, trainer._backward_critic_epoch),
        ):
            module.train()
            measure(f"long_history_packed_{name}", lambda backward=backward: {
                "metrics": backward(capacity_samples), "samples": len(capacity_samples),
                "sequences": 8, "context_tokens": args.context_tokens})
            optimizer.zero_grad(set_to_none=True)
        states = diverse_states(settings.mode, settings.model, args.context_tokens, 4 * args.games)

        def rollout():
            actor = FrozenPolicyActor(trainer.policy.eval(), amp_dtype=trainer.amp_dtype, max_batch_size=args.games)
            critic = FrozenValueActor(trainer.critic, amp_dtype=trainer.amp_dtype, max_batch_size=args.games)
            for start in range(0, len(states), args.games):
                chunk = states[start:start + args.games]
                actor.sample(chunk)
                critic.values(chunk)
            return {"cached_player_histories": len(states), "context_tokens": args.context_tokens,
                    "effective_actor_batch": actor.max_batch_size,
                    "effective_critic_batch": critic.max_batch_size}

        measure("long_history_dual_model_rollout", rollout)

    try:
        for module, optimizer in ((trainer.policy, trainer.policy_optimizer),
                                  (trainer.critic, trainer.critic_optimizer),
                                  (trainer.layout, trainer.layout_optimizer)):
            initialize_optimizer_state(module, optimizer)
        if args.capacity_only:
            result["capacity_only"] = True
            measure_capacity()
            return
        trainer.settings = replace(settings, target_environment_plies=args.updates * args.transitions)
        measure("real_training_loop_without_checkpoint", lambda: trainer.train())
        result["real_updates"] = [json.loads(line) for line in trainer.logger.jsonl_path.read_text().splitlines()]
        result["real_history_batches"] = trainer.timing_histories
        result["effective_sequence_microbatch_after_training"] = trainer.effective_policy_microbatch

        def save():
            path = SelfPlayTrainer.save_checkpoint(trainer, reason="sequence_timing_probe", archive=False)
            size = path.stat().st_size
            if not args.keep_checkpoint:
                path.unlink()
            return {"bytes": size, "gib": size / 2**30,
                    "checkpoint": str(path), "removed_disposable_checkpoint": not args.keep_checkpoint}
        measure("checkpoint_to_workspace_drive", save)

        trainer.settings = settings
        samples = list(trainer._ppo_optimizer_batches(trainer.last_samples)[0][:128])
        result["comparison"] = {"samples": len(samples),
                                "mean_history_tokens": sum(len(x.state.records) for x in samples) / len(samples),
                                "min_history_tokens": min(len(x.state.records) for x in samples),
                                "max_history_tokens": max(len(x.state.records) for x in samples)}
        for packed in (() if args.skip_learner_comparison else (False, True)):
            trainer.settings = replace(settings, ppo_sequence_training=packed)
            for name, module, optimizer, backward in (
                ("policy", trainer.policy, trainer.policy_optimizer, trainer._backward_policy_epoch),
                ("critic", trainer.critic, trainer.critic_optimizer, trainer._backward_critic_epoch),
            ):
                module.train()
                measure(f"same_real_samples_{name}_sequence_{packed}", lambda backward=backward: {"metrics": backward(samples)})
                optimizer.zero_grad(set_to_none=True)

        if args.capacity:
            measure_capacity()
        persist()
    finally:
        trainer.logger.close()


if __name__ == "__main__":
    main()
