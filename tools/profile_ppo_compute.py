"""Bounded PPO CPU/CUDA audit on real trajectories, without production saves.

The profiler's matmul FLOPs omit fused attention and many elementwise operators.
CUDA event durations are kernel/copy service times, not tensor-core utilization.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch

from benchmark_cuda import initialize_optimizer_state
from benchmark_ppo_sequences import ProbeTrainer
from junqi.training import models
from junqi.training.metrics import MetricLogger
from junqi.training.settings import TrainingSettings


def block_flops(batch, tokens, width, hidden):
    # Q/K/V/output projections, SwiGLU's three matrices, dense QK/AV.
    return batch * (8 * tokens * width**2 + 6 * tokens * width * hidden
                    + 4 * tokens**2 * width)


@contextmanager
def count_blocks(module):
    counts = defaultdict(lambda: {"calls": 0, "batch_tokens": 0,
                                  "estimated_forward_flops": 0})
    handles = []
    for name, child in module.named_modules():
        if name == "board_encoder.projection" and isinstance(child, torch.nn.Linear):
            def board_hook(layer, args):
                boards = args[0].shape[0]
                item = counts["board"]
                item["calls"] += 1
                item["batch_tokens"] += boards
                item["estimated_forward_flops"] += 2 * boards * layer.in_features * layer.out_features
            handles.append(child.register_forward_pre_hook(board_hook))
        if isinstance(child, models.PreNormEncoderBlock):
            kind = "board" if name.startswith("board_encoder.") else "temporal"

            def hook(block, args, kind=kind):
                b, s, d = args[0].shape
                item = counts[kind]
                item["calls"] += 1
                item["batch_tokens"] += b * s
                item["estimated_forward_flops"] += block_flops(
                    b, s, d, block.ffn.down.in_features)
            handles.append(child.register_forward_pre_hook(hook))
    try:
        yield counts
    finally:
        for handle in handles:
            handle.remove()


def summarize_profiler(prof):
    rows = [{"name": item.key, "calls": item.count,
             "self_cpu_ms": item.self_cpu_time_total / 1000,
             "self_cuda_ms": getattr(item, "self_device_time_total", 0) / 1000,
             "recognized_flops": item.flops or 0}
            for item in prof.key_averages()]
    cuda_events = [event for event in prof.events()
                   if event.device_type == torch.autograd.DeviceType.CUDA]
    intervals = sorted((e.time_range.start, e.time_range.end) for e in cuda_events)
    union_us = 0
    end = float("-inf")
    for start, stop in intervals:
        union_us += max(0, stop - max(start, end))
        end = max(end, stop)
    return {"recognized_flops_lower_bound": sum(row["recognized_flops"] for row in rows),
            "cuda_kernel_copy_events": len(cuda_events),
            "cuda_event_service_seconds": sum(stop - start for start, stop in intervals) / 1e6,
            "cuda_event_union_seconds": union_us / 1e6,
            "top_cpu": sorted(rows, key=lambda x: x["self_cpu_ms"], reverse=True)[:25],
            "top_cuda": sorted(rows, key=lambda x: x["self_cuda_ms"], reverse=True)[:25]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--transitions", type=int, default=4096)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--profile-samples", type=int, default=128)
    parser.add_argument("--checkpoint-ablation-samples", type=int, default=0,
                        help="opt-in small subset; 0 skips checkpoint-off memory stress")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.games, args.transitions, args.updates, args.profile_samples) <= 0:
        parser.error("counts must be positive")
    if args.checkpoint_ablation_samples < 0:
        parser.error("checkpoint ablation count must be nonnegative")
    torch.set_num_threads(4)
    # Fail with an allocator OOM before WDDM/WSL can silently page GPU memory.
    torch.cuda.set_per_process_memory_fraction(0.85)
    root = Path(__file__).resolve().parents[1]
    settings = TrainingSettings.from_yaml(
        root / "configs/bootstrap.yaml", "four_dark", model_scale="main",
        overrides={"device": "cuda", "base_game_pool_size": args.games,
                   "actor_inference_batch": args.games, "anchor_batch": args.transitions,
                   "policy_microbatch": 8, "ppo_minibatch_samples": 512,
                   "ppo_sequence_training": True, "ppo_max_samples_per_sequence": 64,
                   "arena_enabled": False},
    )
    settings = replace(settings, target_environment_plies=args.updates * args.transitions,
                       model=replace(settings.model, inference_temporal_cache_entries=12 * args.games))
    with patch.object(MetricLogger, "start_resource_monitor"):
        trainer = ProbeTrainer(settings, run_directory=args.run_dir, auto_resume=False)
    trainer.timing_histories = []
    result = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "games": args.games, "transitions_per_update": args.transitions,
              "updates": args.updates, "weights": "fresh random, no strength conclusion",
              "schedule_total_updates": settings.total_updates,
              "schedule_warmup_updates": settings.warmup_updates,
              "allocator_memory_fraction": 0.85,
              "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in ("src/junqi/training/models.py", "src/junqi/training/trainer.py",
                                             "src/junqi/training/encoding.py", "src/junqi/training/ppo.py")},
              "parameters": {}, "phases": []}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def measure(name, action, module=None):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if module is not None:
            module.reset_board_encoding_stats()
        start = time.perf_counter()
        with count_blocks(module) if module is not None else count_blocks(torch.nn.Identity()) as counts:
            extra = action() or {}
        torch.cuda.synchronize()
        phase = {"phase": name, "seconds": time.perf_counter() - start,
                 "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "blocks_including_checkpoint_replays": dict(counts), **extra}
        if module is not None:
            phase["encoding"] = module.board_encoding_metrics()
        result["phases"].append(phase)
        print(json.dumps(phase), flush=True)
        persist()

    try:
        for name, module, optimizer in (("policy", trainer.policy, trainer.policy_optimizer),
                                        ("critic", trainer.critic, trainer.critic_optimizer),
                                        ("layout", trainer.layout, trainer.layout_optimizer)):
            result["parameters"][name] = {
                "total": sum(p.numel() for p in module.parameters()),
                "children": {n: sum(p.numel() for p in m.parameters()) for n, m in module.named_children()}}
            initialize_optimizer_state(module, optimizer)
        measure("real_training_without_checkpoint", lambda: trainer.train())
        result["real_updates"] = [json.loads(line) for line in trainer.logger.jsonl_path.read_text().splitlines()]
        result["real_history_batches"] = trainer.timing_histories
        samples = list(trainer._ppo_optimizer_batches(trainer.last_samples)[0])
        result["sample_batch"] = {"samples": len(samples),
                                  "mean_history": sum(len(s.state.records) for s in samples) / len(samples),
                                  "min_history": min(len(s.state.records) for s in samples),
                                  "max_history": max(len(s.state.records) for s in samples)}
        for name, module, optimizer, backward in (
            ("policy", trainer.policy, trainer.policy_optimizer, trainer._backward_policy_epoch),
            ("critic", trainer.critic, trainer.critic_optimizer, trainer._backward_critic_epoch),
        ):
            module.train()
            measure(f"{name}_one_epoch_{len(samples)}", lambda: {"metrics": backward(samples)}, module)
            optimizer.zero_grad(set_to_none=True)

        small = samples[:args.profile_samples]
        result["profile_sample_batch"] = {
            "samples": len(small), "mean_history": sum(len(s.state.records) for s in small) / len(small),
            "min_history": min(len(s.state.records) for s in small),
            "max_history": max(len(s.state.records) for s in small)}
        measure("policy_profile_subset_uninstrumented", lambda: {
            "metrics": trainer._backward_policy_epoch(small)}, trainer.policy)
        trainer.policy_optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, with_flops=True) as prof:
            measure("policy_profile_subset_instrumented", lambda: {
                "metrics": trainer._backward_policy_epoch(small)}, trainer.policy)
        result["profiler"] = summarize_profiler(prof)
        persist()
        trainer.policy_optimizer.zero_grad(set_to_none=True)
        del prof
        gc.collect()
        torch.cuda.empty_cache()

        # Same real subset and parameters, only the memory/recompute tradeoff differs.
        if not args.checkpoint_ablation_samples:
            result["checkpoint_off_subset"] = "skipped; explicit opt-in only"
            result["status"] = "completed"
            persist()
            return
        small = samples[:args.checkpoint_ablation_samples]
        result["checkpoint_ablation_sample_count"] = len(small)
        measure("policy_checkpoint_ablation_on", lambda: {
            "metrics": trainer._backward_policy_epoch(small)}, trainer.policy)
        trainer.policy_optimizer.zero_grad(set_to_none=True)
        previous = trainer.policy.config
        previous_board = trainer.policy.board_encoder.config
        try:
            trainer.policy.config = replace(previous, activation_checkpointing=False)
            trainer.policy.board_encoder.config = replace(previous_board, activation_checkpointing=False)
            measure("policy_same_subset_checkpoint_off", lambda: {
                "metrics": trainer._backward_policy_epoch(small)}, trainer.policy)
        except torch.cuda.OutOfMemoryError:
            result["checkpoint_off_subset"] = "OOM; no production setting changed"
            persist()
        finally:
            trainer.policy.config = previous
            trainer.policy.board_encoder.config = previous_board
            trainer.policy_optimizer.zero_grad(set_to_none=True)
        result["status"] = "completed"
        persist()
    finally:
        trainer.logger.close()


if __name__ == "__main__":
    main()
