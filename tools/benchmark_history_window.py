"""Bounded four-dark context-window diagnostic with fresh random weights.

Synthetic fixed histories isolate sequence length and loss of exact-prefix reuse.
This is not self-play, a complete PPO update, or an estimate of playing strength.
No training configuration or existing checkpoint is changed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time

import torch

from benchmark_ppo import diverse_states
from junqi.training.encoding import history_prefix_groups
from junqi.training.models import GamePolicyTransformer
from junqi.training.modes import TrainingMode
from junqi.training.ppo import PPOSample, policy_ppo_loss, sequence_training_batches
from junqi.training.settings import TrainingSettings


def window(state, length, limit):
    records = state.records[:length]
    if len(records) > limit + 1:
        records = (records[0], *records[-limit:])
    return replace(state, records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    import fcntl
    probe_lock = open(Path(tempfile.gettempdir()) / "siguozero-cuda-probe.lock", "a+b")
    try:
        fcntl.flock(probe_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("another PPO GPU probe holds the exclusive benchmark lock")

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(0.85)
    settings = TrainingSettings.from_yaml(
        Path(__file__).parents[1] / "configs/bootstrap.yaml",
        "four_dark", model_scale="main", overrides={"device": "cuda"},
    )
    config = replace(settings.model, max_transitions=1000,
                     inference_temporal_cache_entries=96)
    states = diverse_states(TrainingMode.FOUR_DARK, config, 1001, 8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": "four_dark", "device": torch.cuda.get_device_name(),
        "torch": torch.__version__, "model_config": asdict(config),
        "fresh_random_weights": True, "checkpoint_loaded": False,
        "synthetic_fixed_histories": True, "policy_only": True,
        "optimizer_step_included": False, "ppo_epochs_measured": 1,
        "sequence_microbatch": 8, "samples_per_prefix": 64,
        "repeats": args.repeats,
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ("src/junqi/training/models.py", "src/junqi/training/encoding.py",
                         "src/junqi/training/ppo.py", "configs/bootstrap.yaml",
                         "tools/benchmark_history_window.py")
        },
        "limitations": [
            "Fixed synthetic histories are not legal self-play trajectories.",
            "Policy phases exclude critic, environments, optimizer step, layout and checkpoints.",
            "Phase ratios must not be applied directly to the 3-billion-step budget.",
            "No existing model or training configuration is modified.",
        ],
        "cases": [],
    }

    def persist():
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def samples_for(limit, lengths):
        return [PPOSample(window(state, length, limit), state.legal_actions[0],
                          -4.0, 0.0, 0.5 if i % 2 else -0.5, 0.25, 0)
                for state in states for i, length in enumerate(lengths)]

    for limit in (1000, 500):
        # Each case starts from random parameters, never a historical checkpoint.
        torch.manual_seed(20260911)
        policy = GamePolicyTransformer(replace(config, max_transitions=limit)).cuda()
        policy.train()
        current = {"max_transitions": limit, "measurements": []}
        result["cases"].append(current)

        def backward(samples):
            policy.train()
            policy.clear_inference_board_cache()
            policy.zero_grad(set_to_none=True)
            batches = sequence_training_batches(samples, sequences_per_batch=8,
                                                max_samples_per_sequence=64)
            for batch in batches:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = policy_ppo_loss(policy, batch, clip_epsilon=0.2,
                                           entropy_coefficient=0.01,
                                           sequence_training=True).loss
                    loss = loss * len(batch) / len(samples)
                loss.backward()
            return batches

        def timed(name, action, repeats):
            times = []
            peaks = []
            for repetition in range(repeats):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                action()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - started)
                peaks.append(torch.cuda.max_memory_allocated() / 2**30)
                print(json.dumps({"limit": limit, "phase": name,
                                  "repetition": repetition, "seconds": times[-1]}), flush=True)
            row = {"phase": name, "seconds": times,
                   "median_seconds": statistics.median(times),
                   "peak_allocated_gib": max(peaks)}
            current["measurements"].append(row)
            persist()
            return row

        # Warm both a short and the case's full history shape before timing.
        backward(samples_for(limit, [257]))
        backward(samples_for(limit, [1001]))
        timed("eight_independent_full_histories_backward",
              lambda: backward(samples_for(limit, [1001])), args.repeats)

        # Same eight streams and 512 decisions, after crossing the 500-step cutoff.
        rolling = samples_for(limit, range(502, 755, 4))
        groups = history_prefix_groups([sample.state for sample in rolling])
        batches = sequence_training_batches(rolling, sequences_per_batch=8,
                                            max_samples_per_sequence=64)
        current["rolling_structure"] = {
            "decision_samples": len(rolling), "exact_prefix_groups": len(groups),
            "microbatches": len(batches),
            "packed_history_tokens": sum(len(rolling[group[0]].state.records) for group in groups),
            "min_history_tokens": min(len(item.state.records) for item in rolling),
            "max_history_tokens": max(len(item.state.records) for item in rolling),
        }
        timed("same_512_rolling_decisions_backward", lambda: backward(rolling), args.repeats)

        policy.zero_grad(set_to_none=True)

        def inference():
            policy.eval()
            policy.start_inference_board_cache()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for length in range(490, 567, 4):
                    batch = [window(state, length, limit) for state in states]
                    policy(batch, [state.legal_actions for state in batch])

        inference()
        timed("rolling_inference_160_decisions", inference, args.repeats)
        current["inference_encoding_metrics"] = policy.board_encoding_metrics()
        persist()
        policy.clear_inference_board_cache()
        del policy
        gc.collect()
        torch.cuda.empty_cache()

    old, new = result["cases"]
    result["phase_time_ratios_500_over_1000"] = {
        newer["phase"]: newer["median_seconds"] / older["median_seconds"]
        for older, newer in zip(old["measurements"], new["measurements"], strict=True)
    }
    persist()
    print(json.dumps(result["phase_time_ratios_500_over_1000"]), flush=True)


if __name__ == "__main__":
    main()
