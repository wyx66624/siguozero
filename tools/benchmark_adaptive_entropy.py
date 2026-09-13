"""Matched frozen-policy forward/backward probe; no training checkpoint writes."""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time

import torch

from junqi.training.entropy import AdaptiveEntropyCoefficient, phase_entropy_ratios
from junqi.training.inference import InferenceEngine
from junqi.training.ppo import PPOSample, policy_ppo_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=6)
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("iterations must be at least two")
    lock = open(Path(tempfile.gettempdir()) / "siguozero-cuda-probe.lock", "a+b")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    checkpoint = Path(args.checkpoint)
    before_stat = checkpoint.stat()
    with checkpoint.open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    torch.manual_seed(20260912)
    engine = InferenceEngine.from_checkpoint(checkpoint, device="cpu", temporal_cache_entries=16)
    games = [engine.new_game(seed=20260912 + i) for i in range(16)]
    samples = []
    for ply in range(32):
        batch = [h.state_for(g) for g, h in games]
        actions, logs = engine.actor.sample(batch, return_log_probs=True)
        for state, action, log, (game, history) in zip(batch, actions, logs, games):
            samples.append(PPOSample(state, action[0], float(log[0]), 0.,
                1. if len(samples) % 2 else -1., 0., engine.checkpoint_update))
            game.step(action[0])
            history.append_after_step(game)
    # Interleave early and later states so each microbatch includes both phases.
    samples = [samples[ply * 16 + game] for game in range(16) for ply in range(32)]
    policy = engine.policy
    policy.clear_inference_board_cache()
    policy.config = replace(policy.config, ppo_compile_mode="off")
    policy.to("cuda").train().requires_grad_(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print(f"Frozen update {engine.checkpoint_update}; {len(samples)} real-state samples; CUDA BF16", flush=True)

    def run(adaptive):
        policy.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        metrics = {}
        for start in range(0, len(samples), 32):
            batch = samples[start:start + 32]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = policy_ppo_loss(policy, batch, clip_epsilon=.2, entropy_coefficient=.01,
                    opening_entropy_coefficient=.01 if adaptive else None,
                    entropy_opening_plies=16, sequence_training=True, defer_metrics=True)
                loss = output.loss * len(batch) / len(samples)
            loss.backward()
            for key, value in output.metrics.items():
                metrics[key] = metrics.get(key, 0.) + value * len(batch) / len(samples)
        values = torch.stack(list(metrics.values())).detach().cpu().tolist()
        metrics = dict(zip(metrics, values, strict=True))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return {"seconds": elapsed, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "metrics": metrics}, policy.source_query[-1].weight.grad.detach().cpu().clone()

    # Warm both branches, then alternate order to reduce a systematic order bias.
    for _ in range(2):
        run(False)
        run(True)
    records = {"fixed": [], "adaptive": []}
    gradients = {}
    for iteration in range(args.iterations):
        for enabled in ((False, True) if iteration % 2 == 0 else (True, False)):
            name = "adaptive" if enabled else "fixed"
            row, gradients[name] = run(enabled)
            records[name].append(row)
        print(f"Pair {iteration + 1}: fixed={records['fixed'][-1]['seconds']:.4f}s adaptive={records['adaptive'][-1]['seconds']:.4f}s", flush=True)
    medians = {name: statistics.median(row["seconds"] for row in values) for name, values in records.items()}
    metrics = records["adaptive"][-1]["metrics"]
    ratios = phase_entropy_ratios(metrics)
    next_coefficients = {}
    for phase, ratio in ratios.items():
        c = AdaptiveEntropyCoefficient(.01, .005, .02, .6, .1, .9)
        c.observe(ratio)
        next_coefficients[phase] = c.state_dict()
    result = {"device": torch.cuda.get_device_name(), "checkpoint_update": engine.checkpoint_update,
        "checkpoint_sha256": checkpoint_hash, "samples": len(samples), "microbatch": 32,
        "optimizer_steps": 0, "scope": "Short-history policy forward/backward only; fixed coefficients matched; no optimizer, rollout, layout or end-to-end ETA measurement.",
        "precision": "BF16 autocast, FP32 entropy reductions", "compile_mode": "off",
        "iterations_per_variant": args.iterations, "median_seconds": medians,
        "relative_overhead": medians["adaptive"] / medians["fixed"] - 1,
        "source_head_gradient_max_difference": float((gradients["adaptive"] - gradients["fixed"]).abs().max()),
        "observed_entropy_ratios": ratios, "controller_after_one_observation": next_coefficients,
        "results": records}
    after_stat = checkpoint.stat()
    assert (before_stat.st_size, before_stat.st_mtime_ns) == (after_stat.st_size, after_stat.st_mtime_ns)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
