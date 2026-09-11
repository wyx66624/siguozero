"""Measure complete, matched serial/batched arena games with real checkpoints.

This is a timing/capacity experiment. Self-play results are not evidence of
strength gain. Every phase uses the same checkpoint(s), group seeds and limits;
game/action digests report whether the sampled workloads were exactly equal.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import threading
import time
from typing import Any

import torch

from junqi.training import arena, models, rollout
from junqi.training.accelerator import (
    empty_cache, get_device_name, max_memory_allocated, max_memory_reserved,
    reset_peak_memory_stats, synchronize,
)
from junqi.training.arena import MatchSettings, atomic_json, play_groups, sha256_file
from junqi.training.inference import InferenceEngine


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def histogram_summary(counts: Counter[int]) -> dict[str, Any]:
    size = sum(counts.values())
    if not size:
        return {"count": 0}
    ordered = sorted(counts.items())

    def quantile(fraction: float) -> int:
        target = max(1, int(size * fraction + 0.999999))
        cumulative = 0
        for value, count in ordered:
            cumulative += count
            if cumulative >= target:
                return value
        return ordered[-1][0]

    return {"count": size, "min": ordered[0][0], "p50": quantile(0.5),
            "p95": quantile(0.95), "max": ordered[-1][0],
            "mean": sum(value * count for value, count in ordered) / size,
            "above_1000": sum(count for value, count in ordered if value > 1000)}


class InferenceProbe:
    def __init__(self) -> None:
        self.histories: Counter[int] = Counter()
        self.batches: Counter[int] = Counter()
        self.batch_limits: Counter[int] = Counter()
        self.started = self.last_progress = time.perf_counter()

    def wrap(self, actor: Any) -> None:
        sample = actor.sample

        def measured(states, *args, **kwargs):
            self.batches[len(states)] += 1
            self.histories.update(len(state.records) for state in states)
            now = time.perf_counter()
            if now - self.last_progress >= 30:
                print(json.dumps({"progress_seconds": now - self.started,
                                  "sampled_plies": sum(self.histories.values()),
                                  "largest_history_tokens": max(self.histories, default=0),
                                  "largest_actor_batch": max(self.batches, default=0)}), flush=True)
                self.last_progress = now
            result = sample(states, *args, **kwargs)
            self.batch_limits[actor.max_batch_size] += 1
            return result

        actor.sample = measured


class NvidiaSampler:
    """Optional whole-device utilization samples; model allocation is separate."""
    def __init__(self, device: torch.device) -> None:
        self.executable = shutil.which("nvidia-smi") if device.type == "cuda" else None
        self.index = device.index or 0
        self.rows: list[tuple[float, float]] = []
        self.error: str | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    [self.executable, f"--id={self.index}",
                     "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=5,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                values = result.stdout.strip().split(",")
                self.rows.append((float(values[0]), float(values[1])))
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                self.error = str(error)
                return
            self.stop_event.wait(1)

    def __enter__(self):
        if self.executable:
            self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        if self.executable:
            self.thread.join(timeout=6)

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {"samples": len(self.rows),
                                   "scope": "whole GPU, including other processes"}
        if self.rows:
            utilization, memory = zip(*self.rows)
            summary.update(utilization_mean_percent=statistics.mean(utilization),
                           utilization_p50_percent=statistics.median(utilization),
                           utilization_max_percent=max(utilization),
                           memory_used_max_mib=max(memory))
        if self.error:
            summary["error"] = self.error
        return summary


def game_key(game: dict[str, Any]) -> tuple[int, int]:
    return (game.get("group_index", game.get("pair_index", -1)),
            game.get("rotation", game.get("leg", game.get("candidate_seat", -1))))


def benchmark_phase(args: argparse.Namespace, parallel_games: int) -> dict[str, Any]:
    load_started = time.perf_counter()
    cache_entries = args.temporal_cache_entries
    candidate = InferenceEngine.from_checkpoint(
        args.checkpoint, device=args.device, temporal_cache_entries=cache_entries)
    opponent = InferenceEngine.from_checkpoint(
        args.opponent or args.checkpoint, device=args.device,
        temporal_cache_entries=cache_entries)
    device = candidate.policy.device
    settings = MatchSettings(
        pairs=args.groups, mode=candidate.mode.value, seed=args.seed,
        max_plies=args.max_plies, temporal_cache_entries=cache_entries,
        smoke_test=True, parallel_games=parallel_games,
        inference_batch_size=args.inference_batch_size or parallel_games,
        environment_workers=args.environment_workers,
    )
    probe = InferenceProbe()
    probe.wrap(candidate.actor)
    probe.wrap(opponent.actor)
    synchronize(device)
    load_seconds = time.perf_counter() - load_started
    empty_cache(device)
    reset_peak_memory_stats(device)
    print(json.dumps({"phase_started": parallel_games, "settings": asdict(settings),
                      "checkpoint_load_seconds": load_seconds}), flush=True)
    with NvidiaSampler(device) as resources:
        synchronize(device)
        started = time.perf_counter()
        cpu_started = time.process_time()
        games = play_groups(candidate, opponent, range(args.groups), settings)
        synchronize(device)
        seconds = time.perf_counter() - started
        cpu_seconds = time.process_time() - cpu_started
    games.sort(key=game_key)
    records = [{**{key: value for key, value in game.items() if key != "actions"},
                "actions_sha256": digest(game["actions"])} for game in games]
    total_plies = sum(game["plies"] for game in games)
    result = {
        "parallel_games": parallel_games, "settings": asdict(settings),
        "effective_temporal_cache_entries": settings.effective_temporal_cache_entries,
        "mode": candidate.mode.value, "device": get_device_name(device),
        "checkpoint_update": getattr(candidate, "checkpoint_update", None),
        "model_config": asdict(candidate.policy.config),
        "checkpoint_load_seconds": load_seconds,
        "seconds": seconds, "parent_cpu_seconds": cpu_seconds,
        "games": len(games), "total_plies": total_plies,
        "seconds_per_game": seconds / len(games), "games_per_second": len(games) / seconds,
        "plies_per_second": total_plies / seconds,
        "peak_allocated_gib": max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": max_memory_reserved(device) / 2**30,
        "game_plies": histogram_summary(Counter(game["plies"] for game in games)),
        "policy_history_tokens": histogram_summary(probe.histories),
        "actor_requested_batches": histogram_summary(probe.batches),
        "actor_batch_limits": histogram_summary(probe.batch_limits),
        "actor_oom_reductions": [candidate.actor.oom_reductions, opponent.actor.oom_reductions],
        "gpu_samples": resources.summary(), "records": records,
        "actions_sha256": digest([game["actions"] for game in games]),
        "results_sha256": digest([{key: value for key, value in game.items() if key != "actions"}
                                  for game in games]),
    }
    del candidate, opponent, games
    gc.collect()
    empty_cache(device)
    return result


def compare(baseline: dict[str, Any], phase: dict[str, Any]) -> dict[str, Any]:
    baseline_games = {game_key(game): game for game in baseline["records"]}
    current_games = {game_key(game): game for game in phase["records"]}
    keys = sorted(baseline_games.keys() | current_games.keys())
    different_actions = [key for key in keys if baseline_games.get(key, {}).get("actions_sha256")
                         != current_games.get(key, {}).get("actions_sha256")]
    different_outcomes = [key for key in keys if
                          (baseline_games.get(key, {}).get("candidate_reward"),
                           baseline_games.get(key, {}).get("terminal_reason")) !=
                          (current_games.get(key, {}).get("candidate_reward"),
                           current_games.get(key, {}).get("terminal_reason"))]
    return {"baseline_parallel_games": baseline["parallel_games"],
            "parallel_games": phase["parallel_games"],
            "wall_time_speedup": baseline["seconds"] / phase["seconds"],
            "plies_per_second_speedup": phase["plies_per_second"] / baseline["plies_per_second"],
            "identical_actions": baseline["actions_sha256"] == phase["actions_sha256"],
            "identical_results": baseline["results_sha256"] == phase["results_sha256"],
            "different_action_game_keys": different_actions,
            "different_outcome_game_keys": different_outcomes,
            "numerical_note": "GPU batched matrix rounding can change stochastic action paths; "
                              "report plies/s as well as full elapsed time when workloads diverge."}


def capacity_phase(args: argparse.Namespace, parallel_games: int) -> dict[str, Any]:
    """Full-window cold/shift inference only; synthetic states are not games."""
    from benchmark_ppo import diverse_states

    engines = [InferenceEngine.from_checkpoint(
        path, device=args.device, temporal_cache_entries=args.temporal_cache_entries)
        for path in (args.checkpoint, args.opponent or args.checkpoint)]
    device = engines[0].policy.device
    settings = MatchSettings(
        pairs=args.groups, mode=engines[0].mode.value, parallel_games=parallel_games,
        inference_batch_size=args.inference_batch_size or parallel_games,
        environment_workers=args.environment_workers,
        temporal_cache_entries=args.temporal_cache_entries,
    )
    for engine in engines:
        engine.policy.config = replace(
            engine.policy.config,
            inference_temporal_cache_entries=settings.effective_temporal_cache_entries,
            inference_board_cache_entries=settings.board_cache_entries(engine.policy.config.max_transitions))
        engine.actor.max_batch_size = settings.inference_batch_size
        engine.policy.start_inference_board_cache()
    tokens = engines[0].policy.config.max_sequence_tokens
    states = diverse_states(engines[0].mode, engines[0].policy.config, tokens, parallel_games)
    reserve = torch.zeros(int(args.reserve_gib * 2**30), dtype=torch.uint8, device=device)
    result: dict[str, Any] = {
        "purpose": "synthetic full-window capacity only; not legal trajectories or games/s",
        "parallel_games": parallel_games, "states_per_step": len(states),
        "history_tokens": tokens, "synthetic_extra_resident_gib": args.reserve_gib,
        "extra_resident_note": "byte allocation approximates headroom; not actual training models/optimizers",
        "effective_temporal_cache_entries": settings.effective_temporal_cache_entries,
        "board_cache_entries_per_model": settings.board_cache_entries(tokens - 1),
        "device": get_device_name(device), "steps": [],
    }
    for step in range(args.capacity_steps):
        if step:
            # Retain the initial observation, drop the oldest transition and
            # append a distinct board record. This forces a full-window shift.
            states = [replace(state, records=(state.records[0], *state.records[2:],
                      replace(state.records[-1], board_codes=(state.records[-1].board_codes[1:]
                              + state.records[-1].board_codes[:1])))) for state in states]
        empty_cache(device)
        reset_peak_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()
        for owner, engine in enumerate(engines):
            selected = states[owner::2]
            if selected:
                engine.actor.sample(
                    selected, return_log_probs=False,
                    sampling_uniforms=torch.full((len(selected), 1, 2), 0.5, dtype=torch.float64))
        synchronize(device)
        row = {"step": step, "phase": "cold_full_window" if step == 0 else "shifted_full_window",
               "seconds": time.perf_counter() - started,
               "peak_allocated_gib": max_memory_allocated(device) / 2**30,
               "peak_reserved_gib": max_memory_reserved(device) / 2**30,
               "actor_oom_reductions": [engine.actor.oom_reductions for engine in engines],
               "actor_max_batch_sizes": [engine.actor.max_batch_size for engine in engines]}
        result["steps"].append(row)
        print(json.dumps({"capacity_parallel_games": parallel_games, **row}), flush=True)
    del engines, states, reserve
    gc.collect()
    empty_cache(device)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--opponent")
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--parallel-games", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--environment-workers", type=int, default=4)
    parser.add_argument("--temporal-cache-entries", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--capacity-steps", type=int, default=0,
                        help="instead of matches, measure one synthetic full window then N-1 shifts")
    parser.add_argument("--reserve-gib", type=float, default=0,
                        help="additional resident byte allocation for the synthetic capacity probe")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.groups, args.max_plies, args.torch_threads, args.temporal_cache_entries,
           args.environment_workers, *args.parallel_games) <= 0:
        parser.error("counts must be positive")
    if args.inference_batch_size is not None and args.inference_batch_size <= 0:
        parser.error("inference-batch-size must be positive")
    if args.capacity_steps < 0 or not 0 <= args.reserve_gib < float("inf"):
        parser.error("capacity-steps and reserve-gib must be nonnegative and finite")
    torch.set_num_threads(args.torch_threads)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint = Path(args.checkpoint)
    opponent = Path(args.opponent) if args.opponent else checkpoint
    result: dict[str, Any] = {
        "purpose": "complete real games for throughput; no strength verdict",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "arena_version": arena.ARENA_VERSION,
        "source_sha256": {Path(module.__file__).name: sha256_file(Path(module.__file__))
                          for module in (arena, models, rollout)},
        "torch": torch.__version__, "host_cpu_count": os.cpu_count(),
        "torch_threads": args.torch_threads, "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint), "opponent": str(opponent),
        "opponent_sha256": sha256_file(opponent) if opponent != checkpoint else None,
        "seed": args.seed, "groups": args.groups, "max_game_plies": args.max_plies,
        "phases": [], "comparisons": [], "capacity_phases": [],
    }
    if args.capacity_steps:
        result["purpose"] = "synthetic full-window capacity; not game throughput or strength"
    atomic_json(args.output, result)
    for parallel_games in args.parallel_games:
        try:
            if args.capacity_steps:
                result["capacity_phases"].append(capacity_phase(args, parallel_games))
                atomic_json(args.output, result)
                continue
            phase = benchmark_phase(args, parallel_games)
        except Exception as error:
            result["failed_phase"] = {"parallel_games": parallel_games,
                                      "error_type": type(error).__name__, "error": str(error)}
            atomic_json(args.output, result)
            raise
        result["phases"].append(phase)
        if len(result["phases"]) > 1:
            result["comparisons"].append(compare(result["phases"][0], phase))
        atomic_json(args.output, result)
        print(json.dumps({key: value for key, value in phase.items()
                          if key not in ("records", "model_config")}), flush=True)
    print(json.dumps({"output": str(args.output), "comparisons": result["comparisons"]}), flush=True)


if __name__ == "__main__":
    main()
