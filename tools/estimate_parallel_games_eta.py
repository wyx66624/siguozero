"""Revision-21 (512-wide) budgets; target GPU/concurrency rates are unmeasured.

Preserves 256 collected transitions per game per outer update, optimizer
minibatch 512 and three policy/critic epochs. More games therefore reduce
checkpoint frequency per environment transition, not optimizer sample reuse.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "docs/benchmarks"
TARGET = 3_000_000_000
AVAILABILITY = 0.8


def read(name):
    path = BENCH / name
    return path, json.loads(path.read_text(encoding="utf-8"))


def summarize(probe, selected):
    rows = probe["real_updates"][selected]
    total = lambda key: sum(row[key] for row in rows)
    steps = total("rollout/environment_plies")
    work = total("timing/update_seconds")
    inference = total("rollout/actor_inference_seconds") + total("rollout/critic_inference_seconds")
    checkpoint = next(p for p in probe["phases"] if p["phase"] == "checkpoint_to_workspace_drive")["seconds"]
    with_save = work + checkpoint * len(rows)
    return {
        "updates": [r["update"] for r in rows],
        "environment_steps": steps,
        "non_save_seconds": work,
        "actor_plus_critic_inference_seconds": inference,
        "other_non_save_seconds": work - inference,
        "checkpoint_seconds_each": checkpoint,
        "inference_share_with_save": inference / with_save,
        "env_steps_per_second_including_amortized_save": steps / with_save,
        "calendar_days_at_80_percent": TARGET / steps * with_save / 86400 / AVAILABILITY,
        "history_batches": [h for h in probe["real_history_batches"] if h["update"] in {r["update"] for r in rows}],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hardware_path, hardware = read("action_linear_hardware_eta_20260910.json")
    actor_path, actor = read("actor_batch_4090_20260911.json")
    inputs = [hardware_path, actor_path, Path(__file__)]
    baselines = {}
    for mode in ("four_dark", "double_open"):
        path, probe = read(f"action_linear_real_{mode}_4090_20260910.json")
        assert probe["games"] == 20 and probe["transitions_per_update"] == 5120
        assert probe["config_revision"] == 21 and len(probe["real_updates"]) == 4
        inputs.append(path)
        baselines[mode] = summarize(probe, slice(2, 4))
    scenarios = []
    candidates = {
        "RTX 4090 24GB": {"games": [24, 28], "inference_gain": [1.0, 1.08]},
        "RTX PRO 6000 Blackwell Workstation 96GB": {"games": [64, 80, 96], "inference_gain": [1.0, 1.2]},
        "H100 SXM 80GB": {"games": [64, 80], "inference_gain": [1.0, 1.2]},
    }
    for name, candidate in candidates.items():
        speedups = hardware["hardware"][name]["non_save_speedup_scenario_range"]
        for cards in (1, 2, 4, 8):
            efficiencies = [1.0] if cards == 1 else [0.7, 0.9]
            by_mode = {}
            for mode, baseline in baselines.items():
                days = []
                for games, gain, hw, efficiency in itertools.product(
                    candidate["games"], candidate["inference_gain"], speedups, efficiencies
                ):
                    work = TARGET / baseline["environment_steps"] * (
                        baseline["other_non_save_seconds"]
                        + baseline["actor_plus_critic_inference_seconds"] / gain
                    ) / (cards * hw * efficiency)
                    saves = math.ceil(TARGET / (games * 256 * cards)) + 1
                    days.append((work + saves * baseline["checkpoint_seconds_each"]) / 86400 / AVAILABILITY)
                by_mode[mode] = [min(days), max(days)]
            scenarios.append({
                "hardware": name, "cards": cards, "games_per_card_candidates": candidate["games"],
                "inference_speedup_assumption_unmeasured": candidate["inference_gain"],
                "calendar_days_by_mode": by_mode,
                "one_mode_calendar_days_envelope": [min(x[0] for x in by_mode.values()), max(x[1] for x in by_mode.values())],
            })
    memory = []
    for games in (20, 24, 28, 32, 40, 64, 80, 96, 128):
        entries = 12 * games
        pages = (entries // 3) * math.ceil(1001 / 16) + 4 * entries
        # Two models, 32 layers, K/V, 16 tokens/page, 8 heads * 64, BF16.
        kv_bytes = 2 * 32 * pages * 2 * 16 * 8 * 64 * 2
        memory.append({"games": games, "player_view_histories": 4 * games,
                       "cache_entries_per_model": entries, "dual_model_kv_arena_gib": kv_bytes / 2**30,
                       "transitions_per_rank_update": games * 256,
                       "excludes_weights_optimizers_board_cache_and_temporary_tensors": True})
    local_path = BENCH / "parallel_games_24_four_dark_4090_20260911.json"
    local = None
    if local_path.exists():
        probe = json.loads(local_path.read_text(encoding="utf-8"))
        if len(probe.get("real_updates", [])) >= 3 and any(
            p["phase"] == "checkpoint_to_workspace_drive" for p in probe["phases"]
        ):
            inputs.append(local_path)
            local = {"source": str(local_path.relative_to(ROOT)),
                     "games": probe["games"], "transitions_per_update": probe["transitions_per_update"],
                     "requested_updates": probe["requested_updates"],
                     "completed_updates": len(probe["real_updates"]),
                     "interruption": "SIGTERM was logged during update 3; sender not determined; update 3 and result export completed",
                     "all_completed_updates": summarize(probe, slice(0, None)),
                     "later_updates_after_first_two": summarize(probe, slice(2, None)),
                     "phases": probe["phases"][:2],
                     "effective_microbatch": probe["effective_sequence_microbatch_after_training"],
                     "effective_actor_batches": [r["rollout/effective_actor_inference_batch"] for r in probe["real_updates"]],
                     "process_cuda_allocator_limit_fraction": 0.90,
                     "same_trajectories_as_baseline": False}
    result = {
        "status": "rough_scenarios_not_target_gpu_measurements_or_confidence_intervals",
        "date": "2026-09-11", "target_environment_steps_per_mode": TARGET,
        "architecture_scope": {"config_revision": 21, "board_dim": 256,
                               "temporal_dim": 512, "temporal_layers": 32,
                               "does_not_apply_to_revision_22_128_plus_128_four_player_models": True},
        "availability": AVAILABILITY,
        "baselines_20_games_later_two_updates": baselines,
        "new_24_game_probe": local,
        "fixed_80_stream_inference_probe": {
            "actual_environment_steps": actor["actual_environment_steps"],
            "rows": [{k: r[k] for k in ("inference_batch", "model_decisions_per_second")} for r in actor["rows"]],
            "gain_20_to_80": actor["rows"][2]["model_decisions_per_second"] / actor["rows"][0]["model_decisions_per_second"],
            "not_80_parallel_four_player_games": True,
        },
        "memory_arena_calculation": memory,
        "scenarios": scenarios,
        "assumptions": {
            "inference_gains_are_sensitivity_parameters_not_fits_or_predictions": True,
            "old_hardware_speedups_and_70_to_90_percent_multigpu_efficiency_remain_unmeasured": True,
            "maintained_samples_per_game_per_outer_update": 256,
            "optimizer_minibatch": 512, "policy_epochs": 3, "critic_epochs": 3,
            "one_checkpoint_per_outer_update_with_baseline_single_card_latency": True,
            "one_mode_budget_not_both_modes_combined": True,
            "formula": "days=[T/S*(other+inference/gain)/(cards*hardware_speedup*efficiency)+(ceil(T/(256*games*cards))+1)*save]/86400/0.8",
            "limitations": ["short random-weight probes, no convergence claim", "larger-game cold prefill capacity unmeasured except the 24-game probe",
                            "same relative hardware utilization assumed", "larger pools/checkpoints and distributed gathers may take longer to save",
                            "selection matches and periodic layout training excluded", "same data reuse does not prove same learning dynamics"],
        },
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"local_probe_available": local is not None,
                      "scenarios": [{"hardware": r["hardware"], "cards": r["cards"], "days": r["one_mode_calendar_days_envelope"]} for r in scenarios]}, indent=2))


if __name__ == "__main__":
    main()
