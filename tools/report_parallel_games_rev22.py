"""Budget scenarios for the measured 128+128 / 256-token four-player profile."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

from estimate_parallel_games_eta import AVAILABILITY, BENCH, ROOT, TARGET, read, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe_path, probe = read("parallel_games_48_four_dark_rev22_4090_20260911.json")
    hardware_path, hardware = read("action_linear_hardware_eta_20260910.json")
    config = probe["model_config"]
    assert probe["config_revision"] == 22 and len(probe["real_updates"]) == 4
    assert config["board_dim"] == 128 and config["temporal_dim"] == 256
    assert config["temporal_layers"] == 32 and config["temporal_ffn_dim"] == 1024
    assert probe["games"] == 48 and probe["transitions_per_update"] == 12288
    # A process start timestamp was previously mistaken for GPU work overlap.
    # Replacement trainer initialization is logged at 00:58:42.883, after
    # this run's update 4 at 00:58:27.037. The provenance record accepts 3/4
    # and rejects the first two updates for a different overlapping probe.
    baseline = summarize(probe, slice(2, 4))
    configurations = {
        "RTX 4090 24GB": ([48], [1.0]),
        "RTX PRO 6000 Blackwell Workstation 96GB": ([128, 160, 192], [1.0, 1.2]),
        "H100 SXM 80GB": ([128, 160], [1.0, 1.2]),
    }
    scenarios = []
    for name, (games_choices, gains) in configurations.items():
        # Hardware factors are conditional Amdahl scenarios. They are NOT
        # target-card timings, nor new data inferred from the larger VRAM.
        hw_gains = hardware["hardware"][name]["non_save_speedup_scenario_range"]
        for cards in (1, 2, 4, 8):
            efficiencies = [1.0] if cards == 1 else [0.7, 0.9]
            values = []
            for games, gain, hw, efficiency in itertools.product(games_choices, gains, hw_gains, efficiencies):
                compute = TARGET / baseline["environment_steps"] * (
                    baseline["other_non_save_seconds"]
                    + baseline["actor_plus_critic_inference_seconds"] / gain
                ) / (cards * hw * efficiency)
                saves = math.ceil(TARGET / (games * 256 * cards)) + 1
                values.append((compute + saves * baseline["checkpoint_seconds_each"]) / 86400 / AVAILABILITY)
            scenarios.append({"hardware": name, "cards": cards,
                              "games_per_card_candidates": games_choices,
                              "additional_inference_speedup_assumption": gains,
                              "calendar_days": [min(values), max(values)]})
    provenance_path = BENCH / "compact128_measurement_provenance_20260911.json"
    sources = [probe_path, hardware_path, provenance_path, Path(__file__), ROOT / "tools/estimate_parallel_games_eta.py"]
    result = {
        "status": "short_probe_and_conditional_scenarios_not_validated_long_term_durations",
        "date": "2026-09-11", "mode_measured": "four_dark", "double_open_measured_here": False,
        "config_revision": 22, "model_config": config,
        "target_environment_steps_per_mode": TARGET, "availability": AVAILABILITY,
        "total_measured_environment_steps": sum(r["rollout/environment_plies"] for r in probe["real_updates"]),
        "longer_history_baseline": baseline,
        "all_updates_including_initial_overlap_not_for_eta": summarize(probe, slice(0, 4)),
        "timing_provenance": {"accepted_updates": [3, 4], "excluded_initial_updates": [1, 2],
                              "previous_update_4_overlap_inference_withdrawn": True,
                              "process_start_is_not_gpu_work_start": True,
                              "replacement_trainer_initialized_local": "2026-09-11 00:58:42.883 -07:00",
                              "current_update_4_completed_local": "2026-09-11 00:58:27.037 -07:00",
                              "source": str(provenance_path.relative_to(ROOT)),
                              "not_a_continuous_gpu_activity_trace": True},
        "effective_actor_batches": [r["rollout/effective_actor_inference_batch"] for r in probe["real_updates"]],
        "effective_microbatch": probe["effective_sequence_microbatch_after_training"],
        "peak_phase": probe["phases"][0],
        "scenarios": scenarios,
        "memory_kv_only": [{"games_per_card": g, "dual_model_kv_arena_gib": 0.29296875 * g,
                            "transitions_per_rank_update": 256 * g} for g in (20, 48, 56, 64, 96, 128, 160, 192, 256)],
        "assumptions": {
            "same_model_architecture_as_probe": True,
            "same_256_environment_actions_per_game_per_outer_update": True,
            "three_policy_and_critic_epochs_optimizer_minibatch_512": True,
            "hardware_speedups": {n: hardware["hardware"][n]["non_save_speedup_scenario_range"] for n in configurations},
            "gpu_accelerable_fraction_for_hardware_estimate_not_measured": [0.6, 0.85],
            "multigpu_efficiency_not_measured": [0.7, 0.9],
            "larger_parallel_game_throughput_and_memory_not_measured_on_target_gpus": True,
            "one_checkpoint_per_update_single_card_latency_assumed_for_all_pools_and_card_counts": True,
            "excluded": ["initial overlapping updates 1/2", "selection matches", "periodic layout update absent in this probe", "changed mature policy history lengths", "failures and retries"],
            "not_same_trajectory_or_same_architecture_comparison_to_rev21": True,
            "double_open_runtime_requires_its_own_probe": True,
        },
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": baseline, "scenarios": scenarios}, indent=2))


if __name__ == "__main__":
    main()
