"""Recompute whole-board PPO time from measured environment steps and wall time.

Only the local RTX 4090 is measured. Faster-device and multi-card rows are
explicit sensitivities, with no assumed mapping from GPU name to speedup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from estimate_ppo_eta import read_probe


def select_update_window(probe, skip_updates):
    """Select a measured history window without inventing throughput gains."""
    if not 0 <= skip_updates < len(probe["updates"]):
        raise ValueError("skip_updates must leave at least one measured update")
    selected = dict(probe)
    selected["source_total_measured_environment_steps"] = probe["total_measured_environment_steps"]
    selected["excluded_initial_updates"] = skip_updates
    if skip_updates:
        rows = probe["updates"][skip_updates:]
        steps = sum(row["environment_steps"] for row in rows)
        seconds = sum(row["seconds_with_save"] for row in rows)
        selected.update({"updates": rows, "total_measured_environment_steps": steps,
                         "weighted_seconds_with_periodic_save": seconds,
                         "weighted_environment_steps_per_second_with_periodic_save": steps / seconds})
    return selected


def estimate(probes, *, target, availability, save_every, arena=None):
    batches = {p["per_rank_transition_batch"] for p in probes}
    if len(batches) != 1:
        raise ValueError("all probes must use the same per-rank transition batch")
    batch = batches.pop()
    updates = math.ceil(target / batch)
    rows = []
    sensitivities = []
    for probe in probes:
        n = probe["total_measured_environment_steps"]
        # Remove the once-per-update save allowance added by the legacy reader.
        measured_seconds = (
            probe["weighted_seconds_with_periodic_save"]
            - len(probe["updates"]) * probe["checkpoint_seconds"]
        )
        seconds_per_step = measured_seconds / n
        save_count = updates // save_every + 1  # periodic plus final save
        save_seconds = save_count * probe["checkpoint_seconds"]
        compute_seconds = target * seconds_per_step
        per_batch = [
            target / x["environment_steps"]
            * (x["seconds_without_save"] + probe["checkpoint_seconds"] / save_every)
            / 86400 for x in probe["updates"]
        ]
        archive_count = updates // math.lcm(save_every, 100) + 1
        row = {
            "mode": probe["mode"],
            "measured_non_save_seconds": measured_seconds,
            "measured_environment_steps": n,
            "amortized_steps_per_second": target / (compute_seconds + save_seconds),
            "continuous_days": (compute_seconds + save_seconds) / 86400,
            "calendar_days": (compute_seconds + save_seconds) / 86400 / availability,
            "individual_batch_continuous_days": per_batch,
            "individual_batch_envelope_is_not_a_bound": True,
            "save_count": save_count,
            "save_only_continuous_days": save_seconds / 86400,
            "archive_every_updates": 100,
            "additional_archive_days_if_each_costs_one_full_save": (
                archive_count * probe["checkpoint_seconds"] / 86400
            ),
            "save_frequency_sensitivity": [
                {"every_updates": every,
                 "continuous_days": (compute_seconds + (updates // every + 1)
                                     * probe["checkpoint_seconds"]) / 86400}
                for every in (1, 10, 100)
            ],
        }
        rows.append(row)
        for speedup in (1, 2, 3, 4):
            for cards in (1, 2, 4, 8):
                efficiencies = (1.0,) if cards == 1 else (0.7, 0.9)
                global_updates = math.ceil(target / (batch * cards))
                io_seconds = (global_updates // save_every + 1) * probe["checkpoint_seconds"]
                values = [(compute_seconds / (cards * speedup * efficiency) + io_seconds)
                          / 86400 / availability for efficiency in efficiencies]
                sensitivities.append({
                    "mode": probe["mode"], "cards": cards,
                    "per_card_non_save_speedup_assumption": speedup,
                    "parallel_efficiency_assumptions": list(efficiencies),
                    "global_transition_batch": batch * cards,
                    "calendar_days": [min(values), max(values)],
                    "status": "conditional_only_not_measured_on_target_hardware",
                })
    evaluation = None
    if arena is not None:
        data = json.loads(arena.read_text(encoding="utf-8"))
        if not data["phases"] or data.get("failed_phase"):
            raise ValueError("arena probe must contain a completed successful phase")
        phase = data["phases"][-1]
        games = len(phase["records"])
        seconds = phase["seconds"]
        rounds = target // 50_000_000
        evaluation = {
            "source": arena.as_posix(),
            "source_sha256": hashlib.sha256(arena.read_bytes()).hexdigest(),
            "measured_complete_games": games,
            "measured_seconds": seconds,
            "rounds_per_mode": rounds, "games_per_round": 100,
            "total_games_per_mode": rounds * 100,
            "estimated_seconds_per_100_game_round": seconds / games * 100,
            "estimated_total_continuous_hours_per_mode": (
                seconds / games * rounds * 100 / 3600
            ),
            "limitation": "four_dark random self-play; linear game-count scaling only; future game lengths, incomplete final wave, training-resident memory and model reloads differ",
        }
    return {
        "status": "short_probe_extrapolation_not_a_validated_training_duration",
        "architecture": "whole_board_one_hot_linear",
        "config_revisions": sorted({p["config_revision"] for p in probes
                                    if p.get("config_revision") is not None}),
        "action_encoder_types": sorted({p["model_config"]["action_encoder_type"] for p in probes
                                        if p.get("model_config")}),
        "target_environment_steps_per_mode": target,
        "availability": availability, "save_every_updates": save_every,
        "observed_device": "RTX 4090 24GB",
        "probes": probes, "single_card": rows,
        "modes_included": [x["mode"] for x in rows],
        "selected_modes_sequential_continuous_days": sum(x["continuous_days"] for x in rows),
        "selected_modes_sequential_calendar_days": sum(x["calendar_days"] for x in rows),
        "two_modes_sequential_continuous_days": (
            sum(x["continuous_days"] for x in rows)
            if {x["mode"] for x in rows} == {"four_dark", "double_open"} else None),
        "two_modes_sequential_calendar_days": (
            sum(x["calendar_days"] for x in rows)
            if {x["mode"] for x in rows} == {"four_dark", "double_open"} else None),
        "selection_estimate": evaluation,
        "hardware_sensitivities": sensitivities,
        "limitations": [
            "Learning/forward passes are not environment steps; PPO currently has no extra simulation branches.",
            "Use the recorded measured history lengths; short random-policy probes do not establish mature-policy steady-state throughput.",
            "Named PRO 6000/H100 devices have no measured speedup in this model; old GPU factors are not reused.",
            "Multi-card sensitivities assume each rank keeps the measured workload; global batches grow with cards.",
            "Periodic layout updates may not occur in a short probe; inspect layout_backward_seconds before treating their amortized cost as measured.",
            "Serial save latency does not speed up with GPU compute. Archive and evaluation overhead are listed separately.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", nargs="+", type=Path)
    parser.add_argument("--arena", type=Path)
    parser.add_argument("--target", type=int, default=3_000_000_000)
    parser.add_argument("--availability", type=float, default=0.8)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--skip-updates", type=int, default=0,
                        help="Exclude opening updates to inspect a later measured history window")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.target <= 0 or args.save_every <= 0 or not 0 < args.availability <= 1:
        parser.error("invalid target, checkpoint frequency or availability")
    probes = [select_update_window(read_probe(p), args.skip_updates) for p in args.probes]
    if len({p["mode"] for p in probes}) != len(probes):
        parser.error("use one probe per mode")
    result = estimate(probes, target=args.target, availability=args.availability,
                      save_every=args.save_every, arena=args.arena)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"single_card": result["single_card"],
                      "selection": result["selection_estimate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
