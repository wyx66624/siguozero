"""Recompute conditional PPO budget scenarios from bounded end-to-end probes.

GPU speedups and DDP efficiencies below are explicit sensitivity assumptions,
not measured performance on the named hardware. The observed batch envelope
is not a confidence interval or a bound on late-game/long-run performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


HARDWARE_SCENARIOS = (
    {"name": "RTX 4090 24GB", "non_save_speedup": [1.0, 1.0], "interconnect": "pcie"},
    {"name": "RTX PRO 6000 Blackwell Workstation 96GB",
     "non_save_speedup": [1.5, 2.5], "interconnect": "pcie"},
    {"name": "H100 PCIe 80GB", "non_save_speedup": [2.0, 3.5], "interconnect": "pcie"},
    {"name": "H100 SXM 80GB (HGX/NVSwitch)",
     "non_save_speedup": [2.5, 4.5], "interconnect": "nvswitch"},
)
DDP_EFFICIENCY = {
    "pcie": {1: [1.0, 1.0], 2: [.80, .95], 4: [.70, .90], 8: [.55, .80]},
    "nvswitch": {1: [1.0, 1.0], 2: [.85, .95], 4: [.80, .95], 8: [.70, .90]},
}


def read_probe(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    phases = {phase["phase"]: phase for phase in raw["phases"]}
    saved = phases["checkpoint_to_workspace_drive"]
    updates = raw["real_updates"]
    rows = []
    for index, update in enumerate(updates):
        environment = int(update["rollout/environment_plies"])
        base = int(update["rollout/base_plies"])
        branches = int(update["rollout/continuation_plies"])
        if environment != base + branches or branches != 0:
            raise ValueError("This estimator expects actual PPO probes without extra branches")
        if environment != raw["transitions_per_update"]:
            raise ValueError("Use full, equal-length probe updates for extrapolation")
        seconds = float(update["timing/update_seconds"])
        rows.append({
            "environment_steps": environment,
            "base_steps": base,
            "branch_steps": branches,
            "history": raw["real_history_batches"][index],
            "seconds_without_save": seconds,
            "seconds_with_save": seconds + saved["seconds"],
            "environment_steps_per_second_with_save": environment / (seconds + saved["seconds"]),
            "policy_epochs": update["optimizer/policy_epochs"],
            "critic_epochs": update["optimizer/critic_epochs"],
            "policy_nonzero_advantage_fraction": update["policy/nonzero_advantage_fraction"],
            "terminal_games": update["rollout/base_games_completed"],
            "policy_backward_seconds": update["timing/policy_backward_seconds"],
            "critic_backward_seconds": update["timing/critic_backward_seconds"],
        })
    steps = sum(row["environment_steps"] for row in rows)
    elapsed = phases["real_training_loop_without_checkpoint"]["seconds"] + len(rows) * saved["seconds"]
    return {
        "source": path.as_posix(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mode": raw["mode"], "device": raw["device"], "torch": raw["torch"],
        "games": raw["games"], "per_rank_transition_batch": raw["transitions_per_update"],
        "sequence_microbatch": raw["sequence_microbatch"],
        "optimizer_minibatch_samples": raw["optimizer_minibatch_samples"],
        "checkpoint_seconds": saved["seconds"], "checkpoint_bytes": saved["bytes"],
        "peak_allocated_gib": phases["real_training_loop_without_checkpoint"]["peak_allocated_gib"],
        "peak_reserved_gib": phases["real_training_loop_without_checkpoint"]["peak_reserved_gib"],
        "total_measured_environment_steps": steps,
        "weighted_environment_steps_per_second_with_periodic_save": steps / elapsed,
        "weighted_seconds_with_periodic_save": elapsed,
        "updates": rows,
    }


def estimate(probes: list[dict], target: int, availability: float) -> dict:
    batches = {probe["per_rank_transition_batch"] for probe in probes}
    if len(batches) != 1:
        raise ValueError("Scenario inputs must use the same per-rank rollout batch")
    batch = batches.pop()
    rows = []
    for hardware in HARDWARE_SCENARIOS:
        for cards in (1, 2, 4, 8):
            values = []
            per_mode = {}
            for probe in probes:
                mode_values = []
                for update in probe["updates"]:
                    for speedup in hardware["non_save_speedup"]:
                        for efficiency in DDP_EFFICIENCY[hardware["interconnect"]][cards]:
                            # Weak scaling: each rank collects `batch` transitions.
                            # Checkpoint I/O stays serial and is NOT divided by GPU speed.
                            seconds = update["seconds_without_save"] / speedup / efficiency + probe["checkpoint_seconds"]
                            days = math.ceil(target / (batch * cards)) * seconds / 86400 / availability
                            mode_values.append(days)
                per_mode[probe["mode"]] = [min(mode_values), max(mode_values)]
                values.extend(mode_values)
            rows.append({
                "hardware": hardware["name"], "cards": cards,
                "non_save_speedup_assumption": hardware["non_save_speedup"],
                "ddp_efficiency_assumption": DDP_EFFICIENCY[hardware["interconnect"]][cards],
                "global_rollout_batch": batch * cards,
                "global_updates": math.ceil(target / (batch * cards)),
                "calendar_days_scenario_envelope": [min(values), max(values)],
                "calendar_months_scenario_envelope": [min(values) / 30.44, max(values) / 30.44],
                "calendar_days_by_mode": per_mode,
                "two_modes_sequential_calendar_days": [
                    sum(v[0] for v in per_mode.values()), sum(v[1] for v in per_mode.values())
                ] if {p["mode"] for p in probes} == {"four_dark", "double_open"} else None,
            })
    checkpoints = []
    for probe in probes:
        for every in (1, 10, 100):
            saves = math.ceil(math.ceil(target / batch) / every)
            checkpoints.append({
                "mode": probe["mode"], "save_every_updates": every, "saves": saves,
                "save_only_continuous_days": saves * probe["checkpoint_seconds"] / 86400,
                "save_only_calendar_days": saves * probe["checkpoint_seconds"] / 86400 / availability,
                "cumulative_checkpoint_payload_pib": saves * probe["checkpoint_bytes"] / 2**50,
                "status": "current launch script" if every == 1 else "sensitivity only; not applied",
            })
    measured = []
    for probe in probes:
        rate = probe["weighted_environment_steps_per_second_with_periodic_save"]
        measured.append({"mode": probe["mode"], "steps_per_second": rate,
                         "linear_continuous_days": target / rate / 86400,
                         "linear_calendar_days": target / rate / 86400 / availability})
    targets = [{"whole_machine_steps_per_second": rate,
                "continuous_days": target / rate / 86400,
                "calendar_days": target / rate / 86400 / availability}
               for rate in (25, 50, 100, 250, 500, 1000)]
    deadlines = [{"calendar_days": days,
                  "required_whole_machine_steps_per_running_second": target / (days * 86400 * availability)}
                 for days in (30, 60, 90, 180, 365)]
    return {
        "status": "conditional_sensitivity_scenarios_not_validated_hardware_etas",
        "target_environment_steps_per_mode": target,
        "definition": "sum of actual training environment transitions across every seat, game, branch and rank; forwards/backwards/GAE/reused epochs add zero",
        "current_ppo_extra_simulation_branches": 0,
        "calendar_training_availability": availability,
        "calendar_month_days": 30.44,
        "assumptions": {
            "hardware_speedups": "manually selected sensitivities for the entire non-save phase; NOT inferred directly from TFLOPS or vendor application claims",
            "ddp": "unmeasured weak scaling, full models per rank, rollout and effective optimizer batches grow with GPU count",
            "save": "one full checkpoint per global update; same measured storage latency, no GPU or DDP speedup assigned to storage",
            "excluded": "arena time, restarts/tuning, model initialization, mature-policy game-length distribution and >1000-transition sliding-window slowdown",
            "range": "combination of individual short-batch measurements and selected assumptions; NOT a confidence interval or upper/lower bound on production time",
        },
        "measured_probes": probes,
        "weighted_short_probe_linear_extrapolations": measured,
        "hardware_scenarios": rows,
        "checkpoint_sensitivity": checkpoints,
        "throughput_sensitivity": targets,
        "deadline_throughput_requirements": deadlines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path, nargs="+")
    parser.add_argument("--target", type=int, default=3_000_000_000)
    parser.add_argument("--availability", type=float, default=.8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.target <= 0 or not 0 < args.availability <= 1:
        parser.error("target must be positive and availability must be in (0, 1]")
    probes = [read_probe(path) for path in args.probes]
    if len({probe["mode"] for probe in probes}) != len(probes):
        parser.error("use one probe per mode")
    result = estimate(probes, args.target, args.availability)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"measured": result["weighted_short_probe_linear_extrapolations"],
                      "scenarios": [{"hardware": row["hardware"], "cards": row["cards"],
                                     "months": row["calendar_months_scenario_envelope"]}
                                    for row in result["hardware_scenarios"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
