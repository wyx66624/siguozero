"""Extrapolate actual four-dark PPO actions from matched, bounded GPU probes.

Uses measured wall time only. No TFLOPS ratios or guessed hardware speedups.
Evaluation latency and changes in future game distributions remain unmeasured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path: Path):
    raw = path.read_bytes()
    data = json.loads(raw)
    assert data["complete"] and data["checkpoint_stat_unchanged"]
    assert data["environment_steps"] == sum(row["metrics"]["rollout/environment_plies"] for row in data["updates"])
    assert all(row["metrics"]["rollout/continuation_plies"] == 0 for row in data["updates"])
    return data, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-steps", type=int, default=3_000_000_000)
    args = parser.parse_args()
    assert args.environment_steps > 0
    before, before_sha = read(args.before)
    after, after_sha = read(args.after)
    for key in ("device", "checkpoint", "checkpoint_update", "games", "policy_epochs", "critic_epochs", "ppo_minibatch_samples"):
        assert before[key] == after[key], key
    for key in ("board_dim", "temporal_dim", "temporal_layers", "temporal_heads", "temporal_ffn_dim", "max_transitions"):
        assert before["model_config"][key] == after["model_config"][key], key
    old, new = before["updates"][0], after["updates"][0]
    transitions = int(new["metrics"]["rollout/environment_plies"])
    assert transitions == old["metrics"]["rollout/environment_plies"]
    layout_events = [row["metrics"]["timing/layout_backward_seconds"] for row in after["updates"]
                     if row["metrics"]["timing/layout_backward_seconds"] > 0]
    interval = after["layout_update_interval"]
    per_update_layout = sum(layout_events) / len(layout_events) / interval if layout_events else None
    measured_layout = sum(layout_events)
    # Keep measured startup/shutdown/logging overhead. Adjust only the sampled
    # layout frequency (e.g. one event in four measured rounds vs one in eight).
    adjusted_seconds = (after["wall_seconds"] - measured_layout
                        + len(after["updates"]) * per_update_layout) if per_update_layout is not None else None
    def projection(steps_per_second):
        seconds = args.environment_steps / steps_per_second
        return {"steps_per_second": steps_per_second,
                "continuous_days": seconds / 86400,
                "calendar_days_at_80_percent": seconds / 86400 / .8}
    matched = {
        "before_wall_seconds": old["wall_seconds"], "after_wall_seconds": new["wall_seconds"],
        "speedup": old["wall_seconds"] / new["wall_seconds"],
        "wall_reduction_percent": (1 - new["wall_seconds"] / old["wall_seconds"]) * 100,
        "before_extrapolation": projection(transitions / old["wall_seconds"]),
        "after_extrapolation": projection(transitions / new["wall_seconds"]),
    }
    result = {
        "mode": "four_dark", "environment_step_target": args.environment_steps,
        "device": after["device"], "games": after["games"], "transitions_per_update": transitions,
        "sources": [{"path": str(path), "sha256": sha} for path, sha in ((args.before, before_sha), (args.after, after_sha))],
        "matched_first_update": matched,
        "measured_updated_window": projection(after["environment_steps"] / after["wall_seconds"]),
        "observed_layout_updates": len(layout_events), "layout_seconds_total": measured_layout,
        "layout_seconds_per_update_at_configured_interval": per_update_layout,
        "cadence_adjusted_updated_window": (projection(after["environment_steps"] / adjusted_seconds)
                                            if adjusted_seconds is not None else None),
        "observed_round_projections": [dict(update=row["update"], wall_seconds=row["wall_seconds"],
                                             history=row["history"],
                                             **projection(transitions / row["wall_seconds"])) for row in after["updates"]],
        "target_15_days_steps_per_second": {"continuous": args.environment_steps / (15 * 86400),
                                           "availability_80_percent": args.environment_steps / (15 * 86400 * .8)},
        "evaluation_events_for_target": math.ceil(args.environment_steps / 50_000_000),
        "limitations": ["Projection of bounded measured gameplay, not a long-run guarantee or strength measurement.",
                        "No PRO 6000 measurement; no assumed cross-hardware speed ratio.",
                        "Scheduled evaluation/save latency must be added; ordinary update saves are disabled.",
                        "Observed round variation is not a confidence interval.",
                        "First-update comparison shares checkpoint and initial setup, but BF16 can change later sampled actions."],
        "complete": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
