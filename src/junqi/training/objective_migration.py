"""Adopt a changed terminal training utility without discarding training state."""
from __future__ import annotations


def reconcile_draw_objective(payload, settings, *, adopt: bool = False):
    config = payload.get("config", {})
    old = float(config.get("draw_reward", config.get("raw_config", {}).get(
        "rules", {}).get("terminal_reward", {}).get("draw", 0.0)))
    target = float(settings.draw_reward)
    if old == target:
        return
    if not adopt:
        raise RuntimeError("draw reward changed across resume; use --adopt-draw-penalty to retain state and adopt it explicitly")
    state = payload["trainer_state"]
    changed = 0
    for item in state.get("layout_buffer", []):
        if float(item["reward"]) == old:
            item["reward"] = target
            changed += 1
    migration = {
        "update": int(payload["update"]),
        "environment_plies": state.get("cumulative", {}).get("environment_plies", 0),
        "from_draw_reward": old, "to_draw_reward": target,
        "action_feature_dim": 6, "layout_outcomes_remapped": changed,
        "gae": "signed_team_outcome_plus_unsigned_shared_draw_penalty",
    }
    state["draw_objective_migration"] = migration
    config["draw_reward"] = target
