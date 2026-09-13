"""Explicitly adopt current draw rules while retaining a complete training state."""
from __future__ import annotations

from dataclasses import replace


def adopt_draw_rules(payload, settings):
    """Update unfinished games only; leave weights, histories and counters intact.

    Every combat in the previous engine removed at least one piece, so its
    no-interaction counter already measures moves since the last capture.
    Completed evaluation rounds require a separate statistical protocol; this
    migration deliberately supports only runs before their first evaluation.
    """
    state = payload["trainer_state"]
    selection = state.get("model_selection", {})
    if selection.get("rounds"):
        raise ValueError("draw-rule migration after evaluated rounds requires a separate evaluation protocol")
    pools = []
    if "base_game_pool" in state:
        pools.append(state["base_game_pool"])
    if "distributed" in state:
        pools.extend(rank["base_game_pool"] for rank in state["distributed"]["rank_states"])
    target = {"max_game_plies": settings.max_game_plies,
              "no_capture_draw_plies": settings.no_capture_draw_plies}
    changes = []
    for pool in pools:
        old = {"max_game_plies": pool["max_game_plies"],
               "no_capture_draw_plies": pool.get("no_capture_draw_plies", 60)}
        if old == target:
            continue
        if old != {"max_game_plies": 2000, "no_capture_draw_plies": 60} or target != {
            "max_game_plies": None, "no_capture_draw_plies": 70,
        }:
            raise ValueError("supported migration is 2000/60 to unlimited/70 capture draw rules")
        for slot in pool["slots"]:
            game = slot["game"]
            config = game["config"]
            if config.max_plies != 2000 or config.no_interaction_draw_plies != 60:
                raise ValueError("unfinished game has inconsistent legacy draw rules")
            game["config"] = replace(config, max_plies=None, no_interaction_draw_plies=70)
        pool.update(target)
        changes.append(old)
    if changes:
        state["draw_rule_migration"] = {
            "update": int(payload["update"]),
            "environment_plies": state.get("cumulative", {}).get("environment_plies", 0),
            "from": changes[0], "to": target,
            "preserved": "models, optimizer moments, RNG, unfinished boards, histories, layout state, cumulative counters",
        }
