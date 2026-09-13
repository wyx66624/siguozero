"""Explicit adoption of four voluntary passes without discarding training state."""
from __future__ import annotations


def reconcile_pass_rule(payload, *, adopt: bool = False):
    if payload["format_version"] >= 9:
        return
    if not adopt:
        raise RuntimeError("legacy checkpoint has no voluntary passes; resume with --adopt-pass-rule to adopt four passes per player")
    state = payload["trainer_state"]
    if state.get("model_selection", {}).get("rounds"):
        raise ValueError("pass-rule adoption after evaluated rounds requires a separate run/protocol")
    pools = [state["base_game_pool"]] if "base_game_pool" in state else []
    if "distributed" in state:
        pools.extend(rank["base_game_pool"] for rank in state["distributed"]["rank_states"])
    games = records = 0
    for pool in pools:
        for slot in pool["slots"]:
            game = slot["game"]
            count = game["config"].player_count
            game["passes_remaining"] = (4,) * count
            for player in slot["history"]["players"]:
                for record in player["records"]:
                    record["passes_remaining"] = (4,) * count + (0,) * (4 - count)
                    records += 1
            games += 1
    state["pass_rule_migration"] = {
        "update": int(payload["update"]),
        "environment_plies": state.get("cumulative", {}).get("environment_plies", 0),
        "from_max_passes_per_player": 0, "to_max_passes_per_player": 4,
        "pass_action": [0, 0], "pass_action_id": 0, "board_pass_input_dim": 4,
        "unfinished_games": games, "history_records": records,
        "preserved": "weights, Adam moments, RNG, boards, histories, layout state, cumulative counters",
    }
    payload["config"]["max_passes_per_player"] = 4
