"""Explicitly retire voluntary passes without rewriting past actions or weights."""
from dataclasses import replace

from ..game import GameVariant
from .modes import TrainingMode


def reconcile_two_player_no_pass(payload, settings, *, adopt=False):
    if settings.mode is not TrainingMode.TWO_PLAYER:
        return False
    state = payload["trainer_state"]
    pools = [state["base_game_pool"]] if "base_game_pool" in state else []
    pools.extend(rank["base_game_pool"] for rank in state.get("distributed", {}).get("rank_states", []))
    slots = [slot for pool in pools for slot in pool["slots"]]
    previous = payload["config"].get("max_passes_per_player", 4)
    if previous not in (0, 4):
        raise ValueError("unsupported saved two-player pass budget")
    if previous == 4 and not adopt:
        raise ValueError("two-player checkpoint permits passes; use --adopt-two-player-no-pass")
    # Validate every rank before mutating any saved state.
    for slot in slots:
        game = slot["game"]
        if (game["config"].variant is not GameVariant.TWO_PLAYER
                or game["config"].max_passes_per_player != previous
                or len(game["passes_remaining"]) != 2
                or any(type(n) is not int or not 0 <= n <= previous for n in game["passes_remaining"])):
            raise ValueError("saved pool contradicts the two-player pass rule")
        for history in _histories(slot):
            if len(history["players"]) != 2 or any(not p["records"] for p in history["players"]):
                raise ValueError("incomplete saved two-player history")
            if previous == 0 and any(any(p["records"][-1]["passes_remaining"]) for p in history["players"]):
                raise ValueError("current history contradicts the no-pass rule")
    if previous == 0:
        return False
    budgets = []
    for slot in slots:
        game = slot["game"]
        budgets.append(list(game["passes_remaining"]))
        game["config"] = replace(game["config"], allow_passes=False)
        game["passes_remaining"] = (0, 0)
        # Earlier records describe real past observations, including past passes.
        # Only the current decision observes the new zero budget.
        for history in _histories(slot):
            for player in history["players"]:
                player["records"][-1] = dict(player["records"][-1], passes_remaining=(0, 0, 0, 0))
    payload["config"]["max_passes_per_player"] = 0
    state["two_player_no_pass_migration"] = dict(
        update=payload["update"], environment_plies=state.get("cumulative", {}).get("environment_plies", 0),
        games=len(slots), previous_budgets=budgets, from_passes=4, to_passes=0,
        preserved="weights, optimizers, RNG, counters, boards and past actions; only current pass budgets change")
    return True


def _histories(slot):
    yield slot["history"]
    if slot.get("critic_history") is not None:
        yield slot["critic_history"]["history"]
