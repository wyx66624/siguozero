"""Explicitly shrink a saved context without changing model or Adam tensors."""
from __future__ import annotations


def reconcile_context_window(payload, settings, *, adopt=False):
    saved = payload["config"]["model"]
    old = int(saved["max_transitions"])
    new = settings.model.max_transitions
    old_stride = saved.get("history_window_stride", 1)
    new_stride = settings.model.history_window_stride
    if old == new and old_stride == new_stride:
        return False
    if not adopt:
        raise ValueError("history window changed across resume; use --shrink-context-window")
    if not 0 < new <= old:
        raise ValueError("context migration only supports shrinking a positive history window")
    capacity = saved.get("position_embedding_capacity") or old + 1
    target_capacity = settings.model.position_embedding_capacity or new + 1
    if target_capacity != capacity:
        raise ValueError("context migration must preserve the saved position embedding capacity")
    state = payload["trainer_state"]
    pools = []
    if "base_game_pool" in state:
        pools.append(state["base_game_pool"])
    if "distributed" in state:
        pools.extend(rank["base_game_pool"] for rank in state["distributed"]["rank_states"])
    if not pools:
        raise ValueError("context migration requires saved unfinished game pools")
    histories = []
    seen = set()
    for pool in pools:
        if int(pool["max_transitions"]) != old:
            raise ValueError("saved pool context differs from the saved model")
        if pool.get("history_window_stride", 1) != old_stride:
            raise ValueError("saved pool stride differs from the saved model")
        for slot in pool["slots"]:
            groups = [slot["history"]]
            if slot.get("critic_history") is not None:
                groups.append(slot["critic_history"]["history"])
            for group in groups:
                for player in group["players"]:
                    if int(player["max_transitions"]) != old or not player["records"]:
                        raise ValueError("saved player context is inconsistent")
                    if player.get("history_window_stride", 1) != old_stride:
                        raise ValueError("saved player history stride is inconsistent")
                    if id(player) not in seen:
                        seen.add(id(player))
                        histories.append(player)
    # Validate the complete payload before making changes. Keep the pinned
    # initial observation and the latest transitions, including the current board.
    removed = 0
    for player in histories:
        records = player["records"]
        if len(records) > new + 1:
            removed += len(records) - new - 1
            player["records"] = [records[0], *records[-new:]]
        player["max_transitions"] = new
        player["history_window_stride"] = new_stride
    for pool in pools:
        pool["max_transitions"] = new
        pool["history_window_stride"] = new_stride
    saved.update(max_transitions=new, position_embedding_capacity=capacity,
                 history_window_stride=new_stride,
                 paged_kv_length_bucket_tokens=settings.model.paged_kv_length_bucket_tokens)
    state["context_window_migration"] = {
        "update": int(payload["update"]),
        "environment_plies": state.get("cumulative", {}).get("environment_plies", 0),
        "from": old, "to": new, "position_embedding_capacity": capacity,
        "stride_from": old_stride, "stride_to": new_stride,
        "removed_history_records": removed,
        "preserved": "all model/optimizer tensors, RNG, unfinished boards, pinned initial states, layout state and counters",
    }
    return True
