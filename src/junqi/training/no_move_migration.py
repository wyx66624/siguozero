"""Adopt immobility elimination without creating a pass or a training action."""
from __future__ import annotations

from dataclasses import replace


def reconcile_no_move_rule(payload, *, adopt=False):
    state = payload["trainer_state"]
    pools = [state["base_game_pool"]] if "base_game_pool" in state else []
    pools.extend(rank["base_game_pool"] for rank in state.get("distributed", {}).get("rank_states", []))
    if payload["config"].get("eliminate_no_moves") is True:
        if any(not slot["game"]["config"].eliminate_no_moves for pool in pools for slot in pool["slots"]):
            raise ValueError("saved pool contradicts the no-move elimination rule")
        return
    if not adopt:
        raise RuntimeError("checkpoint permits forced passes; resume with --adopt-no-move-elimination")

    from .encoding import GameHistory, _record_state_dict
    from .history_arrays import HistoryArrayView, observation_width
    from .mappo_encoding import CentralizedHistory, centralized_observation_rows
    from .packed_observation import observation_rows
    from .rollout import BaseGamePool

    changes, terminal_games, games = [], [], 0
    for rank, raw in enumerate(pools):
        pool = BaseGamePool(raw["mode"], pool_size=raw["pool_size"], max_transitions=raw["max_transitions"],
            max_game_plies=raw["max_game_plies"], no_capture_draw_plies=raw["no_capture_draw_plies"],
            dead_rules_enabled=raw["dead_rules_enabled"], centralized_critic=raw.get("centralized_critic", False), seed=0)
        retained = []
        for index, saved in enumerate(raw["slots"]):
            # Restore with the original rule, including every historical token.
            slot = pool._slot_from_state_dict(saved)
            game = slot.game
            game.config = replace(game.config, eliminate_no_moves=True)
            game._invalidate_legal_actions()
            eliminated = game._resolve_current_turn()
            games += 1
            if not eliminated:
                retained.append(dict(saved, game=dict(saved["game"], config=game.config)))
                continue
            change = dict(rank=rank, slot=index, ply=game.ply_count, eliminated=list(eliminated))
            changes.append(change)
            if game.is_terminal:
                # A rule change can finish a board without a new actor action.
                # Refill this slot normally; do not invent PPO samples/plies or
                # attach this result to another game's layout or action.
                terminal_games.append(dict(change, rewards=list(game.rewards())))
                continue
            if game._public_history:
                event = game._public_history[-1]
                game._public_history[-1] = replace(event,
                    eliminated_players=tuple(dict.fromkeys((*event.eliminated_players, *eliminated))))
            public = observation_rows(game, pool.mode)
            streams = [(saved["history"], public)]
            if slot.critic_history is not None:
                streams.append((saved["critic_history"]["history"],
                                centralized_observation_rows(game, pool.mode, public_rows=public)))
            histories = []
            for history, rows in streams:
                players = []
                for viewer, player in enumerate(history["players"]):
                    records = list(player["records"])
                    if len(records) == 1:
                        width = observation_width(pool.mode, game.config.dead_rules_enabled)
                        rows[viewer, width:width+6] = 0
                    record = HistoryArrayView(pool.mode, game.config.dead_rules_enabled,
                        (0, viewer), 0, (rows[viewer:viewer+1],))[0]
                    records[-1] = _record_state_dict(record)
                    players.append(dict(player, records=records))
                histories.append(GameHistory.from_state_dict(dict(history, players=players)))
            slot.history = histories[0]
            if slot.critic_history is not None:
                slot.critic_history = CentralizedHistory(histories[1])
            retained.append(pool._slot_state_dict(slot))
        raw["slots"] = retained
    payload["config"]["eliminate_no_moves"] = True
    state["no_move_rule_migration"] = dict(update=payload["update"],
        environment_plies=state.get("cumulative", {}).get("environment_plies", 0),
        games=games, resolved_positions=changes, terminal_games=terminal_games,
        from_rule="pass_can_keep_immobile_player_alive", to_rule="eliminate_before_decision",
        preserved="weights, optimizers, RNG, counters, continuing games and prior history; only current snapshots change when a seat is eliminated")
