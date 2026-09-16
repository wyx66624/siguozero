"""Referee-only histories for centralized value learning.

These arrays never replace the visibility-filtered GameHistory used by actors.
Each team has a fixed seat-relative global view, even after that seat is out.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from ..pieces import DEPLOYMENT_PIECE_SEQUENCE
from .encoding import EXACT_PIECE_CODE_STRIDE, OWN_PIECE_CODES, GameHistory, PlayerHistory
from .history_arrays import HistoryArrayView
from .modes import TrainingMode, mode_spec, normalize_mode
from .packed_observation import observation_rows


def centralized_observation_rows(game, mode):
    """Encode true current pieces, casualties and public rule state, per seat.

    Start with a separate allocation of public auxiliary fields; reading hidden
    identities here is intentional and confined to the training-only critic.
    No future randomness or future actions are included.
    """
    mode = normalize_mode(mode)
    rows = observation_rows(game, mode)
    points = mode_spec(mode).point_count
    inventory = Counter(DEPLOYMENT_PIECE_SEQUENCE)
    remaining = [Counter() for _ in range(game.config.player_count)]
    for piece in game._pieces.values():
        remaining[piece.owner][piece.kind] += 1
    for viewer, row in enumerate(rows):
        order = game._relative_player_order(viewer)
        relative = {owner: index for index, owner in enumerate(order)}
        board = game._boards[viewer]
        for point, piece in game._pieces.items():
            owner = relative[piece.owner]
            block = (2 if owner else 0) if mode is TrainingMode.TWO_PLAYER else owner
            row[board.encode(point)] = OWN_PIECE_CODES[piece.kind] + EXACT_PIECE_CODE_STRIDE * block
        if game.config.dead_rules_enabled:
            # Preserve the mode's canonical casualty slots through replay and
            # serialization. The full board already identifies every survivor.
            for owner in game._casualty_owner_order(viewer):
                slot = 1 if mode is TrainingMode.TWO_PLAYER else relative[owner] - 1
                begin = points + slot * 25
                row[begin:begin + 25] = game._casualty_bits(inventory - remaining[owner])
    return rows


class CentralizedHistory:
    """A separately owned global history, never an actor observation."""

    def __init__(self, history: GameHistory):
        self.history = history

    @classmethod
    def initialize(cls, game, mode, *, max_transitions=1000):
        mode = normalize_mode(mode)
        rows = centralized_observation_rows(game, mode)
        width = mode_spec(mode).point_count + (75 if game.config.dead_rules_enabled else 0)
        rows[:, width:width + 6] = 0  # The pinned initial token has no preceding action.
        players = tuple(PlayerHistory(
            mode, HistoryArrayView(mode, game.config.dead_rules_enabled, (0, seat), 0,
                                   (rows[seat:seat + 1],))[0],
            max_transitions=max_transitions,
        ) for seat in range(game.config.player_count))
        return cls(GameHistory(mode, players))

    def enable_array_storage(self):
        # GameHistory allocates a new identity; actor and critic prefixes cannot
        # alias even when their public portions happen to be equal.
        self.history.enable_array_storage()

    def append_encoded_rows(self, rows: np.ndarray):
        self.history.append_encoded_rows(rows)

    def value_state_for(self, team: int):
        if team not in (0, 1):
            raise ValueError("centralized value perspective must be team 0 or 1")
        return replace(self.history.players[team].as_value_state(), centralized=True)

    def state_dict(self):
        return {"observation_contract": "referee_global_v1", "history": self.history.state_dict()}

    @classmethod
    def from_state_dict(cls, state):
        if state.get("observation_contract") != "referee_global_v1":
            raise ValueError("MAPPO requires an explicitly saved global history")
        return cls(GameHistory.from_state_dict(state["history"]))
