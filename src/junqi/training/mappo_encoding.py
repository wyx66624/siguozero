"""Referee-only histories for centralized value learning.

These arrays never replace the visibility-filtered GameHistory used by actors.
Each player has a fixed seat-relative global view, even after that seat is out.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import os

from ..pieces import DEPLOYMENT_PIECE_SEQUENCE
from .encoding import EXACT_PIECE_CODE_STRIDE, OWN_PIECE_CODES, GameHistory, PlayerHistory, PolicyState
from .history_arrays import HistoryArrayView, observation_width
from .modes import TrainingMode, mode_spec, normalize_mode
from .packed_observation import observation_rows


def centralized_observation_rows(game, mode, *, public_rows=None):
    if normalize_mode(mode) is TrainingMode.TWO_PLAYER and os.environ.get("SIGUOZERO_FAST_OBSERVATIONS", "1") != "0":
        from .fast_observation import centralized_rows
        return centralized_rows(game,normalize_mode(mode),public_rows)
    return _centralized_observation_rows_reference(game,mode,public_rows=public_rows)

def _centralized_observation_rows_reference(game, mode, *, public_rows=None):
    """Encode true current pieces, casualties and public rule state, per seat.

    Start with a separate allocation of public auxiliary fields; reading hidden
    identities here is intentional and confined to the training-only critic.
    No future randomness or future actions are included.
    """
    mode = normalize_mode(mode)
    game.enable_inventory_tracking()
    game.enable_critic_inventory_tracking()
    # A worker has already encoded the actors' current observations. Copy that
    # allocation before replacing private fields; never mutate/share actor rows.
    rows = observation_rows(game, mode) if public_rows is None else public_rows.copy()
    points = mode_spec(mode).point_count
    if rows.shape != (game.config.player_count, observation_width(mode, game.config.dead_rules_enabled) + 14):
        raise ValueError('centralized observations require all current public rows')
    inventory = Counter(DEPLOYMENT_PIECE_SEQUENCE)
    remaining = [Counter() for _ in range(game.config.player_count)]
    for piece in game._pieces.values():
        remaining[piece.owner][piece.kind] += 1
    casualties = [game._casualty_bits(inventory - counts) for counts in remaining]
    mask_begin = points + (75 if game.config.dead_rules_enabled else 0)
    pieces = tuple(game._pieces.items())
    owners = np.asarray([piece.owner for _, piece in pieces], dtype=np.intp)
    codes = np.asarray([OWN_PIECE_CODES[piece.kind] for _, piece in pieces], dtype=np.int16)
    masks = np.asarray([1 << game._critic_inventory_slots[point] for point, _ in pieces], dtype=np.int32)
    low_masks, high_masks = masks & 8191, masks >> 13
    for viewer, row in enumerate(rows):
        order = game._relative_player_order(viewer)
        relative = {owner: index for index, owner in enumerate(order)}
        board = game._boards[viewer]
        relative_owners = np.asarray([relative[owner] for owner in range(game.config.player_count)])[owners]
        blocks = relative_owners * 2 if mode is TrainingMode.TWO_PLAYER else relative_owners
        indices = np.fromiter((board.encode(point) for point, _ in pieces),
                              dtype=np.intp, count=len(pieces))
        row[indices] = codes + EXACT_PIECE_CODE_STRIDE * blocks
        row[mask_begin + indices] = low_masks
        row[mask_begin + points + indices] = high_masks
        if game.config.dead_rules_enabled:
            # Preserve the mode's canonical casualty slots through replay and
            # serialization. The full board already identifies every survivor.
            for owner in game._casualty_owner_order(viewer):
                slot = 1 if mode is TrainingMode.TWO_PLAYER else relative[owner] - 1
                begin = points + slot * 25
                row[begin:begin + 25] = casualties[owner]
    return rows


class CentralizedHistory:
    """A separately owned global history, never an actor observation."""

    def __init__(self, history: GameHistory):
        self.history = history

    @classmethod
    def initialize(cls, game, mode, *, max_transitions=1000, history_window_stride=1):
        mode = normalize_mode(mode)
        rows = centralized_observation_rows(game, mode)
        width = observation_width(mode, game.config.dead_rules_enabled)
        rows[:, width:width + 6] = 0  # The pinned initial token has no preceding action.
        players = tuple(PlayerHistory(
            mode, HistoryArrayView(mode, game.config.dead_rules_enabled, (0, seat), 0,
                                   (rows[seat:seat + 1],))[0],
            max_transitions=max_transitions,
            history_window_stride=history_window_stride,
        ) for seat in range(game.config.player_count))
        return cls(GameHistory(mode, players))

    def enable_array_storage(self):
        # GameHistory allocates a new identity; actor and critic prefixes cannot
        # alias even when their public portions happen to be equal.
        self.history.enable_array_storage()

    def append_encoded_rows(self, rows: np.ndarray):
        self.history.append_encoded_rows(rows)

    def value_state_for(self, player: int):
        if type(player) is not int or not 0 <= player < len(self.history.players):
            raise ValueError("centralized value perspective must be a valid player")
        return PolicyState(self.history.mode, self.history.players[player].records, (), centralized=True)

    def state_dict(self):
        return {"observation_contract": "referee_global_inventory_instances_v2", "history": self.history.state_dict()}

    @classmethod
    def from_state_dict(cls, state):
        if state.get("observation_contract") != "referee_global_inventory_instances_v2":
            raise ValueError("MAPPO critic requires an explicitly saved global history with inventory instance encoding")
        history = GameHistory.from_state_dict(state["history"])
        return cls(history)
