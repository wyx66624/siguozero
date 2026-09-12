"""Numerical player observations using the referee's visibility filter.

Avoid allocating ObservedPiece/ObservedEvent/StateTokenRecord for every seat
at every PPO step. This shares the exact knowledge lookup used by observe();
hidden piece identities are never read to construct a player's board codes.
"""
from __future__ import annotations

import numpy as np

from .encoding import ACTION_PLAYER_PAD, EXACT_PIECE_CODE_STRIDE, OWN_PIECE_CODES, ActionFeatures
from .modes import TrainingMode, mode_spec


def observation_rows(game, mode):
    points = mode_spec(mode).point_count
    dead = game.config.dead_rules_enabled
    width = points + (75 if dead else 0)
    rows = np.zeros((game.config.player_count, width + 10), dtype=np.int16)
    event = game._public_history[-1] if game._public_history else None
    for viewer, row in enumerate(rows):
        board = game._boards[viewer]
        order = game._relative_player_order(viewer)
        relative = {owner: index for index, owner in enumerate(order)}
        for point, piece in game._pieces.items():
            owner = relative[piece.owner]
            # Reuse the public-observation knowledge contract, including
            # exact dead-rule deductions and double-open teammate visibility.
            kind = game._known_piece_kind(viewer, point, piece)
            if kind is not None:
                block = (2 if owner else 0) if mode is TrainingMode.TWO_PLAYER else owner
                code = OWN_PIECE_CODES[kind] + EXACT_PIECE_CODE_STRIDE * block
            else:
                if owner == 0 or (mode is TrainingMode.DOUBLE_OPEN and owner == 2):
                    raise ValueError('visible own/teammate identity unexpectedly hidden')
                code = 2 if mode is TrainingMode.TWO_PLAYER else owner
            row[board.encode(point)] = code
        if dead:
            for owner in game._casualty_owner_order(viewer):
                slot = 1 if mode is TrainingMode.TWO_PLAYER else relative[owner] - 1
                begin = points + slot * 25
                row[begin:begin + 25] = game._casualty_bits(game._known_casualties[viewer][owner])
        if event is not None:
            action = ActionFeatures(board.encode(event.start), board.encode(event.end), relative[event.actor])
            row[width:width + 5] = action.as_vector(mode)
        row[width + 5:] = (
            event is not None, min(game.no_interaction_plies, 60),
            sum(1 << i for i, owner in enumerate(order) if game._active[owner]),
            sum(1 << i for i, owner in enumerate(order) if game._flag_revealed[owner]),
            ACTION_PLAYER_PAD if game.current_player is None else relative[game.current_player],
        )
    return rows
