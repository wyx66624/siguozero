"""Numerical player observations using the referee's visibility filter.

Avoid allocating ObservedPiece/ObservedEvent/StateTokenRecord for every seat
at every PPO step. This shares the exact knowledge lookup used by observe();
hidden piece identities are never read to construct a player's board codes.
"""
from __future__ import annotations

import numpy as np

from .encoding import ACTION_PLAYER_PAD, EXACT_PIECE_CODE_STRIDE, NO_CAPTURE_COUNTER_MAX, OWN_PIECE_CODES, ActionFeatures
from .modes import TrainingMode, mode_spec
from .history_arrays import observation_width


def observation_rows(game, mode):
    if game._inventory is None:
        raise ValueError("actor observations require inventory tracking from game initialization")
    points = mode_spec(mode).point_count
    dead = game.config.dead_rules_enabled
    width = observation_width(mode, dead)
    rows = np.zeros((game.config.player_count, width + 14), dtype=np.int16)
    event = game._public_history[-1] if game._public_history else None
    pieces = tuple(game._pieces.items())
    for viewer, row in enumerate(rows):
        board = game._boards[viewer]
        order = game._relative_player_order(viewer)
        relative = {owner: index for index, owner in enumerate(order)}
        masks = game._inventory.masks(game, viewer)
        begin = points + (75 if dead else 0)
        indices = np.fromiter((board.encode(point) for point, _ in pieces),
                              dtype=np.intp, count=len(pieces))
        codes, inventory_masks = [], []
        for point, piece in pieces:
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
            codes.append(code)
            inventory_masks.append(masks[point])
        # Scatter each field once; the public knowledge and copy-slot lookup
        # above remains viewer-local and never reads hidden piece kinds.
        packed_masks = np.asarray(inventory_masks, dtype=np.int32)
        row[indices] = codes
        row[begin + indices] = packed_masks & 8191
        row[begin + points + indices] = packed_masks >> 13
        if dead:
            for owner in game._casualty_owner_order(viewer):
                slot = 1 if mode is TrainingMode.TWO_PLAYER else relative[owner] - 1
                begin = points + slot * 25
                row[begin:begin + 25] = game._casualty_bits(game._known_casualties[viewer][owner])
        if event is not None:
            action = (ActionFeatures(0, 0, relative[event.actor]) if event.start is None else
                      ActionFeatures(board.encode(event.start), board.encode(event.end), relative[event.actor]))
            row[width:width + 5] = action.as_vector(mode)[:5]
        row[width + 10:width + 10 + len(order)] = [game._passes_remaining[owner] for owner in order]
        row[width + 5:width + 10] = (
            event is not None, min(game.no_interaction_plies, NO_CAPTURE_COUNTER_MAX),
            sum(1 << i for i, owner in enumerate(order) if game._active[owner]),
            sum(1 << i for i, owner in enumerate(order) if game._flag_revealed[owner]),
            ACTION_PLAYER_PAD if game.current_player is None else relative[game.current_player],
        )
    return rows
