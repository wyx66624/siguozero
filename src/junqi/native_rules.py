"""Optional two-player integer geometry; Python referee owns all knowledge."""
from functools import lru_cache
import os

from . import inventory
from .board import PointKind
from .pieces import PieceType


def enabled(board):
    return (board.point_count == 60 and inventory._native_inventory is not None
            and hasattr(inventory._native_inventory, "legal60")
            and os.environ.get("SIGUOZERO_NATIVE_RULES", "0") == "1")


@lru_cache(maxsize=8)
def geometry(board):
    return inventory._native_inventory.geometry60(
        board._neighbors_all, board._neighbors_railway,
        tuple((a, b, int(d), int(e)) for (a, b), (d, e) in board._railway_directions.items()),
        tuple(1 if p.kind is PointKind.CAMP else 2 if p.kind is PointKind.HEADQUARTERS else 0
              for p in board.points))


def legal_actions(game, player):
    board = game._boards[player]
    encoded = bytearray(60)
    lookup = getattr(board, "_encoded_points", None)
    encode = lookup.__getitem__ if lookup is not None else board.encode
    for point, piece in game._pieces.items():
        encoded[encode(point)] = (piece.owner + 1) | (4 if piece.movable else 0) | (8 if piece.kind is PieceType.ENGINEER else 0)
    return inventory._native_inventory.legal60(
        geometry(board), encoded, player,
        game.config.allow_passes and game._passes_remaining[player] > 0,
        game.config.eliminate_no_moves)


def railway_targets(board, start, occupied, can_turn):
    mask = sum(1 << i for i in occupied)
    return set(inventory._native_inventory.railway60(geometry(board), start, mask, can_turn))
