"""One adaptive history-pool member: current policy with public tactical rules.

No game/referee object is accepted. Geometry, visible identities and public
inventory candidates come exclusively from the acting player's observation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ..board import FourPlayerBoard, PASS_ACTION, PointKind
from ..game import CombatOutcome, JunqiGame
from ..inventory import INVENTORY_SLOTS
from ..pieces import PIECE_RANKS, PieceType
from .modes import TrainingMode
from .tactical_opponent import _EXACT


LIVE_CURRENT_ID = "live_current_tactical"
CURRENT_RULE_VERSION = 1
CURRENT_RULE_PRIORITY = ("defend_flag", "bomb_known_high", "capture_known_engineer", "free_capture")
_BOARD = FourPlayerBoard()
_REMOVES = frozenset((CombatOutcome.ATTACKER_WINS, CombatOutcome.BOTH_REMOVED))


@dataclass(frozen=True)
class _PublicPiece:
    owner: int
    kinds: frozenset[PieceType]
    known: PieceType | None


@lru_cache(maxsize=4096)
def _kinds(mask):
    return frozenset(kind for i, kind in enumerate(INVENTORY_SLOTS) if mask & (1 << i))


def _pieces(state):
    masks = state.records[-1].inventory_masks
    pieces = {}
    for point, code in enumerate(state.current_board):
        exact = _EXACT.get(int(code))
        if exact is not None:
            owner, known = exact
            pieces[point] = _PublicPiece(owner, frozenset((known,)), known)
        elif code in (1, 2, 3):
            kinds = _kinds(masks[point]) if masks is not None else frozenset()
            pieces[point] = _PublicPiece(int(code), kinds, next(iter(kinds)) if len(kinds) == 1 else None)
    return pieces


def _threats_to(target, pieces, active_mask):
    """Confirmed one-move enemy removals under every public candidate identity."""
    defender = pieces[target]
    if _BOARD.point(target).kind is PointKind.CAMP:
        return ()
    occupied = frozenset(pieces)
    threats = []
    for start, enemy in pieces.items():
        if (enemy.owner % 2 == defender.owner % 2 or not active_mask & (1 << enemy.owner)
                or not enemy.kinds or enemy.kinds.intersection((PieceType.MINE, PieceType.FLAG))
                or _BOARD.point(start).kind is PointKind.HEADQUARTERS):
            continue
        if not all(JunqiGame._combat_outcome_for_kinds(a, d) in _REMOVES
                   for a in enemy.kinds for d in defender.kinds):
            continue
        if target not in _BOARD.neighbors(start) and not all(
                target in JunqiGame._railway_reachable_targets(_BOARD, start, occupied, can_turn=engineer)
                for engineer in {kind is PieceType.ENGINEER for kind in enemy.kinds}):
            continue
        threats.append(start)
    return tuple(threats)


def current_tactical_actions(state):
    if state.centralized:
        raise ValueError("tactical opponent cannot read centralized observations")
    if state.mode is TrainingMode.TWO_PLAYER:
        raise ValueError("current tactical opponent requires a four-player mode")
    pieces = _pieces(state)
    active_mask = state.records[-1].active_mask
    captures = []
    for action in state.legal_actions:
        if action == PASS_ACTION:
            continue
        own, enemy = pieces.get(action[0]), pieces.get(action[1])
        if own and own.owner == 0 and own.known and enemy and enemy.owner in (1, 3):
            captures.append((action, own, enemy))
    if not captures:
        return None, ()

    threats = {attacker for point, piece in pieces.items()
               if piece.owner == 0 and piece.known is PieceType.FLAG
               for attacker in _threats_to(point, pieces, active_mask)}
    defense = [(action, own, enemy) for action, own, enemy in captures if action[1] in threats]
    if defense:
        # Prefer a publicly guaranteed removal. If none is certain, attempt
        # defense with the largest legal piece instead of silently ignoring it.
        certain = [(a, own, enemy) for a, own, enemy in defense if enemy.kinds and all(
            JunqiGame._combat_outcome_for_kinds(own.known, kind) in _REMOVES for kind in enemy.kinds)]
        choices = certain or defense
        largest = max(PIECE_RANKS.get(own.known, 0) for _, own, _ in choices)
        return "defend_flag", tuple(a for a, own, _ in choices if PIECE_RANKS.get(own.known, 0) == largest)

    bombs = tuple(a for a, own, enemy in captures
                  if own.known is PieceType.BOMB and PIECE_RANKS.get(enemy.known, 0) >= 7)
    if bombs:
        return "bomb_known_high", bombs
    engineers = tuple(a for a, own, enemy in captures if enemy.known is PieceType.ENGINEER
                      and JunqiGame._combat_outcome_for_kinds(own.known, enemy.known) in _REMOVES)
    if engineers:
        return "capture_known_engineer", engineers

    free = []
    for action, own, enemy in captures:
        if not enemy.kinds or not all(JunqiGame._combat_outcome_for_kinds(own.known, kind)
                                      is CombatOutcome.ATTACKER_WINS for kind in enemy.kinds):
            continue
        after = dict(pieces)
        del after[action[0]]
        if enemy.known is PieceType.FLAG:
            after = {point: piece for point, piece in after.items() if piece.owner != enemy.owner}
        after[action[1]] = own
        if not _threats_to(action[1], after, active_mask):
            free.append(action)
    return ("free_capture", tuple(free)) if free else (None, ())
