"""One-step tactical rewards, kept outside rule outcomes and actor observations."""
from __future__ import annotations

from ..board import ArmPoint, PASS_ACTION, PointKind
from ..game import CombatOutcome, StepResult
from ..pieces import PieceType


BOMB_REWARD_FIELDS = {
    PieceType.COMMANDER: "bomb_commander_reward",
    PieceType.ARMY_COMMANDER: "bomb_army_commander_reward",
    PieceType.DIVISION_COMMANDER: "bomb_division_commander_reward",
}

NONFLAG_HQ_PENALIZED_KINDS = frozenset({
    PieceType.REGIMENT_COMMANDER, PieceType.BRIGADE_COMMANDER,
    PieceType.DIVISION_COMMANDER, PieceType.ARMY_COMMANDER, PieceType.COMMANDER,
    PieceType.ENGINEER, PieceType.BOMB,
})


def known_nonflag_hq_entry(game, action) -> bool:
    """Judge the chosen HQ entry/attack from the acting player's prior knowledge.

    Attempts count even if the attacker dies. A visible flag in the other HQ
    must establish the location before combat; an unknown flag never qualifies.
    This also covers an empty nonflag HQ on one's own or an ally's arm.
    """
    if action == 0 or tuple(action) == PASS_ACTION or game.current_player is None:
        return False
    actor = game.current_player
    board = game.board_for(actor)
    if board.point(action[1]).kind is not PointKind.HEADQUARTERS:
        return False
    start, end = (board.decode(code) for code in action)
    attacker = game.piece_at(start)
    if (attacker is None or attacker.owner != actor
            or attacker.kind not in NONFLAG_HQ_PENALIZED_KINDS):
        return False
    other = ArmPoint(end.seat, 6, 6 - end.column)
    flag = game.piece_at(other)
    # Use the identical public/private identity boundary as Actor observations.
    return (flag is not None and flag.owner == end.seat
            and game._known_piece_kind(actor, other, flag) is PieceType.FLAG)


def bomb_event(result: StepResult) -> tuple[int | None, str | None]:
    """Return the bomb owner and actual high-rank casualty, once per explosion.

    Either side can initiate combat. These referee labels travel only in the
    training reward channel; they never change an actor's knowledge or history.
    """
    if result.combat is not CombatOutcome.BOTH_REMOVED:
        return None, None
    for bomb, victim in ((result.attacker, result.defender), (result.defender, result.attacker)):
        if (bomb is not None and victim is not None and bomb.kind is PieceType.BOMB
                and victim.kind in BOMB_REWARD_FIELDS):
            return bomb.owner, victim.kind.value
    return None, None


def halfway_reward_scale(environment_plies: int, target_environment_plies: int | None, *, world_size: int = 1) -> float:
    """Linear global-step decay; do not let a DDP wave cross the half-budget cut."""
    stop = (target_environment_plies or 0) // 2
    remaining = stop - environment_plies
    if stop <= 0 or remaining < world_size:
        return 0.0
    return min(1.0, remaining / stop)
