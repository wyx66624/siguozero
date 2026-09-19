"""One-step bomb rewards, kept outside rule outcomes and actor observations."""
from __future__ import annotations

from ..game import CombatOutcome, StepResult
from ..pieces import PieceType


BOMB_REWARD_FIELDS = {
    PieceType.COMMANDER: "bomb_commander_reward",
    PieceType.ARMY_COMMANDER: "bomb_army_commander_reward",
    PieceType.DIVISION_COMMANDER: "bomb_division_commander_reward",
}


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
