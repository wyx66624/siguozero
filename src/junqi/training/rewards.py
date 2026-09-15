"""Training utility is separate from game outcomes and arena scores."""

DEFAULT_DRAW_REWARD = -0.15


def terminal_utility(outcome: float, *, draw_reward: float = DEFAULT_DRAW_REWARD) -> float:
    """Map a completed game's +1/0/-1 outcome to its training reward.

    Call only for terminal games, separately from event rewards. A rollout/batch
    boundary is neither a terminal state nor a draw.
    """
    return float(draw_reward if outcome == 0 else outcome)


def flag_capture_utility(game, captured_owner: int | None, *, player: int,
                         coefficient: float = 0.0) -> float:
    """Score one actual flag-capture event from the value owner's team view.

    The rules engine supplies captured_owner for this step only. Losing a
    teammate's flag also costs the team; elimination without capture scores zero.
    """
    if captured_owner is None:
        return 0.0
    sign = -1 if game.team_of(captured_owner) == game.team_of(player) else 1
    return sign * float(coefficient)
