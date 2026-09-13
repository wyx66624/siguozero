"""Training utility is separate from game outcomes and arena scores."""

DEFAULT_DRAW_REWARD = -0.15


def terminal_utility(outcome: float, *, draw_reward: float = DEFAULT_DRAW_REWARD) -> float:
    """Map a completed game's +1/0/-1 outcome to its training reward.

    Call only for terminal games. Intermediate transitions still receive zero;
    a rollout/batch boundary is neither a terminal state nor a draw.
    """
    return float(draw_reward if outcome == 0 else outcome)
