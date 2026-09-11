"""Independent two-player historical evaluation entry point (two-game pairs)."""

from __future__ import annotations

from .evaluate_history import evaluate_history
from .modes import TrainingMode


def evaluate(argv: list[str] | None = None):
    return evaluate_history(argv, default_mode=TrainingMode.TWO_PLAYER,
                            allowed_modes=(TrainingMode.TWO_PLAYER.value,))


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
