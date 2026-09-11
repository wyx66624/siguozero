"""Independent four-country historical evaluation entry point (team rotations).

four_dark and double_open each require their own frozen baseline/output suite.
"""

from __future__ import annotations

from .evaluate_history import evaluate_history
from .modes import TrainingMode


def evaluate(argv: list[str] | None = None):
    return evaluate_history(argv, default_mode=TrainingMode.FOUR_DARK,
                            allowed_modes=(TrainingMode.FOUR_DARK.value, TrainingMode.DOUBLE_OPEN.value))


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
