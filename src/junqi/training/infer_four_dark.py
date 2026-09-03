from .inference import main as run
from .modes import TrainingMode


def main() -> None:
    run(default_mode=TrainingMode.FOUR_DARK)


if __name__ == "__main__":
    main()
