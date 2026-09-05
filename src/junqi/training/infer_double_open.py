from .inference import main as run
from .modes import TrainingMode


def main() -> None:
    run(default_mode=TrainingMode.DOUBLE_OPEN)


if __name__ == "__main__":
    main()
