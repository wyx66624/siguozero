"""Command-line entry points for the three independent training profiles."""

from __future__ import annotations

import argparse
from typing import Sequence

from .modes import TrainingMode
from .settings import TrainingSettings
from .trainer import SelfPlayTrainer


def build_parser(default_mode: TrainingMode | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SiguoZero with terminal-only K=4, M=2 Game-GRPO"
    )
    parser.add_argument(
        "--mode",
        choices=[item.value for item in TrainingMode],
        default=None if default_mode is None else default_mode.value,
        required=default_mode is None,
    )
    parser.add_argument("--config", default="configs/bootstrap.yaml")
    parser.add_argument(
        "--run-directory",
        "--run-dir",
        dest="run_directory",
        default=None,
        help=(
            "output root or mode directory; mode and with_dead_rules/"
            "without_dead_rules are appended as needed"
        ),
    )
    dead_rules = parser.add_mutually_exclusive_group()
    dead_rules.add_argument(
        "--dead-rules",
        dest="dead_rules_enabled",
        action="store_true",
        help="enable persistent deterministic identities and casualty priors",
    )
    dead_rules.add_argument(
        "--no-dead-rules",
        dest="dead_rules_enabled",
        action="store_false",
        help="disable deterministic identities and remove the casualty branch",
    )
    parser.set_defaults(dead_rules_enabled=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--model-scale",
        choices=("bootstrap", "main", "extended"),
        default="bootstrap",
    )
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument(
        "--target-continuation-plies",
        type=int,
        default=None,
        help="stop after at least this many terminal-continuation environment plies",
    )
    parser.add_argument("--anchor-batch", type=int, default=None)
    parser.add_argument("--microbatch", type=int, default=None)
    parser.add_argument("--actor-batch", type=int, default=None)
    parser.add_argument("--base-game-pool", type=int, default=None)
    parser.add_argument("--max-game-plies", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--archive-every", type=int, default=None)
    parser.add_argument("--keep-checkpoint-archives", type=int, default=None)
    parser.add_argument("--resource-monitor-seconds", type=float, default=None)
    parser.add_argument(
        "--no-auto-microbatch-fallback",
        action="store_true",
        help="fail instead of halving the learner microbatch after CUDA OOM",
    )
    parser.add_argument(
        "--amp",
        choices=("bfloat16", "float16", "float32"),
        default=None,
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="require an empty/new run directory; existing checkpoints are never overwritten",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="use a tiny model, one anchor, eight short terminal continuations",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    default_mode: TrainingMode | None = None,
) -> None:
    args = build_parser(default_mode).parse_args(argv)
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.updates is not None:
        overrides["total_updates"] = args.updates
    if args.target_continuation_plies is not None:
        overrides["target_continuation_plies"] = args.target_continuation_plies
    if args.anchor_batch is not None:
        overrides["anchor_batch"] = args.anchor_batch
    if args.microbatch is not None:
        overrides["policy_microbatch"] = args.microbatch
    if args.actor_batch is not None:
        overrides["actor_inference_batch"] = args.actor_batch
    if args.base_game_pool is not None:
        overrides["base_game_pool_size"] = args.base_game_pool
    if args.max_game_plies is not None:
        overrides["max_game_plies"] = args.max_game_plies
    if args.checkpoint_every is not None:
        overrides["checkpoint_every_updates"] = args.checkpoint_every
    if args.archive_every is not None:
        overrides["archive_every_updates"] = args.archive_every
    if args.keep_checkpoint_archives is not None:
        overrides["keep_checkpoint_archives"] = args.keep_checkpoint_archives
    if args.resource_monitor_seconds is not None:
        overrides["resource_monitor_interval_seconds"] = (
            args.resource_monitor_seconds
        )
    if args.no_auto_microbatch_fallback:
        overrides["auto_reduce_microbatch_on_oom"] = False
    if args.amp is not None:
        overrides["amp"] = args.amp
    settings = TrainingSettings.from_yaml(
        args.config,
        args.mode,
        tiny=args.smoke_test,
        model_scale=args.model_scale,
        dead_rules_enabled=args.dead_rules_enabled,
        overrides=overrides,
    )
    run_directory = settings.resolve_run_directory(args.run_directory)
    trainer = SelfPlayTrainer(
        settings,
        run_directory=run_directory,
        auto_resume=not args.no_resume,
    )
    trainer.train()


if __name__ == "__main__":
    main()
