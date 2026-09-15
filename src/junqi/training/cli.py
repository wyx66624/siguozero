"""Command-line entry points for the three independent training profiles."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Sequence

from .distributed import DistributedContext
from .modes import TrainingMode
from .settings import TrainingSettings
from .trainer import SelfPlayTrainer


def build_parser(default_mode: TrainingMode | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train four-player PPO with a critic, or two-player K=4/M=2 GRPO"
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
    parser.add_argument(
        "--device",
        default=None,
        help="auto, cpu, cuda[:N], or npu[:N] (auto prefers an available NPU)",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model-scale",
        choices=("bootstrap", "main", "extended"),
        default="bootstrap",
    )
    parser.add_argument("--updates", type=int, default=None)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--target-environment-plies",
        type=int,
        default=None,
        help="total training environment transitions across all games/ranks, including simulated branches; network passes and repeated epochs add no steps",
    )
    budget.add_argument(
        "--target-continuation-plies", type=int, default=None,
        help="legacy GRPO branch-only budget; for PPO an alias for --target-environment-plies",
    )
    parser.add_argument("--anchor-batch", "--transition-batch", type=int, default=None,
                        help="global PPO transitions or GRPO anchor groups per update")
    parser.add_argument("--microbatch", type=int, default=None)
    parser.add_argument("--ppo-minibatch", type=int, default=None,
                        help="PPO decision samples per optimizer step per rank")
    parser.add_argument("--no-ppo-sequences", action="store_true",
                        help="disable exact prefix sequence training for A/B measurements")
    parser.add_argument("--grpo-equivalent-plies", type=int, default=None,
                        help="legacy estimated root-state coverage, not an environment interaction count; overrides the default budget")
    parser.add_argument("--grpo-mean-remaining-plies", type=float, default=None)
    parser.add_argument("--warmup-updates", type=int, default=None)
    parser.add_argument("--actor-batch", type=int, default=None)
    parser.add_argument("--rollout-anchor-wave", type=int, default=None)
    parser.add_argument("--environment-workers", type=int, default=None,
                        help='PPO CPU worker processes per rank; 1 selects serial collection (GRPO uses threads)')
    parser.add_argument('--activation-checkpointing', action=argparse.BooleanOptionalAction, default=None,
                        help='recompute full temporal blocks to reduce activation memory')
    parser.add_argument('--causal-sdpa', action=argparse.BooleanOptionalAction, default=None,
                        help='use direct causal SDPA for temporal attention')
    parser.add_argument('--board-chunk-size', type=int, default=None)
    parser.add_argument('--layout-prefetch-games', type=int, default=None)
    parser.add_argument('--ppo-deferred-values', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--ppo-fused-optimizer', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--ppo-pipeline-groups', type=int, default=None)
    parser.add_argument('--ppo-tensor-learner', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--ppo-varlen-attention', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--ppo-sampling-graphs', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--ppo-low-precision-residual', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--ppo-compile-mode', choices=('off', 'default', 'reduce-overhead', 'max-autotune-no-cudagraphs'), default=None)
    parser.add_argument("--temporal-cache-entries", type=int, default=None)
    parser.add_argument("--paged-kv-length-bucket", type=int, default=None)
    parser.add_argument(
        "--no-incremental-inference",
        action="store_true",
        help="disable rollout KV reuse for an exact A/B benchmark",
    )
    parser.add_argument(
        "--no-paged-kv",
        action="store_true",
        help="use the legacy contiguous rollout KV cache for an A/B benchmark",
    )
    parser.add_argument("--no-ppo-array-history", action="store_true",
                        help="use replay-record histories for a PPO pipeline A/B check")
    parser.add_argument("--no-ppo-fixed-kv", action="store_true",
                        help="use the legacy prefix KV backend for PPO")
    parser.add_argument("--no-ppo-cuda-graphs", action="store_true",
                        help="run fixed-slot PPO decode eagerly without CUDA graph replay")
    parser.add_argument("--base-game-pool", type=int, default=None)
    parser.add_argument("--max-game-plies", type=int, default=None)
    parser.add_argument("--checkpoint-policy", choices=("periodic", "evaluation"), default=None,
                        help="evaluation saves only for scheduled matches; periodic also saves at update intervals and on exit")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="update interval for the periodic checkpoint policy")
    parser.add_argument("--checkpoint-interval-environment-plies", type=int, default=None,
                        help="save every N global environment plies; 0 selects the update interval")
    parser.add_argument("--adopt-draw-penalty", action="store_true",
                        help="explicitly adopt the configured draw reward while preserving the complete latest checkpoint")
    parser.add_argument("--adopt-flag-capture-reward", action="store_true",
                        help="adopt the configured PPO flag reward for future steps, preserving the complete latest checkpoint")
    parser.add_argument("--adopt-champion-evaluation", action="store_true",
                        help="switch future evaluations to the historical champion while preserving training state and completed reports")
    parser.add_argument("--adopt-pass-rule", action="store_true",
                        help="explicitly resume a pre-evaluation legacy checkpoint with four voluntary passes per player")
    parser.add_argument("--expand-game-pool", action="store_true",
                        help="allow a single-rank checkpoint to grow its parallel pool while preserving every saved game and RNG")
    parser.add_argument("--reset-oom-batch-limits", action="store_true",
                        help="explicitly retry configured actor/learner batch limits after a memory optimization")
    parser.add_argument("--adopt-current-draw-rules", action="store_true",
                        help="resume a pre-evaluation 2000/60 run with unlimited/70 capture rules, preserving training state")
    parser.add_argument("--archive-every", type=int, default=None)
    parser.add_argument("--keep-checkpoint-archives", type=int, default=None)
    parser.add_argument(
        "--no-model-selection",
        action="store_true",
        help="disable scheduled current-versus-best matches (also disables automatic saves with the evaluation checkpoint policy)",
    )
    parser.add_argument("--arena-games", type=int, default=None,
                        help="total games per selection, divisible by 2 (two-player) or 4 (four-player)")
    parser.add_argument("--arena-start-percent", type=int, default=None,
                        help="first training progress percentage to evaluate (default: 30)")
    parser.add_argument("--arena-interval-percent", type=int, default=None,
                        help="training progress between selections (default: 5 percentage points)")
    parser.add_argument("--arena-interval-environment-plies", type=int, default=None,
                        help="select every N global training environment transitions (four-player default: 50000000)")
    parser.add_argument("--arena-after-half-interval-environment-plies", type=int, default=None,
                        help="evaluation interval after half the environment budget; 0 disables the second interval")
    parser.add_argument("--arena-after-half-historical-only", action=argparse.BooleanOptionalAction, default=None,
                        help="use only the fixed historical panel after half; keep the latest model regardless of scores")
    parser.add_argument("--arena-max-plies", type=int, default=None,
                        help="maximum moves per arena game; 0 disables the cap (default: unlimited)")
    parser.add_argument("--arena-parallel-games", type=int, default=None,
                        help="active arena games per rank (default: 32)")
    parser.add_argument("--arena-inference-batch", dest="arena_inference_batch_size", type=int,
                        default=None, help="maximum arena inference requests per batch (default: 32)")
    parser.add_argument("--arena-environment-workers", type=int, default=None,
                        help="arena environment threads per rank (default: 4)")
    parser.add_argument("--resource-monitor-seconds", type=float, default=None)
    parser.add_argument(
        "--no-auto-microbatch-fallback",
        action="store_true",
        help="fail instead of halving the learner microbatch after accelerator OOM",
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
        "--init-from", default=None,
        help="initialize weights from a checkpoint in a new run, with fresh optimizers/counters",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="use a tiny model and one short PPO/GRPO update; disable model selection",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    default_mode: TrainingMode | None = None,
) -> None:
    parser = build_parser(default_mode)
    args = parser.parse_args(argv)
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.updates is not None:
        overrides["total_updates"] = args.updates
    if args.target_continuation_plies is not None:
        overrides["target_continuation_plies"] = args.target_continuation_plies
    if args.target_environment_plies is not None:
        overrides["target_environment_plies"] = args.target_environment_plies
    if args.anchor_batch is not None:
        overrides["anchor_batch"] = args.anchor_batch
    if args.microbatch is not None:
        overrides["policy_microbatch"] = args.microbatch
    if args.ppo_minibatch is not None:
        overrides["ppo_minibatch_samples"] = args.ppo_minibatch
    if args.no_ppo_sequences:
        overrides["ppo_sequence_training"] = False
    if args.no_ppo_array_history and args.ppo_pipeline_groups is None:
        overrides['ppo_pipeline_groups'] = 1
    if args.grpo_equivalent_plies is not None:
        overrides["grpo_equivalent_plies"] = args.grpo_equivalent_plies
    if args.grpo_mean_remaining_plies is not None:
        overrides["grpo_mean_remaining_plies"] = args.grpo_mean_remaining_plies
    if args.warmup_updates is not None:
        overrides["warmup_updates"] = args.warmup_updates
    if args.actor_batch is not None:
        overrides["actor_inference_batch"] = args.actor_batch
    if args.rollout_anchor_wave is not None:
        overrides["rollout_anchor_wave_size"] = args.rollout_anchor_wave
    if args.environment_workers is not None:
        overrides["rollout_environment_workers"] = args.environment_workers
    for name in ('layout_prefetch_games', 'ppo_deferred_values', 'ppo_pipeline_groups', 'ppo_fused_optimizer'):
        if getattr(args, name) is not None:
            overrides[name] = getattr(args, name)
    if args.base_game_pool is not None:
        overrides["base_game_pool_size"] = args.base_game_pool
    if args.max_game_plies is not None:
        overrides["max_game_plies"] = None if args.max_game_plies == 0 else args.max_game_plies
    if args.checkpoint_every is not None:
        overrides["checkpoint_every_updates"] = args.checkpoint_every
        overrides["checkpoint_interval_environment_plies"] = None
    if args.checkpoint_interval_environment_plies is not None:
        overrides["checkpoint_interval_environment_plies"] = args.checkpoint_interval_environment_plies or None
    if args.checkpoint_policy is not None:
        overrides["checkpoint_policy"] = args.checkpoint_policy
    if args.archive_every is not None:
        overrides["archive_every_updates"] = args.archive_every
    if args.keep_checkpoint_archives is not None:
        overrides["keep_checkpoint_archives"] = args.keep_checkpoint_archives
    if args.no_model_selection:
        overrides["arena_enabled"] = False
    if (args.arena_interval_environment_plies is not None or args.arena_after_half_interval_environment_plies not in (None, 0)
            or args.arena_after_half_historical_only is True) and (
        args.arena_start_percent is not None or args.arena_interval_percent is not None
    ):
        parser.error("choose environment-step or percentage model selection, not both")
    if args.arena_start_percent is not None or args.arena_interval_percent is not None:
        overrides["arena_interval_environment_plies"] = None
        overrides["arena_after_half_interval_environment_plies"] = None
        overrides["arena_after_half_historical_only"] = False
    for name in (
        "arena_games", "arena_start_percent", "arena_interval_percent", "arena_max_plies",
        "arena_interval_environment_plies",
        "arena_after_half_interval_environment_plies",
        "arena_after_half_historical_only",
        "arena_parallel_games", "arena_inference_batch_size", "arena_environment_workers",
    ):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    if args.arena_after_half_interval_environment_plies == 0:
        overrides["arena_after_half_interval_environment_plies"] = None
    if args.resource_monitor_seconds is not None:
        overrides["resource_monitor_interval_seconds"] = (
            args.resource_monitor_seconds
        )
    if args.arena_max_plies == 0:
        overrides["arena_max_plies"] = None
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
    if (
        args.temporal_cache_entries is not None
        or args.paged_kv_length_bucket is not None
        or args.no_incremental_inference
        or args.no_paged_kv
        or args.no_ppo_array_history
        or args.no_ppo_fixed_kv
        or args.no_ppo_cuda_graphs
        or args.activation_checkpointing is not None
        or args.causal_sdpa is not None
        or any(getattr(args, name) is not None for name in (
            'board_chunk_size', 'ppo_tensor_learner', 'ppo_varlen_attention',
            'ppo_low_precision_residual', 'ppo_sampling_graphs', 'ppo_compile_mode'))
    ):
        settings = replace(
            settings,
            model=replace(
                settings.model,
                inference_temporal_cache_entries=(
                    settings.model.inference_temporal_cache_entries
                    if args.temporal_cache_entries is None
                    else args.temporal_cache_entries
                ),
                incremental_inference=(
                    False
                    if args.no_incremental_inference
                    else settings.model.incremental_inference
                ),
                paged_kv_cache=(
                    False if args.no_paged_kv else settings.model.paged_kv_cache
                ),
                ppo_array_history=False if args.no_ppo_array_history else settings.model.ppo_array_history,
                ppo_fixed_kv=False if args.no_ppo_fixed_kv else settings.model.ppo_fixed_kv,
                ppo_cuda_graphs=False if args.no_ppo_cuda_graphs else settings.model.ppo_cuda_graphs,
                activation_checkpointing=(settings.model.activation_checkpointing if args.activation_checkpointing is None
                                          else args.activation_checkpointing),
                temporal_causal_sdpa=(settings.model.temporal_causal_sdpa if args.causal_sdpa is None else args.causal_sdpa),
                **{name: getattr(args, name) for name in (
                    'board_chunk_size', 'ppo_tensor_learner', 'ppo_varlen_attention',
                    'ppo_low_precision_residual', 'ppo_sampling_graphs', 'ppo_compile_mode') if getattr(args, name) is not None},
                paged_kv_length_bucket_tokens=(
                    settings.model.paged_kv_length_bucket_tokens
                    if args.paged_kv_length_bucket is None
                    else args.paged_kv_length_bucket
                ),
            ),
        )
        settings.validate()
    distributed = DistributedContext.initialize(settings.device)
    try:
        settings = replace(settings, device=str(distributed.device))
        run_directory = settings.resolve_run_directory(args.run_directory)
        trainer = SelfPlayTrainer(
            settings,
            run_directory=run_directory,
            auto_resume=not args.no_resume,
            distributed=distributed,
            initialize_from=args.init_from,
            adopt_current_draw_rules=args.adopt_current_draw_rules,
            adopt_draw_penalty=args.adopt_draw_penalty,
            adopt_flag_capture_reward=args.adopt_flag_capture_reward,
            adopt_champion_evaluation=args.adopt_champion_evaluation,
            adopt_pass_rule=args.adopt_pass_rule,
            expand_game_pool=args.expand_game_pool,
            reset_oom_batch_limits=args.reset_oom_batch_limits,
        )
        trainer.train()
    finally:
        distributed.close()


if __name__ == "__main__":
    main()
