"""Validated YAML settings for reproducible training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
from typing import Any

import yaml

from .encoding import ACTION_FEATURE_DIM
from .models import ModelConfig
from .modes import TrainingMode, normalize_mode
from .rollout import REPLICAS_PER_CANDIDATE, ROOT_CANDIDATE_COUNT
from .rewards import DEFAULT_DRAW_REWARD
from .clipping import DEFAULT_CLIP_SCHEDULE


PLAYER_PARAMETER_SHARING = "single_policy_and_layout_instance_all_seats"
DEAD_RULES_ON_DIRECTORY = "with_dead_rules"
DEAD_RULES_OFF_DIRECTORY = "without_dead_rules"


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    mode: TrainingMode
    model: ModelConfig
    dead_rules_enabled: bool
    source_path: Path
    raw_config: dict[str, Any]
    output_root: Path
    total_updates: int
    target_continuation_plies: int | None
    seed: int
    device: str
    amp: str
    enable_tf32: bool
    enable_torch_compile: bool
    base_game_pool_size: int
    max_game_plies: int | None
    anchor_batch: int
    policy_microbatch: int
    actor_inference_batch: int
    rollout_anchor_wave_size: int
    rollout_environment_workers: int
    learner_length_bucketing: bool
    policy_epochs: int
    policy_learning_rate: float
    minimum_learning_rate: float
    layout_learning_rate: float
    warmup_updates: int
    clip_epsilon: float
    gradient_norm_clip: float
    entropy_coefficient: float
    kl_coefficient: float
    target_kl: float
    early_stop_kl_multiple: float
    early_stop_clip_fraction: float
    advantage_epsilon: float
    layout_update_interval: int
    layout_outcomes_per_update: int
    reference_refresh_updates: int
    checkpoint_every_updates: int
    archive_every_updates: int
    keep_checkpoint_archives: int
    metrics_every_updates: int
    resource_monitor_interval_seconds: float
    auto_reduce_microbatch_on_oom: bool
    minimum_policy_microbatch: int
    # One game.step on any training trajectory, including simulated branches.
    # target_continuation_plies remains a legacy branch-only GRPO option.
    target_environment_plies: int | None = None
    algorithm: str = "grpo"
    discount: float = 1.0
    gae_lambda: float = 0.95
    value_coefficient: float = 0.5
    value_clip_epsilon: float = 0.2
    critic_learning_rate: float = 1.0e-4
    critic_epochs: int = 3
    ppo_sequence_training: bool = True
    ppo_max_samples_per_sequence: int = 64
    ppo_minibatch_samples: int = 512
    layout_prefetch_games: int = 1
    ppo_deferred_values: bool = False
    ppo_pipeline_groups: int = 1
    ppo_fused_optimizer: bool = False
    ppo_learner_cuda_graphs: bool = False
    ppo_cuda_memory_fraction: float | None = None
    ppo_adaptive_clip: bool = False
    ppo_clip_minimum: float = 0.1
    ppo_clip_maximum: float = 0.45
    ppo_clip_advantage_scale: float = 0.5
    ppo_clip_schedule: tuple[tuple[float, float, float], ...] = DEFAULT_CLIP_SCHEDULE
    grpo_equivalent_plies: int | None = None
    grpo_mean_remaining_plies: float = 334.5
    arena_enabled: bool = True
    arena_start_percent: int = 30
    arena_interval_percent: int = 5
    arena_interval_environment_plies: int | None = None
    arena_after_half_interval_environment_plies: int | None = None
    arena_after_half_historical_only: bool = False
    arena_observational_only: bool = False
    arena_champion_only: bool = False
    arena_historical_teammate_fraction: float = 0.0
    arena_games: int = 1000
    arena_max_plies: int | None = None
    arena_seed: int = 20260910
    arena_temporal_cache_entries: int = 8
    arena_parallel_games: int = 32
    arena_inference_batch_size: int = 32
    arena_environment_workers: int = 4
    checkpoint_policy: str = "periodic"
    inference_snapshot_every_updates: int = 0
    no_capture_draw_plies: int = 70
    max_passes_per_player: int = 4
    checkpoint_interval_environment_plies: int | None = None
    draw_reward: float = DEFAULT_DRAW_REWARD
    flag_capture_reward: float = 0.0
    historical_enabled: bool = False
    historical_start_fraction: float = 0.5
    historical_training_fraction: float = 0.2
    historical_teammate_fraction: float = 0.2
    historical_snapshot_fractions: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4)
    historical_cohort_games: int = 1024
    historical_eval_games: int = 400
    historical_eval_total_games: int | None = None
    historical_uniform_fraction: float = 0.05
    historical_learning_rate: float = 0.01
    historical_checkpoint_start_fraction: float | None = None
    historical_stage_mix_fraction: float = 0.0
    historical_cache_gib: float = 0.0
    historical_cache_reserve_gib: float = 16.0
    historical_pinned_mib: int = 256
    layout_buffer_capacity: int = 16384
    layout_max_behavior_age: int = 0
    layout_microbatch_size: int = 32
    adaptive_entropy: bool = False
    entropy_minimum: float = 0.005
    entropy_maximum: float = 0.02
    entropy_opening_maximum: float | None = None
    entropy_target_ratio: float = 0.6
    entropy_adaptation_rate: float = 0.1
    entropy_ema_decay: float = 0.9
    entropy_opening_plies: int = 16
    adaptive_learning_rate: bool = False
    maximum_learning_rate: float = 1.0e-4
    lr_increase_factor: float = 1.03
    lr_decrease_factor: float = 0.8
    lr_stable_updates: int = 5
    lr_cooldown_updates: int = 5
    lr_ema_decay: float = 0.8
    lr_probe_samples: int = 2048
    lr_recovery_clip_fraction: float = 0.1

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        mode: TrainingMode | str,
        *,
        tiny: bool = False,
        model_scale: str = "bootstrap",
        dead_rules_enabled: bool | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> TrainingSettings:
        source = Path(path).resolve()
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        normalized = normalize_mode(mode)
        sharing = data.get("models", {}).get("common", {}).get(
            "player_parameter_sharing"
        )
        if sharing != PLAYER_PARAMETER_SHARING:
            raise ValueError(
                "all seats must share one Policy and one Layout model instance"
            )
        rollout = data["rollout"]
        grpo = data["grpo"]
        learner = data["learner"]
        runtime = data.get("runtime", {})
        ppo = data.get("ppo", {})
        adaptive_clip = ppo.get("adaptive_clip", {})
        if not isinstance(adaptive_clip, dict):
            raise ValueError("ppo.adaptive_clip must be a mapping")
        historical = data.get("historical_opponents", {})
        if not isinstance(historical, dict):
            raise ValueError("historical_opponents must be a mapping")
        model_selection = data.get("model_selection", {})
        if not isinstance(model_selection, dict):
            raise ValueError("model_selection must be a mapping")
        four_player_selection = model_selection.get("four_player", {})
        if not isinstance(four_player_selection, dict):
            raise ValueError("model_selection.four_player must be a mapping")
        if normalized is not TrainingMode.TWO_PLAYER:
            model_selection = {**model_selection, **four_player_selection}
        algorithm = "grpo" if normalized is TrainingMode.TWO_PLAYER else "ppo"
        configured_dead_rules = runtime.get("dead_rules_enabled", True)
        if not isinstance(configured_dead_rules, bool):
            raise ValueError("runtime.dead_rules_enabled must be a boolean")
        selected_dead_rules = (
            configured_dead_rules
            if dead_rules_enabled is None
            else dead_rules_enabled
        )
        if not isinstance(selected_dead_rules, bool):
            raise ValueError("dead_rules_enabled override must be a boolean")
        if overrides and "dead_rules_enabled" in overrides:
            raise ValueError(
                "pass dead_rules_enabled as its named argument, not in overrides"
            )
        if algorithm == "grpo" and rollout["root_candidate_count"] != ROOT_CANDIDATE_COUNT:
            raise ValueError("runtime requires exactly four root candidates")
        if (
            algorithm == "grpo" and rollout["environment_replicas_per_root_candidate"]
            != REPLICAS_PER_CANDIDATE
        ):
            raise ValueError("runtime requires exactly two replicas per candidate")

        if tiny:
            model = ModelConfig.tiny(dead_rules_enabled=selected_dead_rules)
        elif model_scale == "bootstrap":
            model = _model_config(
                data, dead_rules_enabled=selected_dead_rules
            )
        elif model_scale == "main":
            model = ModelConfig.main(dead_rules_enabled=selected_dead_rules)
        elif model_scale == "extended":
            model = ModelConfig.extended(dead_rules_enabled=selected_dead_rules)
        else:
            raise ValueError(
                "model_scale must be bootstrap, main, or extended"
            )
        if normalized is not TrainingMode.TWO_PLAYER and not tiny:
            profile = data["models"].get("four_player", {})
            if not isinstance(profile, dict):
                raise ValueError("models.four_player must be a mapping")
            board_dim = int(profile.get("board_embedding_dim", model.board_dim))
            action_dim = int(profile.get("action_embedding_dim", model.board_dim))
            if action_dim != board_dim:
                raise ValueError("four-player action and board encoders must have equal output dimensions")
            model = replace(
                model,
                board_dim=board_dim,
                temporal_dim=int(profile.get("transition_token_dim", model.temporal_dim)),
                temporal_ffn_dim=int(profile.get("temporal_ffn_dim", model.temporal_ffn_dim)),
            )
        model = replace(
            model,
            inference_board_cache_entries=int(
                runtime.get("inference_board_cache_entries", 65536)
            ),
            inference_temporal_cache_entries=int(
                ppo.get("inference_temporal_cache_entries", 96)
                if algorithm == "ppo"
                else runtime.get("inference_temporal_cache_entries", 192)
            ),
            incremental_inference=bool(
                runtime.get("incremental_inference", True)
            ),
            paged_kv_cache=bool(runtime.get("paged_kv_cache", True)),
            ppo_array_history=ppo.get("array_history", True),
            ppo_fixed_kv=ppo.get("fixed_kv", True),
            ppo_cuda_graphs=ppo.get("cuda_graphs", True),
            ppo_tensor_learner=ppo.get('tensor_learner', False) if algorithm == 'ppo' else False,
            ppo_varlen_attention=ppo.get('varlen_attention', False) if algorithm == 'ppo' else False,
            ppo_sampling_graphs=ppo.get('sampling_graphs', False) if algorithm == 'ppo' else False,
            ppo_low_precision_residual=ppo.get('low_precision_residual', False) if algorithm == 'ppo' else False,
            ppo_compile_mode=ppo.get('compile_mode', 'off') if algorithm == 'ppo' and not tiny else 'off',
            board_chunk_size=int(ppo.get('board_chunk_size', model.board_chunk_size))
                if algorithm == 'ppo' and not tiny else model.board_chunk_size,
            temporal_causal_sdpa=ppo.get('causal_sdpa', True) if algorithm == 'ppo' else model.temporal_causal_sdpa,
            activation_checkpointing=ppo.get('activation_checkpointing', model.activation_checkpointing)
                if algorithm == 'ppo' else model.activation_checkpointing,
            paged_kv_length_bucket_tokens=int(
                runtime.get("paged_kv_length_bucket_tokens", 1001)
            ),
        )
        profile_pool = runtime.get("base_game_pool_size", {})
        microbatch_profile = runtime.get("policy_microbatch", 1)
        policy_microbatch = (
            int(microbatch_profile.get(model_scale, 1))
            if isinstance(microbatch_profile, dict)
            else int(microbatch_profile)
        )
        values: dict[str, Any] = {
            "algorithm": algorithm,
            "discount": float(ppo.get("discount", 1.0)),
            "gae_lambda": float(ppo.get("gae_lambda", 0.95)),
            "value_coefficient": float(ppo.get("value_coefficient", 0.5)),
            "value_clip_epsilon": float(ppo.get("value_clip_epsilon", 0.2)),
            "critic_learning_rate": float(ppo.get("critic_learning_rate", 1.0e-4)),
            "critic_epochs": int(ppo.get("critic_epochs", 3)),
            "ppo_sequence_training": bool(ppo.get("sequence_training", True)),
            "ppo_max_samples_per_sequence": int(ppo.get("max_samples_per_sequence", 64)),
            "ppo_minibatch_samples": int(ppo.get("optimizer_minibatch_samples", 512)),
            "layout_prefetch_games": int(ppo.get("layout_prefetch_games", 1)) if algorithm == "ppo" else 1,
            "ppo_deferred_values": ppo.get("deferred_values", False) if algorithm == 'ppo' else False,
            "ppo_pipeline_groups": int(ppo.get("pipeline_groups", 1)) if algorithm == 'ppo' else 1,
            "historical_enabled": historical.get("enabled", False) if algorithm == "ppo" else False,
            "historical_start_fraction": historical.get("start_fraction", 0.5),
            "historical_training_fraction": historical.get("training_fraction", 0.2),
            "historical_teammate_fraction": historical.get("teammate_fraction", 0.2),
            "historical_snapshot_fractions": tuple(historical.get("snapshot_fractions", (0.05, 0.1, 0.2, 0.3, 0.4))),
            "historical_cohort_games": historical.get("cohort_games", 1024),
            "historical_eval_games": historical.get("eval_games", 400),
            "historical_eval_total_games": historical.get("eval_total_games"),
            "historical_uniform_fraction": historical.get("uniform_fraction", 0.05),
            "historical_learning_rate": historical.get("learning_rate", 0.01),
            "historical_checkpoint_start_fraction": historical.get("checkpoint_start_fraction"),
            "historical_stage_mix_fraction": historical.get("stage_mix_fraction", 0.0),
            "historical_cache_gib": historical.get("cache_gib", 0.0),
            "historical_cache_reserve_gib": historical.get("cache_reserve_gib", 16.0),
            "historical_pinned_mib": historical.get("pinned_mib", 256),
            "ppo_fused_optimizer": ppo.get("fused_optimizer", False) if algorithm == 'ppo' else False,
            "ppo_learner_cuda_graphs": ppo.get("learner_cuda_graphs", False) if algorithm == 'ppo' else False,
            "ppo_cuda_memory_fraction": ppo.get("cuda_memory_fraction") if algorithm == 'ppo' else None,
            "ppo_adaptive_clip": adaptive_clip.get("enabled", False) if algorithm == 'ppo' else False,
            "ppo_clip_minimum": adaptive_clip.get("minimum", 0.1),
            "ppo_clip_maximum": adaptive_clip.get("maximum", 0.45),
            "ppo_clip_advantage_scale": adaptive_clip.get("advantage_scale", 0.5),
            "ppo_clip_schedule": adaptive_clip.get("schedule", DEFAULT_CLIP_SCHEDULE),
            "grpo_equivalent_plies": None,
            "grpo_mean_remaining_plies": float(ppo.get("grpo_mean_remaining_plies", 334.5)),
            # Preserve the YAML types so malformed booleans and fractional
            # schedules fail validation instead of changing meaning silently.
            "arena_enabled": model_selection.get("enabled", True),
            "arena_start_percent": model_selection.get("start_percent", 30),
            "arena_interval_percent": model_selection.get("interval_percent", 5),
            "arena_interval_environment_plies": model_selection.get("interval_environment_plies"),
            "arena_after_half_interval_environment_plies": model_selection.get("after_half_interval_environment_plies"),
            "arena_after_half_historical_only": model_selection.get("after_half_historical_only", False),
            "arena_observational_only": model_selection.get("observational_only", False),
            "arena_champion_only": model_selection.get("champion_only", False),
            "arena_historical_teammate_fraction": model_selection.get("historical_teammate_fraction", 0.0),
            "arena_games": model_selection.get("games", 1000),
            "arena_max_plies": model_selection.get("max_plies"),
            "no_capture_draw_plies": data.get("rules", {}).get("no_capture_draw_plies", 70),
            "max_passes_per_player": data.get("rules", {}).get("max_passes_per_player", 4),
            "draw_reward": data.get("rules", {}).get("terminal_reward", {}).get("draw", DEFAULT_DRAW_REWARD),
            "flag_capture_reward": ppo.get("flag_capture_reward", 0.0) if algorithm == "ppo" else 0.0,
            "arena_seed": model_selection.get("seed", 20260910),
            "arena_temporal_cache_entries": model_selection.get(
                "temporal_cache_entries", 8
            ),
            "arena_parallel_games": model_selection.get("parallel_games", 32),
            "arena_inference_batch_size": model_selection.get("inference_batch_size", 32),
            "arena_environment_workers": model_selection.get("environment_workers", 4),
            "output_root": Path(runtime.get("output_root", "runs")),
            "total_updates": int(runtime.get("total_updates", 200000)),
            "target_continuation_plies": runtime.get(
                "target_continuation_plies"
            ),
            "target_environment_plies": runtime.get("target_environment_plies"),
            "seed": int(runtime.get("seed", 20260902)),
            "device": str(runtime.get("device", "auto")),
            "amp": str(runtime.get("amp", "bfloat16")),
            "enable_tf32": bool(runtime.get("enable_tf32", True)),
            "enable_torch_compile": bool(
                runtime.get("enable_torch_compile", False)
            ),
            "base_game_pool_size": int(profile_pool.get(normalized.value, 32)),
            "max_game_plies": runtime.get("max_game_plies"),
            "anchor_batch": int(learner["policy_batch"]["global_target"]),
            "policy_microbatch": policy_microbatch,
            "actor_inference_batch": int(
                runtime.get("actor_inference_batch", 64)
            ),
            "rollout_anchor_wave_size": int(
                runtime.get("rollout_anchor_wave_size", 8)
            ),
            "rollout_environment_workers": int(
                runtime.get("rollout_environment_workers", 2)
            ),
            "learner_length_bucketing": bool(
                runtime.get("learner_length_bucketing", True)
            ),
            "policy_epochs": int(
                runtime.get("policy_epochs", learner["max_epochs_per_batch"])
            ),
            "policy_learning_rate": float(learner["learning_rate"]["initial"]),
            "minimum_learning_rate": float(learner["learning_rate"]["minimum"]),
            "adaptive_learning_rate": (learner["learning_rate"].get("adaptive", False)
                                       if algorithm == "ppo" else False),
            "maximum_learning_rate": float(learner["learning_rate"].get(
                "maximum", learner["learning_rate"]["initial"])),
            "lr_increase_factor": learner["learning_rate"].get("increase_factor", 1.03),
            "lr_decrease_factor": learner["learning_rate"].get("decrease_factor", .8),
            "lr_stable_updates": learner["learning_rate"].get("stable_updates", 5),
            "lr_cooldown_updates": learner["learning_rate"].get("cooldown_updates", 5),
            "lr_ema_decay": learner["learning_rate"].get("ema_decay", .8),
            "lr_probe_samples": learner["learning_rate"].get("probe_samples", 2048),
            "lr_recovery_clip_fraction": learner["learning_rate"].get("recovery_clip_fraction", .1),
            "layout_learning_rate": float(
                runtime.get("layout_learning_rate", 5.0e-5)
            ),
            "warmup_updates": int(runtime.get("warmup_updates", 2000)),
            "clip_epsilon": float(learner["clip_epsilon"]),
            "gradient_norm_clip": float(learner["gradient_norm_clip"]),
            "entropy_coefficient": float(
                learner["entropy_coefficient"]["initial"]
            ),
            "adaptive_entropy": learner["entropy_coefficient"].get("adaptive_to_target_entropy", False),
            "entropy_minimum": float(learner["entropy_coefficient"].get("minimum", 0.005)),
            "entropy_maximum": float(learner["entropy_coefficient"].get("maximum", 0.02)),
            "entropy_opening_maximum": learner["entropy_coefficient"].get("opening_maximum"),
            "entropy_target_ratio": float(learner["entropy_coefficient"].get("target_ratio", 0.6)),
            "entropy_adaptation_rate": float(learner["entropy_coefficient"].get("adaptation_rate", 0.1)),
            "entropy_ema_decay": float(learner["entropy_coefficient"].get("ema_decay", 0.9)),
            "entropy_opening_plies": learner["entropy_coefficient"].get("opening_plies", 16),
            "kl_coefficient": float(learner["kl"]["initial_coefficient"]),
            "target_kl": float(learner["kl"]["target_per_action"]),
            "early_stop_kl_multiple": float(
                learner["early_stop"]["observed_kl_multiple"]
            ),
            "early_stop_clip_fraction": float(
                learner["early_stop"]["clipped_sample_fraction"]
            ),
            "advantage_epsilon": float(grpo["advantage_epsilon"]),
            "layout_update_interval": int(
                runtime.get("layout_update_interval", 8)
            ),
            "layout_outcomes_per_update": int(
                runtime.get("layout_outcomes_per_update", 64)
            ),
            "layout_buffer_capacity": int(runtime.get("layout_buffer_capacity", 16384)),
            # GRPO games can span many learner updates; PPO's freshness window
            # must not silently expire those outcomes in two-player runs.
            "layout_max_behavior_age": int(runtime.get("layout_max_behavior_age", 16 if algorithm == "ppo" else 0)),
            "layout_microbatch_size": int(runtime.get("layout_microbatch_size", 32)),
            "reference_refresh_updates": int(
                runtime.get("reference_refresh_updates", 50000)
            ),
            "checkpoint_every_updates": int(
                runtime.get("checkpoint_every_updates", 5)
            ),
            "checkpoint_policy": model_selection.get("checkpoint_policy", "periodic"),
            "checkpoint_interval_environment_plies": model_selection.get("checkpoint_interval_environment_plies"),
            "inference_snapshot_every_updates": int(runtime.get("inference_snapshot_every_updates", 0)),
            "archive_every_updates": int(
                runtime.get("archive_every_updates", 500)
            ),
            "keep_checkpoint_archives": int(
                runtime.get("keep_checkpoint_archives", 10)
            ),
            "metrics_every_updates": int(runtime.get("metrics_every_updates", 1)),
            "resource_monitor_interval_seconds": float(
                runtime.get("resource_monitor_interval_seconds", 30.0)
            ),
            "auto_reduce_microbatch_on_oom": bool(
                runtime.get("auto_reduce_microbatch_on_oom", True)
            ),
            "minimum_policy_microbatch": int(
                runtime.get("minimum_policy_microbatch", 1)
            ),
        }
        if algorithm == "ppo":
            values.update(
                base_game_pool_size=int(ppo.get("base_game_pool_size", 8)),
                actor_inference_batch=int(ppo.get("actor_inference_batch", 8)),
                anchor_batch=int(ppo.get("transition_batch", 512)),
                policy_microbatch=int(ppo.get('learner_microbatch_by_model_scale', {}).get(
                    model_scale, policy_microbatch)),
                rollout_environment_workers=int(ppo.get('environment_workers', 1)),
                target_environment_plies=ppo.get(
                    "target_environment_plies",
                    values["target_environment_plies"] if values["target_environment_plies"] is not None
                    else values["target_continuation_plies"],
                ),
                target_continuation_plies=None,
            )
        if tiny:
            values.update(
                {
                    "total_updates": 1,
                    "target_continuation_plies": None,
                    "target_environment_plies": None,
                    "base_game_pool_size": 2,
                    "max_game_plies": 4,
                    "anchor_batch": 1,
                    "policy_microbatch": 1,
                    "actor_inference_batch": 16,
                    "rollout_anchor_wave_size": 1,
                    "rollout_environment_workers": 1,
                    "ppo_pipeline_groups": 1,
                    "policy_epochs": 1,
                    "critic_epochs": 1,
                    "warmup_updates": 1,
                    "layout_update_interval": 1,
                    "layout_outcomes_per_update": 2,
                    "checkpoint_every_updates": 1,
                    "checkpoint_policy": "periodic",
                    "checkpoint_interval_environment_plies": None,
                    "ppo_adaptive_clip": False,
                    "archive_every_updates": 1,
                    "arena_enabled": False,
                    "historical_enabled": False,
                    "arena_interval_environment_plies": None,
                    "arena_after_half_interval_environment_plies": None,
                    "arena_after_half_historical_only": False,
                    "arena_observational_only": False,
                    "arena_champion_only": False,
                    "arena_historical_teammate_fraction": 0.0,
                }
            )
        if overrides:
            selected_overrides = dict(overrides)
            # A caller explicitly selecting serial rules or online values also
            # selects the compatible collector, unless it explicitly requested
            # a contradictory pipeline configuration (validated below).
            if 'ppo_pipeline_groups' not in selected_overrides and (
                selected_overrides.get('rollout_environment_workers') == 1
                or selected_overrides.get('ppo_deferred_values') is False
            ):
                values['ppo_pipeline_groups'] = 1
            if algorithm == "ppo" and "target_continuation_plies" in selected_overrides:
                if "target_environment_plies" in selected_overrides:
                    raise ValueError("choose one environment budget option, not both aliases")
                # Old PPO callers used the GRPO field name for the same counter.
                selected_overrides["target_environment_plies"] = selected_overrides.pop(
                    "target_continuation_plies"
                )
            elif "target_environment_plies" in selected_overrides and "target_continuation_plies" not in selected_overrides:
                values["target_continuation_plies"] = None
            values.update(selected_overrides)
        if values["grpo_equivalent_plies"] is not None:
            if algorithm != "ppo":
                raise ValueError("GRPO-equivalent coverage is only a PPO planning option")
            if overrides and any(overrides.get(key) is not None for key in (
                "target_environment_plies", "target_continuation_plies"
            )):
                raise ValueError("choose an environment step target or a GRPO-equivalent budget, not both")
            values["target_environment_plies"] = equivalent_ppo_decisions(
                values["grpo_equivalent_plies"], values["grpo_mean_remaining_plies"]
            )
            if not tiny and (not overrides or "warmup_updates" not in overrides):
                # Former GRPO warmup: 2000 updates x 128 root decision states.
                values["warmup_updates"] = max(1, math.ceil(256_000 / values["anchor_batch"]))
        if values["target_continuation_plies"] is not None:
            values["target_continuation_plies"] = int(
                values["target_continuation_plies"]
            )
        if values["target_environment_plies"] is not None:
            target = values["target_environment_plies"]
            if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
                raise ValueError("target environment plies must be a positive integer or None")
            if algorithm == "ppo" and not tiny and (not overrides or "total_updates" not in overrides):
                if values["anchor_batch"] <= 0:
                    raise ValueError("anchor_batch must be positive")
                # This horizon assumes the current branch-free PPO collector.
                # Actual environment interactions, including any extra branches,
                # are the stopping counter; optimizer epochs never add steps.
                values["total_updates"] = (
                    target + values["anchor_batch"] - 1
                ) // values["anchor_batch"]
        result = cls(
            mode=normalized,
            model=model,
            dead_rules_enabled=selected_dead_rules,
            source_path=source,
            raw_config=data,
            **values,
        )
        result.validate()
        return result

    def validate(self) -> None:
        clip_values = (self.clip_epsilon, self.ppo_clip_minimum, self.ppo_clip_maximum,
                       self.ppo_clip_advantage_scale)
        if (type(self.ppo_adaptive_clip) is not bool or any(
                type(x) not in (int, float) or not math.isfinite(x) for x in clip_values)
                or not 0 < self.clip_epsilon < 1
                or not 0 < self.ppo_clip_minimum <= self.ppo_clip_maximum < 1
                or not 0 <= self.ppo_clip_advantage_scale <= 1):
            raise ValueError("invalid bounded PPO clipping settings")
        schedule = self.ppo_clip_schedule
        if (not isinstance(schedule, (list, tuple)) or len(schedule) < 2
                or any(not isinstance(row, (list, tuple)) or len(row) != 3 or any(
                    type(x) not in (int, float) or not math.isfinite(x) for x in row)
                    for row in schedule)):
            raise ValueError("PPO clip schedule must contain finite [progress, lower, upper] rows")
        if (schedule[0][0] != 0 or schedule[-1][0] != 1
                or any(not 0 <= p <= 1 or not self.ppo_clip_minimum <= low <= high <= self.ppo_clip_maximum
                       for p, low, high in schedule)
                or any(a[0] >= b[0] or a[1] < b[1] or a[2] < b[2]
                       for a, b in zip(schedule, schedule[1:]))):
            raise ValueError("PPO clipping must narrow monotonically from progress 0 to 1 within hard bounds")
        if self.ppo_adaptive_clip and (self.algorithm != 'ppo' or not self.target_environment_plies):
            raise ValueError("adaptive PPO clipping requires an environment-step target and PPO")
        rates = (self.policy_learning_rate, self.minimum_learning_rate,
                 self.maximum_learning_rate, self.lr_increase_factor,
                 self.lr_decrease_factor, self.lr_ema_decay, self.lr_recovery_clip_fraction,
                 self.critic_learning_rate, self.target_kl, self.early_stop_kl_multiple,
                 self.early_stop_clip_fraction)
        if (type(self.adaptive_learning_rate) is not bool
                or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                       or not math.isfinite(v) for v in rates)
                or not 0 < self.minimum_learning_rate <= self.maximum_learning_rate
                or self.minimum_learning_rate > self.policy_learning_rate
                or not 1 < self.lr_increase_factor <= 1.5
                or not 0 < self.lr_decrease_factor < 1
                or not 0 <= self.lr_ema_decay < 1
                or not 0 < self.lr_recovery_clip_fraction <= self.early_stop_clip_fraction
                or type(self.lr_stable_updates) is not int or self.lr_stable_updates < 1
                or type(self.lr_cooldown_updates) is not int or self.lr_cooldown_updates < 0
                or type(self.lr_probe_samples) is not int or self.lr_probe_samples < 1
                or not math.isfinite(self.target_kl) or self.target_kl <= 0
                or not math.isfinite(self.early_stop_kl_multiple) or self.early_stop_kl_multiple <= .5
                or (self.adaptive_learning_rate and self.algorithm != "ppo")):
            raise ValueError("invalid bounded adaptive learning rate settings")
        entropy_values = (self.entropy_coefficient, self.entropy_minimum, self.entropy_maximum,
                          self.entropy_target_ratio, self.entropy_adaptation_rate, self.entropy_ema_decay)
        if (type(self.adaptive_entropy) is not bool or any(
                type(x) not in (int, float) or not math.isfinite(x) for x in entropy_values)
                or self.entropy_coefficient < 0 or not 0 < self.entropy_minimum <= self.entropy_maximum
                or not 0 < self.entropy_target_ratio <= 1 or not 0 < self.entropy_adaptation_rate <= 1
                or not 0 <= self.entropy_ema_decay < 1 or type(self.entropy_opening_plies) is not int
                or self.entropy_opening_plies < 0):
            raise ValueError("invalid adaptive entropy settings")
        if self.adaptive_entropy and not self.entropy_minimum <= self.entropy_coefficient <= self.entropy_maximum:
            raise ValueError("adaptive entropy initial coefficient must be within minimum and maximum")
        if self.entropy_opening_maximum is not None:
            maximum = self.entropy_opening_maximum
            if (type(maximum) not in (int, float) or not math.isfinite(maximum)
                    or maximum < self.entropy_minimum
                    or (self.adaptive_entropy and maximum < self.entropy_coefficient)):
                raise ValueError("invalid opening entropy maximum")
        if (isinstance(self.draw_reward, bool) or not isinstance(self.draw_reward, (int, float))
                or not math.isfinite(self.draw_reward) or not -1 < self.draw_reward <= 0):
            raise ValueError("draw_reward must be finite and in (-1, 0]")
        if (isinstance(self.flag_capture_reward, bool) or not isinstance(self.flag_capture_reward, (int, float))
                or not math.isfinite(self.flag_capture_reward) or self.flag_capture_reward < 0):
            raise ValueError("flag_capture_reward must be finite and nonnegative")
        if self.algorithm != "ppo" and self.flag_capture_reward != 0:
            raise ValueError("flag_capture_reward requires PPO")
        expected_algorithm = "grpo" if self.mode is TrainingMode.TWO_PLAYER else "ppo"
        if self.algorithm != expected_algorithm:
            raise ValueError(f"{self.mode.value} requires {expected_algorithm}")
        if not 0 < self.discount <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("discount must be in (0, 1] and gae_lambda in [0, 1]")
        if self.value_coefficient <= 0 or self.value_clip_epsilon <= 0:
            raise ValueError("value loss coefficient and clipping must be positive")
        if self.critic_learning_rate <= 0:
            raise ValueError("critic learning rate must be positive")
        if self.model.dead_rules_enabled != self.dead_rules_enabled:
            raise ValueError("model and training dead-rule modes must match")
        positive = {
            "total_updates": self.total_updates,
            "base_game_pool_size": self.base_game_pool_size,
            "anchor_batch": self.anchor_batch,
            "policy_microbatch": self.policy_microbatch,
            "actor_inference_batch": self.actor_inference_batch,
            "rollout_anchor_wave_size": self.rollout_anchor_wave_size,
            "rollout_environment_workers": self.rollout_environment_workers,
            "policy_epochs": self.policy_epochs,
            "critic_epochs": self.critic_epochs,
            "ppo_max_samples_per_sequence": self.ppo_max_samples_per_sequence,
            "ppo_minibatch_samples": self.ppo_minibatch_samples,
            "layout_prefetch_games": self.layout_prefetch_games,
            "ppo_pipeline_groups": self.ppo_pipeline_groups,
            "layout_update_interval": self.layout_update_interval,
            "layout_outcomes_per_update": self.layout_outcomes_per_update,
            "layout_buffer_capacity": self.layout_buffer_capacity,
            "layout_microbatch_size": self.layout_microbatch_size,
            "checkpoint_every_updates": self.checkpoint_every_updates,
            "archive_every_updates": self.archive_every_updates,
            "minimum_policy_microbatch": self.minimum_policy_microbatch,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.layout_buffer_capacity < max(2, self.layout_outcomes_per_update):
            raise ValueError("layout buffer capacity must cover an update batch")
        if self.layout_max_behavior_age < 0:
            raise ValueError("layout maximum behavior age must be nonnegative (0 disables age expiry)")
        if self.policy_microbatch > self.anchor_batch:
            raise ValueError("policy microbatch cannot exceed anchor batch")
        if self.minimum_policy_microbatch > self.policy_microbatch:
            raise ValueError(
                "minimum policy microbatch cannot exceed the initial microbatch"
            )
        if self.resource_monitor_interval_seconds <= 0:
            raise ValueError("resource monitor interval must be positive")
        if self.inference_snapshot_every_updates < 0:
            raise ValueError("inference snapshot interval must be non-negative")
        if (
            self.target_continuation_plies is not None
            and self.target_continuation_plies <= 0
        ):
            raise ValueError("target continuation plies must be positive or None")
        if self.target_environment_plies is not None:
            if (isinstance(self.target_environment_plies, bool)
                    or not isinstance(self.target_environment_plies, int)
                    or self.target_environment_plies <= 0):
                raise ValueError("target environment plies must be a positive integer or None")
            if self.target_continuation_plies is not None:
                raise ValueError("choose a total environment budget or a legacy continuation budget, not both")
        if self.amp not in ("bfloat16", "float16", "float32"):
            raise ValueError("amp must be bfloat16, float16, or float32")
        if self.keep_checkpoint_archives < 0:
            raise ValueError("keep checkpoint archives cannot be negative")
        if self.checkpoint_policy not in ("periodic", "evaluation"):
            raise ValueError("checkpoint_policy must be periodic or evaluation")
        for name in ("max_game_plies", "arena_max_plies", "checkpoint_interval_environment_plies"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        if type(self.max_passes_per_player) is not int or self.max_passes_per_player != 4:
            raise ValueError("the current rules require exactly four passes per player")
        if type(self.no_capture_draw_plies) is not int or not 1 <= self.no_capture_draw_plies <= 70:
            raise ValueError("no_capture_draw_plies must be an integer in [1, 70]")
        if not isinstance(self.learner_length_bucketing, bool):
            raise ValueError("learner_length_bucketing must be a boolean")
        if not isinstance(self.ppo_sequence_training, bool):
            raise ValueError("ppo_sequence_training must be a boolean")
        if type(self.ppo_deferred_values) is not bool:
            raise ValueError("ppo_deferred_values must be a boolean")
        if type(self.ppo_fused_optimizer) is not bool:
            raise ValueError("ppo_fused_optimizer must be a boolean")
        if type(self.ppo_learner_cuda_graphs) is not bool:
            raise ValueError("ppo_learner_cuda_graphs must be a boolean")
        if self.ppo_cuda_memory_fraction is not None and (
                type(self.ppo_cuda_memory_fraction) not in (int, float)
                or not math.isfinite(self.ppo_cuda_memory_fraction)
                or not 0 < self.ppo_cuda_memory_fraction <= 1):
            raise ValueError('ppo_cuda_memory_fraction must be in (0, 1]')
        if self.ppo_pipeline_groups > 1 and (not self.ppo_deferred_values or not self.model.ppo_array_history
                                            or self.rollout_environment_workers < 2):
            raise ValueError("PPO pipeline needs deferred values, array histories and parallel workers")
        if type(self.arena_enabled) is not bool:
            raise ValueError("arena_enabled must be a boolean")
        if type(self.historical_enabled) is not bool:
            raise ValueError("historical_enabled must be a boolean")
        if type(self.arena_after_half_historical_only) is not bool:
            raise ValueError("arena_after_half_historical_only must be a boolean")
        if self.arena_after_half_historical_only and (
                not self.historical_enabled or self.historical_start_fraction > .5
                or self.target_environment_plies is None or self.arena_interval_environment_plies is None):
            raise ValueError("historical-only evaluation after half requires an environment schedule and historical opponents active by half")
        if self.historical_enabled:
            fraction = self.historical_checkpoint_start_fraction
            if fraction is not None and (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                                         or not 0 < fraction < 1):
                raise ValueError("historical_checkpoint_start_fraction must be in (0, 1)")
            mix = self.historical_stage_mix_fraction
            if isinstance(mix, bool) or not isinstance(mix, (int, float)) or not 0 <= mix <= 1:
                raise ValueError("historical_stage_mix_fraction must be in [0, 1]")
            for name in ("historical_cache_gib", "historical_cache_reserve_gib"):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be finite and nonnegative")
            if type(self.historical_pinned_mib) is not int or not 0 <= self.historical_pinned_mib <= 1024:
                raise ValueError("historical_pinned_mib must be an integer in [0, 1024]")
            if (self.algorithm != "ppo" or self.target_environment_plies is None
                    or not self.arena_enabled or self.ppo_pipeline_groups < 2):
                raise ValueError("historical opponents require four-player pipelined PPO, an environment budget and arena")
            if not self.model.ppo_fixed_kv or not self.model.incremental_inference:
                raise ValueError("historical opponents require bounded shared fixed KV storage")
            for name in ("historical_start_fraction", "historical_training_fraction", "historical_uniform_fraction"):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
                    raise ValueError(f"{name} must be in (0, 1)")
            value = self.historical_teammate_fraction
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ValueError("historical_teammate_fraction must be in [0, 1)")
            fractions = self.historical_snapshot_fractions
            if (not 2 <= len(fractions) <= 7 or list(fractions) != sorted(set(fractions))
                    or any(not 0 < f < self.historical_start_fraction for f in fractions)):
                raise ValueError("historical snapshots require 2..7 increasing fractions before activation")
            for name in ("historical_cohort_games", "historical_eval_games"):
                if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            if self.historical_eval_games % 4:
                raise ValueError("historical evaluation requires complete four-game rotations")
            total = self.historical_eval_total_games
            if total is not None and (type(total) is not int or total % 4
                                      or total < 4 * (1 + len(fractions))):
                raise ValueError("historical_eval_total_games must cover every opponent with complete four-game rotations")
            if not 0 < self.historical_learning_rate <= 1:
                raise ValueError("historical learning rate must be in (0, 1]")
        if self.arena_interval_environment_plies is not None:
            if (type(self.arena_interval_environment_plies) is not int
                    or self.arena_interval_environment_plies <= 0):
                raise ValueError("arena_interval_environment_plies must be a positive integer or None")
            if self.arena_enabled and self.target_environment_plies is None:
                raise ValueError("environment-step model selection requires target_environment_plies")
        if self.arena_after_half_interval_environment_plies is not None:
            interval = self.arena_after_half_interval_environment_plies
            if type(interval) is not int or interval <= 0:
                raise ValueError("arena_after_half_interval_environment_plies must be a positive integer or None")
            if self.arena_interval_environment_plies is None or self.target_environment_plies is None:
                raise ValueError("after-half evaluation requires an environment schedule and budget")
            if interval > self.arena_interval_environment_plies:
                raise ValueError("after-half evaluation interval must not exceed the initial interval")
        for name in ("arena_start_percent", "arena_interval_percent"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 100:
                raise ValueError(f"{name} must be an integer in [1, 100]")
        if type(self.arena_observational_only) is not bool:
            raise ValueError("arena_observational_only must be a boolean")
        if self.arena_observational_only and (self.mode is TrainingMode.TWO_PLAYER or self.algorithm != "ppo"):
            raise ValueError("observational evaluation requires four-player PPO")
        if type(self.arena_champion_only) is not bool:
            raise ValueError("arena_champion_only must be a boolean")
        if self.arena_champion_only:
            if self.mode is TrainingMode.TWO_PLAYER or self.algorithm != "ppo":
                raise ValueError("champion-only evaluation requires four-player PPO")
            if self.arena_observational_only or self.arena_after_half_historical_only:
                raise ValueError("champion-only evaluation cannot use observational or historical-only evaluation")
        fraction = self.arena_historical_teammate_fraction
        if (type(fraction) not in (int, float) or fraction not in (0, .5)
                or (fraction and not (self.arena_observational_only or self.arena_champion_only))):
            raise ValueError("historical teammate evaluation requires observational or champion-only mode and fraction 0 or 0.5")
        for name in (
            "arena_games", "arena_temporal_cache_entries",
            "arena_parallel_games", "arena_inference_batch_size", "arena_environment_workers",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        games_per_group = 2 if self.mode is TrainingMode.TWO_PLAYER else 4
        if self.arena_games % games_per_group:
            raise ValueError(
                f"arena_games must be divisible by {games_per_group} for {self.mode.value}"
            )
        match_seed_span = (2 + games_per_group) * (self.arena_games // games_per_group)
        final_seed_index = (len(self.arena_milestones)
                            if self.arena_interval_environment_plies is not None
                            else max(self.arena_milestones))
        if (
            type(self.arena_seed) is not int
            or self.arena_seed < 0
            or self.arena_seed + (final_seed_index + 1) * match_seed_span >= 2**63
        ):
            raise ValueError(
                "arena_seed and all scheduled matches must fit a nonnegative signed 64-bit integer"
            )

    @property
    def arena_milestones(self) -> range | tuple[int, ...]:
        if self.arena_interval_environment_plies is not None:
            interval = self.arena_interval_environment_plies
            if self.arena_after_half_interval_environment_plies is not None:
                target = self.target_environment_plies or 0
                half = (target + 1) // 2
                # Include the half-budget boundary exactly once, even when
                # the first interval does not divide it.
                return (*range(interval, half, interval),
                        *range(half, target + 1, self.arena_after_half_interval_environment_plies))
            return range(interval, (self.target_environment_plies or 0) + 1, interval)
        return range(self.arena_start_percent, 101, self.arena_interval_percent)

    def serializable(self) -> dict[str, Any]:
        values = asdict(self)
        values["mode"] = self.mode.value
        values["source_path"] = str(self.source_path)
        values["output_root"] = str(self.output_root)
        values["step_budget_counter"] = self.step_budget_counter
        values["step_budget_unit"] = (
            "training_environment_transitions" if self.step_budget_counter == "environment_plies"
            else "legacy_grpo_branch_transitions"
        )
        return values

    @property
    def step_budget_target(self) -> int | None:
        return (self.target_environment_plies if self.target_environment_plies is not None
                else self.target_continuation_plies)

    @property
    def step_budget_counter(self) -> str:
        return ("environment_plies" if self.target_environment_plies is not None or self.algorithm == "ppo"
                else "continuation_plies")

    @property
    def run_variant_directory(self) -> str:
        return (
            DEAD_RULES_ON_DIRECTORY
            if self.dead_rules_enabled
            else DEAD_RULES_OFF_DIRECTORY
        )

    def resolve_run_directory(
        self, base: str | Path | None = None
    ) -> Path:
        """Return a path that cannot mix the two incompatible model variants.

        ``base`` is intentionally a parent directory.  The feature tag is
        always appended unless the caller already supplied that exact tag.
        Supplying the opposite tag is rejected rather than silently writing a
        checkpoint into the wrong experiment.
        """

        root = self.output_root if base is None else Path(base)
        known_tags = {DEAD_RULES_ON_DIRECTORY, DEAD_RULES_OFF_DIRECTORY}
        mode_names = {mode.value for mode in TrainingMode}
        if root.name in known_tags:
            if root.name != self.run_variant_directory:
                raise ValueError(
                    f"run directory tag {root.name!r} contradicts "
                    f"dead_rules_enabled={self.dead_rules_enabled}"
                )
            if root.parent.name != self.mode.value:
                raise ValueError(
                    "an explicit variant directory must be nested under its "
                    f"mode directory {self.mode.value!r}"
                )
            return root
        if root.name in mode_names:
            if root.name != self.mode.value:
                raise ValueError(
                    f"run directory mode {root.name!r} contradicts "
                    f"training mode {self.mode.value!r}"
                )
            mode_root = root
        else:
            mode_root = root / self.mode.value
        return mode_root / self.run_variant_directory


def equivalent_ppo_decisions(continuation_plies: int, mean_remaining_plies: float) -> int:
    """Approximate equal root-state coverage, never equal strength or real steps."""
    if continuation_plies <= 0 or not math.isfinite(mean_remaining_plies) or mean_remaining_plies <= 0:
        raise ValueError("equivalent budget and mean remaining length must be positive and finite")
    return math.ceil(continuation_plies / (ROOT_CANDIDATE_COUNT * REPLICAS_PER_CANDIDATE * mean_remaining_plies))


def _model_config(
    data: dict[str, Any], *, dead_rules_enabled: bool
) -> ModelConfig:
    models = data["models"]
    common = models["common"]
    policy = models["policy"]
    board = policy["board_encoder"]
    action = policy["action_encoder"]
    if int(action["input_dim"]) != ACTION_FEATURE_DIM:
        raise ValueError("action encoder requires exactly six coordinate/player/countdown inputs")
    if int(action["output_dim"]) != int(board["output_dim"]):
        raise ValueError("action and board encoders must have equal output dimensions")
    temporal = policy["temporal_transformer"]
    layout = models["layout"]
    history = data["history"]
    return ModelConfig(
        board_dim=int(board["output_dim"]),
        board_encoder_type=str(board["architecture"]),
        action_encoder_type=str(action["architecture"]),
        temporal_dim=int(temporal["d_model"]),
        temporal_layers=int(temporal["transformer_layers"]),
        temporal_heads=int(temporal["attention_heads"]),
        temporal_ffn_dim=int(temporal["ffn_dim"]),
        layout_dim=int(layout["d_model"]),
        layout_layers=int(layout["transformer_layers"]),
        layout_heads=int(layout["attention_heads"]),
        layout_ffn_dim=int(layout["ffn_dim"]),
        max_transitions=int(history["max_previous_decisions"]),
        dropout=float(common["dropout"]),
        dead_rules_enabled=dead_rules_enabled,
    )
