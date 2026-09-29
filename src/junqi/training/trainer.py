"""Resumable two/four-player PPO/MAPPO and legacy terminal-rollout GRPO training."""

from __future__ import annotations

from dataclasses import replace

import copy
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from typing import Any, Sequence, TypeVar

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .accelerator import (
    empty_cache,
    is_accelerator,
    is_bf16_supported,
    is_out_of_memory,
    make_grad_scaler,
    manual_seed_all,
    reset_peak_memory_stats,
    resolve_device,
    set_device,
)
from .checkpoint import (
    CheckpointManager,
    capture_rng_state,
    require_current_checkpoint,
    restore_rng_state,
    restore_training_state,
)
from .distributed import DistributedContext
from .losses import layout_grpo_loss, policy_grpo_loss, prepare_layout_batch
from .layout_buffer import LayoutOutcomeBuffer
from .learning_metrics import ppo_signal_metrics, ppo_signal_sums
from .learning_rate import BoundedLearningRate, diagnostic_sequences
from .clipping import scheduled_policy_clip
from .metrics import MetricLogger
from .model_selection import ModelSelection
from .modes import mode_spec
from .inference_snapshot import export_inference_snapshot
from .models import (
    GamePolicyTransformer,
    GameValueTransformer,
    GameCentralValueTransformer,
    PieceConditionedLayoutPointerDecoder,
    parameter_count,
)
from .rollout import (
    BaseGamePool,
    FrozenPolicyActor,
    LayoutOutcome,
    PolicyGroup,
    RolloutMetrics,
    collect_policy_groups,
)
from .settings import TrainingSettings
from .ppo import (
    FrozenValueActor, PPOSample, collect_ppo_samples, critic_ppo_loss,
    normalize_advantages, policy_ppo_loss, sequence_training_batches,
)
from .encoding import history_prefix_groups
from .mappo import collect_mappo_samples, critic_training_samples
from .tactical_logging import MAPPOTrainingAudit, TacticalLogSettings
from .entropy import AdaptiveEntropyCoefficient, phase_entropy_ratios, scheduled_entropy_targets


T = TypeVar("T")


def _chunks(values: Sequence[T], size: int) -> list[Sequence[T]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


class SelfPlayTrainer:
    def __init__(
        self,
        settings: TrainingSettings,
        *,
        run_directory: str | Path | None = None,
        auto_resume: bool = True,
        distributed: DistributedContext | None = None,
        initialize_from: str | Path | None = None,
        adopt_current_draw_rules: bool = False,
        shrink_context_window: bool = False,
        adopt_draw_penalty: bool = False,
        adopt_flag_capture_reward: bool = False,
        adopt_futile_attack_penalty: bool = False,
        adopt_bomb_rewards: bool = False,
        adopt_flag_defense_rewards: bool = False,
        adopt_known_nonflag_hq_penalty: bool = False,
        adopt_player_advantages: bool = False,
        adopt_champion_evaluation: bool = False,
        adopt_pass_rule: bool = False,
        adopt_no_move_elimination: bool = False,
        adopt_two_player_no_pass: bool = False,
        expand_game_pool: bool = False,
        reset_oom_batch_limits: bool = False,
    ) -> None:
        self.settings = settings
        self.session_started_unix = time.time()
        self.resumed_from_update = 0
        self.resumed_from_cumulative = {}
        self.adopt_current_draw_rules = adopt_current_draw_rules
        self.shrink_context_window = shrink_context_window
        self.context_window_migration = None
        self.context_shrunk_on_resume = False
        self.adopt_draw_penalty = adopt_draw_penalty
        self.adopt_flag_capture_reward = adopt_flag_capture_reward
        self.adopt_futile_attack_penalty = adopt_futile_attack_penalty
        self.adopt_bomb_rewards = adopt_bomb_rewards
        self.adopt_flag_defense_rewards = adopt_flag_defense_rewards
        self.adopt_known_nonflag_hq_penalty = adopt_known_nonflag_hq_penalty
        self.adopt_player_advantages = adopt_player_advantages
        self.adopt_pass_rule = adopt_pass_rule
        self.adopt_no_move_elimination = adopt_no_move_elimination
        self.adopt_two_player_no_pass = adopt_two_player_no_pass
        self.two_player_no_pass_migration = None
        self.expand_game_pool = expand_game_pool
        self.reset_oom_batch_limits = reset_oom_batch_limits
        self.pool_expanded_on_resume = False
        self.pass_rule_migration = None
        self.no_move_rule_migration = None
        self.last_checkpoint_environment_plies = 0
        self.draw_rule_migration = None
        self.draw_objective_migration = None
        self.flag_capture_objective_migration = None
        self.futile_attack_objective_migration = None
        self.bomb_objective_migration = None
        self.flag_defense_objective_migration = None
        self.known_nonflag_hq_objective_migration = None
        self.player_advantage_migration = None
        from .futile_attack import FutileAttackPenalty
        self.futile_attack_controller = FutileAttackPenalty(
            settings.futile_attack_penalty, settings.target_environment_plies, settings.futile_attack_ema_decay)
        self.initialize_from = None if initialize_from is None else Path(initialize_from).resolve()
        self.device = (
            resolve_device(settings.device)
            if distributed is None
            else distributed.device
        )
        self.distributed = distributed or DistributedContext(
            rank=0,
            world_size=1,
            local_rank=0,
            device=self.device,
        )
        if expand_game_pool and self.distributed.enabled:
            raise ValueError("explicit pool expansion currently requires a single-rank resume")
        if is_accelerator(self.device):
            self.device = set_device(self.device)
        if self.device.type == 'cuda' and settings.ppo_cuda_memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(settings.ppo_cuda_memory_fraction, self.device)
        if settings.anchor_batch % self.distributed.world_size:
            raise ValueError(
                "global anchor_batch must be divisible by distributed world_size"
            )
        if settings.base_game_pool_size < self.distributed.world_size:
            raise ValueError(
                "global base_game_pool_size must be at least distributed world_size"
            )
        self.local_anchor_batch = (
            settings.anchor_batch // self.distributed.world_size
        )
        pool_base, pool_remainder = divmod(
            settings.base_game_pool_size, self.distributed.world_size
        )
        self.local_base_game_pool_size = pool_base + int(
            self.distributed.rank < pool_remainder
        )

        self.run_directory = settings.resolve_run_directory(run_directory).resolve()
        if self.distributed.primary:
            self.run_directory.mkdir(parents=True, exist_ok=True)
            # Storage may be provisioned as empty mounts/junctions before the
            # first run. Any file or nonempty directory still requires latest.pt.
            had_existing_run_artifacts = any(not (
                path.name in ("checkpoints", "historical_opponents")
                and path.is_dir() and not any(path.iterdir())) for path in self.run_directory.iterdir())
        else:
            had_existing_run_artifacts = False
        self.distributed.barrier()
        self._had_existing_run_artifacts = bool(
            self.distributed.broadcast_object(had_existing_run_artifacts)
        )
        if not auto_resume and self._had_existing_run_artifacts:
            raise RuntimeError(
                "resume is disabled but the run directory is not empty; "
                "choose a new empty run directory"
            )
        if initialize_from is not None and self._had_existing_run_artifacts:
            raise RuntimeError("--init-from requires a new empty run directory")
        expected_latest = self.run_directory / "checkpoints" / "latest.pt"
        if (
            auto_resume
            and self._had_existing_run_artifacts
            and not expected_latest.exists()
        ):
            raise RuntimeError(
                "run directory contains prior artifacts but latest.pt is missing; "
                "refusing to restart from random weights"
            )
        self.logger = MetricLogger(
            self.run_directory,
            device=self.device,
            rank=self.distributed.rank,
            write_training_metrics=self.distributed.primary,
        )
        self.checkpoints = CheckpointManager(
            self.run_directory,
            keep_archives=settings.keep_checkpoint_archives,
        )
        self.stop_requested = False
        self.update = 0
        self.ppo_rollout_metrics: dict[str, float] = {}
        self.tactical_log_settings = TacticalLogSettings.from_config(settings.raw_config)
        self.training_audit = None
        self._ppo_environment = None
        self.phase = "initializing"
        self.effective_policy_microbatch = min(
            settings.policy_microbatch, self.local_anchor_batch
        )
        self.effective_actor_inference_batch = settings.actor_inference_batch
        self.policy_lr_scale = 1.0
        self.lr_controller = (BoundedLearningRate(
            minimum=settings.minimum_learning_rate, maximum=settings.maximum_learning_rate,
            target_kl=settings.target_kl, high_kl_multiple=settings.early_stop_kl_multiple,
            increase_factor=settings.lr_increase_factor, decrease_factor=settings.lr_decrease_factor,
            stable_updates=settings.lr_stable_updates, cooldown_updates=settings.lr_cooldown_updates,
            ema_decay=settings.lr_ema_decay,
            kl_band_lower=settings.lr_kl_band_lower, kl_band_upper=settings.lr_kl_band_upper,
            epoch_completion=settings.lr_epoch_completion,
            minimum_completed_epochs=settings.lr_minimum_completed_epochs,
            increase_kl_threshold=settings.lr_increase_kl_threshold,
            decrease_kl_threshold=settings.lr_decrease_kl_threshold,
        ) if settings.adaptive_learning_rate else None)
        if settings.lr_epoch_completion or settings.lr_increase_on_full_epochs or settings.lr_kl_band_lower is not None:
            # Fresh runs honor the configured initial rate independently of the
            # feedback bounds; resumes below restore the saved next rate.
            self.lr_controller.scale = settings.policy_learning_rate / settings.maximum_learning_rate
        self.entropy_opening_plies = min(settings.entropy_opening_plies, settings.model.max_transitions)
        entropy_targets = scheduled_entropy_targets(settings, 0)
        self.entropy_controllers = {
            phase: AdaptiveEntropyCoefficient(
                initial=settings.entropy_coefficient, minimum=settings.entropy_minimum,
                maximum=(settings.entropy_opening_maximum
                         if phase == "opening" and settings.entropy_opening_maximum is not None
                         else settings.entropy_maximum),
                target_ratio=entropy_targets[phase],
                adaptation_rate=settings.entropy_adaptation_rate, ema_decay=settings.entropy_ema_decay,
                enabled=settings.adaptive_entropy, deadband=settings.entropy_deadband,
                max_relative_change=settings.entropy_max_relative_change)
            for phase in ("opening", "other")
        }
        if settings.entropy_control_mode == "joint_source":
            # One shared beta for all phases and the extra source bonus.
            self.entropy_controllers["opening"] = self.entropy_controllers["other"]
        self.cumulative: dict[str, int] = {
            "policy_samples": 0,
            "environment_plies": 0,
            "anchors": 0,
            "root_candidates": 0,
            "terminal_continuations": 0,
            "continuation_plies": 0,
            "base_plies": 0,
            "base_games": 0,
        }
        self.layout_buffer = LayoutOutcomeBuffer(settings.layout_buffer_capacity, settings.layout_max_behavior_age)
        self.model_selection = ModelSelection(settings, self.run_directory, self.distributed)
        self._saved_model_selection: dict[str, Any] = {}

        random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        if is_accelerator(self.device):
            manual_seed_all(self.device.type, settings.seed)
        if settings.enable_tf32 and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self.policy = GamePolicyTransformer(settings.model).to(self.device)
        self.layout = PieceConditionedLayoutPointerDecoder(settings.model).to(
            self.device
        )
        self.distributed.broadcast_module(self.policy)
        self.distributed.broadcast_module(self.layout)
        self.reference_policy = (
            copy.deepcopy(self.policy).eval().requires_grad_(False)
            if settings.algorithm == "grpo" else None
        )
        self.critic = None
        if settings.algorithm == "mappo":
            self.critic = GameCentralValueTransformer(
                replace(settings.model, temporal_layers=settings.critic_layers)).to(self.device)
        elif settings.algorithm == "ppo":
            self.critic = GameValueTransformer(settings.model).to(self.device)
        if self.critic is not None:
            self.critic.initialize_from_policy(self.policy)
            self.distributed.broadcast_module(self.critic)
        self.reference_layout = copy.deepcopy(self.layout).eval().requires_grad_(False)
        self.policy_parallel: nn.Module
        if self.distributed.enabled:
            self.policy_parallel = DistributedDataParallel(
                self.policy,
                device_ids=(
                    [self.distributed.local_rank]
                    if is_accelerator(self.device)
                    else None
                ),
                output_device=(
                    self.distributed.local_rank
                    if is_accelerator(self.device)
                    else None
                ),
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        else:
            self.policy_parallel = self.policy
        self.critic_parallel: nn.Module | None = self.critic
        if self.critic is not None and self.distributed.enabled:
            self.critic_parallel = DistributedDataParallel(
                self.critic,
                device_ids=([self.distributed.local_rank] if is_accelerator(self.device) else None),
                output_device=(self.distributed.local_rank if is_accelerator(self.device) else None),
                broadcast_buffers=False, gradient_as_bucket_view=True,
            )
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=settings.policy_learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.05,
            fused=True if settings.algorithm in ("ppo", "mappo") and settings.ppo_fused_optimizer and self.device.type == 'cuda' else None,
        )
        self.layout_optimizer = torch.optim.AdamW(
            self.layout.parameters(),
            lr=settings.layout_learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.05,
            fused=True if settings.algorithm in ("ppo", "mappo") and settings.ppo_fused_optimizer and self.device.type == 'cuda' else None,
        )
        self.critic_optimizer = (
            torch.optim.AdamW(
                self.critic.parameters(), lr=settings.critic_learning_rate,
                betas=(0.9, 0.95), eps=1e-8, weight_decay=0.05,
                fused=True if settings.ppo_fused_optimizer and self.device.type == 'cuda' else None,
            ) if self.critic is not None else None
        )
        self.amp_dtype = self._amp_dtype()
        self.grad_scaler = make_grad_scaler(
            self.device.type, enabled=self.amp_dtype is torch.float16
        )
        self.layout_grad_scaler = make_grad_scaler(
            self.device.type, enabled=self.amp_dtype is torch.float16
        )
        self.critic_grad_scaler = make_grad_scaler(
            self.device.type, enabled=self.amp_dtype is torch.float16
        )
        self._seed_runtime()
        from .historical_opponents import HistoricalOpponents
        self.historical = HistoricalOpponents(settings, self.run_directory, self.distributed)
        self.pool = BaseGamePool(
            settings.mode,
            pool_size=self.local_base_game_pool_size,
            max_transitions=settings.model.max_transitions,
            history_window_stride=settings.model.history_window_stride,
            max_game_plies=settings.max_game_plies,
            dead_rules_enabled=settings.dead_rules_enabled,
            seed=settings.seed + 17 + 1_000_003 * self.distributed.rank,
            layout_prefetch_games=settings.layout_prefetch_games,
            no_capture_draw_plies=settings.no_capture_draw_plies,
            centralized_critic=settings.algorithm == "mappo",
        )
        self.pool.historical = self.historical
        try:
            if self.initialize_from is not None:
                self._initialize_weights(self.initialize_from)
            resumed = self._resume_if_available() if auto_resume else False
            self._sync_entropy_targets()
            self.resumed_from_update = self.update
            self.resumed_from_cumulative = dict(self.cumulative)
            self._write_training_progress()
            if settings.arena_enabled:
                if self._saved_model_selection and not self.model_selection.state_path.exists():
                    raise RuntimeError("model selection state is missing from the resumed run")
                self.model_selection.initialize(
                    self.policy, self.layout, update=self.update, cumulative=self.cumulative,
                    adopt_current_draw_rules=adopt_current_draw_rules,
                    adopt_pass_rule=adopt_pass_rule,
                    adopt_two_player_no_pass=adopt_two_player_no_pass,
                    adopt_champion_evaluation=adopt_champion_evaluation,
                )
            self.historical.update_progress(self.policy, self.layout,
                environment_plies=self.cumulative["environment_plies"], update=self.update)
            self.historical.write_status(self.pool)
        except BaseException:
            self.logger.close()
            raise
        self._configure_optimizer_backend()
        self._write_run_config()
        self._export_live_inference()
        self._install_signal_handlers()
        if resumed and self.context_shrunk_on_resume:
            self.save_checkpoint(reason="context_window_migration", archive=False)
        if resumed and (adopt_current_draw_rules or adopt_draw_penalty or adopt_pass_rule or adopt_no_move_elimination or adopt_two_player_no_pass or
                        adopt_flag_capture_reward or adopt_champion_evaluation or adopt_futile_attack_penalty or
                        adopt_bomb_rewards or adopt_flag_defense_rewards or adopt_player_advantages or adopt_known_nonflag_hq_penalty):
            self.save_checkpoint(reason="two_player_no_pass_migration" if adopt_two_player_no_pass else
                                 "no_move_rule_migration" if adopt_no_move_elimination else
                                 "known_nonflag_hq_objective_migration" if adopt_known_nonflag_hq_penalty else
                                 "player_advantage_migration" if adopt_player_advantages else
                                 "champion_evaluation_migration" if adopt_champion_evaluation else
                                 "flag_defense_objective_migration" if adopt_flag_defense_rewards else
                                 "bomb_objective_migration" if adopt_bomb_rewards else
                                 "futile_attack_objective_migration" if adopt_futile_attack_penalty else
                                 "flag_capture_objective_migration" if adopt_flag_capture_reward else
                                 "pass_rule_migration" if adopt_pass_rule else
                                 "draw_objective_migration" if adopt_draw_penalty else "draw_rule_migration", archive=False)
        if not resumed and settings.checkpoint_policy == "periodic":
            self.save_checkpoint(reason="initialized", archive=False)
        if settings.checkpoint_policy == "evaluation":
            self.logger.event(
                "checkpoint policy=evaluation: save only at scheduled matches; "
                "stopping between matches retains the last evaluation checkpoint"
            )
            if not settings.arena_enabled:
                self.logger.event("model selection disabled; no automatic checkpoints will be written")
        self.logger.event(
            "initialized mode=%s device=%s policy_params=%d layout_params=%d "
            "resume_update=%d dead_rules_enabled=%s step_budget=%s:%s "
            "player_models=1(shared_across_all_seats) rank=%d/%d "
            "local_anchor_batch=%d local_pool=%d per_device_microbatch=%d"
            % (
                settings.mode.value,
                self.device,
                parameter_count(self.policy),
                parameter_count(self.layout),
                self.update,
                settings.dead_rules_enabled,
                settings.step_budget_counter,
                settings.step_budget_target,
                self.distributed.rank,
                self.distributed.world_size,
                self.local_anchor_batch,
                self.local_base_game_pool_size,
                self.effective_policy_microbatch,
            )
        )
        self.logger.event(
            f"algorithm={settings.algorithm} critic_params="
            f"{0 if self.critic is None else parameter_count(self.critic)} "
            f"frozen_reference_policy={self.reference_policy is not None}"
        )
        if settings.arena_enabled:
            self.logger.event(
                f"model selection enabled: milestones={self.model_selection.milestones} "
                f"games={settings.arena_games} best_update={self.model_selection.state['best_update']} "
                f"best_checkpoint={self.model_selection.best_path} "
                f"baseline_pending_in_memory={self.model_selection.state['best_sha256'] is None}"
            )
        if settings.grpo_equivalent_plies is not None:
            self.logger.event(
                f"planning_grpo_equivalent_plies={settings.grpo_equivalent_plies} "
                f"assumed_mean_remaining_plies={settings.grpo_mean_remaining_plies} "
                f"actual_ppo_environment_target={settings.target_environment_plies}; "
                "equivalent root-state coverage only, not equivalent playing strength"
            )
        self.logger.start_resource_monitor(
            lambda: {
                "training/algorithm": self.settings.algorithm,
                "training/environment_plies": self.cumulative["environment_plies"],
                "training/step_budget_counter": self.settings.step_budget_counter,
                "training/target_environment_plies": (
                    -1 if self.settings.target_environment_plies is None
                    else self.settings.target_environment_plies
                ),
                "training/update": self.update,
                "training/phase": self.phase,
                "training/continuation_plies": self.cumulative[
                    "continuation_plies"
                ],
                "training/target_continuation_plies": (
                    -1
                    if self.settings.target_continuation_plies is None
                    else self.settings.target_continuation_plies
                ),
                "training/effective_policy_microbatch": (
                    self.effective_policy_microbatch
                ),
                "training/effective_actor_batch": (
                    self.effective_actor_inference_batch
                ),
                "distributed/rank": self.distributed.rank,
                "distributed/world_size": self.distributed.world_size,
            },
            interval_seconds=settings.resource_monitor_interval_seconds,
        )
        self.phase = "ready"

    def _export_live_inference(self) -> None:
        if self.settings.inference_snapshot_every_updates and self.distributed.primary:
            started = time.perf_counter()
            try:
                path = export_inference_snapshot(
                    self.run_directory, self.policy, self.layout, self.settings, self.update,
                )
                self.logger.event(f"CPU play snapshot update={self.update} path={path} seconds={time.perf_counter()-started:.3f}")
            except Exception as error:
                self.logger.event(f"CPU play snapshot export failed: {type(error).__name__}: {error}")

    def _configure_optimizer_backend(self) -> None:
        # Loading an old state restores its old param-group flags. Reapply the
        # requested backend and migrate only scalar step counters, not moments.
        if self.settings.algorithm not in ("ppo", "mappo"):
            return
        fused = self.settings.ppo_fused_optimizer and self.device.type == 'cuda'
        for optimizer in (self.policy_optimizer, self.critic_optimizer, self.layout_optimizer):
            if optimizer is None:
                continue
            for group in optimizer.param_groups:
                group['fused'], group['foreach'] = (True, False) if fused else (None, None)
                for parameter in group['params']:
                    state = optimizer.state.get(parameter, {})
                    if 'step' in state:
                        state['step'] = state['step'].to(self.device if fused else 'cpu')

    def _seed_runtime(self) -> None:
        runtime_seed = self.settings.seed + 1_000_003 * self.distributed.rank
        random.seed(runtime_seed)
        torch.manual_seed(runtime_seed)
        if is_accelerator(self.device):
            manual_seed_all(self.device.type, runtime_seed)

    def _amp_dtype(self) -> torch.dtype | None:
        if not is_accelerator(self.device) or self.settings.amp == "float32":
            return None
        if self.settings.amp == "bfloat16":
            if not is_bf16_supported(self.device):
                self.logger.event("BF16 unsupported; falling back to float16")
                return torch.float16
            return torch.bfloat16
        return torch.float16

    def _write_training_progress(self) -> None:
        """Publish completed work independently of stale append-only metrics.

        This small record is NOT a recovery checkpoint. On a new process it
        immediately replaces the abandoned log suffix with the restored count.
        """
        if not self.distributed.primary:
            return
        record = {"pid": os.getpid(), "session_started_unix": self.session_started_unix,
                  "timestamp_unix": time.time(), "update": self.update,
                  "cumulative": dict(self.cumulative),
                  "resumed_from_update": self.resumed_from_update,
                  "resumed_from_cumulative": self.resumed_from_cumulative}
        temporary = self.run_directory / ".training_progress.json.tmp"
        temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.run_directory / "training_progress.json")

    def _write_run_config(self) -> None:
        if not self.distributed.primary:
            return
        destination = self.run_directory / "resolved_config.json"
        temporary = self.run_directory / ".resolved_config.json.tmp"
        resolved = self.settings.serializable()
        resolved["draw_objective_migration"] = self.draw_objective_migration
        resolved["context_window_migration"] = self.context_window_migration
        resolved["flag_capture_objective_migration"] = self.flag_capture_objective_migration
        resolved["futile_attack_objective_migration"] = self.futile_attack_objective_migration
        resolved["bomb_objective_migration"] = self.bomb_objective_migration
        resolved["flag_defense_objective_migration"] = self.flag_defense_objective_migration
        resolved["known_nonflag_hq_objective_migration"] = self.known_nonflag_hq_objective_migration
        resolved["player_advantage_migration"] = self.player_advantage_migration
        resolved["pass_rule_migration"] = self.pass_rule_migration
        resolved["no_move_rule_migration"] = self.no_move_rule_migration
        resolved["two_player_no_pass_migration"] = self.two_player_no_pass_migration
        resolved["initialized_from"] = (
            None if self.initialize_from is None else str(self.initialize_from)
        )
        resolved["distributed_runtime"] = {
            "world_size": self.distributed.world_size,
            "anchor_batch_semantics": "global",
            "configured_anchor_batch": self.settings.anchor_batch,
            "local_anchor_batch": self.local_anchor_batch,
            "microbatch_semantics": "per_device",
            "per_device_policy_microbatch": self.effective_policy_microbatch,
            "global_base_game_pool_size": self.settings.base_game_pool_size,
            "local_base_game_pool_size_rank0": (
                self.settings.base_game_pool_size // self.distributed.world_size
                + int(self.settings.base_game_pool_size % self.distributed.world_size > 0)
            ),
        }
        temporary.write_text(
            json.dumps(
                resolved, ensure_ascii=False, indent=2, default=str
            ),
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    def _initialize_weights(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require_current_checkpoint(payload)
        if payload.get("mode") != self.settings.mode.value:
            raise RuntimeError("initialization checkpoint mode does not match")
        if payload.get("dead_rules_enabled") is not self.settings.dead_rules_enabled:
            raise RuntimeError("initialization checkpoint dead-rule variant does not match")
        self.policy.load_state_dict(payload["policy"], strict=True)
        self.layout.load_state_dict(payload["layout"], strict=True)
        if self.reference_policy is not None:
            self.reference_policy.load_state_dict(self.policy.state_dict())
        self.reference_layout.load_state_dict(self.layout.state_dict())
        if self.critic is not None:
            if payload.get("algorithm") == self.settings.algorithm and "critic" in payload:
                self.critic.load_state_dict(payload["critic"], strict=True)
            else:
                self.critic.initialize_from_policy(self.policy)
        self.logger.event(f"initialized model weights from {path}; new optimizers and counters")

    def _resume_if_available(self) -> bool:
        # Avoid a second accelerator-resident copy of every model and optimizer.
        payload = self.checkpoints.load_latest(map_location="cpu")
        if payload is None:
            if self._had_existing_run_artifacts:
                raise RuntimeError(
                    "run directory contains prior artifacts but latest.pt is missing; "
                    "refusing to restart from random weights"
                )
            self.logger.event("no checkpoint found; starting a new run")
            return False
        from .context_migration import reconcile_context_window
        self.context_shrunk_on_resume = reconcile_context_window(
            payload, self.settings, adopt=self.shrink_context_window)
        if self.adopt_current_draw_rules:
            from .rule_migration import adopt_draw_rules
            adopt_draw_rules(payload, self.settings)
        from .pass_migration import reconcile_pass_rule
        reconcile_pass_rule(payload, adopt=self.adopt_pass_rule)
        from .no_move_migration import reconcile_no_move_rule
        reconcile_no_move_rule(payload, adopt=self.adopt_no_move_elimination)
        from .two_player_no_pass_migration import reconcile_two_player_no_pass
        reconcile_two_player_no_pass(payload, self.settings, adopt=self.adopt_two_player_no_pass)
        from .checkpoint_format import checkpoint_format_for_model
        from .no_pass_features import remove_two_player_pass_inputs
        if (payload.get("mode") == "two_player" and payload.get("format_version") == 17
                and checkpoint_format_for_model(self.settings.model) == 18):
            if not self.adopt_two_player_no_pass:
                raise ValueError("removing pass input dimensions requires --adopt-two-player-no-pass")
            remove_two_player_pass_inputs(payload, optimizer_modules={"policy": self.policy, "critic": self.critic})
            payload["trainer_state"].setdefault("two_player_no_pass_migration", {"update": payload["update"]}).update(
                action_inputs_from=14, action_inputs_to=10,
                projected="Actor/Critic action weight and Adam moment columns; all other tensors unchanged")
        from .objective_migration import reconcile_draw_objective, reconcile_flag_capture_objective
        reconcile_draw_objective(payload, self.settings, adopt=self.adopt_draw_penalty)
        reconcile_flag_capture_objective(payload, self.settings, adopt=self.adopt_flag_capture_reward)
        # Check the original saved budget before the older penalty migration
        # updates it, so neither objective silently adopts a new decay horizon.
        from .objective_migration import reconcile_bomb_objective
        reconcile_bomb_objective(payload, self.settings, adopt=self.adopt_bomb_rewards)
        from .objective_migration import reconcile_flag_defense_objective
        reconcile_flag_defense_objective(payload, self.settings, adopt=self.adopt_flag_defense_rewards)
        from .objective_migration import reconcile_futile_attack_objective
        reconcile_futile_attack_objective(payload, self.settings, adopt=self.adopt_futile_attack_penalty)
        from .objective_migration import reconcile_player_advantages
        reconcile_player_advantages(payload, self.settings, adopt=self.adopt_player_advantages)
        from .objective_migration import reconcile_known_nonflag_hq_objective
        reconcile_known_nonflag_hq_objective(payload, self.settings, adopt=self.adopt_known_nonflag_hq_penalty)
        self.update, trainer_state = restore_training_state(
            payload,
            expected_mode=self.settings.mode.value,
            expected_dead_rules_enabled=self.settings.dead_rules_enabled,
            policy=self.policy,
            layout=self.layout,
            reference_policy=self.reference_policy,
            reference_layout=self.reference_layout,
            policy_optimizer=self.policy_optimizer,
            layout_optimizer=self.layout_optimizer,
            accelerator_device=self.device,
            expected_algorithm=self.settings.algorithm,
            critic=self.critic,
            critic_optimizer=self.critic_optimizer,
        )
        if self.distributed.primary:
            # Repair the tiny sidecar if the process died after latest.pt's
            # atomic replacement but before its manifest could be committed.
            self.checkpoints.write_manifest(payload)
        self.policy_lr_scale = float(trainer_state.get("policy_lr_scale", 1.0))
        if self.lr_controller is not None:
            self.lr_controller.load_state_dict(trainer_state.get("adaptive_learning_rate"),
                legacy_scale=self.policy_lr_scale,
                scheduled=self._scheduled_learning_rate(self.update + 1))
            if self.settings.lr_epoch_completion or self.settings.lr_increase_on_full_epochs or self.settings.lr_kl_band_lower is not None:
                saved_config = payload.get("config", {})
                saved_controller = trainer_state.get("adaptive_learning_rate") or {}
                adopting_band = (self.settings.lr_kl_band_lower is not None
                                 and saved_config.get("lr_kl_band_lower") is None)
                if self.settings.lr_epoch_completion and not saved_config.get("lr_epoch_completion", False):
                    rate = self.policy_optimizer.param_groups[0]["lr"]
                    self.logger.event(f"LR feedback changed to epoch completion; preserved actual LR={rate:.8g}")
                elif adopting_band:
                    # Discard the old controller's pending 50%/40% jump. Start
                    # the new observation window at the actual saved rate.
                    rate = self.policy_optimizer.param_groups[0]["lr"]
                    self.logger.event(f"LR feedback changed to KL band "
                        f"[{self.settings.lr_kl_band_lower}, {self.settings.lr_kl_band_upper}]; "
                        f"preserved actual LR={rate:.8g}; reset KL trend observations")
                elif (saved_config.get("lr_epoch_completion", False)
                      or saved_config.get("lr_increase_on_full_epochs", False)
                      or saved_config.get("lr_kl_band_lower") is not None):
                    # The multiplier is relative to the ceiling in this mode.
                    # Preserve the saved next rate even if that ceiling changes.
                    saved_scale = saved_controller.get("scale", self.policy_lr_scale)
                    saved_maximum = saved_config["maximum_learning_rate"]
                    rate = min(saved_maximum, max(saved_config["minimum_learning_rate"],
                                                 saved_maximum * saved_scale))
                else:
                    # Switching from warmup/cosine must not reinterpret scale=1
                    # as permission to jump immediately to the new 1e-4 bound.
                    rate = self.policy_optimizer.param_groups[0]["lr"]
                    self.logger.event(f"LR feedback changed to constant-reference control; preserved actual LR={rate:.8g}")
                self.lr_controller.scale = rate / self.settings.maximum_learning_rate
                self.lr_controller.rate(self.settings.maximum_learning_rate)
            self.policy_lr_scale = self.lr_controller.scale
        if self.settings.entropy_control_mode == "joint_source":
            if payload.get("config", {}).get("entropy_control_mode") == "joint_source":
                self.entropy_controllers["other"].load_state_dict(
                    trainer_state.get("adaptive_entropy", {}).get("other"))
                saved_feedback = payload.get("config", {}).get("entropy_feedback", "joint")
                if saved_feedback != self.settings.entropy_feedback:
                    controller = self.entropy_controllers["other"]
                    controller.entropy_ema = None
                    controller.updates = 0
                    self.logger.event(
                        f"entropy feedback changed {saved_feedback} -> {self.settings.entropy_feedback}; "
                        f"preserved beta={controller.coefficient:.9g}; new EMA starts at first final probe")
            else:
                self.logger.event("enabled shared entropy feedback from configured beta; new EMA starts at first final probe")
        else:
            for phase, controller in self.entropy_controllers.items():
                controller.load_state_dict(trainer_state.get("adaptive_entropy", {}).get(phase))
        saved_source_coefficient = payload.get("config", {}).get("source_entropy_coefficient", 0.0)
        if saved_source_coefficient != self.settings.source_entropy_coefficient:
            self.logger.event(
                f"source entropy bonus changed on resume: {saved_source_coefficient:g} -> "
                f"{self.settings.source_entropy_coefficient:g}; raw H(source), added to joint entropy bonus"
            )
        self._saved_model_selection = dict(trainer_state.get("model_selection", {}))
        restored_microbatch = int(
            trainer_state.get(
                "effective_policy_microbatch",
                self.settings.policy_microbatch,
            )
        )
        if restored_microbatch < self.settings.minimum_policy_microbatch:
            raise RuntimeError("checkpoint effective microbatch is below configured minimum")
        saved_microbatch = int(payload.get('config', {}).get('policy_microbatch', restored_microbatch))
        if restored_microbatch == saved_microbatch and self.settings.policy_microbatch > saved_microbatch:
            restored_microbatch = self.settings.policy_microbatch
        self.effective_policy_microbatch = min(
            restored_microbatch,
            self.settings.policy_microbatch,
            self.local_anchor_batch,
        )
        restored_actor_batch = int(
            trainer_state.get(
                "effective_actor_inference_batch",
                self.settings.actor_inference_batch,
            )
        )
        if restored_actor_batch <= 0:
            raise RuntimeError("checkpoint effective actor batch must be positive")
        saved_actor_batch = int(
            payload.get("config", {}).get(
                "actor_inference_batch", restored_actor_batch
            )
        )
        if (
            restored_actor_batch == saved_actor_batch
            and self.settings.actor_inference_batch > saved_actor_batch
        ):
            # A larger explicit setting is an intentional retune, rather than
            # an OOM recovery cap inherited from the old configuration.
            self.effective_actor_inference_batch = (
                self.settings.actor_inference_batch
            )
        else:
            self.effective_actor_inference_batch = min(
                restored_actor_batch,
                self.settings.actor_inference_batch,
            )
        if self.reset_oom_batch_limits:
            self.effective_actor_inference_batch = self.settings.actor_inference_batch
            self.effective_policy_microbatch = min(self.settings.policy_microbatch, self.local_anchor_batch)
            self.logger.event('explicitly reset saved OOM batch caps to configured limits after retuning')
        if "grad_scaler" in trainer_state:
            self.grad_scaler.load_state_dict(trainer_state["grad_scaler"])
        if "layout_grad_scaler" in trainer_state:
            self.layout_grad_scaler.load_state_dict(
                trainer_state["layout_grad_scaler"]
            )
        if "critic_grad_scaler" in trainer_state:
            self.critic_grad_scaler.load_state_dict(trainer_state["critic_grad_scaler"])
        restored_cumulative = trainer_state.get("cumulative", {})
        self.draw_rule_migration = trainer_state.get("draw_rule_migration")
        self.context_window_migration = trainer_state.get("context_window_migration")
        if self.context_shrunk_on_resume:
            self.logger.event(f"context window shrunk {self.context_window_migration['from']} -> "
                              f"{self.context_window_migration['to']}; all model/optimizer tensors retained")
        self.pass_rule_migration = trainer_state.get("pass_rule_migration")
        self.no_move_rule_migration = trainer_state.get("no_move_rule_migration")
        self.two_player_no_pass_migration = trainer_state.get("two_player_no_pass_migration")
        self.draw_objective_migration = trainer_state.get("draw_objective_migration")
        self.flag_capture_objective_migration = trainer_state.get("flag_capture_objective_migration")
        self.futile_attack_objective_migration = trainer_state.get("futile_attack_objective_migration")
        self.bomb_objective_migration = trainer_state.get("bomb_objective_migration")
        self.flag_defense_objective_migration = trainer_state.get("flag_defense_objective_migration")
        self.known_nonflag_hq_objective_migration = trainer_state.get("known_nonflag_hq_objective_migration")
        self.player_advantage_migration = trainer_state.get("player_advantage_migration")
        if self.settings.futile_attack_penalty and "futile_attack_penalty" not in trainer_state:
            raise RuntimeError("enabled futile attack penalty has no saved decay state")
        self.futile_attack_controller.load_state_dict(trainer_state.get("futile_attack_penalty"))
        self.last_checkpoint_environment_plies = int(trainer_state.get(
            "last_checkpoint_environment_plies", restored_cumulative.get("environment_plies", 0),
        ))
        self.cumulative.update(
            {key: int(value) for key, value in restored_cumulative.items()}
        )
        if "environment_plies" not in restored_cumulative:
            # Older checkpoints retain the disjoint base/branch counts. Rebuild
            # the total once, never add it to an already saved environment total.
            self.cumulative["environment_plies"] = (
                self.cumulative["base_plies"] + self.cumulative["continuation_plies"]
            )
        if self.distributed.primary:
            self.layout_buffer.restore(trainer_state.get("layout_buffer", []), version=self.update,
                minimum_version=self._layout_minimum_version(), totals=trainer_state.get("layout_buffer_totals"))
            self.logger.event(f"layout buffer restored: retained={len(self.layout_buffer)} "
                              f"expired={self.layout_buffer.totals['expired']} overflow={self.layout_buffer.totals['overflow']}")
        distributed_state = trainer_state.get("distributed")
        if distributed_state is not None:
            saved_world_size = int(distributed_state["world_size"])
            rank_states = distributed_state["rank_states"]
            if len(rank_states) != saved_world_size:
                raise RuntimeError("checkpoint rank-state count is inconsistent")
            if saved_world_size == self.distributed.world_size:
                local_state = rank_states[self.distributed.rank]
                self._restore_local_pool(local_state["base_game_pool"])
                restore_rng_state(local_state["rng_state"], self.device)
            else:
                saved_pools = [item["base_game_pool"] for item in rank_states]
                if any(p.get("historical_opponents", {}).get("active_id") for p in saved_pools
                       if p.get("historical_opponents")):
                    raise RuntimeError("active historical cohorts require the same world size on resume")
                all_slots = [
                    slot
                    for saved_pool in saved_pools
                    for slot in saved_pool["slots"]
                ]
                saved_global_pool_size = sum(
                    int(saved_pool["pool_size"])
                    for saved_pool in saved_pools
                )
                if saved_global_pool_size != self.settings.base_game_pool_size:
                    raise RuntimeError(
                        "checkpoint global base-game pool size does not match config"
                    )
                sizes = [
                    self.settings.base_game_pool_size
                    // self.distributed.world_size
                    + int(
                        rank
                        < self.settings.base_game_pool_size
                        % self.distributed.world_size
                    )
                    for rank in range(self.distributed.world_size)
                ]
                start = sum(sizes[: self.distributed.rank])
                repartitioned_pool = dict(saved_pools[0])
                repartitioned_pool.pop("historical_opponents", None)
                # Rank RNG streams restart after repartition; never duplicate
                # rank 0's unused random layouts across every new rank.
                repartitioned_pool.pop('layout_queue', None)
                repartitioned_pool.pop('layout_queue_version', None)
                repartitioned_pool["pool_size"] = self.local_base_game_pool_size
                repartitioned_pool["slots"] = all_slots[
                    start : start + self.local_base_game_pool_size
                ]
                repartitioned_pool["rng_state"] = self.pool.rng.getstate()
                self.pool.load_state_dict(repartitioned_pool)
                self._seed_runtime()
                self.logger.event(
                    "repartitioned distributed checkpoint from world_size=%d "
                    "to world_size=%d; unfinished games preserved and rank RNG "
                    "streams deterministically restarted"
                    % (saved_world_size, self.distributed.world_size)
                )
        elif "base_game_pool" in trainer_state and self.distributed.enabled:
            # One-time migration from a legacy single-process checkpoint: split
            # its global pool without duplicating unfinished games.  New rank
            # RNG streams are deliberately made independent.
            raw_pool = dict(trainer_state["base_game_pool"])
            raw_pool.pop('layout_queue', None)
            raw_pool.pop('layout_queue_version', None)
            if int(raw_pool["pool_size"]) != self.settings.base_game_pool_size:
                raise RuntimeError(
                    "legacy checkpoint pool size does not match configured global pool"
                )
            sizes = [
                self.settings.base_game_pool_size // self.distributed.world_size
                + int(
                    rank
                    < self.settings.base_game_pool_size
                    % self.distributed.world_size
                )
                for rank in range(self.distributed.world_size)
            ]
            start = sum(sizes[: self.distributed.rank])
            raw_pool["pool_size"] = self.local_base_game_pool_size
            raw_pool["slots"] = raw_pool["slots"][
                start : start + self.local_base_game_pool_size
            ]
            raw_pool["rng_state"] = self.pool.rng.getstate()
            self.pool.load_state_dict(raw_pool)
            self._seed_runtime()
            self.logger.event(
                "migrated legacy single-process checkpoint pool to distributed shard"
            )
        elif "base_game_pool" in trainer_state:
            self._restore_local_pool(trainer_state["base_game_pool"])
        elif "base_pool_rng_state" in trainer_state:
            # Backward compatibility with checkpoints written before unfinished
            # base games and their history windows were persisted.
            self.pool.rng.setstate(trainer_state["base_pool_rng_state"])
        self.logger.event(
            f"resumed checkpoint update={self.update}; models, optimizers, "
            f"rank-{self.distributed.rank} RNG/pool, and layout state restored"
        )
        return True

    def _install_signal_handlers(self) -> None:
        def request_stop(signum: int, _frame: Any) -> None:
            self.stop_requested = True
            action = (
                "stopping after current update; retaining the last evaluation checkpoint"
                if self.settings.checkpoint_policy == "evaluation"
                else "saving after current update"
            )
            self.logger.event(f"received signal {signum}; {action}")

        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(signum, request_stop)
            except (OSError, ValueError):
                pass

    def _restore_local_pool(self, state):
        self.pool.load_state_dict(state, allow_expansion=self.expand_game_pool)
        if int(state["pool_size"]) != self.pool.pool_size:
            self.pool_expanded_on_resume = True
            self.logger.event(f"expanded parallel pool {state['pool_size']} -> {self.pool.pool_size}; "
                              f"preserved {len(self.pool.slots)} unfinished games, layout queue and RNG; "
                              "new slots will be filled before the next rollout")

    def _scheduled_learning_rate(self, update: int) -> float:
        settings = self.settings
        if settings.lr_epoch_completion or settings.lr_increase_on_full_epochs or settings.lr_kl_band_lower is not None:
            # A constant reference makes feedback changes actual LR changes,
            # with no independent warmup rise or late cosine reduction.
            return settings.maximum_learning_rate
        if update <= settings.warmup_updates:
            fraction = update / max(settings.warmup_updates, 1)
            scheduled = settings.policy_learning_rate * fraction
        else:
            progress = (update - settings.warmup_updates) / max(
                settings.total_updates - settings.warmup_updates, 1
            )
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            scheduled = settings.minimum_learning_rate + cosine * (
                settings.policy_learning_rate - settings.minimum_learning_rate
            )
        return max(settings.minimum_learning_rate, scheduled)

    def _learning_rate(self, update: int) -> float:
        scheduled = self._scheduled_learning_rate(update)
        if self.lr_controller is not None:
            value = self.lr_controller.rate(scheduled)
            self.policy_lr_scale = self.lr_controller.scale
            return value
        return min(self.settings.maximum_learning_rate, max(
            self.settings.minimum_learning_rate, scheduled * self.policy_lr_scale))

    def _set_policy_lr(self, update: int) -> float:
        value = self._learning_rate(update)
        for group in self.policy_optimizer.param_groups:
            group["lr"] = value
        if self.critic_optimizer is not None:
            for group in self.critic_optimizer.param_groups:
                group["lr"] = value * (
                    self.settings.critic_learning_rate / self.settings.policy_learning_rate
                )
        return value

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype or torch.float32,
            enabled=self.amp_dtype is not None,
        )

    def _backward_policy_epoch(
        self, groups: Sequence[PolicyGroup] | Sequence[PPOSample], *, defer_metrics: bool = False,
    ) -> dict[str, float]:
        """Accumulate one epoch so an OOM can safely restart before step()."""

        self.policy_optimizer.zero_grad(set_to_none=True)
        # Phase statistics remain observable with fixed entropy. Disabled
        # controllers both retain the configured initial coefficient on resume.
        entropy_options = {
            "opening_entropy_coefficient": self.entropy_controllers["opening"].coefficient,
            "entropy_opening_plies": self.entropy_opening_plies,
        }
        epoch_metrics: dict[str, float] = {}
        gradient_balance = 1.0
        global_owned = len(groups)
        if self.settings.algorithm in ("ppo", "mappo") and self.historical.active:
            owned = [item for item in groups if item.learnable]
            mean_owned = self.distributed.reduce_float(float(len(owned)), operation="mean")
            global_owned = mean_owned
            gradient_balance = len(owned) / mean_owned if mean_owned > 0 else 0.
            groups = owned or groups[:1]
        chunks = self._learner_chunks(groups)
        varlen_before = getattr(self.policy, '_varlen_calls', 0)
        token_lengths = [
            [len(chunk[group[0]].state.records) for group in
             history_prefix_groups([item.state for item in chunk])]
            if self._sequence_training_enabled() else [len(item.state.records) for item in chunk]
            for chunk in chunks
        ]
        padded_tokens = sum(len(lengths) * max(lengths) for lengths in token_lengths)
        real_tokens = sum(sum(lengths) for lengths in token_lengths)
        epoch_metrics["optimizer/temporal_padding_fraction"] = (
            1.0 - real_tokens / max(padded_tokens, 1)
        )
        epoch_metrics["optimizer/learner_sequences"] = float(sum(map(len, token_lengths)))
        epoch_metrics["optimizer/learner_samples"] = float(len(groups))
        epoch_metrics["optimizer/history_token_reuse_fraction"] = (
            1.0 - real_tokens / max(sum(len(item.state.records) for item in groups), 1)
        )
        # Repeating the fixed-size action heads across graph subbatches costs
        # more than the dynamic packed Actor path; accumulate only value graphs.
        captured = (self._try_learner_graph(groups, critic=False)
                    if self.settings.algorithm in ("ppo", "mappo") and gradient_balance == 1.0
                    and all(item.learnable for item in groups) else None)
        if captured is not None:
            capacity = captured.pop('_graph_tokens')
            epoch_metrics.update(captured)
            epoch_metrics['optimizer/learner_graph_fraction'] = 1.0
            epoch_metrics['optimizer/policy_graph_subbatches'] = 1.0
            epoch_metrics['optimizer/temporal_padding_fraction'] = 1.0 - real_tokens / capacity
            epoch_metrics['optimizer/no_policy_samples'] = 0.0
            return epoch_metrics if defer_metrics else self._materialize_metrics(epoch_metrics)
        epoch_metrics['optimizer/learner_graph_fraction'] = 0.0
        for chunk_index, chunk in enumerate(chunks):
            synchronization = (
                self.policy_parallel.no_sync()
                if self.distributed.enabled and chunk_index < len(chunks) - 1
                else nullcontext()
            )
            # DDP requires both forward and backward to be inside no_sync().
            # The final normal pass synchronizes all accumulated gradients.
            with synchronization:
                with self._autocast():
                    if self.settings.algorithm in ("ppo", "mappo"):
                        output = policy_ppo_loss(
                            self.policy_parallel, chunk,
                            clip_epsilon=self.settings.clip_epsilon,
                            entropy_coefficient=self.entropy_controllers["other"].coefficient,
                            source_entropy_coefficient=self._source_entropy_coefficient(),
                            policy_clip=self._policy_clip(),
                            sequence_training=self._sequence_training_enabled(),
                            defer_metrics=True,
                            **entropy_options,
                        )
                    else:
                        output = policy_grpo_loss(
                            self.policy_parallel, self.reference_policy, chunk,
                            clip_epsilon=self.settings.clip_epsilon,
                            kl_coefficient=self.settings.kl_coefficient,
                            entropy_coefficient=self.entropy_controllers["other"].coefficient,
                            **entropy_options,
                        )
                    scaled_loss = output.loss * (len(chunk) / len(groups)) * gradient_balance
                self.grad_scaler.scale(scaled_loss).backward()
            for key, value in output.metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value * (
                    len(chunk) / len(groups)
                ) * gradient_balance
        epoch_metrics["optimizer/no_policy_samples"] = float(global_owned == 0)
        if getattr(self.policy, '_varlen_calls', 0) - varlen_before == len(chunks):
            epoch_metrics['optimizer/temporal_padding_fraction'] = 0.0
        return epoch_metrics if defer_metrics else self._materialize_metrics(epoch_metrics)

    @staticmethod
    def _materialize_metrics(metrics):
        """One device transfer per optimizer batch, after all microbatch backward."""
        keys = [key for key, value in metrics.items() if isinstance(value, torch.Tensor)]
        if keys:
            values = torch.stack([metrics[key] for key in keys]).detach().cpu().tolist()
            metrics.update(zip(keys, values, strict=True))
        return metrics

    def _clip_and_materialize(self, module, metrics):
        """One host synchronization for PPO metrics and the finite-gradient guard.

        Nonfinite gradients still abort before Adam/weight decay is applied.
        Clipping uses PyTorch's existing norm and coefficient calculation.
        """
        norm = nn.utils.clip_grad_norm_(module.parameters(), self.settings.gradient_norm_clip,
                                        error_if_nonfinite=False)
        values = self._materialize_metrics({**metrics, "_gradient_norm": norm.detach()})
        norm_value = values.pop("_gradient_norm")
        if not math.isfinite(norm_value):
            raise RuntimeError("The total norm of gradients is non-finite; optimizer step was not applied")
        return values, norm_value

    def _sequence_training_enabled(self) -> bool:
        return (self.settings.algorithm in ("ppo", "mappo") and self.settings.ppo_sequence_training
                and self.settings.model.dropout == 0.0)

    def _try_learner_graph(self, samples, *, critic):
        """Capture one bounded single-device backward pass.

        History/source buckets share one scratch pool. Small token buckets
        limit padding and leave room for the next rollout's full KV cache;
        oversized optimizer batches use weighted backward accumulation.
        """
        if (not self.settings.ppo_learner_cuda_graphs or self.device.type != 'cuda'
                or self.distributed.enabled or self.amp_dtype is not torch.bfloat16
                or not self._sequence_training_enabled()
                or not self.settings.model.ppo_varlen_attention
                or self.settings.model.activation_checkpointing):
            return None
        from .learner_graph import PackedLearnerGraph
        from .packed_attention import AVAILABLE
        if not AVAILABLE:
            return None
        name = '_critic_learner_graph' if critic else '_policy_learner_graph'
        graphs = getattr(self, name, None)
        if graphs is False:
            return None
        model = self.critic if critic else self.policy
        states = [s.value_state or s.state for s in samples] if critic else [s.state for s in samples]
        groups = history_prefix_groups(states)
        tokens = sum(len(states[g[0]].records) for g in groups)
        sequences = 32 if critic else getattr(self.settings, 'ppo_actor_graph_sequences', 32)
        token_limit = 4096 if critic else getattr(self.settings, 'ppo_actor_graph_tokens', 4096)
        required_tokens = tokens + sequences - len(groups)
        capacity = next((size for size in (1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384)
                         if required_tokens <= size <= token_limit), None)
        if capacity is None or len(groups) >= sequences:
            return None
        action_metadata = None
        source_capacity = 16384
        if not critic:
            action_metadata = PackedLearnerGraph.action_metadata(states)
            required_sources = len(action_metadata[3])
            source_capacity = next((size for size in (4096, 8192, 16384)
                                    if required_sources <= size), 16384)
        value_capacity = (4096 if len(samples) > 2048 else 2048 if len(samples) > 1024 else None) if critic else None
        policy_capacity = 1024 if not critic and len(samples) > 512 else None
        key = (capacity, source_capacity, value_capacity, policy_capacity, sequences)
        if graphs is None:
            graphs = {}
            setattr(self, name, graphs)
        graph = graphs.get(key)
        if graph is None:
            if not hasattr(self, '_learner_graph_pool'):
                self._learner_graph_pool = torch.cuda.graph_pool_handle()
            if not hasattr(self, '_learner_graph_gradient_caches'):
                self._learner_graph_gradient_caches = {}
            # Graphs never run concurrently. Losses/gradients are consumed by
            # Adam or copied into independent accumulators before the next
            # replay. Inputs and optimizer state live outside the scratch pool.
            graph = graphs[key] = PackedLearnerGraph(model, critic=critic, amp_dtype=self.amp_dtype,
                value_samples=value_capacity, policy_samples=policy_capacity,
                tokens=capacity, sequences=sequences, sources=source_capacity, pool=self._learner_graph_pool,
                gradient_cache=self._learner_graph_gradient_caches.setdefault(critic, {}))
        try:
            ready = graph.stage(model, samples,
                clip=self.settings.value_clip_epsilon if critic else self.settings.clip_epsilon,
                coefficient=self.settings.value_coefficient if critic else self.entropy_controllers['other'].coefficient,
                source_coefficient=0.0 if critic else self._source_entropy_coefficient(),
                opening_coefficient=(self.entropy_controllers['opening'].coefficient
                                     if not critic else None),
                opening_plies=self.entropy_opening_plies, action_metadata=action_metadata, history_groups=groups,
                policy_clip=None if critic else self._policy_clip())
            if not ready:
                return None
            result = graph.run(model)
            if critic:
                result['optimizer/critic_learner_graph_tokens'] = float(capacity)
                result['optimizer/critic_shared_gradient_buffers'] = 1.
            if not critic:
                result['_graph_tokens'] = float(capacity)
                result['optimizer/policy_shared_gradient_buffers'] = 1.
                result['optimizer/learner_source_capacity'] = float(source_capacity)
                result['optimizer/learner_source_padding_fraction'] = 1.0 - graph.used_sources / source_capacity
            return result
        except RuntimeError as error:
            if not is_out_of_memory(error, self.device.type):
                raise
            # Do not retry capture with increasingly many resident private pools.
            setattr(self, name, False)
            model.zero_grad(set_to_none=True)
            graphs.clear()
            del graph
            self._learner_graph_gradient_caches.pop(critic, None)
            empty_cache(self.device)
            self.logger.event(f"{'critic' if critic else 'policy'} learner CUDA graph exceeded memory; using packed learner")
            return None

    def _backward_learner_graphs(self, samples, *, critic):
        """Keep one optimizer batch while accumulating bounded graph backwards.

        A graph replay owns and overwrites its gradient buffers. Independent
        accumulators must hold weighted gradients before the next replay uses
        the shared scratch pool. Adam and gradient clipping still run once.
        """
        result = self._try_learner_graph(samples, critic=critic)
        if result is not None:
            result['optimizer/critic_graph_subbatches' if critic else 'optimizer/policy_graph_subbatches'] = 1.
            return result
        name = '_critic_learner_graph' if critic else '_policy_learner_graph'
        if (self.device.type != 'cuda' or not self.settings.ppo_learner_cuda_graphs
                or getattr(self, name, None) is False or self.distributed.enabled):
            return None
        states = [s.value_state or s.state for s in samples] if critic else [s.state for s in samples]
        groups = history_prefix_groups(states)
        parts, current, tokens, sequences = [], [], 0, 0
        for group in groups:
            length = len(states[group[0]].records)
            if current and (tokens + length > 4096 - 32 or sequences == 31):
                parts.append(current)
                current, tokens, sequences = [], 0, 0
            current.extend(samples[i] for i in group)
            tokens += length
            sequences += 1
        if current:
            parts.append(current)
        if len(parts) < 2:
            return None
        model = self.critic if critic else self.policy
        parameters = tuple(model.parameters())
        gradients, metrics, total_capacity = {}, {}, 0.
        for part in parts:
            output = self._try_learner_graph(part, critic=critic)
            if output is None:
                # No optimizer step has occurred; the complete ordinary batch
                # can be recomputed safely even after a capture/shape fallback.
                model.zero_grad(set_to_none=True)
                return None
            weight = len(part) / len(samples)
            total_capacity += output.pop('_graph_tokens', 0.)
            for key, value in output.items():
                metrics[key] = metrics.get(key, 0.) + value * weight
            existing, new = [], []
            for i, parameter in enumerate(parameters):
                if parameter.grad is not None:
                    (existing if i in gradients else new).append(i)
            if existing:
                torch._foreach_add_([gradients[i] for i in existing],
                                    [parameters[i].grad for i in existing], alpha=weight)
            if new:
                values = torch._foreach_mul([parameters[i].grad for i in new], weight)
                gradients.update(zip(new, values, strict=True))
        for i, parameter in enumerate(parameters):
            parameter.grad = gradients.get(i)
        if not critic:
            metrics['_graph_tokens'] = total_capacity
        metrics['optimizer/critic_graph_subbatches' if critic else 'optimizer/policy_graph_subbatches'] = float(len(parts))
        return metrics

    def _learner_chunks(self, samples):
        if self._sequence_training_enabled():
            return sequence_training_batches(
                samples, sequences_per_batch=self.effective_policy_microbatch,
                max_samples_per_sequence=self.settings.ppo_max_samples_per_sequence,
            )
        ordered = (sorted(samples, key=lambda item: len(item.state.records))
                   if self.settings.learner_length_bucketing else list(samples))
        return _chunks(ordered, self.effective_policy_microbatch)

    def _policy_clip(self):
        # Completed environment steps at the start of this update. Resume uses
        # the restored counter, never a stale log or the process lifetime.
        return scheduled_policy_clip(self.settings, self.cumulative["environment_plies"])

    def _source_entropy_coefficient(self) -> float:
        if self.settings.entropy_control_mode == "joint_source":
            return self.entropy_controllers["other"].coefficient
        return self.settings.source_entropy_coefficient

    def _sync_entropy_targets(self) -> dict[str, float]:
        targets = scheduled_entropy_targets(self.settings, self.cumulative["environment_plies"])
        for phase, controller in self.entropy_controllers.items():
            controller.target_ratio = targets[phase]
        return targets

    def _update_policy(
        self, groups: Sequence[PolicyGroup] | Sequence[PPOSample]
    ) -> dict[str, float]:
        # One target per complete learner update, held across minibatches/epochs.
        targets = self._sync_entropy_targets()
        if self.settings.algorithm in ("ppo", "mappo") and (
            len(groups) > self.settings.ppo_minibatch_samples or self.settings.lr_increase_on_full_epochs
            or self.settings.lr_kl_band_lower is not None or self.settings.lr_epoch_completion
        ):
            metrics = self._update_ppo_minibatches(groups, critic=False)
        else:
            metrics = self._update_policy_batch(groups)
        ratios = phase_entropy_ratios(metrics)
        source_coefficient = self._source_entropy_coefficient()
        shared_entropy = self.settings.entropy_control_mode == "joint_source"
        metrics.update({"policy/adaptive_entropy_enabled": float(self.settings.adaptive_entropy),
                        "policy/entropy_joint_source_control": float(shared_entropy),
                        "policy/entropy_feedback_source": float(self.settings.entropy_feedback == "source"),
                        "policy/source_entropy_coefficient": source_coefficient,
                        "policy/entropy_target_ratio": targets["other"],
                        "policy/opening_entropy_target_ratio": targets["opening"],
                        "policy/other_entropy_target_ratio": targets["other"],
                        "policy/entropy_target_schedule_enabled": float(bool(self.settings.entropy_target_schedule)),
                        "policy/entropy_target_schedule_environment_plies": float(self.cumulative["environment_plies"]),
                        "policy/entropy_opening_plies": float(self.entropy_opening_plies)})
        if self.settings.entropy_target_schedule:
            end, opening_final, other_final = self.settings.entropy_target_schedule[-1]
            metrics.update({"policy/entropy_target_schedule_end_environment_plies": float(end),
                            "policy/opening_entropy_target_final": opening_final,
                            "policy/other_entropy_target_final": other_final})
        if shared_entropy and metrics.get("optimizer/policy_steps", 0) > 0:
            # Reuse the final PPO probe when available; never feed epoch averages
            # or a tactical subset into the shared controller.
            if "policy/final_probe_samples" not in metrics:
                metrics.update(self._final_policy_diagnostics(groups))
            feedback = metrics.get("policy/final_source_entropy_ratio" if self.settings.entropy_feedback == "source"
                                   else "policy/final_entropy_ratio")
        else:
            feedback = None
        used_coefficients = {phase: controller.coefficient
                             for phase, controller in self.entropy_controllers.items()}
        if shared_entropy:
            self.entropy_controllers["other"].observe(feedback)
            metrics.update({"policy/entropy_deadband": self.settings.entropy_deadband,
                            "policy/entropy_ema_decay": self.settings.entropy_ema_decay,
                            "policy/entropy_adaptation_rate": self.settings.entropy_adaptation_rate,
                            "policy/entropy_max_relative_change": self.settings.entropy_max_relative_change or 0.,
                            "policy/effective_source_entropy_coefficient": 2 * source_coefficient,
                            "policy/effective_source_entropy_coefficient_next": 2 * self._source_entropy_coefficient()})
            if feedback is not None:
                metrics["policy/entropy_feedback_ratio"] = feedback
        metrics["policy/source_entropy_coefficient_next"] = self._source_entropy_coefficient()
        for phase, controller in self.entropy_controllers.items():
            prefix = "policy/opening_entropy" if phase == "opening" else "policy/entropy"
            metrics[f"{prefix}_coefficient"] = used_coefficients[phase]
            metrics[f"{prefix}_coefficient_min"] = controller.minimum
            metrics[f"{prefix}_coefficient_max"] = controller.maximum
            if not shared_entropy and metrics.get("optimizer/policy_steps", 0) > 0:
                controller.observe(ratios[phase])
            metrics[f"{prefix}_coefficient_next"] = controller.coefficient
            if controller.entropy_ema is not None:
                metrics[f"{prefix}_ratio_ema"] = controller.entropy_ema
        metrics["optimizer/adaptive_lr_enabled"] = float(self.lr_controller is not None)
        metrics.update({"policy/target_kl": self.settings.target_kl,
                         "policy/early_stop_kl_threshold": self._policy_kl_stop_threshold(),
                         "policy/early_stop_kl_probe_only": float(self.settings.lr_kl_band_lower is not None or self.settings.lr_epoch_completion),
                        "policy/early_stop_clip_fraction": self.settings.early_stop_clip_fraction,
                        "optimizer/policy_epochs_max": self.settings.policy_epochs})
        if self.lr_controller is not None:
            metrics.update(self._finish_learning_rate(groups, metrics))
        if self.settings.algorithm in ("ppo", "mappo"):
            metrics.update(self._policy_clip().metrics())
            metrics["policy/adaptive_clip_enabled"] = float(self.settings.ppo_adaptive_clip)
        return metrics

    def _final_policy_diagnostics(self, samples) -> dict[str, float]:
        started = time.perf_counter()
        key = (self.update, id(samples), len(samples))
        cached = getattr(self, '_kl_probe_cache', None)
        if cached is None or cached[0] != key:
            probe = diagnostic_sequences(samples, limit=self.settings.lr_probe_samples,
                sequence_length=self.settings.ppo_max_samples_per_sequence,
                seed=self.settings.seed + 1_000_003 * (self.update + 1) + self.distributed.rank)
            cached = self._kl_probe_cache = (key, probe, self._learner_chunks(probe))
        _, probe, chunks = cached
        moment_names = ("policy/approx_kl_old", "policy/clip_fraction", "policy/entropy",
                        "policy/opening_entropy_ratio_mass", "policy/opening_entropy_fraction",
                        "policy/other_entropy_ratio_mass", "policy/other_entropy_fraction")
        sum_names = ("kl", "clip", "entropy", "opening_ratio", "opening_count", "other_ratio", "other_count")
        source_feedback = self.settings.entropy_feedback == "source"
        if source_feedback:
            moment_names += ("policy/source_entropy_ratio_mass", "policy/source_entropy_fraction")
            sum_names += ("source_ratio", "source_count")
        totals = torch.zeros(len(moment_names), device=self.device)
        was_training = self.policy.training
        self.policy.eval()
        try:
            with torch.no_grad():
                # Retain the same bounded sequence/microbatch packing as the
                # learner. No reference network or full old logits are stored.
                for chunk in chunks:
                    with self._autocast():
                        output = policy_ppo_loss(self.policy, chunk,
                            clip_epsilon=self.settings.clip_epsilon, entropy_coefficient=0.,
                            opening_entropy_coefficient=0., entropy_opening_plies=self.entropy_opening_plies,
                            policy_clip=self._policy_clip(),
                            source_entropy_diagnostics=source_feedback,
                            sequence_training=self._sequence_training_enabled(), defer_metrics=True)
                    totals += torch.stack([output.metrics[key] for key in moment_names]) * len(chunk)
        finally:
            self.policy.train(was_training)
        values = totals.cpu().tolist()  # one transfer, scalar moments only
        sums = self.distributed.sum_metrics(dict(zip(sum_names, values, strict=True)) | {"count": len(probe)})
        count = sums["count"]
        result = {"policy/final_probe_samples": count,
                  "timing/lr_probe_seconds": self.distributed.reduce_float(
                      time.perf_counter() - started, operation="max")}
        if source_feedback:
            result["policy/final_source_entropy_samples"] = sums["source_count"]
            if sums["source_count"]:
                result["policy/final_source_entropy_ratio"] = sums["source_ratio"] / sums["source_count"]
        if count:
            result.update({"policy/final_kl": sums["kl"] / count,
                           "policy/final_clip_fraction": sums["clip"] / count,
                           "policy/final_entropy": sums["entropy"] / count})
            eligible = sums["opening_count"] + sums["other_count"]
            if eligible:
                result["policy/final_entropy_ratio"] = (sums["opening_ratio"] + sums["other_ratio"]) / eligible
            for phase in ("opening", "other"):
                if sums[f"{phase}_count"]:
                    result[f"policy/final_{phase}_entropy_ratio"] = sums[f"{phase}_ratio"] / sums[f"{phase}_count"]
        return result

    def _finish_learning_rate(self, samples, metrics) -> dict:
        controller = self.lr_controller
        diagnostics = ({key: value for key, value in metrics.items()
                        if key.startswith("policy/final_") or key == "timing/lr_probe_seconds"}
                       if "policy/final_probe_samples" in metrics else
                       self._final_policy_diagnostics(samples)
                       if metrics.get("optimizer/policy_steps", 0) > 0 else
                       {"policy/final_probe_samples": 0., "timing/lr_probe_seconds": 0.})
        next_base = self._scheduled_learning_rate(self.update + 2)
        completion_mode = self.settings.lr_increase_on_full_epochs
        epoch_completion = self.settings.lr_epoch_completion
        band_mode = self.settings.lr_kl_band_lower is not None
        kl_feedback = diagnostics.get("policy/final_kl")
        if completion_mode and kl_feedback is not None:
            kl_feedback = max(kl_feedback, metrics.get("policy/minibatch_kl_max", 0.),
                              metrics.get("policy/epoch_kl_max", 0.))
        next_lr, direction = controller.observe(scheduled=next_base,
            kl=kl_feedback,
            # A clip-only early stop is not a learning-rate feedback signal.
            early_stopped=bool(metrics.get("optimizer/policy_stop_kl", 0)),
            allow_increase=epoch_completion or completion_mode or band_mode or self.update + 1 > self.settings.warmup_updates,
            full_epochs_completed=(metrics.get("optimizer/policy_epochs", 0) >= self.settings.policy_epochs
                and metrics.get("optimizer/policy_first_pass_coverage", 0) == 1.
                if completion_mode or epoch_completion else None),
            completed_epochs=metrics.get("optimizer/policy_epochs", 0.))
        self.policy_lr_scale = controller.scale
        diagnostics.update({"optimizer/policy_lr_next": next_lr,
            "optimizer/policy_lr_min": self.settings.minimum_learning_rate,
            "optimizer/policy_lr_max": self.settings.maximum_learning_rate,
            "optimizer/policy_lr_ceiling_next": min(self.settings.maximum_learning_rate, next_base),
            "optimizer/policy_lr_scale": controller.scale,
            "optimizer/lr_direction": direction,
            "optimizer/lr_reason": controller.reason,
            "optimizer/lr_stable_count": controller.stable_count,
            "optimizer/lr_stable_direction": controller.stable_direction,
            "optimizer/lr_stable_required": controller.stable_updates,
            "optimizer/lr_cooldown": controller.cooldown,
            "optimizer/lr_scale_reduced": float(direction < 0),
            "optimizer/lr_scale_increased": float(direction > 0),
            "optimizer/critic_lr": self.critic_optimizer.param_groups[0]["lr"],
            "optimizer/lr_feedback_kl_only": 1.,
            "policy/lr_kl_feedback": kl_feedback or 0.,
            "optimizer/lr_increase_on_full_epochs": float(completion_mode),
            "optimizer/lr_kl_band_enabled": float(band_mode),
            "optimizer/lr_epoch_completion": float(epoch_completion),
            "optimizer/lr_minimum_completed_epochs": self.settings.lr_minimum_completed_epochs,
            "optimizer/lr_increase_kl_threshold": self.settings.lr_increase_kl_threshold,
            "optimizer/lr_decrease_kl_threshold": self.settings.lr_decrease_kl_threshold,
            "policy/kl_probe_interval_steps": self.settings.ppo_kl_probe_interval_steps,
            "optimizer/lr_increase_factor": self.settings.lr_increase_factor,
            "optimizer/lr_decrease_factor": self.settings.lr_decrease_factor,
        })
        if band_mode:
            diagnostics.update({"optimizer/lr_kl_band_lower": self.settings.lr_kl_band_lower,
                "optimizer/lr_kl_band_upper": self.settings.lr_kl_band_upper,
                "optimizer/lr_increase_factor": self.settings.lr_increase_factor,
                "optimizer/lr_decrease_factor": self.settings.lr_decrease_factor,
                "optimizer/lr_cooldown_updates": self.settings.lr_cooldown_updates})
        if controller.kl_ema is not None:
            diagnostics["policy/final_kl_ema"] = controller.kl_ema
        return diagnostics

    def _policy_kl_stop_threshold(self) -> float:
        explicit = getattr(self.settings, "early_stop_kl_threshold", None)
        if explicit is not None:
            return explicit
        upper = getattr(self.settings, "lr_kl_band_upper", None)
        return upper if upper is not None else self.settings.target_kl * self.settings.early_stop_kl_multiple

    def _ppo_optimizer_batches(self, samples, *, minibatch_samples: int | None = None,
                               full_histories: bool = False):
        batch_size = minibatch_samples or self.settings.ppo_minibatch_samples
        if self._sequence_training_enabled():
            chains = sequence_training_batches(
                samples, sequences_per_batch=1,
                # Value heads can gather many labels from one complete causal
                # history. Shuffling whole bounded histories avoids repeatedly
                # encoding their shared prefix in different optimizer steps.
                # Actor ordering retains its configured trajectory chunk size.
                max_samples_per_sequence=(batch_size if full_histories else
                                          self.settings.ppo_max_samples_per_sequence),
            )
            random.shuffle(chains)
            ordered = [sample for chain in chains for sample in chain]
        else:
            ordered = list(samples)
            random.shuffle(ordered)
        # Fixed sample counts keep DDP optimizer/collective counts equal even
        # when ranks have different game lengths or numbers of prefix chains.
        return _chunks(ordered, batch_size)

    def _update_ppo_minibatches(self, samples, *, critic: bool,
                              minibatch_samples: int | None = None) -> dict[str, float]:
        epochs = self.settings.critic_epochs if critic else self.settings.policy_epochs
        aggregate: dict[str, float] = {}
        processed = steps = 0
        eligible_total = sum(getattr(sample, "learnable", True) for sample in samples)
        eligible_processed = 0
        stop = False
        max_kl = max_clip = 0.
        stop_kl = stop_clip = False
        epoch_diagnostics: dict[str, float] = {}
        final_diagnostics: dict[str, float] = {}
        probe_seconds = max_epoch_kl = 0.
        intermediate_probes = 0
        band_mode = getattr(self.settings, "lr_kl_band_lower", None) is not None
        epoch_completion = getattr(self.settings, 'lr_epoch_completion', False)
        probe_interval = getattr(self.settings, 'ppo_kl_probe_interval_steps', 0) if epoch_completion else 0
        kl_stop_threshold = self._policy_kl_stop_threshold()
        probe_epochs = not critic and (getattr(self.settings, "lr_increase_on_full_epochs", False) or band_mode or epoch_completion)
        for epoch in range(epochs):
            last_probe_step = -1
            epoch_start = processed
            epoch_steps = steps
            entropy_sum = 0.
            batch_options = ({'minibatch_samples': minibatch_samples} if minibatch_samples is not None else {})
            if critic and getattr(self.settings, 'ppo_critic_full_history_batches', False):
                batch_options['full_histories'] = True
            batches = self._ppo_optimizer_batches(samples, **batch_options)
            for batch in batches:
                metrics = (self._update_critic_batch(batch, epochs=1) if critic
                           else self._update_policy_batch(batch, epochs=1))
                for key, value in metrics.items():
                    aggregate[key] = aggregate.get(key, 0.) + value * len(batch)
                processed += len(batch)
                entropy_sum += metrics.get("policy/entropy", 0.) * len(batch)
                eligible_processed += sum(getattr(sample, "learnable", True) for sample in batch)
                steps += int(metrics.get("optimizer/critic_steps" if critic else "optimizer/policy_steps", 1))
                if not critic:
                    max_kl = max(max_kl, metrics["policy/approx_kl_old"])
                    max_clip = max(max_clip, metrics["policy/clip_fraction"])
                    if not band_mode and not epoch_completion:
                        stop_kl |= metrics["policy/approx_kl_old"] > kl_stop_threshold
                    stop_clip |= metrics["policy/clip_fraction"] > self.settings.early_stop_clip_fraction
                if not critic and (
                    stop_kl or stop_clip
                ) and processed >= self.settings.ppo_min_policy_epochs * len(samples):
                    stop = True
                    break
                if (not critic and probe_interval and epoch > 0
                        and steps > epoch_steps and steps != last_probe_step
                        and (steps - epoch_steps) % probe_interval == 0
                        and processed - epoch_start < len(samples)):
                    final_diagnostics = self._final_policy_diagnostics(samples)
                    last_probe_step = steps
                    intermediate_probes += 1
                    probe_seconds += final_diagnostics['timing/lr_probe_seconds']
                    probe_kl = final_diagnostics.get('policy/final_kl', 0.)
                    max_epoch_kl = max(max_epoch_kl, probe_kl)
                    if probe_kl > kl_stop_threshold:
                        stop = stop_kl = True
                        break
            if probe_epochs:
                # The deterministic sample selection is identical within this
                # update, so epoch endpoints compare the same observations.
                if last_probe_step != steps:
                    final_diagnostics = self._final_policy_diagnostics(samples)
                    probe_seconds += final_diagnostics["timing/lr_probe_seconds"]
                epoch_kl = final_diagnostics.get("policy/final_kl", 0.)
                max_epoch_kl = max(max_epoch_kl, epoch_kl)
                stop_kl |= epoch_kl > kl_stop_threshold
                stop_clip |= final_diagnostics.get("policy/final_clip_fraction", 0.) > self.settings.early_stop_clip_fraction
                if (stop_kl or stop_clip) and processed >= self.settings.ppo_min_policy_epochs * len(samples):
                    stop = True
                prefix = f"policy/epoch_{epoch + 1}/"
                epoch_diagnostics.update({prefix + key.removeprefix("policy/"): value
                    for key, value in final_diagnostics.items() if key.startswith("policy/")})
                epoch_diagnostics.update({prefix + "coverage": (processed - epoch_start) / len(samples),
                    prefix + "optimizer_steps": float(steps - epoch_steps),
                    prefix + "training_entropy": entropy_sum / max(1, processed - epoch_start)})
                self.logger.event(f"update={self.update + 1} ppo_epoch={epoch + 1}/{epochs} "
                    f"coverage={(processed - epoch_start) / len(samples):.6f} "
                    f"KL={epoch_kl:.6f} entropy={final_diagnostics.get('policy/final_entropy', 0.):.6f} "
                    f"clip={final_diagnostics.get('policy/final_clip_fraction', 0.):.6f} stop_kl={int(stop_kl)}")
            if stop:
                break
        aggregate = {key: value / processed for key, value in aggregate.items()}
        name = "critic" if critic else "policy"
        aggregate[f"optimizer/{name}_epochs"] = processed / len(samples)
        aggregate[f"optimizer/{name}_steps"] = float(steps)
        if critic:
            aggregate['optimizer/critic_full_history_batches'] = float(
                getattr(self.settings, 'ppo_critic_full_history_batches', False))
        if not critic:
            aggregate.update({"policy/minibatch_kl_max": max_kl,
                "policy/minibatch_clip_max": max_clip,
                "optimizer/policy_early_stopped": float(stop and processed < epochs * len(samples)),
                "optimizer/policy_first_pass_samples": float(min(eligible_processed, eligible_total)),
                "optimizer/policy_first_pass_coverage": min(1., eligible_processed / max(1, eligible_total)),
                "optimizer/policy_minimum_epochs": float(self.settings.ppo_min_policy_epochs),
                "optimizer/policy_stop_kl": float(stop_kl),
                "optimizer/policy_stop_clip": float(stop_clip)})
            if probe_epochs:
                aggregate.update(epoch_diagnostics)
                aggregate.update(final_diagnostics)
                aggregate["policy/epoch_kl_max"] = max_epoch_kl
                aggregate["timing/lr_probe_seconds"] = probe_seconds
                aggregate['policy/intermediate_kl_probes'] = intermediate_probes
        self._kl_probe_cache = None
        return aggregate

    def _update_policy_batch(
        self, groups: Sequence[PolicyGroup] | Sequence[PPOSample], *, epochs: int | None = None,
    ) -> dict[str, float]:
        if not self.policy.training:
            self.policy.train()
        coalesce = self.settings.algorithm in ("ppo", "mappo") and not self.distributed.enabled
        aggregate: dict[str, float] = {}
        epochs_completed = 0
        grad_norm_value = 0.0
        for _epoch in range(self.settings.policy_epochs if epochs is None else epochs):
            while True:
                try:
                    epoch_metrics = self._backward_policy_epoch(groups, defer_metrics=coalesce)
                    break
                except RuntimeError as error:
                    if not is_out_of_memory(error, self.device.type):
                        raise
                    self.policy_optimizer.zero_grad(set_to_none=True)
                    empty_cache(self.device)
                    current = self.effective_policy_microbatch
                    minimum = self.settings.minimum_policy_microbatch
                    if self.distributed.enabled:
                        self.logger.event(
                            "distributed learner OOM; aborting this launch to "
                            "avoid asymmetric DDP retry. Relaunch every rank with "
                            "a smaller --microbatch"
                        )
                        raise
                    if (
                        not self.settings.auto_reduce_microbatch_on_oom
                        or current <= minimum
                    ):
                        self.logger.event(
                            "%s OOM with policy microbatch=%d; automatic "
                            "fallback unavailable"
                            % (self.device.type.upper(), current)
                        )
                        raise
                    reduced = max(minimum, current // 2)
                    self.effective_policy_microbatch = reduced
                    self.logger.event(
                        "%s OOM with policy microbatch=%d; retrying the "
                        "unapplied epoch with microbatch=%d"
                        % (self.device.type.upper(), current, reduced)
                    )
            # Include entropy moments in one collective, avoiding one extra
            # synchronization per phase statistic on multi-rank runs.
            if not coalesce:
                epoch_metrics = {key: value / self.distributed.world_size for key, value in
                                 self.distributed.sum_metrics(epoch_metrics).items()}
            if epoch_metrics.get("optimizer/no_policy_samples"):
                # Even AdamW weight decay must not update parameters when no
                # rank has a learner-owned action in this optimizer batch.
                self.policy_optimizer.zero_grad(set_to_none=True)
                epoch_metrics = self._materialize_metrics(epoch_metrics)
                return {**epoch_metrics, "optimizer/policy_epochs": 0.,
                        "optimizer/policy_steps": 0., "optimizer/policy_grad_norm": 0.}
            self.grad_scaler.unscale_(self.policy_optimizer)
            if coalesce:
                epoch_metrics, grad_norm_value = self._clip_and_materialize(self.policy, epoch_metrics)
            else:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.settings.gradient_norm_clip,
                    error_if_nonfinite=self.settings.algorithm in ("ppo", "mappo"),
                )
                grad_norm_value = self.distributed.reduce_float(float(grad_norm), operation="mean")
            audit = getattr(self, "training_audit", None)
            hook = (self.policy_optimizer.register_step_post_hook(lambda *_: audit.commit_step(groups))
                    if audit is not None else None)
            try:
                self.grad_scaler.step(self.policy_optimizer)
            finally:
                if hook is not None:
                    hook.remove()
            self.grad_scaler.update()
            # Gradients are not part of optimizer/checkpoint state.  Keeping
            # them alive through the next rollout pins allocator segments and
            # prevents the retired paged-KV arena from being returned at the
            # following update boundary.
            self.policy_optimizer.zero_grad(set_to_none=True)
            epochs_completed += 1
            aggregate = epoch_metrics
            if (
                (self.settings.lr_kl_band_lower is None and
                 epoch_metrics.get("policy/approx_kl_old", epoch_metrics.get("policy/kl_reference", 0.0))
                 > self._policy_kl_stop_threshold())
                or epoch_metrics["policy/clip_fraction"]
                > self.settings.early_stop_clip_fraction
            ):
                break
        aggregate["optimizer/policy_epochs"] = float(epochs_completed)
        aggregate["optimizer/policy_steps"] = float(epochs_completed)
        aggregate["optimizer/policy_grad_norm"] = grad_norm_value
        if self.settings.algorithm in ("ppo", "mappo"):
            kl = aggregate.get("policy/approx_kl_old", 0.)
            clipped = aggregate.get("policy/clip_fraction", 0.)
            stop_for_kl = self.settings.lr_kl_band_lower is None and kl > self._policy_kl_stop_threshold()
            aggregate.update({"policy/minibatch_kl_max": kl, "policy/minibatch_clip_max": clipped,
                "optimizer/policy_stop_kl": float(stop_for_kl),
                "optimizer/policy_stop_clip": float(clipped > self.settings.early_stop_clip_fraction),
                "optimizer/policy_early_stopped": float(
                    stop_for_kl
                    or clipped > self.settings.early_stop_clip_fraction)})
        if self.settings.algorithm not in ("ppo", "mappo") and aggregate.get("policy/kl_reference", 0.0) > 2 * self.settings.target_kl:
            self.policy_lr_scale = max(0.1, self.policy_lr_scale * 0.5)
            aggregate["optimizer/lr_scale_reduced"] = 1.0
        return aggregate

    def _normalize_ppo_batch(self, samples: Sequence[PPOSample]) -> list[PPOSample]:
        # Reduce moments globally: averaging per-microbatch/rank EV is incorrect.
        sums = self.distributed.sum_metrics(ppo_signal_sums([s for s in samples if s.learnable]))
        self.ppo_rollout_metrics = (ppo_signal_metrics(sums) if sums["count"] else {
            "ppo/rollout_samples": 0., "ppo/no_advantage_signal": 1.,
            "ppo/raw_advantage_mean": 0., "ppo/raw_advantage_std": 0.,
        })
        pass_totals = self.distributed.sum_metrics({
            "ppo/voluntary_passes": sum(s.action == (0, 0) for s in samples),
            "ppo/pass_decisions": len(samples),
        })
        self.ppo_rollout_metrics.update(pass_totals)
        if self.settings.algorithm == "mappo" and self.settings.mappo_advantage_mode == "player_decisions":
            for player in range(mode_spec(self.settings.mode).player_count):
                advantages = [s.advantage for s in samples if s.learnable and s.value_player == player]
                moments = self.distributed.sum_metrics({"count": len(advantages),
                    "sum": math.fsum(advantages), "square_sum": math.fsum(a*a for a in advantages)})
                count = moments["count"]
                mean = moments["sum"] / max(1, count)
                prefix = f"ppo/player_{player}/"
                self.ppo_rollout_metrics.update({prefix + "samples": count,
                    prefix + "raw_advantage_mean": mean,
                    prefix + "raw_advantage_std": math.sqrt(max(0., moments["square_sum"] / max(1,count) - mean*mean))})
        self.ppo_rollout_metrics["ppo/pass_fraction"] = (
            pass_totals["ppo/voluntary_passes"] / max(1, pass_totals["ppo/pass_decisions"]))
        return normalize_advantages(
            samples, mean=self.ppo_rollout_metrics["ppo/raw_advantage_mean"],
            std=self.ppo_rollout_metrics["ppo/raw_advantage_std"],
            epsilon=self.settings.advantage_epsilon,
        )

    @staticmethod
    def _critic_graph_padding_fraction(samples) -> tuple[float, int]:
        """Estimate the extra temporal work in bounded graph accumulation.

        Long full windows fill 4096-token graphs efficiently. Medium histories
        can leave a large dummy tail in every part; the exact packed learner
        then avoids both padding and repeated gradient accumulation.
        """
        states = [sample.state for sample in samples]
        groups = history_prefix_groups(states)
        real = padded = tokens = sequences = parts = 0
        for group in groups:
            length = len(states[group[0]].records)
            if sequences and (tokens + length > 4096 - 32 or sequences == 31):
                padded += next(size for size in (1024, 2048, 3072, 4096)
                               if tokens + 32 - sequences <= size)
                parts += 1
                tokens = sequences = 0
            tokens += length
            real += length
            sequences += 1
        if sequences:
            capacity = next((size for size in (1024, 2048, 3072, 4096)
                             if tokens + 32 - sequences <= size), None)
            if capacity is None:
                return 1., 2
            padded += capacity
            parts += 1
        return 1. - real / max(1, padded), parts

    def _backward_critic_epoch(self, samples: Sequence[PPOSample], *, defer_metrics: bool = False) -> dict[str, float]:
        if self.critic_parallel is None or self.critic_optimizer is None:
            raise RuntimeError("PPO critic was not initialized")
        self.critic_optimizer.zero_grad(set_to_none=True)
        value_samples = [replace(s, state=s.value_state, value_state=None) if s.value_state is not None else s for s in samples]
        padding, parts = (self._critic_graph_padding_fraction(value_samples)
                   if self.settings.ppo_learner_cuda_graphs and self.device.type == 'cuda'
                   and self._sequence_training_enabled() else (0., 0))
        # Many short graph replays also lose to one variable-length batch even
        # when padding is low. Keep graphs for at most four bounded parts;
        # every value label and complete history still contributes to one Adam
        # batch, with unchanged sample weights and optimizer step counts.
        captured = (self._backward_learner_graphs(value_samples, critic=True)
                    if parts <= 1 or (parts <= 4 and padding <= .125) else None)
        if captured is not None:
            captured['optimizer/critic_learner_graph_fraction'] = 1.0
            captured['optimizer/critic_graph_planned_padding_fraction'] = padding
            captured['optimizer/critic_graph_planned_subbatches'] = float(parts)
            return captured if defer_metrics else self._materialize_metrics(captured)
        metrics: dict[str, float] = {}
        metrics['optimizer/critic_learner_graph_fraction'] = 0.0
        metrics['optimizer/critic_graph_planned_padding_fraction'] = padding
        metrics['optimizer/critic_graph_planned_subbatches'] = float(parts)
        chunks = self._learner_chunks(value_samples)
        for index, chunk in enumerate(chunks):
            synchronization = (
                self.critic_parallel.no_sync()
                if self.distributed.enabled and index < len(chunks) - 1
                else nullcontext()
            )
            with synchronization:
                with self._autocast():
                    output = critic_ppo_loss(
                        self.critic_parallel, chunk,
                        clip_epsilon=self.settings.value_clip_epsilon,
                        value_coefficient=self.settings.value_coefficient,
                        sequence_training=self._sequence_training_enabled(),
                        defer_metrics=True,
                    )
                    loss = output.loss * len(chunk) / len(samples)
                self.critic_grad_scaler.scale(loss).backward()
            for key, value in output.metrics.items():
                metrics[key] = metrics.get(key, 0.0) + value * len(chunk) / len(samples)
        return metrics if defer_metrics else self._materialize_metrics(metrics)

    def _update_critic(self, samples: Sequence[PPOSample]) -> dict[str, float]:
        minibatch = self.settings.ppo_minibatch_samples
        diagnostics = {}
        if self.settings.algorithm == "mappo":
            transitions = len(samples)
            samples = critic_training_samples(samples)
            # One label for every represented player/team, with the same Adam-step budget.
            # Prefix packing shares the histories among adjacent value targets.
            perspectives = len(samples) // transitions
            if len(samples) != perspectives * transitions or perspectives not in (2, 4):
                raise ValueError("inconsistent MAPPO critic perspective coverage")
            minibatch *= perspectives
            moments = self.distributed.sum_metrics(ppo_signal_sums(samples))
            prefix = "critic/all_players_" if samples[0].value_player is not None else "critic/both_teams_"
            diagnostics.update({prefix + key.removeprefix("critic/rollout_"): value
                for key, value in ppo_signal_metrics(moments).items()
                if key.startswith("critic/rollout_")})
            diagnostics.update({"optimizer/critic_value_samples": float(len(samples)),
                "optimizer/critic_auxiliary_value_samples": float(len(samples) - transitions),
                "optimizer/critic_perspectives_per_transition": float(perspectives),
                "optimizer/critic_minibatch_value_samples": float(minibatch)})
            if prefix == "critic/both_teams_":
                diagnostics["optimizer/critic_opposing_value_samples"] = float(transitions)
        metrics = (self._update_ppo_minibatches(samples, critic=True, minibatch_samples=minibatch)
                   if len(samples) > minibatch else self._update_critic_batch(samples))
        return {**metrics, **diagnostics}

    def _update_critic_batch(self, samples: Sequence[PPOSample], *, epochs: int | None = None) -> dict[str, float]:
        if self.critic is None or self.critic_optimizer is None:
            raise RuntimeError("PPO critic was not initialized")
        if not self.critic.training:
            self.critic.train()
        coalesce = not self.distributed.enabled
        metrics: dict[str, float] = {}
        applied_epochs = self.settings.critic_epochs if epochs is None else epochs
        for _epoch in range(applied_epochs):
            while True:
                try:
                    metrics = self._backward_critic_epoch(samples, defer_metrics=coalesce)
                    break
                except RuntimeError as error:
                    if not is_out_of_memory(error, self.device.type):
                        raise
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    empty_cache(self.device)
                    if (self.distributed.enabled
                            or not self.settings.auto_reduce_microbatch_on_oom
                            or self.effective_policy_microbatch <= self.settings.minimum_policy_microbatch):
                        raise
                    self.effective_policy_microbatch = max(
                        self.settings.minimum_policy_microbatch,
                        self.effective_policy_microbatch // 2,
                    )
                    self.logger.event(
                        f"critic OOM; retrying unapplied critic epoch with microbatch="
                        f"{self.effective_policy_microbatch}"
                    )
            self.critic_grad_scaler.unscale_(self.critic_optimizer)
            if coalesce:
                metrics, norm = self._clip_and_materialize(self.critic, metrics)
            else:
                norm = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.settings.gradient_norm_clip,
                    error_if_nonfinite=True,
                )
            self.critic_grad_scaler.step(self.critic_optimizer)
            self.critic_grad_scaler.update()
            self.critic_optimizer.zero_grad(set_to_none=True)
            metrics = self.distributed.mean_metrics(metrics)
            metrics["optimizer/critic_grad_norm"] = self.distributed.reduce_float(float(norm))
        metrics["optimizer/critic_epochs"] = float(applied_epochs)
        metrics["optimizer/critic_steps"] = float(applied_epochs)
        return metrics

    def _layout_minimum_version(self) -> int:
        return max((int((getattr(self, name, None) or {}).get("update", 0)) for name in
                    ("draw_rule_migration", "draw_objective_migration", "pass_rule_migration", "no_move_rule_migration",
                     "two_player_no_pass_migration")), default=0)

    def _update_layout(self) -> dict[str, float]:
        entropy_coefficient = (self.settings.entropy_coefficient
                               if self.settings.layout_entropy_coefficient is None
                               else self.settings.layout_entropy_coefficient)
        self.layout_buffer.advance(self.update + 1, minimum_version=self._layout_minimum_version())
        if len(self.layout_buffer) < 2:
            return {"layout/update_skipped_buffer_size": float(len(self.layout_buffer)),
                    "optimizer/layout_steps": 0., **self.layout_buffer.metrics()}
        outcomes = self.layout_buffer.take(self.settings.layout_outcomes_per_update)
        batch = prepare_layout_batch(outcomes, device=self.device, advantage_epsilon=self.settings.advantage_epsilon)
        self.layout.train()
        self.layout_optimizer.zero_grad(set_to_none=True)
        aggregate: dict[str, torch.Tensor] = {}
        for start in range(0, len(outcomes), self.settings.layout_microbatch_size):
            stop = min(start + self.settings.layout_microbatch_size, len(outcomes))
            weight = (stop - start) / len(outcomes)
            with self._autocast():
                output = layout_grpo_loss(self.layout, self.reference_layout, (),
                    clip_epsilon=self.settings.clip_epsilon, kl_coefficient=self.settings.kl_coefficient,
                    entropy_coefficient=entropy_coefficient,
                    advantage_epsilon=self.settings.advantage_epsilon,
                    prepared=batch.slice(start, stop), tensor_metrics=True)
            self.layout_grad_scaler.scale(output.loss * weight).backward()
            for key, value in output.metrics.items():
                aggregate[key] = aggregate.get(key, 0.) + value.detach() * weight
        self.layout_grad_scaler.unscale_(self.layout_optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.layout.parameters(), self.settings.gradient_norm_clip
        )
        self.layout_grad_scaler.step(self.layout_optimizer)
        self.layout_grad_scaler.update()
        self.pool.invalidate_layout_queue()
        aggregate["optimizer/layout_grad_norm"] = grad_norm.detach()
        metrics = self._materialize_metrics(aggregate)
        self.layout_optimizer.zero_grad(set_to_none=True)
        metrics.update(self.layout_buffer.metrics(outcomes))
        metrics["optimizer/layout_steps"] = 1.
        metrics["optimizer/layout_samples"] = len(outcomes)
        metrics["layout/entropy_coefficient"] = entropy_coefficient
        return metrics

    def _local_rank_state(self) -> dict[str, Any]:
        return {
            "rank": self.distributed.rank,
            "rng_state": capture_rng_state(self.device),
            "base_game_pool": self.pool.state_dict(),
        }

    def _trainer_state(
        self, rank_states: Sequence[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "model_selection": self.model_selection.state_dict(),
            "last_checkpoint_environment_plies": self.cumulative["environment_plies"],
            "draw_rule_migration": self.draw_rule_migration,
            "context_window_migration": self.context_window_migration,
            "pass_rule_migration": self.pass_rule_migration,
            "no_move_rule_migration": self.no_move_rule_migration,
            "two_player_no_pass_migration": self.two_player_no_pass_migration,
            "draw_objective_migration": self.draw_objective_migration,
            "flag_capture_objective_migration": self.flag_capture_objective_migration,
            "futile_attack_objective_migration": self.futile_attack_objective_migration,
            "bomb_objective_migration": self.bomb_objective_migration,
            "flag_defense_objective_migration": self.flag_defense_objective_migration,
            "known_nonflag_hq_objective_migration": self.known_nonflag_hq_objective_migration,
            "player_advantage_migration": self.player_advantage_migration,
            "futile_attack_penalty": self.futile_attack_controller.state_dict(),
            "policy_lr_scale": self.policy_lr_scale,
            "adaptive_learning_rate": (self.lr_controller.state_dict()
                                       if self.lr_controller is not None else None),
            "adaptive_entropy": {phase: controller.state_dict()
                                 for phase, controller in self.entropy_controllers.items()},
            "effective_policy_microbatch": self.effective_policy_microbatch,
            "effective_actor_inference_batch": (
                self.effective_actor_inference_batch
            ),
            "grad_scaler": self.grad_scaler.state_dict(),
            "layout_grad_scaler": self.layout_grad_scaler.state_dict(),
            "critic_grad_scaler": self.critic_grad_scaler.state_dict(),
            "cumulative": dict(self.cumulative),
            "environment_step_definition": "base_and_simulated_branch_transitions_v1",
            "layout_buffer": self.layout_buffer.records(),
            "layout_buffer_totals": dict(self.layout_buffer.totals),
        }
        if self.distributed.enabled:
            if rank_states is None or len(rank_states) != self.distributed.world_size:
                raise ValueError("all rank states are required for distributed checkpoint")
            state["distributed"] = {
                "world_size": self.distributed.world_size,
                "rank_states": list(rank_states),
            }
        else:
            state["base_game_pool"] = self.pool.state_dict()
        return state

    def save_checkpoint(self, *, reason: str, archive: bool) -> Path:
        # Automatic saves in evaluation mode only reach here at a match. A
        # manual save must also persist its initial opponent for exact resume.
        self.model_selection.persist_baseline()
        inference_state = self.historical.archive_checkpoint(self.policy, self.layout,
            environment_plies=self.cumulative["environment_plies"], update=self.update)
        gathered = self.distributed.gather_object(self._local_rank_state())
        path = self.checkpoints.latest_path
        if self.distributed.primary:
            rank_states = (
                None
                if not self.distributed.enabled
                else [dict(item) for item in gathered or []]
            )
            path = self.checkpoints.save(
                update=self.update,
                mode=self.settings.mode.value,
                dead_rules_enabled=self.settings.dead_rules_enabled,
                policy=self.policy,
                layout=self.layout,
                reference_policy=self.reference_policy,
                reference_layout=self.reference_layout,
                policy_optimizer=self.policy_optimizer,
                layout_optimizer=self.layout_optimizer,
                trainer_state=self._trainer_state(rank_states),
                config=self.settings.serializable(),
                archive=archive,
                reason=reason,
                rng_state=capture_rng_state(self.device),
                algorithm=self.settings.algorithm,
                critic=self.critic,
                critic_optimizer=self.critic_optimizer,
                inference_state=inference_state,
            )
            self.logger.event(f"checkpoint saved: {path} reason={reason}")
        self.distributed.barrier()
        self.last_checkpoint_environment_plies = self.cumulative["environment_plies"]
        return path

    def _periodic_checkpoint_due(self) -> bool:
        if self.settings.checkpoint_policy != "periodic":
            return False
        interval = self.settings.checkpoint_interval_environment_plies
        if interval is not None:
            return (self.cumulative["environment_plies"] // interval
                    > self.last_checkpoint_environment_plies // interval)
        return self.update % self.settings.checkpoint_every_updates == 0

    def _training_budget_reached(self) -> bool:
        target = self.settings.step_budget_target
        return (
            target is not None
            and self.cumulative[self.settings.step_budget_counter] >= target
        )

    def _maybe_evaluate_model(self) -> dict[str, Any] | None:
        milestone = self.model_selection.due_milestone(
            update=self.update, cumulative=self.cumulative,
        )
        if milestone is None or self.distributed.any(self.stop_requested):
            return None
        # Evaluations allocate their own policies/caches. These training-only
        # graphs can be rebuilt after the infrequent match without retaining
        # several GiB of otherwise idle scratch memory throughout evaluation.
        self.policy.zero_grad(set_to_none=True)
        if self.critic is not None:
            self.critic.zero_grad(set_to_none=True)
        for name in ('_policy_learner_graph', '_critic_learner_graph'):
            if getattr(self, name, None) is not False:
                setattr(self, name, None)
        getattr(self, '_learner_graph_gradient_caches', {}).clear()
        empty_cache(self.device)
        # Make the completed update durable before a possibly long match. If
        # interrupted, resume this candidate and retry the uncommitted round.
        self.phase = "model_selection_checkpoint"
        self.save_checkpoint(reason="before_model_selection", archive=False)
        self.phase = "model_selection"
        historical_only = self.model_selection.historical_only(self.cumulative)
        observational = historical_only or self.settings.arena_observational_only
        if historical_only:
            panel_games = (f"games_total={self.settings.historical_eval_total_games}"
                           if self.settings.historical_eval_total_games is not None
                           else f"games_per_opponent={self.settings.historical_eval_games}")
            self.logger.event(
                f"historical panel {self.model_selection.milestone_label(milestone)}: update={self.update} "
                f"opponents={len(self.historical.evaluation_panel())} {panel_games} "
                f"parallel_games_per_rank={self.settings.arena_parallel_games}; champion evaluation disabled, use latest model"
            )
        elif observational:
            self.logger.event(
                f"fixed reference evaluation {self.model_selection.milestone_label(milestone)}: "
                f"update={self.update} reference_update={self.model_selection.state['best_update']} "
                f"games={self.settings.arena_games} "
                f"historical_teammate_fraction={self.settings.arena_historical_teammate_fraction}; use latest model"
            )
        else:
            self.logger.event(
                f"model selection {self.model_selection.milestone_label(milestone)}: update={self.update} versus "
                f"best_update={self.model_selection.state['best_update']} "
                f"games={self.settings.arena_games} parallel_games_per_rank={self.settings.arena_parallel_games} "
                f"inference_batch={self.settings.arena_inference_batch_size}"
            )
        try:
            result = self.model_selection.evaluate(
                self.policy, self.layout, update=self.update, cumulative=self.cumulative,
                amp_dtype=self.amp_dtype, stop_requested=lambda: self.stop_requested,
                historical=self.historical,
            )
        except InterruptedError:
            self.stop_requested = True
            self.phase = "ready"
            self.logger.event("evaluation interrupted; preserving current model and retrying this round on resume")
            return None
        if result is not None:
            metrics = {
                f"arena/{self.model_selection.milestone_key}": milestone,
                "arena/evaluation_type": result.get("evaluation_type", "champion"),
                "arena/environment_plies": self.cumulative["environment_plies"],
                "arena/progress_percent": result["progress_percent"],
                "arena/games": result["games"], "arena/wins": result["wins"],
                "arena/draws": result["draws"], "arena/losses": result["losses"],
                "arena/win_rate": result["win_rate"], "arena/score": result["score"],
                "arena/parallel_games_per_rank": self.settings.arena_parallel_games,
                "arena/inference_batch_size": self.settings.arena_inference_batch_size,
                "arena/wall_seconds": result.get("wall_seconds", 0.0),
                "arena/plies_per_second": result.get("plies_per_second", 0.0),
                "arena/games_per_second": result.get("games_per_second", 0.0),
            }
            if observational:
                panel = result["historical_panel"]
                metrics.update({"historical_eval/minimum_score": panel["minimum_score"],
                                "historical_eval/confirmed_regression": int(panel["confirmed_regression"]),
                                "historical_eval/opponents": len(panel["results"]),
                                "historical_eval/candidate_update": result["candidate_update"]})
                self.logger.event(
                    f"{result['evaluation_type']} {self.model_selection.milestone_label(milestone)} complete: "
                    f"games={result['games']} minimum_score={panel['minimum_score']:.3%} "
                    f"regression_warning={panel['confirmed_regression']} decision=use_latest update={self.update}"
                )
            else:
                metrics.update({"arena/score_ci_lower": result["score_ci"][0],
                                "arena/score_ci_upper": result["score_ci"][1],
                                "arena/promoted": int(result["promoted"]),
                                "arena/best_update": result["best_update"],
                                "arena/score_delta_vs_best": result["score_delta_vs_best"],
                                "arena/strength_verdict": result["strength_verdict"]})
                self.logger.event(
                    f"model selection {self.model_selection.milestone_label(milestone)} complete: "
                    f"W/D/L={result['wins']}/{result['draws']}/{result['losses']} "
                    f"score={result['score']:.3%} decision={result['decision']} "
                    f"best_update={result['best_update']} strength={result['strength_verdict']}"
                )
            self.logger.log(self.update, metrics)
            self.phase = "model_selection_checkpoint"
            self.save_checkpoint(reason="after_model_selection", archive=False)
        self.phase = "ready"
        return result

    def _aggregate_rollout_metrics(
        self, metrics: RolloutMetrics
    ) -> RolloutMetrics:
        summed_fields = (
            "policy_samples",
            "environment_plies",
            "anchors",
            "root_candidates",
            "terminal_continuations",
            "continuation_plies",
            "base_plies",
            "base_games_completed",
            "wins",
            "draws",
            "losses",
            "flag_captures",
            "nonterminal_flag_captures",
            "futile_attacks",
            "known_nonflag_hq_entries",
            "flag_defense_missed",
            "flag_defense_successes",
            "bombed_commanders",
            "bombed_army_commanders",
            "bombed_division_commanders",
            "actor_deferred_batches",
        )
        aggregated = RolloutMetrics()
        for field in summed_fields:
            setattr(
                aggregated,
                field,
                self.distributed.reduce_int(int(getattr(metrics, field))),
            )
        aggregated.wall_seconds = self.distributed.reduce_float(
            metrics.wall_seconds, operation="max"
        )
        aggregated.flag_capture_reward_abs_sum = self.distributed.reduce_float(
            metrics.flag_capture_reward_abs_sum, operation="sum"
        )
        aggregated.futile_attack_reward_abs_sum = self.distributed.reduce_float(
            metrics.futile_attack_reward_abs_sum, operation="sum"
        )
        aggregated.bomb_reward_sum = self.distributed.reduce_float(metrics.bomb_reward_sum, operation="sum")
        aggregated.flag_defense_reward_sum = self.distributed.reduce_float(metrics.flag_defense_reward_sum, operation="sum")
        aggregated.known_nonflag_hq_reward_sum = self.distributed.reduce_float(
            metrics.known_nonflag_hq_reward_sum, operation="sum")
        aggregated.actor_inference_seconds = self.distributed.reduce_float(
            metrics.actor_inference_seconds, operation="max"
        )
        aggregated.actor_result_wait_seconds = self.distributed.reduce_float(
            metrics.actor_result_wait_seconds, operation="max"
        )
        aggregated.critic_inference_seconds = self.distributed.reduce_float(
            metrics.critic_inference_seconds, operation="max"
        )
        aggregated.environment_step_seconds = self.distributed.reduce_float(
            metrics.environment_step_seconds, operation="max"
        )
        aggregated.environment_workers = self.distributed.reduce_int(metrics.environment_workers)
        aggregated.environment_worker_seconds = self.distributed.reduce_float(
            metrics.environment_worker_seconds, operation="sum"
        )
        aggregated.environment_sync_seconds = self.distributed.reduce_float(
            metrics.environment_sync_seconds, operation="max"
        )
        return aggregated

    def _ppo_collection_count(self) -> int:
        target = self.settings.step_budget_target
        if target is None:
            return self.local_anchor_batch
        remaining = target - self.cumulative["environment_plies"]
        # Current PPO has no branch simulations, so one card stops exactly.
        # DDP keeps equal positive counts: at most world_size - 1 excess steps.
        # A collector with extra branches may finish its batch over the rough
        # budget; every actually executed branch transition must still count.
        count = min(self.local_anchor_batch, math.ceil(remaining / self.distributed.world_size))
        cutoff_remaining = self.futile_attack_controller.stop_environment_plies - self.cumulative["environment_plies"]
        from .tactical_rewards import BOMB_REWARD_FIELDS
        shaping = self.settings.futile_attack_penalty or any(
            getattr(self.settings, field, 0.0) for field in BOMB_REWARD_FIELDS.values())
        if shaping and cutoff_remaining >= self.distributed.world_size:
            count = min(count, cutoff_remaining // self.distributed.world_size)
        return count

    def _futile_attack_coefficient(self):
        progress = self.cumulative["environment_plies"]
        # DDP disables at most world_size - 1 steps early rather than allowing
        # any penalized transition to cross the exact halfway cutoff.
        if self.futile_attack_controller.stop_environment_plies - progress < self.distributed.world_size:
            return 0.0
        return self.futile_attack_controller.coefficient(progress)

    def _bomb_reward_coefficients(self):
        from .tactical_rewards import BOMB_REWARD_FIELDS, halfway_reward_scale
        scale = halfway_reward_scale(self.cumulative["environment_plies"], self.settings.target_environment_plies,
                                     world_size=self.distributed.world_size)
        return {field: getattr(self.settings, field) * scale for field in BOMB_REWARD_FIELDS.values()}

    def _gather_layout_outcomes(self, outcomes: Sequence[LayoutOutcome]) -> None:
        self.layout_buffer.advance(self.update + 1, minimum_version=self._layout_minimum_version())
        if not self.distributed.enabled:
            self.layout_buffer.extend(outcomes)
            return
        # PlayerSetup owns an immutable mappingproxy, which cannot be pickled
        # by gather_object. Transfer its already checkpointed pointer trace.
        records = [{"mode": item.sample.mode.value,
                    "position_indices": item.sample.position_indices,
                    "old_log_probs": item.sample.old_log_probs,
                    "reward": item.reward, "seat": item.seat,
                    "behavior_version": item.behavior_version} for item in outcomes]
        gathered = self.distributed.gather_object(records)
        if self.distributed.primary:
            for rank_records in gathered or []:
                for row in rank_records:
                    self.layout_buffer.append_record(row)

    def train(self) -> None:
        try:
            if self.pool_expanded_on_resume:
                # Persist the new capacity before collecting any new actions so
                # ordinary restarts no longer need the one-time expansion flag.
                self.save_checkpoint(reason='parallel_pool_expanded', archive=False)
                self.pool_expanded_on_resume = False
            # Also retries an interrupted last (100%) match when no training
            # budget remains, instead of silently skipping the final selection.
            self._maybe_evaluate_model()
            while (
                self.update < self.settings.total_updates
                and not self._training_budget_reached()
            ):
                if self.distributed.any(self.stop_requested):
                    break
                started = time.perf_counter()
                if is_accelerator(self.device):
                    # policy.train() drops the rollout-only paged-KV arena, but
                    # the device caching allocator may keep its large block and
                    # fragment it during learner backward.  Return those free
                    # blocks at the update boundary before constructing the
                    # next arena; otherwise reserved VRAM can ratchet upward by
                    # one arena per update under WDDM.
                    empty_cache(self.device)
                    reset_peak_memory_stats(self.device)
                next_update = self.update + 1
                policy_lr = self._set_policy_lr(next_update)
                # The learner's seats share these instances. Historical games
                # pin a frozen opponent in the bounded cohort scheduler. Each
                # actor stays frozen throughout collection; only learner-owned
                # action log-probabilities define the policy update's pi_old.
                self.policy.eval()
                self.layout.eval()
                self.phase = "base_game_collection"
                actor = FrozenPolicyActor(
                    self.policy,
                    amp_dtype=self.amp_dtype,
                    max_batch_size=self.effective_actor_inference_batch,
                )
                base_collection_started = time.perf_counter()
                futile_attack_coefficient = self._futile_attack_coefficient()
                bomb_rewards = self._bomb_reward_coefficients()
                self.training_audit = (MAPPOTrainingAudit(self.tactical_log_settings,
                    update=next_update, rank=self.distributed.rank)
                    if self.settings.algorithm == "mappo" and self.tactical_log_settings.enabled else None)
                if self.settings.algorithm in ("ppo", "mappo"):
                    self.phase = f"{self.settings.algorithm}_trajectory_collection"
                    if (self.settings.rollout_environment_workers > 1 or self.settings.algorithm == "mappo") and self._ppo_environment is None:
                        from .ppo_environment import ParallelPPOEnvironment
                        self._ppo_environment = ParallelPPOEnvironment(self.settings.rollout_environment_workers)
                    assert self.critic is not None
                    value_actor = FrozenValueActor(
                        self.critic, amp_dtype=self.amp_dtype,
                        max_batch_size=self.effective_actor_inference_batch,
                        deferred=self.settings.ppo_deferred_values,
                    )
                    collector = collect_mappo_samples if self.settings.algorithm == "mappo" else collect_ppo_samples
                    samples, layout_outcomes, rollout_metrics = collector(
                        self.pool, actor, value_actor, self.layout,
                        count=self._ppo_collection_count(), behavior_version=self.update,
                        environment=self._ppo_environment,
                        environment_workers=self.settings.rollout_environment_workers,
                        pipeline_groups=self.settings.ppo_pipeline_groups,
                        discount=self.settings.discount, gae_lambda=self.settings.gae_lambda,
                        draw_reward=self.settings.draw_reward,
                        flag_capture_reward=self.settings.flag_capture_reward,
                        futile_attack_penalty=futile_attack_coefficient,
                        **({**bomb_rewards, "advantage_mode": self.settings.mappo_advantage_mode,
                            "known_nonflag_hq_penalty": self.settings.known_nonflag_hq_penalty,
                            "flag_defense_missed_penalty": self.settings.flag_defense_missed_penalty,
                            "flag_defense_success_reward": self.settings.flag_defense_success_reward,
                            "reward_scope": self.settings.mappo_reward_scope,
                            "training_audit": self.training_audit}
                           if self.settings.algorithm == "mappo" else {}),
                    )
                    groups = self._normalize_ppo_batch(samples)
                    if self.training_audit is not None:
                        self.training_audit.bind(groups, normalization=dict(
                            mean=self.ppo_rollout_metrics["ppo/raw_advantage_mean"],
                            std=self.ppo_rollout_metrics["ppo/raw_advantage_std"],
                            epsilon=self.settings.advantage_epsilon), objective=dict(
                            mode=self.settings.mode.value, advantage_mode=self.settings.mappo_advantage_mode,
                            reward_scope=self.settings.mappo_reward_scope, discount=self.settings.discount,
                            gae_lambda=self.settings.gae_lambda, draw_reward=self.settings.draw_reward,
                            flag_capture_reward=self.settings.flag_capture_reward,
                            futile_attack_penalty=futile_attack_coefficient,
                            flag_defense_missed_penalty=self.settings.flag_defense_missed_penalty,
                            flag_defense_success_reward=self.settings.flag_defense_success_reward,
                            known_nonflag_hq_penalty=self.settings.known_nonflag_hq_penalty, **bomb_rewards))
                    del samples  # Normalized samples retain every required history.
                    if not self.settings.ppo_deferred_values:
                        actor.max_batch_size = min(actor.max_batch_size, value_actor.max_batch_size)
                    base_collection_seconds = 0.0
                else:
                    anchors, layout_outcomes, base_plies, completed_games = (
                        self.pool.collect_anchors(
                            actor, self.layout, count=self.local_anchor_batch,
                            behavior_version=self.update,
                            draw_reward=self.settings.draw_reward,
                        )
                    )
                    base_collection_seconds = self.distributed.reduce_float(
                        time.perf_counter() - base_collection_started, operation="max",
                    )
                    self.phase = "terminal_rollouts"
                    groups, rollout_metrics = collect_policy_groups(
                        anchors, actor, behavior_version=self.update,
                        advantage_epsilon=self.settings.advantage_epsilon,
                        draw_reward=self.settings.draw_reward,
                        anchor_wave_size=self.settings.rollout_anchor_wave_size,
                        environment_workers=self.settings.rollout_environment_workers,
                    )
                    rollout_metrics.record_environment_steps(base_plies)
                    rollout_metrics.base_games_completed = completed_games
                    # The total-transition throughput includes the base-game
                    # actions too, so include their collection time as well.
                    rollout_metrics.wall_seconds += base_collection_seconds
                rollout_metrics = self._aggregate_rollout_metrics(rollout_metrics)
                encoding_metrics = self.distributed.mean_metrics(
                    self.policy._rollout_cache_snapshot or self.policy.board_encoding_metrics()
                )
                previous_actor_batch = self.effective_actor_inference_batch
                self.effective_actor_inference_batch = self.distributed.reduce_int(
                    actor.max_batch_size, operation="min"
                )
                if self.effective_actor_inference_batch < previous_actor_batch:
                    self.logger.event(
                        "%s OOM reduced actor inference batch from %d to %d "
                        "during rollout"
                        % (
                            self.device.type.upper(),
                            previous_actor_batch,
                            self.effective_actor_inference_batch,
                        )
                    )
                # Release inference KV arenas before learner backward. Learner
                # graphs retain their bounded, shared scratch allocation.
                self.policy.clear_inference_board_cache()
                if self.critic is not None:
                    self.critic.clear_inference_board_cache()
                empty_cache(self.device)
                self.phase = "policy_backward"
                if self.training_audit is not None:
                    self.training_audit.measure(self, "before")
                policy_backward_started = time.perf_counter()
                policy_metrics = self._update_policy(groups)
                policy_backward_seconds = self.distributed.reduce_float(
                    time.perf_counter() - policy_backward_started,
                    operation="max",
                )
                if self.training_audit is not None:
                    self.training_audit.measure(self, "after")
                critic_metrics: dict[str, float] = {}
                critic_backward_seconds = 0.0
                if self.critic is not None:
                    self.phase = "critic_backward"
                    critic_started = time.perf_counter()
                    # Long Actor batches can leave most VRAM in unused eager
                    # allocator blocks. Release them before the separate
                    # Critic phase; live model and captured graph storage stays
                    # intact, while WDDM has room for the new packed work.
                    reserved_before = (torch.cuda.memory_reserved(self.device)
                                       if self.device.type == 'cuda' else 0)
                    empty_cache(self.device)
                    released_cache = (max(0, reserved_before - torch.cuda.memory_reserved(self.device))
                                      if self.device.type == 'cuda' else 0)
                    critic_metrics = self._update_critic(groups)
                    critic_metrics['optimizer/critic_entry_cache_released_gib'] = released_cache / 2**30
                    critic_backward_seconds = self.distributed.reduce_float(
                        time.perf_counter() - critic_started, operation="max"
                    )
                self._gather_layout_outcomes(layout_outcomes)
                layout_metrics: dict[str, float] = {}
                layout_backward_seconds = 0.0
                if next_update % self.settings.layout_update_interval == 0:
                    self.phase = "layout_backward"
                    layout_backward_started = time.perf_counter()
                    if self.distributed.primary:
                        layout_metrics = self._update_layout()
                    self.distributed.broadcast_module(self.layout)
                    layout_metrics = dict(
                        self.distributed.broadcast_object(layout_metrics)
                    )
                    layout_backward_seconds = self.distributed.reduce_float(
                        time.perf_counter() - layout_backward_started,
                        operation="max",
                    )
                # Do not keep a completed batch alive while collecting the next
                # one. Histories still needed by live games/layout buffers have
                # their own references; this drops only consumed training data.
                del groups, layout_outcomes
                self.update = next_update

                for key, value in (
                    ("policy_samples", rollout_metrics.policy_samples),
                    ("environment_plies", rollout_metrics.environment_plies),
                    ("anchors", rollout_metrics.anchors),
                    ("root_candidates", rollout_metrics.root_candidates),
                    (
                        "terminal_continuations",
                        rollout_metrics.terminal_continuations,
                    ),
                    ("continuation_plies", rollout_metrics.continuation_plies),
                    ("base_plies", rollout_metrics.base_plies),
                    ("base_games", rollout_metrics.base_games_completed),
                ):
                    self.cumulative[key] += int(value)
                self._write_training_progress()
                tactical_metrics = {}
                if self.training_audit is not None:
                    self.training_audit.finish()
                    tactical_metrics = self.training_audit.write(self.run_directory,
                        environment_plies=self.cumulative["environment_plies"])
                    # Do not retain sampled histories into the next rollout.
                    self.training_audit = None
                if futile_attack_coefficient > 0:
                    self.futile_attack_controller.observe(rollout_metrics.futile_attacks, rollout_metrics.policy_samples)
                if self.update % self.settings.reference_refresh_updates == 0:
                    if self.reference_policy is not None:
                        self.reference_policy.load_state_dict(self.policy.state_dict())
                    self.reference_layout.load_state_dict(self.layout.state_dict())
                    self.logger.event(f"reference refreshed at update {self.update}")

                metrics: dict[str, float | int | str] = {
                    "algorithm": self.settings.algorithm,
                    "mode": self.settings.mode.value,
                    "model/dead_rules_enabled": int(
                        self.settings.dead_rules_enabled
                    ),
                    "model/shared_policy_instances_for_players": 1,
                    "model/shared_layout_instances_for_players": 1,
                    "model/critic_instances": int(self.critic is not None),
                    "model/centralized_critic": int(self.settings.algorithm == "mappo"),
                    "training/mappo_advantage_mode": self.settings.mappo_advantage_mode,
                    "training/mappo_reward_scope": self.settings.mappo_reward_scope,
                    "model/actor_temporal_layers": self.policy.config.temporal_layers,
                    "model/critic_temporal_layers": self.critic.config.temporal_layers if self.critic is not None else 0,
                    "model/attention_head_dim": self.policy.config.temporal_dim // self.policy.config.temporal_heads,
                    "model/token_fusion_hidden_dim": self.policy.config.token_fusion_hidden_dim,
                    "model/action_input_dim": self.policy.action_encoder.projection.in_features,
                    "model/board_pass_input_dim": self.policy.board_encoder.pass_feature_dim,
                    "model/board_casualty_input_dim": self.policy.board_encoder.casualty_feature_dim,
                    "model/board_input_dim": self.policy.board_encoder.input_dim,
                    "model/action_pass_input_dim": self.policy.action_encoder.pass_feature_dim,
                    "rules/max_passes_per_player": self.settings.max_passes_per_player,
                    "rules/eliminate_no_moves": 1.,
                    "training/draw_reward": self.settings.draw_reward,
                    "training/flag_capture_reward": self.settings.flag_capture_reward,
                    "training/known_nonflag_hq_penalty": self.settings.known_nonflag_hq_penalty,
                    "training/flag_defense_missed_penalty": self.settings.flag_defense_missed_penalty,
                    "training/flag_defense_success_reward": self.settings.flag_defense_success_reward,
                    **{f"training/{field}": value for field, value in bomb_rewards.items()},
                    **{f"training/{field}_initial": getattr(self.settings, field) for field in bomb_rewards},
                    "training/bomb_reward_stop_environment_plies": (self.settings.target_environment_plies or 0) // 2,
                    "training/futile_attack_penalty": futile_attack_coefficient,
                    "training/futile_attack_penalty_initial": self.settings.futile_attack_penalty,
                    "training/futile_attack_penalty_next": self._futile_attack_coefficient(),
                    "training/futile_attack_rate_ema": self.futile_attack_controller.rate_ema or 0.0,
                    "training/futile_attack_reference_rate": self.futile_attack_controller.reference_rate or 0.0,
                    "training/futile_attack_frequency_scale": self.futile_attack_controller.frequency_scale,
                    "training/futile_attack_stop_environment_plies": self.futile_attack_controller.stop_environment_plies,
                    "evaluation/observational_only": int(self.settings.arena_observational_only),
                    "evaluation/champion_only": int(self.settings.arena_champion_only),
                    "evaluation/historical_teammate_fraction": self.settings.arena_historical_teammate_fraction,
                    "evaluation/games": self.settings.arena_games,
                    "distributed/world_size": self.distributed.world_size,
                    "distributed/local_anchor_batch": self.local_anchor_batch,
                    "distributed/model_replicas": self.distributed.world_size,
                    "training/target_continuation_plies": (
                        -1
                        if self.settings.target_continuation_plies is None
                        else self.settings.target_continuation_plies
                    ),
                    "training/target_environment_plies": (
                        self.settings.target_environment_plies
                        if self.settings.target_environment_plies is not None
                        else -1
                    ),
                    "training/step_budget_counter": self.settings.step_budget_counter,
                    "training/torch_num_threads": torch.get_num_threads(),
                    "optimizer/effective_policy_microbatch": (
                        self.effective_policy_microbatch
                    ),
                    "rollout/effective_actor_inference_batch": (
                        self.effective_actor_inference_batch
                    ),
                    "optimizer/policy_lr": policy_lr,
                    "timing/base_collection_seconds": base_collection_seconds,
                    "timing/policy_backward_seconds": policy_backward_seconds,
                    "timing/critic_backward_seconds": critic_backward_seconds,
                    "timing/layout_backward_seconds": layout_backward_seconds,
                    "timing/update_seconds": self.distributed.reduce_float(
                        time.perf_counter() - started, operation="max"
                    ),
                    **rollout_metrics.as_dict(),
                    **encoding_metrics,
                    **policy_metrics,
                    **critic_metrics,
                    **(self.ppo_rollout_metrics if self.settings.algorithm in ("ppo", "mappo") else {}),
                    **layout_metrics,
                    **tactical_metrics,
                }
                metrics.update(
                    {
                        f"cumulative/{key}": value
                        for key, value in self.cumulative.items()
                    }
                )
                self.historical.update_progress(self.policy, self.layout,
                    environment_plies=self.cumulative["environment_plies"], update=self.update)
                self.historical.write_status(self.pool)
                if self.historical.enabled:
                    metrics.update(self.historical.metrics())
                if self.update % self.settings.metrics_every_updates == 0:
                    self.logger.log(self.update, metrics)

                selection_result = self._maybe_evaluate_model()
                if (self.settings.inference_snapshot_every_updates
                        and self.update % self.settings.inference_snapshot_every_updates == 0):
                    self.phase = "inference_snapshot"
                    self._export_live_inference()
                if selection_result is None and self._periodic_checkpoint_due():
                    self.phase = "checkpoint"
                    self.save_checkpoint(
                        reason="periodic",
                        archive=(
                            self.update % self.settings.archive_every_updates == 0
                        ),
                    )
                self.phase = "ready"
            if self.settings.checkpoint_policy == "periodic":
                self.phase = "checkpoint"
                self.save_checkpoint(reason="completed_or_stopped", archive=True)
            else:
                self.logger.event(
                    "training completed or stopped; no extra checkpoint outside evaluation; "
                    + ("resume uses the last saved checkpoint" if self.checkpoints.latest_path.exists()
                       else "no checkpoint is available; this progress cannot be resumed")
                )
        except BaseException:
            # Never save partial PPO/distributed updates or write an extra
            # checkpoint outside evaluation when that policy is selected.
            self.phase = "emergency_checkpoint"
            if (self.settings.checkpoint_policy == "evaluation"
                    or self.distributed.enabled or self.settings.algorithm in ("ppo", "mappo")):
                self.logger.event(
                    "training exception: retaining the last completed atomic checkpoint; "
                    "partial actor/critic updates are not saved"
                )
            else:
                self.save_checkpoint(reason="emergency_exception", archive=False)
            raise
        finally:
            self.phase = "stopped"
            self.historical.close()
            if self._ppo_environment is not None:
                self._ppo_environment.close()
                self._ppo_environment = None
            self.logger.close()
