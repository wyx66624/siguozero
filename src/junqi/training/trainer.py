"""Resumable four-player PPO and two-player terminal-rollout GRPO training."""

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
from .inference_snapshot import export_inference_snapshot
from .models import (
    GamePolicyTransformer,
    GameValueTransformer,
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
from .entropy import AdaptiveEntropyCoefficient, phase_entropy_ratios


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
        adopt_draw_penalty: bool = False,
        adopt_flag_capture_reward: bool = False,
        adopt_champion_evaluation: bool = False,
        adopt_pass_rule: bool = False,
        expand_game_pool: bool = False,
        reset_oom_batch_limits: bool = False,
    ) -> None:
        self.settings = settings
        self.session_started_unix = time.time()
        self.resumed_from_update = 0
        self.resumed_from_cumulative = {}
        self.adopt_current_draw_rules = adopt_current_draw_rules
        self.adopt_draw_penalty = adopt_draw_penalty
        self.adopt_flag_capture_reward = adopt_flag_capture_reward
        self.adopt_pass_rule = adopt_pass_rule
        self.expand_game_pool = expand_game_pool
        self.reset_oom_batch_limits = reset_oom_batch_limits
        self.pool_expanded_on_resume = False
        self.pass_rule_migration = None
        self.last_checkpoint_environment_plies = 0
        self.draw_rule_migration = None
        self.draw_objective_migration = None
        self.flag_capture_objective_migration = None
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
            had_existing_run_artifacts = any(self.run_directory.iterdir())
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
            ema_decay=settings.lr_ema_decay, recovery_clip_fraction=settings.lr_recovery_clip_fraction,
        ) if settings.adaptive_learning_rate else None)
        self.entropy_opening_plies = min(settings.entropy_opening_plies, settings.model.max_transitions)
        self.entropy_controllers = {
            phase: AdaptiveEntropyCoefficient(
                initial=settings.entropy_coefficient, minimum=settings.entropy_minimum,
                maximum=(settings.entropy_opening_maximum
                         if phase == "opening" and settings.entropy_opening_maximum is not None
                         else settings.entropy_maximum),
                target_ratio=settings.entropy_target_ratio,
                adaptation_rate=settings.entropy_adaptation_rate, ema_decay=settings.entropy_ema_decay,
                enabled=settings.adaptive_entropy)
            for phase in ("opening", "other")
        }
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
        self.critic = (
            GameValueTransformer(settings.model).to(self.device)
            if settings.algorithm == "ppo" else None
        )
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
            fused=True if settings.algorithm == 'ppo' and settings.ppo_fused_optimizer and self.device.type == 'cuda' else None,
        )
        self.layout_optimizer = torch.optim.AdamW(
            self.layout.parameters(),
            lr=settings.layout_learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.05,
            fused=True if settings.algorithm == 'ppo' and settings.ppo_fused_optimizer and self.device.type == 'cuda' else None,
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
            max_game_plies=settings.max_game_plies,
            dead_rules_enabled=settings.dead_rules_enabled,
            seed=settings.seed + 17 + 1_000_003 * self.distributed.rank,
            layout_prefetch_games=settings.layout_prefetch_games,
            no_capture_draw_plies=settings.no_capture_draw_plies,
        )
        self.pool.historical = self.historical
        try:
            if self.initialize_from is not None:
                self._initialize_weights(self.initialize_from)
            resumed = self._resume_if_available() if auto_resume else False
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
        if resumed and (adopt_current_draw_rules or adopt_draw_penalty or adopt_pass_rule or
                        adopt_flag_capture_reward or adopt_champion_evaluation):
            self.save_checkpoint(reason="champion_evaluation_migration" if adopt_champion_evaluation else
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
        if self.settings.algorithm != 'ppo':
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
        resolved["flag_capture_objective_migration"] = self.flag_capture_objective_migration
        resolved["pass_rule_migration"] = self.pass_rule_migration
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
            if payload.get("algorithm") == "ppo" and "critic" in payload:
                # Weight-only initialization also accepts the pre-draw head.
                weights = dict(payload["critic"])
                if payload["format_version"] in (6, 7):
                    for name in ("draw_value_head.weight", "draw_value_head.bias"):
                        if name not in weights:
                            weights[name] = torch.zeros_like(self.critic.state_dict()[name])
                self.critic.load_state_dict(weights, strict=True)
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
        if self.adopt_current_draw_rules:
            from .rule_migration import adopt_draw_rules
            adopt_draw_rules(payload, self.settings)
        from .pass_migration import reconcile_pass_rule
        reconcile_pass_rule(payload, adopt=self.adopt_pass_rule)
        from .objective_migration import reconcile_draw_objective, reconcile_flag_capture_objective
        reconcile_draw_objective(payload, self.settings, adopt=self.adopt_draw_penalty)
        reconcile_flag_capture_objective(payload, self.settings, adopt=self.adopt_flag_capture_reward)
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
            self.policy_lr_scale = self.lr_controller.scale
        for phase, controller in self.entropy_controllers.items():
            controller.load_state_dict(trainer_state.get("adaptive_entropy", {}).get(phase))
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
        self.pass_rule_migration = trainer_state.get("pass_rule_migration")
        self.draw_objective_migration = trainer_state.get("draw_objective_migration")
        self.flag_capture_objective_migration = trainer_state.get("flag_capture_objective_migration")
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
        entropy_options = ({
            "opening_entropy_coefficient": self.entropy_controllers["opening"].coefficient,
            "entropy_opening_plies": self.entropy_opening_plies,
        } if self.settings.adaptive_entropy else {})
        epoch_metrics: dict[str, float] = {}
        gradient_balance = 1.0
        global_owned = len(groups)
        if self.settings.algorithm == "ppo" and self.historical.active:
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
        captured = (self._try_learner_graph(chunks[0], critic=False)
                    if self.settings.algorithm == 'ppo' and len(chunks) == 1 and gradient_balance == 1.0
                    and all(item.learnable for item in groups) else None)
        if captured is not None:
            capacity = captured.pop('_graph_tokens')
            epoch_metrics.update(captured)
            epoch_metrics['optimizer/learner_graph_fraction'] = 1.0
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
                    if self.settings.algorithm == "ppo":
                        output = policy_ppo_loss(
                            self.policy_parallel, chunk,
                            clip_epsilon=self.settings.clip_epsilon,
                            entropy_coefficient=self.entropy_controllers["other"].coefficient,
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
        return (self.settings.algorithm == "ppo" and self.settings.ppo_sequence_training
                and self.settings.model.dropout == 0.0)

    def _try_learner_graph(self, samples, *, critic):
        """Capture only a complete single-device optimizer batch, never accumulation.

        The actor owns at most six graphs (two history by three source sizes),
        and the critic two. Oversized histories fall back intact to packing.
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
        capacity = 2560 if tokens + 32 - len(groups) <= 2560 else 4096
        action_metadata = None
        source_capacity = 16384
        if not critic:
            action_metadata = PackedLearnerGraph.action_metadata(states)
            required_sources = len(action_metadata[3])
            source_capacity = next((size for size in (4096, 8192, 16384)
                                    if required_sources <= size), 16384)
        key = (capacity, source_capacity)
        if graphs is None:
            graphs = {}
            setattr(self, name, graphs)
        graph = graphs.get(key)
        if graph is None:
            if not hasattr(self, '_learner_graph_pool'):
                self._learner_graph_pool = torch.cuda.graph_pool_handle()
            # Graphs never run concurrently. Every loss/gradient is consumed by
            # the finite guard and Adam before another graph can replay. Inputs,
            # parameters and Adam state live outside the shared scratch pool.
            graph = graphs[key] = PackedLearnerGraph(model, critic=critic, amp_dtype=self.amp_dtype,
                tokens=capacity, sources=source_capacity, pool=self._learner_graph_pool)
        try:
            ready = graph.stage(model, samples,
                clip=self.settings.value_clip_epsilon if critic else self.settings.clip_epsilon,
                coefficient=self.settings.value_coefficient if critic else self.entropy_controllers['other'].coefficient,
                opening_coefficient=(self.entropy_controllers['opening'].coefficient
                                     if not critic and self.settings.adaptive_entropy else None),
                opening_plies=self.entropy_opening_plies, action_metadata=action_metadata,
                policy_clip=None if critic else self._policy_clip())
            if not ready:
                return None
            result = graph.run(model)
            if not critic:
                result['_graph_tokens'] = float(capacity)
                result['optimizer/learner_source_capacity'] = float(source_capacity)
                result['optimizer/learner_source_padding_fraction'] = 1.0 - graph.used_sources / source_capacity
            if not critic and not self.settings.adaptive_entropy:
                result = {k:v for k,v in result.items() if '_entropy_' not in k}
            return result
        except RuntimeError as error:
            if not is_out_of_memory(error, self.device.type):
                raise
            # Do not retry capture with increasingly many resident private pools.
            setattr(self, name, False)
            model.zero_grad(set_to_none=True)
            graphs.clear()
            del graph
            empty_cache(self.device)
            self.logger.event(f"{'critic' if critic else 'policy'} learner CUDA graph exceeded memory; using packed learner")
            return None

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

    def _update_policy(
        self, groups: Sequence[PolicyGroup] | Sequence[PPOSample]
    ) -> dict[str, float]:
        if self.settings.algorithm == "ppo" and len(groups) > self.settings.ppo_minibatch_samples:
            metrics = self._update_ppo_minibatches(groups, critic=False)
        else:
            metrics = self._update_policy_batch(groups)
        ratios = phase_entropy_ratios(metrics)
        metrics.update({"policy/adaptive_entropy_enabled": float(self.settings.adaptive_entropy),
                        "policy/entropy_target_ratio": self.settings.entropy_target_ratio,
                        "policy/entropy_opening_plies": float(self.entropy_opening_plies)})
        for phase, controller in self.entropy_controllers.items():
            prefix = "policy/opening_entropy" if phase == "opening" else "policy/entropy"
            metrics[f"{prefix}_coefficient"] = controller.coefficient
            metrics[f"{prefix}_coefficient_max"] = controller.maximum
            if metrics.get("optimizer/policy_steps", 0) > 0:
                controller.observe(ratios[phase])
            metrics[f"{prefix}_coefficient_next"] = controller.coefficient
            if controller.entropy_ema is not None:
                metrics[f"{prefix}_ratio_ema"] = controller.entropy_ema
        metrics["optimizer/adaptive_lr_enabled"] = float(self.lr_controller is not None)
        if self.lr_controller is not None:
            metrics.update(self._finish_learning_rate(groups, metrics))
        if self.settings.algorithm == "ppo":
            metrics.update(self._policy_clip().metrics())
            metrics["policy/adaptive_clip_enabled"] = float(self.settings.ppo_adaptive_clip)
        return metrics

    def _final_policy_diagnostics(self, samples) -> dict[str, float]:
        started = time.perf_counter()
        probe = diagnostic_sequences(samples, limit=self.settings.lr_probe_samples,
            sequence_length=self.settings.ppo_max_samples_per_sequence,
            seed=self.settings.seed + 1_000_003 * (self.update + 1) + self.distributed.rank)
        totals = torch.zeros(2, device=self.device)
        was_training = self.policy.training
        self.policy.eval()
        try:
            with torch.no_grad():
                # Retain the same bounded sequence/microbatch packing as the
                # learner. No reference network or full old logits are stored.
                for chunk in self._learner_chunks(probe):
                    with self._autocast():
                        output = policy_ppo_loss(self.policy, chunk,
                            clip_epsilon=self.settings.clip_epsilon, entropy_coefficient=0.,
                            policy_clip=self._policy_clip(),
                            sequence_training=self._sequence_training_enabled(), defer_metrics=True)
                    totals += torch.stack([output.metrics["policy/approx_kl_old"],
                                           output.metrics["policy/clip_fraction"]]) * len(chunk)
        finally:
            self.policy.train(was_training)
        values = totals.cpu().tolist()  # one transfer, only two scalar moments
        sums = self.distributed.sum_metrics(dict(kl=values[0], clip=values[1], count=len(probe)))
        count = sums["count"]
        result = {"policy/final_probe_samples": count,
                  "timing/lr_probe_seconds": self.distributed.reduce_float(
                      time.perf_counter() - started, operation="max")}
        if count:
            result.update({"policy/final_kl": sums["kl"] / count,
                           "policy/final_clip_fraction": sums["clip"] / count})
        return result

    def _finish_learning_rate(self, samples, metrics) -> dict:
        controller = self.lr_controller
        diagnostics = (self._final_policy_diagnostics(samples)
                       if metrics.get("optimizer/policy_steps", 0) > 0 else
                       {"policy/final_probe_samples": 0., "timing/lr_probe_seconds": 0.})
        next_base = self._scheduled_learning_rate(self.update + 2)
        next_lr, direction = controller.observe(scheduled=next_base,
            kl=diagnostics.get("policy/final_kl"),
            clip_fraction=diagnostics.get("policy/final_clip_fraction"),
            early_stopped=bool(metrics.get("optimizer/policy_early_stopped", 0)),
            allow_increase=self.update + 1 > self.settings.warmup_updates)
        self.policy_lr_scale = controller.scale
        diagnostics.update({"optimizer/policy_lr_next": next_lr,
            "optimizer/policy_lr_min": self.settings.minimum_learning_rate,
            "optimizer/policy_lr_max": self.settings.maximum_learning_rate,
            "optimizer/policy_lr_ceiling_next": min(self.settings.maximum_learning_rate, next_base),
            "optimizer/policy_lr_scale": controller.scale,
            "optimizer/lr_direction": direction,
            "optimizer/lr_reason": controller.reason,
            "optimizer/lr_stable_count": controller.stable_count,
            "optimizer/lr_stable_required": controller.stable_updates,
            "optimizer/lr_cooldown": controller.cooldown,
            "optimizer/lr_scale_reduced": float(direction < 0),
            "optimizer/lr_scale_increased": float(direction > 0),
            "optimizer/critic_lr": self.critic_optimizer.param_groups[0]["lr"],
        })
        if controller.kl_ema is not None:
            diagnostics["policy/final_kl_ema"] = controller.kl_ema
        return diagnostics

    def _ppo_optimizer_batches(self, samples):
        if self._sequence_training_enabled():
            chains = sequence_training_batches(
                samples, sequences_per_batch=1,
                max_samples_per_sequence=self.settings.ppo_max_samples_per_sequence,
            )
            random.shuffle(chains)
            ordered = [sample for chain in chains for sample in chain]
        else:
            ordered = list(samples)
            random.shuffle(ordered)
        # Fixed sample counts keep DDP optimizer/collective counts equal even
        # when ranks have different game lengths or numbers of prefix chains.
        return _chunks(ordered, self.settings.ppo_minibatch_samples)

    def _update_ppo_minibatches(self, samples, *, critic: bool) -> dict[str, float]:
        epochs = self.settings.critic_epochs if critic else self.settings.policy_epochs
        aggregate: dict[str, float] = {}
        processed = steps = 0
        stop = False
        max_kl = max_clip = 0.
        stop_kl = stop_clip = False
        for _ in range(epochs):
            for batch in self._ppo_optimizer_batches(samples):
                metrics = (self._update_critic_batch(batch, epochs=1) if critic
                           else self._update_policy_batch(batch, epochs=1))
                for key, value in metrics.items():
                    aggregate[key] = aggregate.get(key, 0.) + value * len(batch)
                processed += len(batch)
                steps += int(metrics.get("optimizer/critic_steps" if critic else "optimizer/policy_steps", 1))
                if not critic:
                    max_kl = max(max_kl, metrics["policy/approx_kl_old"])
                    max_clip = max(max_clip, metrics["policy/clip_fraction"])
                    stop_kl = metrics["policy/approx_kl_old"] > self.settings.early_stop_kl_multiple * self.settings.target_kl
                    stop_clip = metrics["policy/clip_fraction"] > self.settings.early_stop_clip_fraction
                if not critic and (
                    stop_kl or stop_clip
                ):
                    stop = True
                    break
            if stop:
                break
        aggregate = {key: value / processed for key, value in aggregate.items()}
        name = "critic" if critic else "policy"
        aggregate[f"optimizer/{name}_epochs"] = processed / len(samples)
        aggregate[f"optimizer/{name}_steps"] = float(steps)
        if not critic:
            aggregate.update({"policy/minibatch_kl_max": max_kl,
                "policy/minibatch_clip_max": max_clip,
                "optimizer/policy_early_stopped": float(stop),
                "optimizer/policy_stop_kl": float(stop_kl),
                "optimizer/policy_stop_clip": float(stop_clip)})
        return aggregate

    def _update_policy_batch(
        self, groups: Sequence[PolicyGroup] | Sequence[PPOSample], *, epochs: int | None = None,
    ) -> dict[str, float]:
        if not self.policy.training:
            self.policy.train()
        coalesce = self.settings.algorithm == "ppo" and not self.distributed.enabled
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
                    error_if_nonfinite=self.settings.algorithm == "ppo",
                )
                grad_norm_value = self.distributed.reduce_float(float(grad_norm), operation="mean")
            self.grad_scaler.step(self.policy_optimizer)
            self.grad_scaler.update()
            # Gradients are not part of optimizer/checkpoint state.  Keeping
            # them alive through the next rollout pins allocator segments and
            # prevents the retired paged-KV arena from being returned at the
            # following update boundary.
            self.policy_optimizer.zero_grad(set_to_none=True)
            epochs_completed += 1
            aggregate = epoch_metrics
            if (
                epoch_metrics.get("policy/approx_kl_old", epoch_metrics.get("policy/kl_reference", 0.0))
                > self.settings.early_stop_kl_multiple * self.settings.target_kl
                or epoch_metrics["policy/clip_fraction"]
                > self.settings.early_stop_clip_fraction
            ):
                break
        aggregate["optimizer/policy_epochs"] = float(epochs_completed)
        aggregate["optimizer/policy_steps"] = float(epochs_completed)
        aggregate["optimizer/policy_grad_norm"] = grad_norm_value
        if self.settings.algorithm == "ppo":
            kl = aggregate.get("policy/approx_kl_old", 0.)
            clipped = aggregate.get("policy/clip_fraction", 0.)
            aggregate.update({"policy/minibatch_kl_max": kl, "policy/minibatch_clip_max": clipped,
                "optimizer/policy_stop_kl": float(kl > self.settings.early_stop_kl_multiple * self.settings.target_kl),
                "optimizer/policy_stop_clip": float(clipped > self.settings.early_stop_clip_fraction),
                "optimizer/policy_early_stopped": float(
                    kl > self.settings.early_stop_kl_multiple * self.settings.target_kl
                    or clipped > self.settings.early_stop_clip_fraction)})
        if self.settings.algorithm != "ppo" and aggregate.get("policy/kl_reference", 0.0) > 2 * self.settings.target_kl:
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
        self.ppo_rollout_metrics["ppo/pass_fraction"] = (
            pass_totals["ppo/voluntary_passes"] / max(1, pass_totals["ppo/pass_decisions"]))
        return normalize_advantages(
            samples, mean=self.ppo_rollout_metrics["ppo/raw_advantage_mean"],
            std=self.ppo_rollout_metrics["ppo/raw_advantage_std"],
            epsilon=self.settings.advantage_epsilon,
        )

    def _backward_critic_epoch(self, samples: Sequence[PPOSample], *, defer_metrics: bool = False) -> dict[str, float]:
        if self.critic_parallel is None or self.critic_optimizer is None:
            raise RuntimeError("PPO critic was not initialized")
        self.critic_optimizer.zero_grad(set_to_none=True)
        value_samples = [replace(s, state=s.value_state, value_state=None) if s.value_state is not None else s for s in samples]
        chunks = self._learner_chunks(value_samples)
        captured = self._try_learner_graph(chunks[0], critic=True) if len(chunks) == 1 else None
        if captured is not None:
            captured['optimizer/critic_learner_graph_fraction'] = 1.0
            return captured if defer_metrics else self._materialize_metrics(captured)
        metrics: dict[str, float] = {}
        metrics['optimizer/critic_learner_graph_fraction'] = 0.0
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
        if len(samples) > self.settings.ppo_minibatch_samples:
            return self._update_ppo_minibatches(samples, critic=True)
        return self._update_critic_batch(samples)

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
                    ("draw_rule_migration", "draw_objective_migration", "pass_rule_migration")), default=0)

    def _update_layout(self) -> dict[str, float]:
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
                    entropy_coefficient=self.settings.entropy_coefficient,
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
            "pass_rule_migration": self.pass_rule_migration,
            "draw_objective_migration": self.draw_objective_migration,
            "flag_capture_objective_migration": self.flag_capture_objective_migration,
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
        aggregated.actor_inference_seconds = self.distributed.reduce_float(
            metrics.actor_inference_seconds, operation="max"
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
        return min(self.local_anchor_batch, math.ceil(remaining / self.distributed.world_size))

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
                if self.settings.algorithm == "ppo":
                    self.phase = "ppo_trajectory_collection"
                    if self.settings.rollout_environment_workers > 1 and self._ppo_environment is None:
                        from .ppo_environment import ParallelPPOEnvironment
                        self._ppo_environment = ParallelPPOEnvironment(self.settings.rollout_environment_workers)
                    assert self.critic is not None
                    value_actor = FrozenValueActor(
                        self.critic, amp_dtype=self.amp_dtype,
                        max_batch_size=self.effective_actor_inference_batch,
                        deferred=self.settings.ppo_deferred_values,
                    )
                    samples, layout_outcomes, rollout_metrics = collect_ppo_samples(
                        self.pool, actor, value_actor, self.layout,
                        count=self._ppo_collection_count(), behavior_version=self.update,
                        environment=self._ppo_environment,
                        pipeline_groups=self.settings.ppo_pipeline_groups,
                        discount=self.settings.discount, gae_lambda=self.settings.gae_lambda,
                        draw_reward=self.settings.draw_reward,
                        flag_capture_reward=self.settings.flag_capture_reward,
                    )
                    groups = self._normalize_ppo_batch(samples)
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
                policy_backward_started = time.perf_counter()
                policy_metrics = self._update_policy(groups)
                policy_backward_seconds = self.distributed.reduce_float(
                    time.perf_counter() - policy_backward_started,
                    operation="max",
                )
                critic_metrics: dict[str, float] = {}
                critic_backward_seconds = 0.0
                if self.critic is not None:
                    self.phase = "critic_backward"
                    critic_started = time.perf_counter()
                    critic_metrics = self._update_critic(groups)
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
                    "model/action_input_dim": self.policy.action_encoder.projection.in_features,
                    "model/board_pass_input_dim": self.policy.board_encoder.pass_feature_dim,
                    "rules/max_passes_per_player": self.settings.max_passes_per_player,
                    "training/draw_reward": self.settings.draw_reward,
                    "training/flag_capture_reward": self.settings.flag_capture_reward,
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
                    **(self.ppo_rollout_metrics if self.settings.algorithm == "ppo" else {}),
                    **layout_metrics,
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
                    or self.distributed.enabled or self.settings.algorithm == "ppo"):
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
