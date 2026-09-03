"""Resumable terminal-only self-play trainer for all three supported modes."""

from __future__ import annotations

import copy
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

from .checkpoint import CheckpointManager, restore_training_state
from .losses import layout_grpo_loss, policy_grpo_loss
from .metrics import MetricLogger
from .models import (
    GamePolicyTransformer,
    PieceConditionedLayoutPointerDecoder,
    layout_sample_from_trace,
    parameter_count,
)
from .rollout import (
    BaseGamePool,
    FrozenPolicyActor,
    LayoutOutcome,
    PolicyGroup,
    collect_policy_groups,
)
from .settings import TrainingSettings


T = TypeVar("T")


def _chunks(values: Sequence[T], size: int) -> list[Sequence[T]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


class SelfPlayTrainer:
    def __init__(
        self,
        settings: TrainingSettings,
        *,
        run_directory: str | Path | None = None,
        auto_resume: bool = True,
    ) -> None:
        self.settings = settings
        self.device = resolve_device(settings.device)
        if self.device.type == "cuda":
            torch.cuda.set_device(
                torch.cuda.current_device()
                if self.device.index is None
                else self.device.index
            )
        self.run_directory = settings.resolve_run_directory(run_directory).resolve()
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self._had_existing_run_artifacts = any(self.run_directory.iterdir())
        if not auto_resume and self._had_existing_run_artifacts:
            raise RuntimeError(
                "resume is disabled but the run directory is not empty; "
                "choose a new empty run directory"
            )
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
        self.logger = MetricLogger(self.run_directory, device=self.device)
        self.checkpoints = CheckpointManager(
            self.run_directory,
            keep_archives=settings.keep_checkpoint_archives,
        )
        self.stop_requested = False
        self.update = 0
        self.phase = "initializing"
        self.effective_policy_microbatch = settings.policy_microbatch
        self.effective_actor_inference_batch = settings.actor_inference_batch
        self.policy_lr_scale = 1.0
        self.cumulative: dict[str, int] = {
            "anchors": 0,
            "root_candidates": 0,
            "terminal_continuations": 0,
            "continuation_plies": 0,
            "base_plies": 0,
            "base_games": 0,
        }
        self.layout_buffer: list[LayoutOutcome] = []

        random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings.seed)
        if settings.enable_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self.policy = GamePolicyTransformer(settings.model).to(self.device)
        self.layout = PieceConditionedLayoutPointerDecoder(settings.model).to(
            self.device
        )
        self.reference_policy = copy.deepcopy(self.policy).eval().requires_grad_(False)
        self.reference_layout = copy.deepcopy(self.layout).eval().requires_grad_(False)
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=settings.policy_learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.05,
        )
        self.layout_optimizer = torch.optim.AdamW(
            self.layout.parameters(),
            lr=settings.layout_learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.05,
        )
        self.amp_dtype = self._amp_dtype()
        self.grad_scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_dtype is torch.float16
        )
        self.pool = BaseGamePool(
            settings.mode,
            pool_size=settings.base_game_pool_size,
            max_transitions=settings.model.max_transitions,
            max_game_plies=settings.max_game_plies,
            dead_rules_enabled=settings.dead_rules_enabled,
            seed=settings.seed + 17,
        )
        resumed = self._resume_if_available() if auto_resume else False
        self._write_run_config()
        self._install_signal_handlers()
        if not resumed:
            self.save_checkpoint(reason="initialized", archive=False)
        self.logger.event(
            "initialized mode=%s device=%s policy_params=%d layout_params=%d "
            "resume_update=%d dead_rules_enabled=%s target_continuation_plies=%s "
            "player_models=1(shared_across_all_seats)"
            % (
                settings.mode.value,
                self.device,
                parameter_count(self.policy),
                parameter_count(self.layout),
                self.update,
                settings.dead_rules_enabled,
                settings.target_continuation_plies,
            )
        )
        self.logger.start_resource_monitor(
            lambda: {
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
            },
            interval_seconds=settings.resource_monitor_interval_seconds,
        )
        self.phase = "ready"

    def _amp_dtype(self) -> torch.dtype | None:
        if self.device.type != "cuda" or self.settings.amp == "float32":
            return None
        if self.settings.amp == "bfloat16":
            if not torch.cuda.is_bf16_supported():
                self.logger.event("BF16 unsupported; falling back to float16")
                return torch.float16
            return torch.bfloat16
        return torch.float16

    def _write_run_config(self) -> None:
        destination = self.run_directory / "resolved_config.json"
        temporary = self.run_directory / ".resolved_config.json.tmp"
        temporary.write_text(
            json.dumps(
                self.settings.serializable(), ensure_ascii=False, indent=2, default=str
            ),
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    def _resume_if_available(self) -> bool:
        payload = self.checkpoints.load_latest(map_location=self.device)
        if payload is None:
            if self._had_existing_run_artifacts:
                raise RuntimeError(
                    "run directory contains prior artifacts but latest.pt is missing; "
                    "refusing to restart from random weights"
                )
            self.logger.event("no checkpoint found; starting a new run")
            return False
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
        )
        self.policy_lr_scale = float(trainer_state.get("policy_lr_scale", 1.0))
        restored_microbatch = int(
            trainer_state.get(
                "effective_policy_microbatch",
                self.settings.policy_microbatch,
            )
        )
        if restored_microbatch < self.settings.minimum_policy_microbatch:
            raise RuntimeError("checkpoint effective microbatch is below configured minimum")
        self.effective_policy_microbatch = min(
            restored_microbatch,
            self.settings.policy_microbatch,
        )
        restored_actor_batch = int(
            trainer_state.get(
                "effective_actor_inference_batch",
                self.settings.actor_inference_batch,
            )
        )
        if restored_actor_batch <= 0:
            raise RuntimeError("checkpoint effective actor batch must be positive")
        self.effective_actor_inference_batch = min(
            restored_actor_batch,
            self.settings.actor_inference_batch,
        )
        if "grad_scaler" in trainer_state:
            self.grad_scaler.load_state_dict(trainer_state["grad_scaler"])
        restored_cumulative = trainer_state.get("cumulative", {})
        self.cumulative.update(
            {key: int(value) for key, value in restored_cumulative.items()}
        )
        self.layout_buffer = [
            LayoutOutcome(
                sample=layout_sample_from_trace(
                    item["mode"],
                    item["position_indices"],
                    item["old_log_probs"],
                ),
                reward=float(item["reward"]),
                seat=int(item["seat"]),
                behavior_version=int(item["behavior_version"]),
            )
            for item in trainer_state.get("layout_buffer", [])
        ]
        if "base_game_pool" in trainer_state:
            self.pool.load_state_dict(trainer_state["base_game_pool"])
        elif "base_pool_rng_state" in trainer_state:
            # Backward compatibility with checkpoints written before unfinished
            # base games and their history windows were persisted.
            self.pool.rng.setstate(trainer_state["base_pool_rng_state"])
        self.logger.event(
            f"resumed checkpoint update={self.update}; models, optimizers, RNG, "
            "layout buffer, and unfinished base games restored"
        )
        return True

    def _install_signal_handlers(self) -> None:
        def request_stop(signum: int, _frame: Any) -> None:
            self.stop_requested = True
            self.logger.event(f"received signal {signum}; saving after current update")

        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(signum, request_stop)
            except (OSError, ValueError):
                pass

    def _learning_rate(self, update: int) -> float:
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
        return max(
            settings.minimum_learning_rate,
            scheduled * self.policy_lr_scale,
        )

    def _set_policy_lr(self, update: int) -> float:
        value = self._learning_rate(update)
        for group in self.policy_optimizer.param_groups:
            group["lr"] = value
        return value

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype or torch.float32,
            enabled=self.amp_dtype is not None,
        )

    def _backward_policy_epoch(
        self, groups: Sequence[PolicyGroup]
    ) -> dict[str, float]:
        """Accumulate one epoch so an OOM can safely restart before step()."""

        self.policy_optimizer.zero_grad(set_to_none=True)
        epoch_metrics: dict[str, float] = {}
        chunks = _chunks(groups, self.effective_policy_microbatch)
        for chunk in chunks:
            with self._autocast():
                output = policy_grpo_loss(
                    self.policy,
                    self.reference_policy,
                    chunk,
                    clip_epsilon=self.settings.clip_epsilon,
                    kl_coefficient=self.settings.kl_coefficient,
                    entropy_coefficient=self.settings.entropy_coefficient,
                )
                scaled_loss = output.loss * (len(chunk) / len(groups))
            self.grad_scaler.scale(scaled_loss).backward()
            for key, value in output.metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value * (
                    len(chunk) / len(groups)
                )
        return epoch_metrics

    def _update_policy(self, groups: Sequence[PolicyGroup]) -> dict[str, float]:
        self.policy.train()
        aggregate: dict[str, float] = {}
        epochs_completed = 0
        grad_norm_value = 0.0
        for _epoch in range(self.settings.policy_epochs):
            while True:
                try:
                    epoch_metrics = self._backward_policy_epoch(groups)
                    break
                except torch.OutOfMemoryError:
                    self.policy_optimizer.zero_grad(set_to_none=True)
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    current = self.effective_policy_microbatch
                    minimum = self.settings.minimum_policy_microbatch
                    if (
                        not self.settings.auto_reduce_microbatch_on_oom
                        or current <= minimum
                    ):
                        self.logger.event(
                            "CUDA OOM with policy microbatch=%d; automatic "
                            "fallback unavailable" % current
                        )
                        raise
                    reduced = max(minimum, current // 2)
                    self.effective_policy_microbatch = reduced
                    self.logger.event(
                        "CUDA OOM with policy microbatch=%d; retrying the "
                        "unapplied epoch with microbatch=%d"
                        % (current, reduced)
                    )
            self.grad_scaler.unscale_(self.policy_optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.settings.gradient_norm_clip
            )
            grad_norm_value = float(grad_norm)
            self.grad_scaler.step(self.policy_optimizer)
            self.grad_scaler.update()
            epochs_completed += 1
            aggregate = epoch_metrics
            if (
                epoch_metrics["policy/kl_reference"]
                > self.settings.early_stop_kl_multiple * self.settings.target_kl
                or epoch_metrics["policy/clip_fraction"]
                > self.settings.early_stop_clip_fraction
            ):
                break
        aggregate["optimizer/policy_epochs"] = float(epochs_completed)
        aggregate["optimizer/policy_grad_norm"] = grad_norm_value
        if aggregate.get("policy/kl_reference", 0.0) > 2 * self.settings.target_kl:
            self.policy_lr_scale = max(0.1, self.policy_lr_scale * 0.5)
            aggregate["optimizer/lr_scale_reduced"] = 1.0
        return aggregate

    def _update_layout(self) -> dict[str, float]:
        if len(self.layout_buffer) < self.settings.layout_outcomes_per_update:
            return {"layout/update_skipped_buffer_size": float(len(self.layout_buffer))}
        outcomes = self.layout_buffer[: self.settings.layout_outcomes_per_update]
        del self.layout_buffer[: self.settings.layout_outcomes_per_update]
        self.layout.train()
        self.layout_optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            output = layout_grpo_loss(
                self.layout,
                self.reference_layout,
                outcomes,
                clip_epsilon=self.settings.clip_epsilon,
                kl_coefficient=self.settings.kl_coefficient,
                entropy_coefficient=self.settings.entropy_coefficient,
                advantage_epsilon=self.settings.advantage_epsilon,
            )
        self.grad_scaler.scale(output.loss).backward()
        self.grad_scaler.unscale_(self.layout_optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.layout.parameters(), self.settings.gradient_norm_clip
        )
        self.grad_scaler.step(self.layout_optimizer)
        self.grad_scaler.update()
        output.metrics["optimizer/layout_grad_norm"] = float(grad_norm)
        output.metrics["layout/buffer_remaining"] = float(len(self.layout_buffer))
        return output.metrics

    def _trainer_state(self) -> dict[str, Any]:
        return {
            "policy_lr_scale": self.policy_lr_scale,
            "effective_policy_microbatch": self.effective_policy_microbatch,
            "effective_actor_inference_batch": (
                self.effective_actor_inference_batch
            ),
            "grad_scaler": self.grad_scaler.state_dict(),
            "cumulative": dict(self.cumulative),
            "layout_buffer": [
                {
                    "mode": item.sample.mode.value,
                    "position_indices": item.sample.position_indices,
                    "old_log_probs": item.sample.old_log_probs,
                    "reward": item.reward,
                    "seat": item.seat,
                    "behavior_version": item.behavior_version,
                }
                for item in self.layout_buffer
            ],
            "base_game_pool": self.pool.state_dict(),
        }

    def save_checkpoint(self, *, reason: str, archive: bool) -> Path:
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
            trainer_state=self._trainer_state(),
            config=self.settings.serializable(),
            archive=archive,
            reason=reason,
        )
        self.logger.event(f"checkpoint saved: {path} reason={reason}")
        return path

    def _continuation_target_reached(self) -> bool:
        target = self.settings.target_continuation_plies
        return (
            target is not None
            and self.cumulative["continuation_plies"] >= target
        )

    def train(self) -> None:
        try:
            while (
                self.update < self.settings.total_updates
                and not self._continuation_target_reached()
                and not self.stop_requested
            ):
                started = time.perf_counter()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                next_update = self.update + 1
                policy_lr = self._set_policy_lr(next_update)
                # Actor and learner are sequential.  During collection, every
                # seat calls these exact same Policy/Layout instances in eval
                # mode; weights cannot change until the terminal batch is done.
                # Therefore no per-player model and no redundant behavior copy
                # is created.  Stored behavior log-probs define pi_old for GRPO.
                self.policy.eval()
                self.layout.eval()
                self.phase = "base_game_collection"
                actor = FrozenPolicyActor(
                    self.policy,
                    amp_dtype=self.amp_dtype,
                    max_batch_size=self.effective_actor_inference_batch,
                )
                anchors, layout_outcomes, base_plies, completed_games = (
                    self.pool.collect_anchors(
                        actor,
                        self.layout,
                        count=self.settings.anchor_batch,
                        behavior_version=self.update,
                    )
                )
                self.phase = "terminal_rollouts"
                groups, rollout_metrics = collect_policy_groups(
                    anchors,
                    actor,
                    behavior_version=self.update,
                    advantage_epsilon=self.settings.advantage_epsilon,
                )
                if actor.max_batch_size < self.effective_actor_inference_batch:
                    self.logger.event(
                        "CUDA OOM reduced actor inference batch from %d to %d "
                        "during rollout"
                        % (
                            self.effective_actor_inference_batch,
                            actor.max_batch_size,
                        )
                    )
                    self.effective_actor_inference_batch = actor.max_batch_size
                rollout_metrics.base_plies = base_plies
                rollout_metrics.base_games_completed = completed_games
                self.phase = "policy_backward"
                policy_metrics = self._update_policy(groups)
                self.layout_buffer.extend(layout_outcomes)
                layout_metrics: dict[str, float] = {}
                if next_update % self.settings.layout_update_interval == 0:
                    self.phase = "layout_backward"
                    layout_metrics = self._update_layout()
                self.update = next_update

                for key, value in (
                    ("anchors", rollout_metrics.anchors),
                    ("root_candidates", rollout_metrics.root_candidates),
                    (
                        "terminal_continuations",
                        rollout_metrics.terminal_continuations,
                    ),
                    ("continuation_plies", rollout_metrics.continuation_plies),
                    ("base_plies", base_plies),
                    ("base_games", completed_games),
                ):
                    self.cumulative[key] += int(value)
                if self.update % self.settings.reference_refresh_updates == 0:
                    self.reference_policy.load_state_dict(self.policy.state_dict())
                    self.reference_layout.load_state_dict(self.layout.state_dict())
                    self.logger.event(f"reference refreshed at update {self.update}")

                metrics: dict[str, float | int | str] = {
                    "mode": self.settings.mode.value,
                    "model/dead_rules_enabled": int(
                        self.settings.dead_rules_enabled
                    ),
                    "model/shared_policy_instances_for_players": 1,
                    "model/shared_layout_instances_for_players": 1,
                    "training/target_continuation_plies": (
                        -1
                        if self.settings.target_continuation_plies is None
                        else self.settings.target_continuation_plies
                    ),
                    "optimizer/effective_policy_microbatch": (
                        self.effective_policy_microbatch
                    ),
                    "rollout/effective_actor_inference_batch": (
                        self.effective_actor_inference_batch
                    ),
                    "optimizer/policy_lr": policy_lr,
                    "timing/update_seconds": time.perf_counter() - started,
                    **rollout_metrics.as_dict(),
                    **policy_metrics,
                    **layout_metrics,
                }
                metrics.update(
                    {
                        f"cumulative/{key}": value
                        for key, value in self.cumulative.items()
                    }
                )
                if self.update % self.settings.metrics_every_updates == 0:
                    self.logger.log(self.update, metrics)

                if self.update % self.settings.checkpoint_every_updates == 0:
                    self.phase = "checkpoint"
                    self.save_checkpoint(
                        reason="periodic",
                        archive=(
                            self.update % self.settings.archive_every_updates == 0
                        ),
                    )
                self.phase = "ready"
            self.phase = "checkpoint"
            self.save_checkpoint(reason="completed_or_stopped", archive=True)
        except BaseException:
            # Preserve the last fully applied optimizer update before surfacing the
            # exception.  SIGKILL and power loss cannot be intercepted, hence the
            # frequent atomic latest checkpoint above.
            self.phase = "emergency_checkpoint"
            self.save_checkpoint(reason="emergency_exception", archive=False)
            raise
        finally:
            self.phase = "stopped"
            self.logger.close()
