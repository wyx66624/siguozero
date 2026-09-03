"""Resumable terminal-only self-play trainer for all three supported modes."""

from __future__ import annotations

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

from .checkpoint import (
    CheckpointManager,
    capture_rng_state,
    restore_rng_state,
    restore_training_state,
)
from .distributed import DistributedContext
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
    RolloutMetrics,
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
        distributed: DistributedContext | None = None,
    ) -> None:
        self.settings = settings
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
        if self.device.type == "cuda":
            torch.cuda.set_device(
                torch.cuda.current_device()
                if self.device.index is None
                else self.device.index
            )
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
        self.phase = "initializing"
        self.effective_policy_microbatch = min(
            settings.policy_microbatch, self.local_anchor_batch
        )
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
        self.distributed.broadcast_module(self.policy)
        self.distributed.broadcast_module(self.layout)
        self.reference_policy = copy.deepcopy(self.policy).eval().requires_grad_(False)
        self.reference_layout = copy.deepcopy(self.layout).eval().requires_grad_(False)
        self.policy_parallel: nn.Module
        if self.distributed.enabled:
            self.policy_parallel = DistributedDataParallel(
                self.policy,
                device_ids=(
                    [self.distributed.local_rank]
                    if self.device.type == "cuda"
                    else None
                ),
                output_device=(
                    self.distributed.local_rank
                    if self.device.type == "cuda"
                    else None
                ),
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        else:
            self.policy_parallel = self.policy
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
        self.layout_grad_scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_dtype is torch.float16
        )
        self._seed_runtime()
        self.pool = BaseGamePool(
            settings.mode,
            pool_size=self.local_base_game_pool_size,
            max_transitions=settings.model.max_transitions,
            max_game_plies=settings.max_game_plies,
            dead_rules_enabled=settings.dead_rules_enabled,
            seed=settings.seed + 17 + 1_000_003 * self.distributed.rank,
        )
        resumed = self._resume_if_available() if auto_resume else False
        self._write_run_config()
        self._install_signal_handlers()
        if not resumed:
            self.save_checkpoint(reason="initialized", archive=False)
        self.logger.event(
            "initialized mode=%s device=%s policy_params=%d layout_params=%d "
            "resume_update=%d dead_rules_enabled=%s target_continuation_plies=%s "
            "player_models=1(shared_across_all_seats) rank=%d/%d "
            "local_anchor_batch=%d local_pool=%d per_device_microbatch=%d"
            % (
                settings.mode.value,
                self.device,
                parameter_count(self.policy),
                parameter_count(self.layout),
                self.update,
                settings.dead_rules_enabled,
                settings.target_continuation_plies,
                self.distributed.rank,
                self.distributed.world_size,
                self.local_anchor_batch,
                self.local_base_game_pool_size,
                self.effective_policy_microbatch,
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
                "distributed/rank": self.distributed.rank,
                "distributed/world_size": self.distributed.world_size,
            },
            interval_seconds=settings.resource_monitor_interval_seconds,
        )
        self.phase = "ready"

    def _seed_runtime(self) -> None:
        runtime_seed = self.settings.seed + 1_000_003 * self.distributed.rank
        random.seed(runtime_seed)
        torch.manual_seed(runtime_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(runtime_seed)

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
        if not self.distributed.primary:
            return
        destination = self.run_directory / "resolved_config.json"
        temporary = self.run_directory / ".resolved_config.json.tmp"
        resolved = self.settings.serializable()
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
        self.effective_actor_inference_batch = min(
            restored_actor_batch,
            self.settings.actor_inference_batch,
        )
        if "grad_scaler" in trainer_state:
            self.grad_scaler.load_state_dict(trainer_state["grad_scaler"])
        if "layout_grad_scaler" in trainer_state:
            self.layout_grad_scaler.load_state_dict(
                trainer_state["layout_grad_scaler"]
            )
        restored_cumulative = trainer_state.get("cumulative", {})
        self.cumulative.update(
            {key: int(value) for key, value in restored_cumulative.items()}
        )
        restored_layout_buffer = [
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
        self.layout_buffer = (
            restored_layout_buffer if self.distributed.primary else []
        )
        distributed_state = trainer_state.get("distributed")
        if distributed_state is not None:
            saved_world_size = int(distributed_state["world_size"])
            rank_states = distributed_state["rank_states"]
            if len(rank_states) != saved_world_size:
                raise RuntimeError("checkpoint rank-state count is inconsistent")
            if saved_world_size == self.distributed.world_size:
                local_state = rank_states[self.distributed.rank]
                self.pool.load_state_dict(local_state["base_game_pool"])
                restore_rng_state(local_state["rng_state"], self.device)
            else:
                saved_pools = [item["base_game_pool"] for item in rank_states]
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
            self.pool.load_state_dict(trainer_state["base_game_pool"])
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
        ordered_groups = (
            sorted(groups, key=lambda group: len(group.state.records))
            if self.settings.learner_length_bucketing
            else list(groups)
        )
        chunks = _chunks(ordered_groups, self.effective_policy_microbatch)
        padded_tokens = sum(
            len(chunk) * max(len(group.state.records) for group in chunk)
            for chunk in chunks
        )
        real_tokens = sum(len(group.state.records) for group in ordered_groups)
        epoch_metrics["optimizer/temporal_padding_fraction"] = (
            1.0 - real_tokens / max(padded_tokens, 1)
        )
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
                    output = policy_grpo_loss(
                        self.policy_parallel,
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
            epoch_metrics = self.distributed.mean_metrics(epoch_metrics)
            self.grad_scaler.unscale_(self.policy_optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.settings.gradient_norm_clip
            )
            grad_norm_value = self.distributed.reduce_float(
                float(grad_norm), operation="mean"
            )
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
        self.layout_grad_scaler.scale(output.loss).backward()
        self.layout_grad_scaler.unscale_(self.layout_optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.layout.parameters(), self.settings.gradient_norm_clip
        )
        self.layout_grad_scaler.step(self.layout_optimizer)
        self.layout_grad_scaler.update()
        output.metrics["optimizer/layout_grad_norm"] = float(grad_norm)
        output.metrics["layout/buffer_remaining"] = float(len(self.layout_buffer))
        return output.metrics

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
            "policy_lr_scale": self.policy_lr_scale,
            "effective_policy_microbatch": self.effective_policy_microbatch,
            "effective_actor_inference_batch": (
                self.effective_actor_inference_batch
            ),
            "grad_scaler": self.grad_scaler.state_dict(),
            "layout_grad_scaler": self.layout_grad_scaler.state_dict(),
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
                rng_state=(
                    capture_rng_state(self.device)
                    if self.distributed.enabled
                    else None
                ),
            )
            self.logger.event(f"checkpoint saved: {path} reason={reason}")
        self.distributed.barrier()
        return path

    def _continuation_target_reached(self) -> bool:
        target = self.settings.target_continuation_plies
        return (
            target is not None
            and self.cumulative["continuation_plies"] >= target
        )

    def _aggregate_rollout_metrics(
        self, metrics: RolloutMetrics
    ) -> RolloutMetrics:
        summed_fields = (
            "anchors",
            "root_candidates",
            "terminal_continuations",
            "continuation_plies",
            "base_plies",
            "base_games_completed",
            "wins",
            "draws",
            "losses",
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
        aggregated.actor_inference_seconds = self.distributed.reduce_float(
            metrics.actor_inference_seconds, operation="max"
        )
        aggregated.environment_step_seconds = self.distributed.reduce_float(
            metrics.environment_step_seconds, operation="max"
        )
        return aggregated

    def train(self) -> None:
        try:
            while (
                self.update < self.settings.total_updates
                and not self._continuation_target_reached()
            ):
                if self.distributed.any(self.stop_requested):
                    break
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
                base_collection_started = time.perf_counter()
                anchors, layout_outcomes, base_plies, completed_games = (
                    self.pool.collect_anchors(
                        actor,
                        self.layout,
                        count=self.local_anchor_batch,
                        behavior_version=self.update,
                    )
                )
                base_collection_seconds = self.distributed.reduce_float(
                    time.perf_counter() - base_collection_started,
                    operation="max",
                )
                self.phase = "terminal_rollouts"
                groups, rollout_metrics = collect_policy_groups(
                    anchors,
                    actor,
                    behavior_version=self.update,
                    advantage_epsilon=self.settings.advantage_epsilon,
                    anchor_wave_size=self.settings.rollout_anchor_wave_size,
                    environment_workers=self.settings.rollout_environment_workers,
                )
                rollout_metrics.base_plies = base_plies
                rollout_metrics.base_games_completed = completed_games
                rollout_metrics = self._aggregate_rollout_metrics(rollout_metrics)
                encoding_metrics = self.distributed.mean_metrics(
                    self.policy.board_encoding_metrics()
                )
                previous_actor_batch = self.effective_actor_inference_batch
                self.effective_actor_inference_batch = self.distributed.reduce_int(
                    actor.max_batch_size, operation="min"
                )
                if self.effective_actor_inference_batch < previous_actor_batch:
                    self.logger.event(
                        "CUDA OOM reduced actor inference batch from %d to %d "
                        "during rollout"
                        % (
                            previous_actor_batch,
                            self.effective_actor_inference_batch,
                        )
                    )
                self.phase = "policy_backward"
                policy_backward_started = time.perf_counter()
                policy_metrics = self._update_policy(groups)
                policy_backward_seconds = self.distributed.reduce_float(
                    time.perf_counter() - policy_backward_started,
                    operation="max",
                )
                gathered_layouts = self.distributed.gather_object(layout_outcomes)
                if self.distributed.primary:
                    for rank_outcomes in gathered_layouts or []:
                        self.layout_buffer.extend(rank_outcomes)
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
                    "distributed/world_size": self.distributed.world_size,
                    "distributed/local_anchor_batch": self.local_anchor_batch,
                    "distributed/model_replicas": self.distributed.world_size,
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
                    "timing/base_collection_seconds": base_collection_seconds,
                    "timing/policy_backward_seconds": policy_backward_seconds,
                    "timing/layout_backward_seconds": layout_backward_seconds,
                    "timing/update_seconds": self.distributed.reduce_float(
                        time.perf_counter() - started, operation="max"
                    ),
                    **rollout_metrics.as_dict(),
                    **encoding_metrics,
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
            if self.distributed.enabled:
                self.logger.event(
                    "distributed exception: retaining the last completed atomic "
                    "checkpoint; no potentially asymmetric emergency save"
                )
            else:
                self.save_checkpoint(reason="emergency_exception", archive=False)
            raise
        finally:
            self.phase = "stopped"
            self.logger.close()
