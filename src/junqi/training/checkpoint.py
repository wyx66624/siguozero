"""Atomic, auto-resumable checkpoints with RNG and optimizer state."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any

import torch
from torch import nn

from .accelerator import (
    ACCELERATOR_DEVICE_TYPES,
    accelerator_available,
    accelerator_module,
    is_accelerator,
)


# Version 4 makes the deterministic dead-rule feature set an explicit,
# architecture-changing training variant.  Cross-variant resume is forbidden.
CHECKPOINT_FORMAT_VERSION = 4


def capture_rng_state(
    accelerator_device: torch.device | str | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if accelerator_device is not None:
        device = torch.device(accelerator_device)
        if is_accelerator(device) and accelerator_available(device.type):
            module = accelerator_module(device.type)
            if device.index is None:
                device = torch.device(device.type, module.current_device())
            state["torch_accelerator"] = {
                "device_type": device.type,
                "device_index": device.index,
                "rng_state": module.get_rng_state(device).cpu(),
            }
    elif torch.cuda.is_available():
        # Preserve compatibility for callers that relied on capturing all CUDA
        # generators before the generic accelerator checkpoint was added.
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(
    state: dict[str, Any],
    accelerator_device: torch.device | str | None = None,
) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    accelerator_state = state.get("torch_accelerator")
    if isinstance(accelerator_state, dict):
        saved_type = str(accelerator_state.get("device_type", ""))
        if saved_type not in ACCELERATOR_DEVICE_TYPES:
            raise RuntimeError("checkpoint contains an invalid accelerator RNG type")
        target = (
            torch.device(accelerator_device)
            if accelerator_device is not None
            else torch.device(
                saved_type,
                int(accelerator_state.get("device_index", 0)),
            )
        )
        if target.type == saved_type and accelerator_available(saved_type):
            accelerator_module(saved_type).set_rng_state(
                accelerator_state["rng_state"].cpu(), target
            )
    elif torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(
            [generator_state.cpu() for generator_state in state["torch_cuda"]]
        )
    elif torch.cuda.is_available() and "torch_cuda_local" in state:
        device = (
            torch.device(accelerator_device)
            if accelerator_device is not None
            else torch.device("cuda", torch.cuda.current_device())
        )
        torch.cuda.set_rng_state(state["torch_cuda_local"].cpu(), device)


class CheckpointManager:
    def __init__(
        self,
        run_directory: str | Path,
        *,
        keep_archives: int = 5,
    ) -> None:
        self.run_directory = Path(run_directory).resolve()
        self.directory = self.run_directory / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.directory / "latest.pt"
        self.keep_archives = keep_archives

    def save(
        self,
        *,
        update: int,
        mode: str,
        dead_rules_enabled: bool,
        policy: nn.Module,
        layout: nn.Module,
        reference_policy: nn.Module | None,
        reference_layout: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        layout_optimizer: torch.optim.Optimizer,
        trainer_state: dict[str, Any],
        config: dict[str, Any],
        archive: bool,
        reason: str,
        rng_state: dict[str, Any] | None = None,
        algorithm: str = "grpo",
        critic: nn.Module | None = None,
        critic_optimizer: torch.optim.Optimizer | None = None,
    ) -> Path:
        if not isinstance(dead_rules_enabled, bool):
            raise ValueError("dead_rules_enabled must be a boolean")
        if algorithm not in ("grpo", "ppo"):
            raise ValueError("invalid checkpoint training algorithm")
        if (algorithm == "ppo") != (critic is not None and critic_optimizer is not None):
            raise ValueError("PPO checkpoints require a critic and its optimizer")
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "update": update,
            "mode": mode,
            "algorithm": algorithm,
            "dead_rules_enabled": dead_rules_enabled,
            "reason": reason,
            "policy": policy.state_dict(),
            "layout": layout.state_dict(),
            "reference_policy": (
                None if reference_policy is None else reference_policy.state_dict()
            ),
            "reference_layout": reference_layout.state_dict(),
            "policy_optimizer": policy_optimizer.state_dict(),
            "layout_optimizer": layout_optimizer.state_dict(),
            "trainer_state": trainer_state,
            "config": config,
            "rng_state": capture_rng_state() if rng_state is None else rng_state,
        }
        if critic is not None and critic_optimizer is not None:
            payload["critic"] = critic.state_dict()
            payload["critic_optimizer"] = critic_optimizer.state_dict()
        temporary = self.directory / ".latest.pt.tmp"
        torch.save(payload, temporary)
        os.replace(temporary, self.latest_path)
        if archive:
            archive_path = self.directory / f"update_{update:09d}.pt"
            archive_temporary = self.directory / f".{archive_path.name}.tmp"
            torch.save(payload, archive_temporary)
            os.replace(archive_temporary, archive_path)
            self._prune_archives()
        manifest = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "latest": self.latest_path.name,
            "update": update,
            "mode": mode,
            "algorithm": algorithm,
            "dead_rules_enabled": dead_rules_enabled,
            "reason": reason,
        }
        manifest_tmp = self.directory / ".manifest.json.tmp"
        manifest_tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(manifest_tmp, self.directory / "manifest.json")
        return self.latest_path

    def _prune_archives(self) -> None:
        if self.keep_archives <= 0:
            return
        archives = sorted(self.directory.glob("update_*.pt"))
        for stale in archives[: -self.keep_archives]:
            stale.unlink()

    def load_latest(
        self,
        *,
        map_location: torch.device | str,
    ) -> dict[str, Any] | None:
        if not self.latest_path.exists():
            return None
        payload = torch.load(
            self.latest_path,
            map_location=map_location,
            weights_only=False,
        )
        if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError(
                f"unsupported checkpoint format: {payload.get('format_version')!r}"
            )
        return payload


def restore_training_state(
    payload: dict[str, Any],
    *,
    expected_mode: str,
    expected_dead_rules_enabled: bool,
    policy: nn.Module,
    layout: nn.Module,
    reference_policy: nn.Module | None,
    reference_layout: nn.Module,
    policy_optimizer: torch.optim.Optimizer,
    layout_optimizer: torch.optim.Optimizer,
    accelerator_device: torch.device | str | None = None,
    expected_algorithm: str = "grpo",
    critic: nn.Module | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, dict[str, Any]]:
    if payload["mode"] != expected_mode:
        raise RuntimeError(
            f"checkpoint mode {payload['mode']!r} does not match {expected_mode!r}"
        )
    if not isinstance(expected_dead_rules_enabled, bool):
        raise ValueError("expected_dead_rules_enabled must be a boolean")
    if payload.get("dead_rules_enabled") is not expected_dead_rules_enabled:
        raise RuntimeError(
            "checkpoint dead-rule variant does not match this training run"
        )
    if payload.get("algorithm", "grpo") != expected_algorithm:
        raise RuntimeError(
            "checkpoint algorithm does not match this training run; "
            "use --init-from with a new run directory for an intentional GRPO-to-PPO migration"
        )
    if expected_algorithm == "ppo":
        if (critic is None or critic_optimizer is None
                or "critic" not in payload or "critic_optimizer" not in payload):
            raise RuntimeError("PPO checkpoint is missing critic training state")
    policy.load_state_dict(payload["policy"], strict=True)
    layout.load_state_dict(payload["layout"], strict=True)
    if reference_policy is not None:
        reference_policy.load_state_dict(payload["reference_policy"], strict=True)
    reference_layout.load_state_dict(payload["reference_layout"], strict=True)
    policy_optimizer.load_state_dict(payload["policy_optimizer"])
    layout_optimizer.load_state_dict(payload["layout_optimizer"])
    if critic is not None and critic_optimizer is not None:
        critic.load_state_dict(payload["critic"], strict=True)
        critic_optimizer.load_state_dict(payload["critic_optimizer"])
    restore_rng_state(payload["rng_state"], accelerator_device)
    return int(payload["update"]), dict(payload.get("trainer_state", {}))
