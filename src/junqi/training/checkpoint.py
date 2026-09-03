"""Atomic, auto-resumable checkpoints with RNG and optimizer state."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any

import torch
from torch import nn


# Version 4 makes the deterministic dead-rule feature set an explicit,
# architecture-changing training variant.  Cross-variant resume is forbidden.
CHECKPOINT_FORMAT_VERSION = 4


def capture_rng_state(
    cuda_device: torch.device | str | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        if cuda_device is None:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        else:
            device = torch.device(cuda_device)
            if device.type == "cuda":
                state["torch_cuda_local"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(
    state: dict[str, Any],
    cuda_device: torch.device | str | None = None,
) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(
            [generator_state.cpu() for generator_state in state["torch_cuda"]]
        )
    elif torch.cuda.is_available() and "torch_cuda_local" in state:
        device = (
            torch.device(cuda_device)
            if cuda_device is not None
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
        reference_policy: nn.Module,
        reference_layout: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        layout_optimizer: torch.optim.Optimizer,
        trainer_state: dict[str, Any],
        config: dict[str, Any],
        archive: bool,
        reason: str,
        rng_state: dict[str, Any] | None = None,
    ) -> Path:
        if not isinstance(dead_rules_enabled, bool):
            raise ValueError("dead_rules_enabled must be a boolean")
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "update": update,
            "mode": mode,
            "dead_rules_enabled": dead_rules_enabled,
            "reason": reason,
            "policy": policy.state_dict(),
            "layout": layout.state_dict(),
            "reference_policy": reference_policy.state_dict(),
            "reference_layout": reference_layout.state_dict(),
            "policy_optimizer": policy_optimizer.state_dict(),
            "layout_optimizer": layout_optimizer.state_dict(),
            "trainer_state": trainer_state,
            "config": config,
            "rng_state": capture_rng_state() if rng_state is None else rng_state,
        }
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
    reference_policy: nn.Module,
    reference_layout: nn.Module,
    policy_optimizer: torch.optim.Optimizer,
    layout_optimizer: torch.optim.Optimizer,
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
    policy.load_state_dict(payload["policy"], strict=True)
    layout.load_state_dict(payload["layout"], strict=True)
    reference_policy.load_state_dict(payload["reference_policy"], strict=True)
    reference_layout.load_state_dict(payload["reference_layout"], strict=True)
    policy_optimizer.load_state_dict(payload["policy_optimizer"])
    layout_optimizer.load_state_dict(payload["layout_optimizer"])
    restore_rng_state(payload["rng_state"])
    return int(payload["update"]), dict(payload.get("trainer_state", {}))
