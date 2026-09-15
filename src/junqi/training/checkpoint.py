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


# Version 6 uses a five-number coordinate/player action projection and histories
# without event outcome features. Version 5's action embeddings are incompatible.
from .checkpoint_format import CHECKPOINT_FORMAT_VERSION, SUPPORTED_CHECKPOINT_FORMAT_VERSIONS


def require_current_checkpoint(payload: dict[str, Any]) -> None:
    if payload.get("format_version") not in SUPPORTED_CHECKPOINT_FORMAT_VERSIONS:
        raise ValueError(
            f"unsupported checkpoint format: {payload.get('format_version')!r}; "
            "the whole-board linear action architecture requires version 6, 7, 8 or 9 weights. "
            "Use a new run directory; old board/action encoder weights are incompatible."
        )
    if payload.get("format_version") in (8, 9):
        for name in ("policy", "critic", "reference_policy"):
            weights = payload.get(name) or {}
            action = weights.get("action_encoder.projection.weight")
            if action is not None and (action.ndim != 2 or action.shape[1] != 6):
                raise ValueError("version 8/9 checkpoint requires six action inputs")
            board = weights.get("board_encoder.projection.weight")
            if payload["format_version"] == 9 and board is not None:
                from .encoding import BOARD_CODE_VOCAB_SIZE
                expected = 129 * BOARD_CODE_VOCAB_SIZE + 3 + (75 if payload["dead_rules_enabled"] else 0) + 4
                if board.ndim != 2 or board.shape[1] != expected:
                    raise ValueError("version 9 checkpoint requires four pass inputs on the board encoder")


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
        inference_state: dict[str, Any] | None = None,
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
            "policy": inference_state["policy"] if inference_state is not None else policy.state_dict(),
            "layout": inference_state["layout"] if inference_state is not None else layout.state_dict(),
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
        self.write_manifest(payload)
        if archive:
            archive_path = self.directory / f"update_{update:09d}.pt"
            archive_temporary = self.directory / f".{archive_path.name}.tmp"
            torch.save(payload, archive_temporary)
            os.replace(archive_temporary, archive_path)
            self._prune_archives()
        return self.latest_path

    def write_manifest(self, payload):
        """Only the primary rank publishes counters read from the full payload."""
        cumulative = payload.get("trainer_state", {}).get("cumulative", {})
        manifest = {
            "format_version": payload["format_version"],
            "environment_plies": cumulative.get("environment_plies"),
            "cumulative": cumulative,
            "saved_at_unix": self.latest_path.stat().st_mtime,
            "latest": self.latest_path.name,
            **{name: payload.get(name) for name in ("update", "mode", "algorithm", "dead_rules_enabled", "reason")},
        }
        manifest_tmp = self.directory / ".manifest.json.tmp"
        manifest_tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(manifest_tmp, self.directory / "manifest.json")

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
        require_current_checkpoint(payload)
        return payload


def _extend_counter_optimizer_state(optimizer: torch.optim.Optimizer, model: nn.Module) -> None:
    """Preserve old Adam moments; new input rows/columns have zero moments."""
    for name, parameter in model.named_parameters():
        if name not in ("no_interaction_embedding.weight", "action_encoder.projection.weight", "board_encoder.projection.weight"):
            continue
        state = optimizer.state.get(parameter, {})
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(key)
            if (name == "no_interaction_embedding.weight" and value is not None and value.shape == (61, parameter.shape[1])
                    and parameter.shape[0] == 71):
                state[key] = torch.cat((value, value.new_zeros((10, value.shape[1]))), dim=0)
            elif (name == "action_encoder.projection.weight" and value is not None
                  and value.shape == (parameter.shape[0], 5) and parameter.shape[1] == 6):
                state[key] = torch.cat((value, value.new_zeros((value.shape[0], 1))), dim=1)
            elif (name == "board_encoder.projection.weight" and value is not None
                  and value.shape == (parameter.shape[0], parameter.shape[1] - 4)):
                state[key] = torch.cat((value, value.new_zeros((value.shape[0], 4))), dim=1)


def _legacy_critic_state(payload, critic):
    """Only known pre-v8 checkpoints may omit the shared draw-value head."""
    weights, optimizer = payload["critic"], payload["critic_optimizer"]
    if payload["format_version"] not in (6, 7):
        return weights, optimizer
    added = [name for name, _ in critic.named_parameters() if name.startswith("draw_value_head.")
             and name not in weights]
    if not added:
        return weights, optimizer
    if set(added) != {"draw_value_head.weight", "draw_value_head.bias"}:
        raise RuntimeError("legacy critic contains a partial draw-value head")
    groups = optimizer["param_groups"]
    if len(groups) != 1 or len(groups[0]["params"]) + len(added) != len(list(critic.parameters())):
        raise RuntimeError("legacy critic optimizer parameter order is incompatible")
    weights = dict(weights)
    parameters = dict(critic.named_parameters())
    for name in added:
        weights[name] = torch.zeros_like(parameters[name], device="cpu")
    next_id = max(groups[0]["params"], default=-1) + 1
    optimizer = {**optimizer, "param_groups": [
        {**groups[0], "params": [*groups[0]["params"], *range(next_id, next_id + len(added))]}]}
    return weights, optimizer


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
    require_current_checkpoint(payload)
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
    _extend_counter_optimizer_state(policy_optimizer, policy)
    layout_optimizer.load_state_dict(payload["layout_optimizer"])
    if critic is not None and critic_optimizer is not None:
        critic_weights, critic_moments = _legacy_critic_state(payload, critic)
        critic.load_state_dict(critic_weights, strict=True)
        critic_optimizer.load_state_dict(critic_moments)
        _extend_counter_optimizer_state(critic_optimizer, critic)
    restore_rng_state(payload["rng_state"], accelerator_device)
    return int(payload["update"]), dict(payload.get("trainer_state", {}))
