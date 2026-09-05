"""Measure policy-step time and peak CUDA/NPU memory for a chosen context.

This is a capacity probe, not a gameplay-quality benchmark.  It constructs a
valid player-view state, expands it to the requested context length, evaluates
all legal actions, runs backward, and takes one AdamW step.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import torch

from junqi.training.accelerator import (
    empty_cache,
    get_device_name,
    is_accelerator,
    is_bf16_supported,
    manual_seed_all,
    max_memory_allocated,
    memory_allocated,
    memory_reserved,
    reset_peak_memory_stats,
    resolve_device,
    synchronize,
)
from junqi.training.encoding import (
    ACTION_PLAYER_PAD,
    ActionFeatures,
    GameHistory,
    PolicyState,
    StateTokenRecord,
)
from junqi.training.models import (
    GamePolicyTransformer,
    ModelConfig,
    PieceConditionedLayoutPointerDecoder,
    parameter_count,
)
from junqi.training.modes import TrainingMode, new_game, normalize_mode
from junqi.training.settings import TrainingSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in TrainingMode],
        default=TrainingMode.FOUR_DARK.value,
    )
    parser.add_argument(
        "--model-scale",
        choices=("bootstrap", "main", "extended"),
        default="bootstrap",
    )
    parser.add_argument("--context-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    dead_rules = parser.add_mutually_exclusive_group()
    dead_rules.add_argument(
        "--dead-rules", dest="dead_rules_enabled", action="store_true"
    )
    dead_rules.add_argument(
        "--no-dead-rules", dest="dead_rules_enabled", action="store_false"
    )
    parser.set_defaults(dead_rules_enabled=True)
    parser.add_argument(
        "--policy-only",
        action="store_true",
        help="do not keep Layout and reference copies resident",
    )
    return parser.parse_args()


def model_config(
    scale: str, mode: TrainingMode, *, dead_rules_enabled: bool
) -> ModelConfig:
    if scale == "main":
        return ModelConfig.main(dead_rules_enabled=dead_rules_enabled)
    if scale == "extended":
        return ModelConfig.extended(dead_rules_enabled=dead_rules_enabled)
    config_path = Path(__file__).parents[1] / "configs" / "bootstrap.yaml"
    return TrainingSettings.from_yaml(
        config_path, mode, dead_rules_enabled=dead_rules_enabled
    ).model


def synthetic_state(
    mode: TrainingMode,
    config: ModelConfig,
    context_tokens: int,
) -> PolicyState:
    if not 1 <= context_tokens <= config.max_sequence_tokens:
        raise ValueError(
            f"context tokens must be in 1..{config.max_sequence_tokens}"
        )
    game = new_game(
        mode,
        seed=20260902,
        max_plies=2000,
        dead_rules_enabled=config.dead_rules_enabled,
    )
    original = GameHistory.initialize(
        game, mode, max_transitions=config.max_transitions
    ).state_for(game)
    initial = original.records[0]
    source, destination = original.legal_actions[0]
    transition = StateTokenRecord(
        board_codes=initial.board_codes,
        known_casualty_bits=initial.known_casualty_bits,
        action=ActionFeatures(
            source=source,
            destination=destination,
            actor=0,
            combat=0,
            was_attack=0,
            flag_captured_owner=ACTION_PLAYER_PAD,
            newly_revealed_count=0,
            eliminated_count=0,
        ),
        no_interaction_plies=0,
        active_mask=initial.active_mask,
        revealed_mask=initial.revealed_mask,
        current_player=initial.current_player,
    )
    return PolicyState(
        mode=mode,
        records=(initial,) + (transition,) * (context_tokens - 1),
        legal_actions=original.legal_actions,
    )


def initialize_optimizer_state(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    dummy = sum(parameter.sum() * 0.0 for parameter in module.parameters())
    dummy.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def main() -> None:
    args = parse_args()
    mode = normalize_mode(args.mode)
    device = resolve_device(args.device)
    if not is_accelerator(device):
        raise RuntimeError("this benchmark requires an available CUDA or NPU device")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    torch.manual_seed(20260902)
    manual_seed_all(device.type, 20260902)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    config = model_config(
        args.model_scale,
        mode,
        dead_rules_enabled=args.dead_rules_enabled,
    )
    state = synthetic_state(mode, config, args.context_tokens)
    states = [state] * args.batch_size
    action_groups = [state.legal_actions] * args.batch_size

    policy = GamePolicyTransformer(config).to(device).train()
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    resident_parameters = parameter_count(policy)
    reference_policy = None
    layout = None
    reference_layout = None
    layout_optimizer = None
    if not args.policy_only:
        layout = PieceConditionedLayoutPointerDecoder(config).to(device)
        reference_policy = copy.deepcopy(policy).eval().requires_grad_(False)
        reference_layout = copy.deepcopy(layout).eval().requires_grad_(False)
        layout_optimizer = torch.optim.AdamW(layout.parameters(), lr=5e-5)
        initialize_optimizer_state(layout, layout_optimizer)
        resident_parameters = (
            parameter_count(policy)
            + parameter_count(reference_policy)
            + parameter_count(layout)
            + parameter_count(reference_layout)
        )

    empty_cache(device)
    reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    amp_dtype = torch.bfloat16 if is_bf16_supported(device) else torch.float16
    with torch.autocast(device_type=device.type, dtype=amp_dtype):
        if reference_policy is not None:
            with torch.no_grad():
                reference_policy.log_probs_for_action_groups(states, action_groups)
        logs = policy.log_probs_for_action_groups(states, action_groups)
        loss = -torch.stack([values.mean() for values in logs]).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    policy_optimizer.step()
    policy_optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    elapsed = time.perf_counter() - started

    result = {
        "accelerator": get_device_name(device),
        "device_type": device.type,
        "torch": torch.__version__,
        "mode": mode.value,
        "model_scale": args.model_scale,
        "dead_rules_enabled": config.dead_rules_enabled,
        "context_tokens": args.context_tokens,
        "batch_size": args.batch_size,
        "full_training_stack_resident": not args.policy_only,
        "policy_parameters": parameter_count(policy),
        "resident_model_parameters": resident_parameters,
        "legal_actions_per_state": len(state.legal_actions),
        "step_seconds": elapsed,
        "memory_allocated_gib": memory_allocated(device) / 2**30,
        "memory_reserved_gib": memory_reserved(device) / 2**30,
        "peak_memory_allocated_gib": max_memory_allocated(device) / 2**30,
        "loss": float(loss.detach()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
