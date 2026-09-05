"""Benchmark the real terminal-rollout path, including rules and KV caching."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path

import torch

from junqi.training.encoding import GameHistory
from junqi.training.models import (
    GamePolicyTransformer,
    ModelConfig,
    PieceConditionedLayoutPointerDecoder,
    parameter_count,
)
from junqi.training.modes import TrainingMode, new_game, normalize_mode
from junqi.training.rollout import AnchorSnapshot, FrozenPolicyActor, collect_policy_groups
from junqi.training.settings import TrainingSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[mode.value for mode in TrainingMode], default="two_player")
    parser.add_argument("--model-scale", choices=("tiny", "bootstrap", "main", "extended"), default="bootstrap")
    parser.add_argument("--anchors", type=int, default=4)
    parser.add_argument("--max-game-plies", type=int, default=32)
    parser.add_argument("--actor-batch", type=int, default=64)
    parser.add_argument("--anchor-wave", type=int, default=8)
    parser.add_argument("--environment-workers", type=int, default=2)
    parser.add_argument("--temporal-cache-entries", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--full-stack", action="store_true")
    parser.add_argument("--incremental", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dead-rules", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def make_config(args: argparse.Namespace, mode: TrainingMode) -> ModelConfig:
    if args.model_scale == "tiny":
        base = ModelConfig.tiny(dead_rules_enabled=args.dead_rules)
    elif args.model_scale == "main":
        base = ModelConfig.main(dead_rules_enabled=args.dead_rules)
    elif args.model_scale == "extended":
        base = ModelConfig.extended(dead_rules_enabled=args.dead_rules)
    else:
        base = TrainingSettings.from_yaml(
            Path(__file__).parents[1] / "configs" / "bootstrap.yaml",
            mode,
            dead_rules_enabled=args.dead_rules,
        ).model
    return replace(
        base,
        incremental_inference=args.incremental,
        inference_temporal_cache_entries=args.temporal_cache_entries,
    )


def main() -> None:
    args = parse_args()
    if args.anchors <= 0 or args.max_game_plies <= 0:
        raise ValueError("anchors and max-game-plies must be positive")
    mode = normalize_mode(args.mode)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config = make_config(args, mode)
    torch.manual_seed(20260903)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260903)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    policy = GamePolicyTransformer(config).to(device).eval()
    residents: list[object] = []
    if args.full_stack:
        reference_policy = copy.deepcopy(policy).requires_grad_(False)
        layout = PieceConditionedLayoutPointerDecoder(config).to(device)
        reference_layout = copy.deepcopy(layout).requires_grad_(False)
        policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
        layout_optimizer = torch.optim.AdamW(layout.parameters(), lr=5e-5)
        for module, optimizer in (
            (policy, policy_optimizer),
            (layout, layout_optimizer),
        ):
            module.train()
            dummy = sum(parameter.sum() * 0.0 for parameter in module.parameters())
            dummy.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            module.eval()
        residents.extend(
            [reference_policy, layout, reference_layout, policy_optimizer, layout_optimizer]
        )
    anchors: list[AnchorSnapshot] = []
    for index in range(args.anchors):
        game = new_game(
            mode,
            seed=20260903 + index,
            max_plies=args.max_game_plies,
            dead_rules_enabled=args.dead_rules,
        )
        history = GameHistory.initialize(
            game, mode, max_transitions=config.max_transitions
        )
        root = game.current_player
        assert root is not None
        anchors.append(
            AnchorSnapshot(
                game=game,
                history=history,
                state=history.state_for(game, root),
                root_player=root,
            )
        )

    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    actor = FrozenPolicyActor(
        policy, amp_dtype=amp_dtype, max_batch_size=args.actor_batch
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        resident_allocated = torch.cuda.memory_allocated(device) / 2**30
    else:
        resident_allocated = 0.0
    groups, metrics = collect_policy_groups(
        anchors,
        actor,
        behavior_version=0,
        anchor_wave_size=args.anchor_wave,
        environment_workers=args.environment_workers,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    output = {
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "mode": mode.value,
        "model_scale": args.model_scale,
        "policy_parameters": parameter_count(policy),
        "incremental_inference": args.incremental,
        "anchors": len(groups),
        "max_game_plies": args.max_game_plies,
        "actor_batch": actor.max_batch_size,
        "anchor_wave": args.anchor_wave,
        "full_training_stack_resident": args.full_stack,
        "resident_allocated_gib": resident_allocated,
        **metrics.as_dict(),
        **policy.board_encoding_metrics(),
        "cuda_peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else 0.0
        ),
        "cuda_peak_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 2**30
            if device.type == "cuda"
            else 0.0
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
