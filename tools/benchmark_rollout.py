"""Benchmark the real terminal-rollout path, including rules and KV caching."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import gc
import json
from pathlib import Path

import torch

from junqi.training.accelerator import (
    empty_cache,
    get_device_name,
    is_accelerator,
    is_bf16_supported,
    manual_seed_all,
    max_memory_allocated,
    max_memory_reserved,
    memory_allocated,
    memory_reserved,
    reset_peak_memory_stats,
    resolve_device,
    synchronize,
)
from junqi.training.encoding import GameHistory
from junqi.training.models import (
    GamePolicyTransformer,
    ModelConfig,
    PieceConditionedLayoutPointerDecoder,
    parameter_count,
)
from junqi.training.modes import TrainingMode, new_game, normalize_mode
from junqi.training.rollout import (
    AnchorSnapshot,
    BaseGamePool,
    FrozenPolicyActor,
    collect_policy_groups,
)
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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="read weights and long-history anchor states from a checkpoint",
    )
    parser.add_argument(
        "--slot-offset",
        type=int,
        default=0,
        help="first base-game-pool slot used with --checkpoint",
    )
    parser.add_argument("--incremental", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--paged-kv", action=argparse.BooleanOptionalAction, default=True)
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
        paged_kv_cache=args.paged_kv,
        inference_temporal_cache_entries=args.temporal_cache_entries,
    )


def main() -> None:
    args = parse_args()
    if args.anchors <= 0 or args.max_game_plies <= 0:
        raise ValueError("anchors and max-game-plies must be positive")
    mode = normalize_mode(args.mode)
    device = resolve_device(args.device)
    checkpoint_payload = None
    if args.checkpoint is None:
        config = make_config(args, mode)
    else:
        checkpoint_payload = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if checkpoint_payload["mode"] != mode.value:
            raise ValueError("checkpoint mode does not match --mode")
        if checkpoint_payload.get("dead_rules_enabled") is not args.dead_rules:
            raise ValueError("checkpoint dead-rule variant does not match arguments")
        config = replace(
            ModelConfig(**checkpoint_payload["config"]["model"]),
            incremental_inference=args.incremental,
            paged_kv_cache=args.paged_kv,
            inference_temporal_cache_entries=args.temporal_cache_entries,
        )
    torch.manual_seed(20260903)
    if is_accelerator(device):
        manual_seed_all(device.type, 20260903)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    policy = GamePolicyTransformer(config).to(device).eval()
    if checkpoint_payload is not None:
        policy.load_state_dict(checkpoint_payload["policy"], strict=True)
    residents: list[object] = []
    if args.full_stack:
        reference_policy = copy.deepcopy(policy).requires_grad_(False)
        layout = PieceConditionedLayoutPointerDecoder(config).to(device)
        reference_layout = copy.deepcopy(layout).requires_grad_(False)
        policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
        layout_optimizer = torch.optim.AdamW(layout.parameters(), lr=5e-5)
        if checkpoint_payload is None:
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
        else:
            layout.load_state_dict(checkpoint_payload["layout"], strict=True)
            reference_policy.load_state_dict(
                checkpoint_payload["reference_policy"], strict=True
            )
            reference_layout.load_state_dict(
                checkpoint_payload["reference_layout"], strict=True
            )
            policy_optimizer.load_state_dict(checkpoint_payload["policy_optimizer"])
            layout_optimizer.load_state_dict(checkpoint_payload["layout_optimizer"])
        residents.extend(
            [reference_policy, layout, reference_layout, policy_optimizer, layout_optimizer]
        )
    anchors: list[AnchorSnapshot] = []
    benchmark_max_game_plies = args.max_game_plies
    if checkpoint_payload is None:
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
    else:
        pool_state = checkpoint_payload["trainer_state"]["base_game_pool"]
        benchmark_max_game_plies = pool_state["max_game_plies"]
        pool = BaseGamePool(
            mode,
            pool_size=int(pool_state["pool_size"]),
            max_transitions=int(pool_state["max_transitions"]),
            max_game_plies=pool_state["max_game_plies"],
            dead_rules_enabled=bool(pool_state["dead_rules_enabled"]),
            seed=0,
        )
        pool.load_state_dict(pool_state)
        if args.anchors > len(pool.slots):
            raise ValueError("--anchors exceeds checkpoint base-game-pool slots")
        start = args.slot_offset % len(pool.slots)
        indices = [(start + index) % len(pool.slots) for index in range(args.anchors)]
        for index in indices:
            slot = pool.slots[index]
            game = slot.game.clone()
            history = slot.history.clone()
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

    checkpoint_update = (
        None if checkpoint_payload is None else int(checkpoint_payload["update"])
    )
    del checkpoint_payload
    gc.collect()
    if is_accelerator(device):
        manual_seed_all(device.type, 20260903)

    amp_dtype = (
        torch.bfloat16
        if is_accelerator(device) and is_bf16_supported(device)
        else torch.float16 if is_accelerator(device) else None
    )
    actor = FrozenPolicyActor(
        policy, amp_dtype=amp_dtype, max_batch_size=args.actor_batch
    )
    if is_accelerator(device):
        empty_cache(device)
        reset_peak_memory_stats(device)
        resident_allocated = memory_allocated(device) / 2**30
    else:
        resident_allocated = 0.0
    groups, metrics = collect_policy_groups(
        anchors,
        actor,
        behavior_version=0,
        anchor_wave_size=args.anchor_wave,
        environment_workers=args.environment_workers,
    )
    if is_accelerator(device):
        synchronize(device)
    output = {
        "accelerator": get_device_name(device),
        "device_type": device.type,
        "mode": mode.value,
        "model_scale": args.model_scale,
        "policy_parameters": parameter_count(policy),
        "incremental_inference": args.incremental,
        "paged_kv_cache": args.paged_kv,
        "anchors": len(groups),
        "max_game_plies": benchmark_max_game_plies,
        "actor_batch": actor.max_batch_size,
        "anchor_wave": args.anchor_wave,
        "full_training_stack_resident": args.full_stack,
        "checkpoint_update": checkpoint_update,
        "resident_allocated_gib": resident_allocated,
        **metrics.as_dict(),
        **policy.board_encoding_metrics(),
        "accelerator_peak_allocated_gib": (
            max_memory_allocated(device) / 2**30
            if is_accelerator(device)
            else 0.0
        ),
        "accelerator_peak_reserved_gib": (
            max_memory_reserved(device) / 2**30
            if is_accelerator(device)
            else 0.0
        ),
        "accelerator_end_allocated_gib": (
            memory_allocated(device) / 2**30
            if is_accelerator(device)
            else 0.0
        ),
        "accelerator_end_reserved_gib": (
            memory_reserved(device) / 2**30
            if is_accelerator(device)
            else 0.0
        ),
        "actor_oom_reductions": actor.oom_reductions,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
