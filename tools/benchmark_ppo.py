"""PPO capacity probe with independent histories and all AdamW states resident.

Synthetic board permutations stress the full history encoder; they are not
legal game trajectories or a strength benchmark. No repeated-state/board
deduplication shortcut is used to represent the requested microbatch.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import time

import torch

from benchmark_cuda import initialize_optimizer_state, synthetic_state
from junqi.training.accelerator import (
    empty_cache, get_device_name, is_accelerator, is_bf16_supported,
    max_memory_allocated, max_memory_reserved, memory_allocated,
    reset_peak_memory_stats, resolve_device, synchronize,
)
from junqi.training.models import (
    GamePolicyTransformer, GameValueTransformer, ModelConfig,
    PieceConditionedLayoutPointerDecoder, parameter_count,
)
from junqi.training.modes import TrainingMode
from junqi.training.ppo import FrozenValueActor, PPOSample, critic_ppo_loss, policy_ppo_loss
from junqi.training.rollout import FrozenPolicyActor


def diverse_states(mode, config, tokens, count):
    template = synthetic_state(mode, config, tokens)
    rng = random.Random(20260910)
    states = []
    for _ in range(count):
        records = []
        for record in template.records:
            board = list(record.board_codes)
            rng.shuffle(board)
            records.append(replace(record, board_codes=tuple(board)))
        states.append(replace(template, records=tuple(records)))
    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("four_dark", "double_open"), default="four_dark")
    parser.add_argument("--model-scale", choices=("bootstrap", "main", "extended"), default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--actor-batch", type=int, default=8)
    parser.add_argument("--context-tokens", type=int, default=1001)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--phase", choices=("learner", "rollout", "both"), default="both")
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.microbatch, args.games, args.actor_batch, args.iterations) <= 0:
        parser.error("batch sizes, games and iterations must be positive")
    device = resolve_device(args.device)
    if not is_accelerator(device):
        parser.error("an available CUDA/NPU device is required")
    torch.manual_seed(20260910)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    config = {"bootstrap": ModelConfig, "main": ModelConfig.main,
              "extended": ModelConfig.extended}[args.model_scale]()
    config = replace(config, inference_temporal_cache_entries=3 * 4 * args.games)
    mode = TrainingMode(args.mode)
    policy = GamePolicyTransformer(config).to(device)
    critic = GameValueTransformer(config).to(device)
    critic.initialize_from_policy(policy)
    layout = PieceConditionedLayoutPointerDecoder(config).to(device)
    reference_layout = copy.deepcopy(layout).eval().requires_grad_(False)
    modules = {"policy": policy, "critic": critic, "layout": layout}
    optimizers = {name: torch.optim.AdamW(module.parameters(), lr=1e-4)
                  for name, module in modules.items()}
    for name, module in modules.items():
        initialize_optimizer_state(module, optimizers[name])
    amp = torch.bfloat16 if is_bf16_supported(device) else torch.float16
    result = {
        "device": get_device_name(device), "torch": torch.__version__,
        "mode": args.mode, "model_scale": args.model_scale,
        "microbatch": args.microbatch, "context_tokens": args.context_tokens,
        "games": args.games, "actor_batch": args.actor_batch,
        "cache_entries_per_model": config.inference_temporal_cache_entries,
        "parameters": {name: parameter_count(module) for name, module in modules.items()},
        "resident_parameters": sum(parameter_count(module) for module in modules.values())
                               + parameter_count(reference_layout),
        "all_optimizer_states_initialized": True,
        "resident_allocated_gib": memory_allocated(device) / 2**30,
        "phases": [],
    }
    print(json.dumps({key: value for key, value in result.items() if key != "phases"}), flush=True)

    def measure(name, action):
        empty_cache(device)
        reset_peak_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()
        extra = action() or {}
        synchronize(device)
        phase = dict(phase=name, seconds=time.perf_counter() - started,
                     peak_allocated_gib=max_memory_allocated(device) / 2**30,
                     peak_reserved_gib=max_memory_reserved(device) / 2**30,
                     **extra)
        result["phases"].append(phase)
        print(json.dumps(phase), flush=True)

    if args.phase in ("learner", "both"):
        states = diverse_states(mode, config, args.context_tokens, args.microbatch)
        policy.eval()
        with torch.no_grad(), torch.autocast(device.type, dtype=amp):
            logs = policy(states, [state.legal_actions for state in states])
            old = [float(row[0]) for row in logs]
        del logs
        samples = [PPOSample(
            state=state, action=state.legal_actions[0], old_log_prob=log,
            old_value=0, advantage=1 if i % 2 else -1,
            value_target=1 if i % 2 else -1, behavior_version=0,
        ) for i, (state, log) in enumerate(zip(states, old, strict=True))]
        for iteration in range(args.iterations):
            for name, module in (("policy", policy), ("critic", critic)):
                def update(name=name, module=module):
                    module.train()
                    optimizer = optimizers[name]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device.type, dtype=amp):
                        output = (policy_ppo_loss(module, samples, clip_epsilon=0.2,
                                                 entropy_coefficient=0.01)
                                  if name == "policy" else
                                  critic_ppo_loss(module, samples, clip_epsilon=0.2,
                                                  value_coefficient=0.5))
                    output.loss.backward()
                    torch.nn.utils.clip_grad_norm_(module.parameters(), 1, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    return {"loss": float(output.loss.detach())}
                measure(f"{name}_backward_{iteration + 1}", update)
    if args.phase in ("rollout", "both"):
        states = diverse_states(mode, config, args.context_tokens, 4 * args.games)
        def rollout():
            actor = FrozenPolicyActor(policy.eval(), amp_dtype=amp, max_batch_size=args.actor_batch)
            value_actor = FrozenValueActor(critic, amp_dtype=amp, max_batch_size=args.actor_batch)
            for start in range(0, len(states), args.actor_batch):
                chunk = states[start:start + args.actor_batch]
                actor.sample(chunk)
                value_actor.values(chunk)
            return {"cached_player_histories": len(states),
                    "policy_kv_gib": policy._paged_kv_store.allocated_bytes / 2**30,
                    "critic_kv_gib": critic._paged_kv_store.allocated_bytes / 2**30}
        measure("both_models_long_history_rollout", rollout)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
