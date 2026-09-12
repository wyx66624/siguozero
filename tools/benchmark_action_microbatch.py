"""Fixed-input action A/B and learner batching probe; never starts training.

Synthetic histories are identical across A/B; old outcome fields are neutral.
All non-action weights are shared and unchanged. This is a compute comparison,
not a self-play or playing-strength comparison. All Adam states stay resident.
"""
from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
from torch import nn, Tensor

from benchmark_cuda import initialize_optimizer_state
from benchmark_ppo import diverse_states
from junqi.training.models import GamePolicyTransformer, GameValueTransformer, PieceConditionedLayoutPointerDecoder
from junqi.training.ppo import PPOSample, policy_ppo_loss, critic_ppo_loss, sequence_training_batches
from junqi.training.settings import TrainingSettings


def legacy_encoder():
    source = subprocess.check_output(['git', 'show', 'HEAD:src/junqi/training/models.py'], text=True)
    node = next(x for x in ast.parse(source).body if isinstance(x, ast.ClassDef) and x.name == 'PublicActionEncoder')
    code = ast.get_source_segment(source, node)
    scope = {'nn': nn, 'Tensor': Tensor, 'torch': torch,
             'ACTION_POINT_PAD': 129, 'ACTION_PLAYER_PAD': 4, 'ACTION_COMBAT_PAD': 4}
    exec(compile(code, '<repository HEAD action encoder>', 'exec'), scope)
    return scope['PublicActionEncoder'](256), hashlib.sha256(code.encode()).hexdigest()


class LegacyFixture(nn.Module):
    """Reuse prebuilt eight-field fixtures after warmup; no coordinate conversion."""
    def __init__(self, encoder, source, destination):
        super().__init__()
        self.encoder = encoder
        self.endpoints = (source, destination)
        self.fixtures = {}

    def forward(self, fields, present):
        key = (tuple(fields.shape[:-1]), fields.device)
        if key not in self.fixtures:
            value = torch.zeros((*fields.shape[:-1], 8), dtype=torch.long, device=fields.device)
            value[..., 0], value[..., 1] = self.endpoints
            value[..., 5] = 4  # no captured flag; all other event fields neutral
            self.fixtures[key] = value
        return self.encoder(self.fixtures[key], present)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(20260911)
    torch.set_num_threads(4)
    settings = TrainingSettings.from_yaml(Path('configs/bootstrap.yaml'), 'four_dark', model_scale='main')
    device = torch.device('cuda')
    policy = GamePolicyTransformer(settings.model).to(device)
    critic = GameValueTransformer(settings.model).to(device)
    critic.initialize_from_policy(policy)
    layout = PieceConditionedLayoutPointerDecoder(settings.model).to(device)
    reference_layout = copy.deepcopy(layout).eval().requires_grad_(False)
    modules = (policy, critic, layout)
    optimizers = [torch.optim.AdamW(module.parameters(), lr=1e-4) for module in modules]
    for module, optimizer in zip(modules, optimizers):
        initialize_optimizer_state(module, optimizer)
    states = diverse_states(settings.mode, settings.model, 1001, 32)
    source, destination = states[0].records[1].action.source, states[0].records[1].action.destination
    new_actions = (policy.action_encoder, critic.action_encoder)
    old_actions = []
    old_digest = None
    for _ in range(2):
        old, old_digest = legacy_encoder()
        old_actions.append(LegacyFixture(old, source, destination).to(device))
    result = {
        'status': 'bounded_fixed_synthetic_workload_not_training_throughput',
        'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'config_revision': settings.raw_config['config_revision'], 'amp': 'bfloat16',
        'old_action_source': 'git HEAD PublicActionEncoder only; current whole-board and temporal model retained',
        'old_action_class_sha256': old_digest,
        'old_event_features': 'neutral combat/attack/reveal/elimination, no captured flag',
        'non_action_weights': 'same objects, no optimizer step in measurements',
        'resident_models': 'policy, critic, layout, frozen layout and initialized policy/critic/layout Adam states',
        'current_optimizer_minibatch_samples': settings.ppo_minibatch_samples,
        'rows': [],
        'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
            'src/junqi/training/models.py', 'src/junqi/training/encoding.py',
            'src/junqi/training/ppo.py', 'configs/bootstrap.yaml', 'tools/benchmark_action_microbatch.py')},
    }

    def persist():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def samples_for(chains, decisions, tokens):
        return [PPOSample(replace(state, records=state.records[:length]), state.legal_actions[0],
                          -4.0, 0.0, 0.5 if index % 2 else -0.5, 0.5 if index % 2 else -0.5, 0)
                for state in states[:chains]
                for index, length in enumerate(range(tokens - 4 * (decisions - 1), tokens + 1, 4))]

    def run_backward(chunks, count):
        for module in (policy, critic):
            module.train()
            module.zero_grad(set_to_none=True)
            for chunk in chunks:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = (policy_ppo_loss(module, chunk, clip_epsilon=0.2, entropy_coefficient=0.01,
                                              sequence_training=True) if module is policy else
                              critic_ppo_loss(module, chunk, clip_epsilon=0.2, value_coefficient=0.5,
                                              sequence_training=True))
                    loss = output.loss * len(chunk) / count
                assert torch.isfinite(loss).item()
                loss.backward()
            module.zero_grad(set_to_none=True)

    def measure(label, samples, microbatch, variant, repeats):
        chunks = sequence_training_batches(samples, sequences_per_batch=microbatch, max_samples_per_sequence=64)
        row = {'label': label, 'variant': variant, 'samples': len(samples),
               'microbatch_sequences': microbatch, 'forward_backward_chunks_per_model': len(chunks),
               'maximum_history_tokens': max(len(x.state.records) for x in samples)}
        try:
            run_backward(chunks, len(samples))  # warmup and fixture preparation
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            elapsed = []
            for _ in range(repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                run_backward(chunks, len(samples))
                torch.cuda.synchronize()
                elapsed.append(time.perf_counter() - start)
            row.update(seconds=elapsed, median_seconds=statistics.median(elapsed),
                       peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                       learner_decisions_per_second=len(samples) / statistics.median(elapsed), status='ok')
        except torch.cuda.OutOfMemoryError:
            for module in (policy, critic):
                module.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            row['status'] = 'oom'
        result['rows'].append(row)
        print(json.dumps(row), flush=True)
        persist()

    for tokens in (512, 1001):
        samples = samples_for(8, 64, tokens)
        # Reverse A/B order on the second pass to expose run-order variation.
        for pass_index, order in enumerate((('new', 'old'), ('old', 'new'))):
            for variant in order:
                policy.action_encoder, critic.action_encoder = new_actions if variant == 'new' else old_actions
                measure(f'action_ab_{tokens}_pass{pass_index + 1}', samples, 8, variant, args.repeats)
    policy.action_encoder, critic.action_encoder = new_actions
    old_actions.clear()
    gc.collect()
    torch.cuda.empty_cache()
    for chains, decisions, label in ((8, 64, '512_decisions_8_full_chains'),
                                     (32, 16, '512_decisions_32_short_chains'),
                                     (32, 64, '2048_decisions_32_full_chains')):
        samples = samples_for(chains, decisions, 1001)
        for microbatch in (8, 16, 32):
            measure(label, samples, microbatch, 'new', args.repeats)
    persist()


if __name__ == '__main__':
    main()
