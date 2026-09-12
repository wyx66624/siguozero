"""Bounded full-context capacity and fixed-KV rollover validation on CUDA.

Uses independent synthetic observation histories, not game-strength evidence.
All three Adam states and the layout reference remain resident. No model saves.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import torch

from benchmark_cuda import initialize_optimizer_state
from benchmark_ppo import diverse_states
from junqi.training.history_arrays import ArrayHistory
from junqi.training.metrics import MetricLogger
from junqi.training.ppo import FrozenValueActor, PPOSample
from junqi.training.rollout import FrozenPolicyActor
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    lock = open(Path(tempfile.gettempdir()) / 'siguozero-cuda-probe.lock', 'a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.manual_seed(91731)
    torch.cuda.set_per_process_memory_fraction(.90)
    root = Path(__file__).resolve().parents[1]
    settings = TrainingSettings.from_yaml(root / 'configs/bootstrap.yaml', 'four_dark', model_scale='main',
        overrides={'device': 'cuda', 'arena_enabled': False, 'checkpoint_policy': 'evaluation'})
    with patch.object(MetricLogger, 'start_resource_monitor'):
        trainer = SelfPlayTrainer(settings, run_directory=args.run_dir, auto_resume=False)
    result = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, model=asdict(settings.model),
                  configured_microbatch=settings.policy_microbatch, phases=[], complete=False,
                  limitations=['Synthetic capacity/rollover stress, not legal-game throughput or strength.'])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')

    def measure(name, fn):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        values = fn()
        torch.cuda.synchronize()
        row = dict(phase=name, seconds=time.perf_counter() - start,
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, **values)
        result['phases'].append(row)
        persist()
        print(json.dumps(row), flush=True)

    try:
        for module, optimizer in ((trainer.policy, trainer.policy_optimizer),
                                  (trainer.critic, trainer.critic_optimizer),
                                  (trainer.layout, trainer.layout_optimizer)):
            initialize_optimizer_state(module, optimizer)
        # 32 independent histories, each with 16 supervised decisions: 512 total.
        states = diverse_states(settings.mode, settings.model, 1001, 32)
        samples = [PPOSample(replace(state, records=state.records[:length]), state.legal_actions[0],
                             -4., 0., .5 if j % 2 else -.5, .5 if j % 2 else -.5, 0)
                   for state in states for j, length in enumerate(range(941, 1002, 4))]
        for name, update in (('policy', trainer._update_policy_batch), ('critic', trainer._update_critic_batch)):
            measure('32_independent_full_context_' + name, lambda update=update: dict(
                metrics=update(samples, epochs=1), samples=len(samples), independent_histories=32,
                max_tokens=1001, effective_microbatch=trainer.effective_policy_microbatch))
        result['effective_microbatch'] = trainer.effective_policy_microbatch
        # Once a 1000-transition window moves, different starts cannot share a
        # causal prefix. Measure a complete 512-decision batch in that regime.
        sliding_samples = []
        for i, state in enumerate(states):
            history = ArrayHistory(settings.mode, state.records, 1000, (i, 0))
            for j in range(16):
                for _ in range(4):
                    history.append(state.records[-1])
                sliding_samples.append(PPOSample(replace(state, records=history.view()), state.legal_actions[0],
                    -4., 0., .5 if j % 2 else -.5, .5 if j % 2 else -.5, 0))
        for name, update in (('policy', trainer._update_policy_batch), ('critic', trainer._update_critic_batch)):
            measure('512_unmergeable_sliding_window_' + name, lambda update=update: dict(
                metrics=update(sliding_samples, epochs=1), samples=len(sliding_samples),
                independent_windows=512, max_tokens=1001,
                effective_microbatch=trainer.effective_policy_microbatch))
        del sliding_samples
        del samples, states

        # Keep 192 player histories live, equivalent to 48 four-player games.
        templates = diverse_states(settings.mode, settings.model, 1001, 192)
        histories = [ArrayHistory(settings.mode, state.records, 1000, (i // 4, i % 4))
                     for i, state in enumerate(templates)]
        actor = FrozenPolicyActor(trainer.policy.eval(), amp_dtype=trainer.amp_dtype, max_batch_size=48)
        critic = FrozenValueActor(trainer.critic, amp_dtype=trainer.amp_dtype, max_batch_size=48)
        for model in (trainer.policy, trainer.critic):
            model.start_ppo_inference_cache(capacity=192, behavior_version=1)

        def rollout(slide):
            if slide:
                for history, template in zip(histories, templates, strict=True):
                    history.append(template.records[-1])
            observed = [replace(template, records=history.view())
                        for history, template in zip(histories, templates, strict=True)]
            for start in range(0, 192, 48):
                batch = observed[start:start + 48]
                actor.sample(batch)
                critic.values(batch)
            return dict(player_histories=192, max_tokens=1001,
                        window_start=observed[0].records.window_start,
                        actor_batch=actor.max_batch_size, critic_batch=critic.max_batch_size,
                        policy_prefills=trainer.policy._fixed_kv_store.prefill_states,
                        critic_prefills=trainer.critic._fixed_kv_store.prefill_states)

        measure('192_full_context_player_histories', lambda: rollout(False))
        measure('192_sliding_window_player_histories', lambda: rollout(True))
        # Exact rollover must drop the retired history. Compare a populated
        # cache with a fresh full forward on the SAME surviving observations.
        subset = [replace(templates[i], records=histories[i].view()) for i in range(3)]
        with torch.inference_mode(), torch.autocast('cuda', dtype=trainer.amp_dtype):
            actions = [state.legal_actions for state in subset]
            cached_policy = [row.clone() for row in trainer.policy(subset, actions)]
            cached_values = trainer.critic(subset).clone()
            trainer.policy.clear_inference_board_cache()
            trainer.critic.clear_inference_board_cache()
            full_policy = trainer.policy(subset, actions)
            full_values = trainer.critic(subset)
        for a, b in zip(cached_policy, full_policy, strict=True):
            torch.testing.assert_close(a, b, rtol=.02, atol=.02)
        torch.testing.assert_close(cached_values, full_values, rtol=.02, atol=.02)
        result['rollover_matches_fresh_forward_bf16_atol_rtol'] = .02
        result['no_model_checkpoints_created'] = not list(trainer.run_directory.rglob('*.pt'))
        assert result['no_model_checkpoints_created']
        result['complete'] = True
        persist()
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        persist()
        raise
    finally:
        trainer.logger.close()


if __name__ == '__main__':
    main()
