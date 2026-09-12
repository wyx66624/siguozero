"""Full-window PPO learner and per-seat KV stress for the optimized profile.

Synthetic histories establish capacity and causal equivalence, not strength.
No production checkpoint is loaded, changed, or created by this probe.
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
    parser.add_argument('--config', default='configs/bootstrap.yaml')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    lock = open(Path(tempfile.gettempdir()) / 'siguozero-cuda-probe.lock', 'a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.manual_seed(91731)
    torch.cuda.set_per_process_memory_fraction(.90)
    settings = TrainingSettings.from_yaml(args.config, 'four_dark', model_scale='main', overrides={
        'device': 'cuda', 'arena_enabled': False, 'checkpoint_policy': 'evaluation'})
    with patch.object(MetricLogger, 'start_resource_monitor'):
        trainer = SelfPlayTrainer(settings, run_directory=args.run_dir, auto_resume=False)
    result = dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                  model=asdict(settings.model), games=settings.base_game_pool_size,
                  phases=[], complete=False,
                  limitations=['Synthetic capacity and equivalence test; not legal-game throughput or strength.'])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        output.write_text(json.dumps(result, indent=2) + '\n')

    def measure(name, function):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        values = function()
        torch.cuda.synchronize()
        row = dict(phase=name, seconds=time.perf_counter() - start,
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, **values)
        result['phases'].append(row)
        persist()
        print(json.dumps(row), flush=True)

    try:
        for model, optimizer in ((trainer.policy, trainer.policy_optimizer),
                                 (trainer.critic, trainer.critic_optimizer),
                                 (trainer.layout, trainer.layout_optimizer)):
            initialize_optimizer_state(model, optimizer)
        templates = diverse_states(settings.mode, settings.model, 1001, 32)
        samples, sliding = [], []
        for index, state in enumerate(templates):
            history = ArrayHistory(settings.mode, state.records, 1000, (index, 0))
            for j, length in enumerate(range(941, 1002, 4)):
                samples.append(PPOSample(replace(state, records=history.view()[:length]),
                    state.legal_actions[0], -4., 0., .5 if j % 2 else -.5, .5 if j % 2 else -.5, 0))
            for j in range(16):
                for _ in range(4):
                    history.append(state.records[-1])
                sliding.append(PPOSample(replace(state, records=history.view()),
                    state.legal_actions[0], -4., 0., .5 if j % 2 else -.5, .5 if j % 2 else -.5, 0))
        for label, data in (('32_shared_prefixes', samples), ('512_distinct_sliding_windows', sliding)):
            for name, update in (('policy', trainer._update_policy_batch), ('critic', trainer._update_critic_batch)):
                # First invocation may compile the full-window shape. Report it
                # separately; the second is the warmed, equally sized workload.
                for repetition in range(2):
                    measure(f'{label}_{name}_{repetition}', lambda update=update, data=data: dict(
                        metrics=update(data, epochs=1), samples=512, max_tokens=1001,
                        effective_microbatch=trainer.effective_policy_microbatch))
        del samples, sliding, templates

        capacity = settings.base_game_pool_size * 4
        templates = diverse_states(settings.mode, settings.model, 1001, capacity)
        histories = [ArrayHistory(settings.mode, state.records[:993], 1000, (i // 4, i % 4))
                     for i, state in enumerate(templates)]
        batch_size = max(1, settings.actor_inference_batch // settings.ppo_pipeline_groups)
        actor = FrozenPolicyActor(trainer.policy.eval(), amp_dtype=trainer.amp_dtype, max_batch_size=batch_size)
        critic = FrozenValueActor(trainer.critic, amp_dtype=trainer.amp_dtype,
                                  max_batch_size=batch_size, deferred=True)
        trainer.policy.start_ppo_inference_cache(capacity=capacity, behavior_version=1)

        def rollout(append):
            for history, template in zip(histories, templates, strict=True):
                for _ in range(append):
                    history.append(template.records[-1])
            observed = [replace(template, records=history.view())
                        for template, history in zip(templates, histories, strict=True)]
            for start in range(0, capacity, batch_size):
                actor.sample(observed[start:start + batch_size])
            store = trainer.policy._fixed_kv_store
            return dict(player_histories=capacity, max_tokens=len(observed[0].records),
                        window_start=observed[0].records.window_start,
                        prefills=store.prefill_states, incremental_states=store.decode_states,
                        graph_replays=store.graph_replays)

        measure('all_seats_cold_prefill_993', lambda: rollout(0))
        measure('all_seats_grouped_decode_997', lambda: rollout(4))
        measure('all_seats_grouped_decode_1001', lambda: rollout(4))
        subset = [replace(templates[i], records=histories[i].view()) for i in range(3)]

        def compare_cached(states, label):
            actions = [s.legal_actions for s in states]
            with torch.inference_mode(), torch.autocast('cuda', dtype=trainer.amp_dtype):
                cached = [row.clone() for row in trainer.policy(states, actions)]
                features = trainer.policy._encode_full(states)
                reference = trainer.policy._all_legal_log_probs(features, states)[0]
            difference = float((torch.cat(cached) - reference).abs().max())
            torch.testing.assert_close(torch.cat(cached), reference, rtol=.02, atol=.02)
            result[label] = dict(max_log_probability_difference=difference, atol=.02, rtol=.02)

        compare_cached(subset, 'grouped_decode_matches_full_forward')
        measure('all_seats_sliding_window_exact_rebuild', lambda: rollout(4))
        subset = [replace(templates[i], records=histories[i].view()) for i in range(3)]
        compare_cached(subset, 'sliding_rebuild_matches_full_forward')
        trainer.policy.clear_inference_board_cache()
        all_states = [replace(template, records=history.view())
                      for template, history in zip(templates, histories, strict=True)]
        measure('deferred_critic_all_seats_full_windows', lambda: dict(values=len(critic.values_packed(all_states))))
        result['complete'] = True
        result['no_model_checkpoints_created'] = not list(trainer.run_directory.rglob('*.pt'))
        assert result['no_model_checkpoints_created']
        persist()
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        persist()
        raise
    finally:
        trainer.logger.close()


if __name__ == '__main__':
    main()
