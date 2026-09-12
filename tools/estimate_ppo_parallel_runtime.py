"""Compare bounded PPO stages and project 3B actual environment transitions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    assert data['complete'] and data['checkpoint_stat_unchanged']
    assert data['environment_steps'] == sum(row['metrics']['rollout/environment_plies'] for row in data['updates'])
    return data, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--learner-only', type=Path, required=True)
    parser.add_argument('--parallel', type=Path, required=True)
    parser.add_argument('--prior-parallel', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = 3_000_000_000
    stages, sources = {}, []
    data_by_name = {}

    def project(seconds, steps):
        rate = steps / seconds
        return dict(environment_steps=steps, wall_seconds=seconds, steps_per_second=rate,
                    continuous_days=target / rate / 86400,
                    calendar_days_at_80_percent=target / rate / 86400 / .8)

    for name, path in (('baseline', args.baseline), ('learner_only', args.learner_only),
                       ('parallel', args.parallel), ('prior_parallel', args.prior_parallel)):
        if path is None:
            continue
        data, source = read(path)
        data_by_name[name] = data
        sources.append(source)
        rows = data['updates']
        worker_diagnostics = all(row['metrics'].get('rollout/environment_workers', 1)
                                 == data.get('environment_workers', 1) for row in rows)
        stages[name] = dict(project(data['wall_seconds'], data['environment_steps']),
            updates=len(rows), configured_microbatch=data.get('configured_microbatch', 8),
            effective_microbatch=data.get('effective_microbatch', 8),
            environment_workers=data.get('environment_workers', 1),
            environment_worker_diagnostics_available=worker_diagnostics,
            policy_optimizer_steps=data['policy_optimizer_steps'],
            critic_optimizer_steps=data['critic_optimizer_steps'],
            layout_optimizer_steps=data['layout_optimizer_steps'],
            peak_allocated_gib=data['peak_allocated_gib'], peak_reserved_gib=data['peak_reserved_gib'],
            max_history_tokens=max(row['history']['max_tokens'] for row in rows),
            full_window_samples=sum(round(row['history']['window_limit_fraction'] * row['history']['samples']) for row in rows),
            mean_phase_seconds={key: (None if not worker_diagnostics and key in
                                      ('rollout/environment_sync_seconds', 'rollout/environment_worker_seconds_sum')
                                      else sum(row['metrics'].get(key, 0.) for row in rows) / len(rows))
                for key in ('rollout/wall_seconds', 'rollout/environment_step_work_seconds',
                            'rollout/environment_sync_seconds', 'rollout/environment_worker_seconds_sum',
                            'rollout/actor_inference_seconds', 'rollout/critic_inference_seconds',
                            'timing/policy_backward_seconds', 'timing/critic_backward_seconds',
                            'timing/layout_backward_seconds')},
            rounds=[dict(update=row['update'], history=row['history'],
                         **project(row['wall_seconds'], row['metrics']['rollout/environment_plies'])) for row in rows])
    baseline = data_by_name['baseline']
    for name, data in data_by_name.items():
        for key in ('device', 'checkpoint', 'checkpoint_update', 'games', 'policy_epochs', 'critic_epochs', 'ppo_minibatch_samples'):
            assert data[key] == baseline[key], (name, key)
        for key in ('board_dim', 'temporal_dim', 'temporal_layers', 'temporal_ffn_dim', 'max_transitions'):
            assert data['model_config'][key] == baseline['model_config'][key], (name, key)
        assert all(row['metrics']['optimizer/policy_steps'] == 72 for row in data['updates'])
        assert all(row['metrics']['optimizer/critic_steps'] == 72 for row in data['updates'])
    first = {name: data['updates'][0] for name, data in data_by_name.items()}
    for data in first.values():
        assert data['history']['samples'] == 12288
    result = dict(complete=True, target_environment_transitions=target, stages=stages, sources=sources,
        matched_first_round_speedup=dict(
            learner_only=first['baseline']['wall_seconds'] / first['learner_only']['wall_seconds'],
            environment_parallelism=first['learner_only']['wall_seconds'] / first['parallel']['wall_seconds'],
            combined=first['baseline']['wall_seconds'] / first['parallel']['wall_seconds']),
        required_steps_per_second_for_days={str(days): target / days / 86400 for days in (10, 15, 20)},
        evaluation_events=60,
        evaluation_overhead_days_formula='60 * measured_seconds_per_evaluation_and_save / 86400',
        limitations=['Bounded four_dark training measurement; conditional projection if future histories have a similar distribution.',
                     'No claim about model strength, other hardware, or double_open throughput.',
                     'Scheduled evaluation and checkpoint save time must be added; normal updates do not save.',
                     'The first round shares checkpoint and initial histories; BF16/atomic gradient rounding can change later sampled games.',
                     'Prior parallel measurements may precede the compact transport optimization; do not pool them as independent training history.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(stages={name: {key: value for key, value in row.items() if key in
          ('steps_per_second', 'continuous_days', 'calendar_days_at_80_percent', 'full_window_samples')}
          for name, row in stages.items()}, speedup=result['matched_first_round_speedup'])), flush=True)


if __name__ == '__main__':
    main()
