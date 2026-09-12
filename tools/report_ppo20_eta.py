"""Audit complete bounded PPO runs and project steady training throughput."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def summarize(path, *, warmup_updates=2):
    raw = path.read_bytes()
    data = json.loads(raw.decode('utf-8-sig'))
    if not data.get('complete') or data.get('candidate_validation_error'):
        raise ValueError(f'not a valid completed measurement: {path}')
    assert data['checkpoint_stat_unchanged']
    updates = data['updates']
    assert len(updates) >= warmup_updates + 2
    steady = updates[warmup_updates:]
    steps = sum(row['metrics']['rollout/environment_plies'] for row in steady)
    seconds = sum(row['wall_seconds'] for row in steady)
    for row in updates:
        n = row['metrics']['rollout/environment_plies']
        assert row['metrics']['optimizer/policy_steps'] == n / data['ppo_minibatch_samples'] * data['policy_epochs']
        assert row['metrics']['optimizer/critic_steps'] == n / data['ppo_minibatch_samples'] * data['critic_epochs']
    rate = steps / seconds
    startup = max(0., sum(row['wall_seconds'] for row in updates[:warmup_updates])
                  - sum(row['metrics']['rollout/environment_plies'] for row in updates[:warmup_updates]) / rate)
    target_seconds = 3_000_000_000 / rate + startup
    return dict(source=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                measured_updates=len(updates), warmup_updates=warmup_updates,
                steady_updates=len(steady), steady_environment_steps=steps,
                steady_seconds=seconds, steps_per_second=rate,
                startup_overhead_seconds=startup,
                continuous_training_days=target_seconds/86400,
                days_at_80_percent_availability=target_seconds/86400/.8,
                additional_speedup_for_20_days=max(1., target_seconds/(20*86400)),
                measured_total_seconds=data['wall_seconds'],
                model_config=data['model_config'], games=data['games'],
                transitions_per_update=updates[0]['metrics']['rollout/environment_plies'],
                policy_epochs=data['policy_epochs'], critic_epochs=data['critic_epochs'],
                optimizer_minibatch=data['ppo_minibatch_samples'],
                environment_workers=data['environment_workers'],
                peak_allocated_gib=data['peak_allocated_gib'], peak_reserved_gib=data['peak_reserved_gib'],
                phase_seconds_per_12288_actions={key: sum(row['metrics'].get(key, 0.) for row in steady) * 12288 / steps
                    for key in ('timing/policy_backward_seconds', 'timing/critic_backward_seconds',
                                'rollout/wall_seconds', 'rollout/actor_inference_seconds',
                                'rollout/critic_inference_seconds', 'rollout/environment_step_work_seconds')},
                max_history=max(row['history']['max_tokens'] for row in steady),
                full_window_samples=sum(round(row['history']['samples'] * row['history']['window_limit_fraction']) for row in steady),
                per_round=[dict(update=row['update'], steps=row['metrics']['rollout/environment_plies'],
                                seconds=row['wall_seconds'], mean_history=row['history']['mean_tokens'],
                                max_history=row['history']['max_tokens']) for row in updates])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(complete=True, target_environment_steps=3_000_000_000,
                  required_steps_per_second_for_20_days=3_000_000_000/20/86400,
                  cases=[summarize(path) for path in args.input],
                  limitations=['Steady projection excludes the first two updates but adds measured startup overhead once.',
                               'All completed optimizer steps/epochs are audited; no early-stop shortcut is included.',
                               'Scheduled evaluation and save time must be added; availability is separate.',
                               'Future history lengths and long-game proportions can change throughput.',
                               'Changing parallel games changes rollout size and data freshness; this is not a paired trajectory experiment.'])
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({c['source']: c['continuous_training_days'] for c in report['cases']}))


if __name__ == '__main__':
    main()
