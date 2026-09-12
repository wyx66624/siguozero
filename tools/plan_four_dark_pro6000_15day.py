"""Recompute four-dark deadline requirements; proposed architectures are unmeasured.

This is arithmetic over existing observations, not a GPU benchmark or a model
implementation. It does not modify training configuration or checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def linear(inputs: int, outputs: int) -> int:
    return inputs * outputs + outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / 'docs/benchmarks/parallel_games_48_four_dark_rev22_4090_20260911.json'
    baseline = json.loads(source.read_text(encoding='utf-8-sig'))
    rows = baseline['real_updates'][2:4]
    steps = sum(row['rollout/environment_plies'] for row in rows)
    save = next(p['seconds'] for p in baseline['phases']
                if p['phase'] == 'checkpoint_to_workspace_drive')
    environment_seconds = sum(row['rollout/environment_step_work_seconds'] for row in rows)
    baseline_seconds = sum(row['timing/update_seconds'] for row in rows) + len(rows) * save
    budget, days, availability = 3_000_000_000, 15, 0.8
    # Both candidates retain board128/action128, the 18009-feature board schema,
    # existing metadata embeddings and factorized policy-head widths (128).
    width = 512
    common = dict(board=linear(18009, 128), action=linear(5, 128),
                  input_projection=linear(256, width),
                  metadata_embeddings=(3 + 61 + 16 + 16 + 5) * width,
                  final_norm=2 * width,
                  source_head=linear(width, 128) + linear(128, 129),
                  destination_head=linear(width + 129, 128) + linear(128, 129),
                  value_head=linear(width, 1))
    recurrent = dict(common)
    # Four pre-norm residual SwiGLU blocks, two standard biased GRU layers.
    recurrent['residual_mlp'] = 4 * (2 * width + 3 * width * 1024)
    recurrent['gru'] = 2 * (3 * width * width + 3 * width * width + 6 * width)
    transformer = dict(common)
    transformer['position_embeddings_1001'] = 1001 * width
    transformer['transformer'] = 8 * (4 * width**2 + 3 * width * 1536 + 8 * width)
    update_steps, proposed_update_seconds = 256 * 128, 11.0
    result = {
        'status': 'design_targets_and_parameter_arithmetic_not_measured_pro6000_performance',
        'scope': 'four_dark only, 3 billion actual environment transitions across all seats',
        'hardware_assumption': 'RTX PRO 6000 Blackwell Workstation Edition 96GB, 600W',
        'budget': budget, 'target_calendar_days': days, 'availability': availability,
        'required_env_steps_per_second': {
            'continuous': budget / (days * 86400),
            'availability_90_percent': budget / (days * 86400 * .9),
            'availability_80_percent': budget / (days * 86400 * .8),
        },
        'baseline': {'file': str(source.relative_to(root)), 'updates': [3, 4],
                     'steps': steps, 'seconds_including_checkpoint': baseline_seconds,
                     'env_steps_per_second': steps / baseline_seconds,
                     'required_ratio_to_4090_baseline':
                         budget / (days * 86400 * availability) / (steps / baseline_seconds)},
        'unchanged_48_game_pipeline_counterfactual': {
            'meaning': 'Hold measured per-step environment cost and every-update save cost fixed; not a new timing.',
            'environment_ms_per_step': environment_seconds / steps * 1000,
            'environment_alone_continuous_days': budget * environment_seconds / steps / 86400,
            'save_alone_continuous_days': budget * len(rows) * save / steps / 86400,
            'combined_continuous_days': budget * (environment_seconds + len(rows) * save) / steps / 86400,
            'combined_calendar_days_at_80_percent':
                budget * (environment_seconds + len(rows) * save) / steps / 86400 / availability,
        },
        'architecture_candidates': {
            'shared_residual_mlp_gru': {'components': recurrent,
                                      'actor_critic_total': sum(recurrent.values()),
                                      'description': '4 residual SwiGLU blocks d512/f1024, 2 GRU layers h512, shared policy/value backbone'},
            'shared_shallow_transformer': {'components': transformer,
                                          'actor_critic_total': sum(transformer.values()),
                                          'description': '8 layers d512/f1536/h8, input256-to512 projection, full 1001-position table'},
        },
        'architecture_notes': [
            'Counts describe the listed proposed modules, not implemented production models.',
            'Layout 17292288 parameters and its frozen reference are excluded from candidate actor-critic totals.',
            'Recurrent memory and truncated gradients change representation and training; strength requires evaluation.',
            'Shared actor-critic requires one combined objective and optimizer for shared parameters.',
            'No independent optimization speedups are multiplied into a claimed end-to-end estimate.',
        ],
        'acceptance_budget_example': {
            'games': 256, 'environment_steps_per_game_per_update': 128,
            'total_environment_steps_per_update': update_steps,
            'per_seat_observation_positions_before_burn_in': update_steps * 4,
            'collection_seconds_max': 5, 'learning_seconds_max_including_all_epochs': 5,
            'save_layout_evaluation_other_seconds_amortized_max': 1,
            'seconds_per_update_max': proposed_update_seconds,
            'resulting_env_steps_per_second': update_steps / proposed_update_seconds,
            'resulting_calendar_days_at_80_percent':
                budget * proposed_update_seconds / update_steps / 86400 / availability,
            'not_observed': True,
        },
        'calendar_days_by_measured_env_steps_per_second_at_80_percent': {
            str(rate): budget / rate / 86400 / availability
            for rate in (1000, 1500, 2000, 2500, 3000, 4000)
        },
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'required_steps_per_second': result['required_env_steps_per_second'],
        'fixed_overhead': result['unchanged_48_game_pipeline_counterfactual'],
        'candidate_parameter_counts': {k: v['actor_critic_total']
                                       for k, v in result['architecture_candidates'].items()},
        'acceptance_budget': result['acceptance_budget_example'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
