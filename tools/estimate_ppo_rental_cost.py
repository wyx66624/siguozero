"""RMB rental sensitivities with dated public quotes and measured A/B evidence.

Target-card runtimes and batching gains are NOT measurements or quotations.
Default billing retains rented GPUs through the 20% non-training time too.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path('docs/benchmarks')
    hardware_path = root / 'action_linear_hardware_eta_20260910.json'
    baseline_path = root / 'action_linear_eta_later_20260910.json'
    micro_path = root / 'action_microbatch_4090_20260911.json'
    hardware = json.loads(hardware_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    micro = json.loads(micro_path.read_text())
    assert hardware['current_five_input_action_timing_measured']
    assert len(micro['rows']) == 17 and all(x['status'] == 'ok' for x in micro['rows'])
    fx = 7.7900 / 1.1616
    quotes = [
        {'hardware': 'RTX 4090 24GB', 'provider': 'AutoDL public homepage', 'rmb_per_gpu_hour': 1.88,
         'quote_status': 'public indexed hourly rate; not a checked stock or multi-GPU order',
         'url': 'https://www.autodl.com/home'},
        {'hardware': 'RTX PRO 6000 Blackwell Workstation 96GB', 'provider': 'AutoDL public homepage',
         'rmb_per_gpu_hour': 6.98, 'quote_status': 'public indexed hourly rate; not a checked stock or multi-GPU order',
         'url': 'https://www.autodl.com/home'},
        {'hardware': 'H100 SXM 80GB', 'provider': 'Runpod Secure Cloud Pods',
         'usd_per_gpu_hour': 3.49, 'rmb_per_gpu_hour': 3.49 * fx,
         'quote_status': 'public pricing page; not a deployment-console stock or cluster order',
         'url': 'https://www.runpod.io/pricing'},
    ]
    variants = []
    for tokens in (512, 1001):
        timings = {kind: [s for r in micro['rows'] if r['label'].startswith(f'action_ab_{tokens}_')
                         and r['variant'] == kind for s in r['seconds']] for kind in ('old', 'new')}
        medians = {k: statistics.median(v) for k, v in timings.items()}
        variants.append({'context_tokens': tokens, 'same_decisions': 512,
                         'old_seconds_median': medians['old'], 'new_seconds_median': medians['new'],
                         'new_change_fraction': medians['new'] / medians['old'] - 1,
                         'repeats_per_variant': len(timings['old'])})
    opening = []
    learner = []
    for probe in baseline['probes']:
        mode = probe['mode']
        old_name = ('whole_board_real_ppo_4090_20260910.json' if mode == 'four_dark'
                    else 'whole_board_real_double_open_4090_20260910.json')
        old = json.loads((root / old_name).read_text())
        new = json.loads(Path(probe['source']).read_text())
        opening.append({'mode': mode, 'steps_each': 10240,
                        'old_first_two_seconds': sum(x['timing/update_seconds'] for x in old['real_updates'][:2]),
                        'new_first_two_seconds': sum(x['timing/update_seconds'] for x in new['real_updates'][:2]),
                        'new_later_two_seconds': sum(x['timing/update_seconds'] for x in new['real_updates'][2:]),
                        'same_step_count_but_not_same_trajectories': True})
        learning = sum(x['policy_backward_seconds'] + x['critic_backward_seconds'] for x in probe['updates'])
        compute = sum(x['seconds_without_save'] for x in probe['updates'])
        save = len(probe['updates']) * probe['checkpoint_seconds']
        fraction = learning / (compute + save)
        learner.append({'mode': mode, 'measured_learner_fraction_including_save': fraction,
                        'whole_pipeline_speedup_if_learner_twice_as_fast': 1 / (1 - fraction / 2),
                        'whole_pipeline_speedup_if_learner_were_free': 1 / (1 - fraction)})
    costs = []
    for quote in quotes:
        for scenario in hardware['scenarios']:
            if scenario['hardware'] != quote['hardware']:
                continue
            cards = scenario['cards']
            modes = {}
            for mode, item in scenario['calendar_days_by_mode'].items():
                times = item['selected_window_average']
                modes[mode] = {
                    'calendar_days': times,
                    'keep_allocated_rmb': [days * 24 * cards * quote['rmb_per_gpu_hour'] for days in times],
                    'release_during_nontraining_rmb': [days * 0.8 * 24 * cards * quote['rmb_per_gpu_hour'] for days in times],
                }
            lows = [v['keep_allocated_rmb'][0] for v in modes.values()]
            highs = [v['keep_allocated_rmb'][1] for v in modes.values()]
            costs.append({'hardware': quote['hardware'], 'cards': cards, 'modes': modes,
                          'one_mode_keep_allocated_rmb_envelope': [min(lows), max(highs)]})
    sensitivities = []
    # These gains are thresholds to validate on a rental pilot, not assigned to microbatch sizes.
    for quote in quotes:
        for cards in (1, 2, 4, 8):
            for gain in (1.5, 2.0):
                values = []
                for probe in baseline['probes']:
                    work = 3e9 * sum(x['seconds_without_save'] for x in probe['updates']) / 10240
                    speedups = hardware['hardware'][quote['hardware']]['non_save_speedup_scenario_range']
                    for speedup in speedups:
                        for efficiency in ((1.0,) if cards == 1 else (0.7, 0.9)):
                            saves = math.ceil(3e9 / (5120 * cards)) + 1
                            seconds = work / (cards * speedup * efficiency * gain) + saves * probe['checkpoint_seconds']
                            values.append(seconds / 3600 / 0.8 * cards * quote['rmb_per_gpu_hour'])
                sensitivities.append({'hardware': quote['hardware'], 'cards': cards,
                                      'additional_non_save_pipeline_speedup_assumption': gain,
                                      'keep_allocated_rmb_envelope': [min(values), max(values)],
                                      'measured_or_assigned_to_a_microbatch': False})
    actor_path = root / 'actor_batch_4090_20260911.json'
    actor = json.loads(actor_path.read_text()) if actor_path.exists() else None
    if actor:
        assert [r['inference_batch'] for r in actor['rows']] == [20, 40, 80]
        assert actor['actual_environment_steps'] == 0
    inputs = [hardware_path, baseline_path, micro_path, Path(__file__)]
    if actor:
        inputs.append(actor_path)
    result = {
        'status': 'conditional_budget_not_a_full_memory_tuned_runtime_or_a_provider_quote',
        'as_of': '2026-09-11', 'target_environment_steps_per_mode': 3_000_000_000,
        'availability': 0.8, 'default_billing': 'rent retained during non-training hours',
        'fx': {'date': '2026-09-10', 'eur_usd': 1.1616, 'eur_cny': 7.7900, 'usd_cny': fx,
               'url': 'https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html'},
        'quotes': quotes,
        'same_provider_alternatives_rmb_per_gpu_hour': {
            'Runpod Secure RTX 4090': 0.74 * fx, 'Runpod Secure RTX Pro 6000': 2.09 * fx},
        'action_parameter_reduction_policy_and_critic': 297472,
        'fraction_of_previous_trainable_parameters': 297472 / 297004547,
        'old_opening_vs_new_opening_and_later': opening, 'controlled_action_ab': variants,
        'learner_share_and_amdahl': learner,
        'microbatch_measurements': [r for r in micro['rows'] if not r['label'].startswith('action_ab')],
        'actor_batch_probe': actor,
        'cost_scenarios': costs, 'unmeasured_additional_throughput_sensitivities': sensitivities,
        'larger_memory_candidate_pilot': {
            '4090': 'microbatch 8/16; optimizer batch 512/1024; games 20; do not assume 32 is faster',
            'PRO_6000_96GB': 'microbatch 16/32/64; optimizer batch 1024/2048/4096; games 40/80; long-context capacity unmeasured',
            'H100_SXM_80GB': 'microbatch 16/32; optimizer batch 1024/2048; games 40/64; long-context capacity unmeasured',
            'criterion': 'actual environment transitions per billed second at equal history and training budget; not maximum memory usage',
        },
        'limitations': [
            'Target-card and multi-GPU throughput have not been measured; baseline hardware ratios/fractions remain assumptions.',
            'No automatic speedup is granted for bigger VRAM or microbatch; local measurements show little learner improvement.',
            'Synthetic learner/model-decision rates are not environment transition throughput.',
            'Public single-card prices are multiplied arithmetically; matching multi-GPU stock, CPU, disks and interconnect were not quoted.',
            'Storage, tax, exchange fees, periodic layout updates, evaluation, retries and mature-policy distributions are not calibrated here.',
            'Saved model and training settings are unchanged; no GPU rental or formal training was started.',
        ],
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'fx': fx, 'controlled_ab': variants, 'learner': learner,
                      'costs': [{k: r[k] for k in ('hardware', 'cards', 'one_mode_keep_allocated_rmb_envelope')}
                                for r in costs]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
