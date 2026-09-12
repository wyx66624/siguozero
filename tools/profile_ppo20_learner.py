"""Profile warmed, complete 512-sample PPO optimizer steps on synthetic histories."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import fcntl
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import torch

from benchmark_ppo import diverse_states
from junqi.training.history_arrays import ArrayHistory
from junqi.training.metrics import MetricLogger
from junqi.training.ppo import PPOSample
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--parallel-streams', action='store_true')
    args = parser.parse_args()
    lock = open(Path(tempfile.gettempdir()) / 'siguozero-cuda-probe.lock', 'a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.manual_seed(177)
    settings = TrainingSettings.from_yaml(args.config, 'four_dark', model_scale='main', overrides={
        'device': 'cuda', 'arena_enabled': False, 'checkpoint_policy': 'evaluation'})
    with patch.object(MetricLogger, 'start_resource_monitor'):
        trainer = SelfPlayTrainer(settings, run_directory=args.run_dir, auto_resume=False)
    templates = diverse_states(settings.mode, settings.model, 769, 8)
    samples = []
    for i, state in enumerate(templates):
        history = ArrayHistory(settings.mode, state.records, 1000, (i, 0))
        for j, length in enumerate(range(257 + i * 32, 513 + i * 32, 4)):
            samples.append(PPOSample(replace(state, records=history.view()[:length]),
                state.legal_actions[0], -4., 0., .1 if j % 2 else -.1, .1 if j % 2 else -.1, 0))
    result = dict(complete=False, cases=[], samples=len(samples), histories=8)
    output = Path(args.output)
    def save():
        output.write_text(json.dumps(result, indent=2) + '\n')
    try:
        for name, update in (('policy', trainer._update_policy_batch), ('critic', trainer._update_critic_batch)):
            for _ in range(3):
                update(samples, epochs=1)
            times = []
            for _ in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                update(samples, epochs=1)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profiler:
                with torch.profiler.record_function('complete_' + name + '_optimizer_step'):
                    update(samples, epochs=1)
                torch.cuda.synchronize()
            events = profiler.key_averages()
            rows = [dict(name=e.key, count=e.count, self_cpu_us=e.self_cpu_time_total,
                         total_cpu_us=e.cpu_time_total, self_device_us=e.self_device_time_total)
                    for e in events]
            result['cases'].append(dict(name=name, unprofiled_seconds=times,
                top_cpu=sorted(rows, key=lambda e: e['self_cpu_us'], reverse=True)[:40],
                top_device=sorted(rows, key=lambda e: e['self_device_us'], reverse=True)[:40]))
            save()
            print(json.dumps(result['cases'][-1]), flush=True)
        if args.parallel_streams:
            modules = (trainer.policy, trainer.critic)
            optimizers = (trainer.policy_optimizer, trainer.critic_optimizer)
            snapshot = [(copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))
                        for model, optimizer in zip(modules, optimizers, strict=True)]
            updates = (trainer._update_policy_batch, trainer._update_critic_batch)
            def restore():
                for model, optimizer, (weights, state) in zip(modules, optimizers, snapshot, strict=True):
                    model.load_state_dict(weights)
                    optimizer.load_state_dict(copy.deepcopy(state))
                trainer._configure_optimizer_backend()
            for update in updates:
                update(samples, epochs=1)
            expected = [(copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))
                        for model, optimizer in zip(modules, optimizers, strict=True)]
            restore()
            streams = [torch.cuda.Stream() for _ in range(2)]
            def concurrent_step(index):
                with torch.cuda.stream(streams[index]):
                    result = updates[index](samples, epochs=1)
                    streams[index].synchronize()
                return result
            times = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                torch.cuda.synchronize()
                futures = [executor.submit(concurrent_step, i) for i in range(2)]
                [future.result() for future in futures]
                max_error = 0.
                for model, optimizer, (weights, state) in zip(modules, optimizers, expected, strict=True):
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value, weights[key], atol=2e-6, rtol=2e-5)
                        max_error = max(max_error, float((value - weights[key]).abs().max()))
                    actual = optimizer.state_dict()['state']
                    for key, value in state['state'].items():
                        for name, expected_value in value.items():
                            if isinstance(expected_value, torch.Tensor):
                                torch.testing.assert_close(actual[key][name], expected_value, atol=2e-6, rtol=2e-5)
                for _ in range(5):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    futures = [executor.submit(concurrent_step, i) for i in range(2)]
                    [future.result() for future in futures]
                    times.append(time.perf_counter() - start)
            result['parallel_streams'] = dict(seconds=times, matches_sequential_parameters_and_adam=True,
                                               max_parameter_difference=max_error)
            print(json.dumps(result['parallel_streams']), flush=True)
        result['complete'] = True
        save()
    finally:
        trainer.logger.close()


if __name__ == '__main__':
    main()
