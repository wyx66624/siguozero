"""Exclusive CUDA diagnostic: compare captured and ordinary PPO gradients."""
import copy
import argparse
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
from junqi.training.learner_graph import PackedLearnerGraph
from junqi.training.metrics import MetricLogger
from junqi.training.ppo import PPOSample, policy_ppo_loss, critic_ppo_loss
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer
from junqi.training.clipping import scheduled_policy_clip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--long-window', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--adaptive-clip', action='store_true')
    args = parser.parse_args()
    lock = open(Path(tempfile.gettempdir()) / 'siguozero-cuda-probe.lock', 'a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.manual_seed(177)
    settings = TrainingSettings.from_yaml('configs/local_4090_training.yaml', 'four_dark', model_scale='main', overrides={
        'device': 'cuda', 'arena_enabled': False, 'checkpoint_policy': 'evaluation',
        'historical_enabled': False, 'arena_after_half_historical_only': False})
    with patch.object(MetricLogger, 'start_resource_monitor'):
        trainer = SelfPlayTrainer(settings, run_directory=tempfile.mkdtemp(prefix='siguozero-graph-validation-'), auto_resume=False)
    templates = diverse_states(settings.mode, settings.model, 1001 if args.long_window else 769, 8)
    samples = []
    for i, state in enumerate(templates):
        history = ArrayHistory(settings.mode, state.records, 1000, (i, 0))
        for j, length in enumerate(range(1 + i * 8, 257 + i * 8, 4)):
            samples.append(PPOSample(replace(state, records=history.view()[:length]),
                state.legal_actions[j % len(state.legal_actions)], -4., .1, .1 if j % 2 else -.1,
                .1 if j % 2 else -.1, 0, old_draw_value=-.03, draw_value_target=-.07))
    if args.long_window:
        history = ArrayHistory(settings.mode, templates[0].records, 1000, (444, 0))
        views = [history.view()]
        history.append(templates[0].records[-1])
        views.append(history.view())
        history.append(templates[0].records[-1])
        views.append(history.view())
        samples = [replace(samples[i], state=replace(templates[0], records=view)) for i,view in enumerate(views)]
    results = []
    shared_pool = torch.cuda.graph_pool_handle()
    graph_cache = {}
    # Return to each model after the other model has overwritten shared scratch
    # memory, and alternate capacities in a different order than capture.
    for stage, (critic, model) in enumerate(((False, trainer.policy), (True, trainer.critic),
                          (False, trainer.policy), (True, trainer.critic))):
        model.train()
        if critic not in graph_cache:
            graph_cache[critic] = {}
        graphs = graph_cache[critic]
        optimizer = trainer.critic_optimizer if critic else trainer.policy_optimizer
        for group in optimizer.param_groups:
            group['lr'] = 2e-5
        for iteration, count in enumerate((3, 3, 3) if args.long_window else (128, 243, 512)):
            tokens = 4096 if args.long_window or iteration % 2 else 2560
            sources = 16384 if critic else (4096, 8192, 16384)[iteration]
            key = (tokens, sources)
            if key not in graphs:
                graphs[key] = PackedLearnerGraph(model, critic=critic, amp_dtype=trainer.amp_dtype,
                    tokens=tokens, sources=sources, pool=shared_pool)
            graph = graphs[key]
            batch = samples[:count]
            original = copy.deepcopy(model.state_dict())
            adam = copy.deepcopy(optimizer.state_dict())
            kwargs = dict(clip=settings.value_clip_epsilon if critic else settings.clip_epsilon,
                          coefficient=settings.value_coefficient if critic else .01,
                          opening_coefficient=None if critic else .02, opening_plies=16)
            # The second actor pass reuses exactly the same graph objects at a
            # later progress knot. Changing device scalars must not recapture.
            bounds = (scheduled_policy_clip(replace(settings, ppo_adaptive_clip=True),
                       (0, 1_000_000_000, 1_500_000_000)[iteration] if stage == 0 else 3_000_000_000)
                      if args.adaptive_clip and not critic else None)
            kwargs['policy_clip'] = bounds
            optimizer.zero_grad(set_to_none=True)
            with trainer._autocast():
                if critic:
                    output = critic_ppo_loss(model, batch, clip_epsilon=kwargs['clip'],
                        value_coefficient=kwargs['coefficient'], sequence_training=True, defer_metrics=True)
                else:
                    output = policy_ppo_loss(model, batch, clip_epsilon=kwargs['clip'], entropy_coefficient=.01,
                        policy_clip=bounds,
                        opening_entropy_coefficient=.02, entropy_opening_plies=16, sequence_training=True, defer_metrics=True)
                output.loss.backward()
            expected = {key: float(value) for key, value in output.metrics.items()}
            del output
            grads = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_norm_clip, error_if_nonfinite=True)
            optimizer.step()
            expected_parameters = [p.detach().clone() for p in model.parameters()]
            expected_adam = copy.deepcopy(optimizer.state_dict())
            model.load_state_dict(original)
            optimizer.load_state_dict(adam)
            trainer._configure_optimizer_backend()
            optimizer.zero_grad(set_to_none=True)
            assert graph.stage(model, batch, **kwargs), 'batch did not fit graph'
            start = time.perf_counter()
            actual = {key: float(value) for key, value in graph.run(model).items()}
            seconds = time.perf_counter() - start
            difference = torch.zeros((), device='cuda')
            denominator = torch.zeros((), device='cuda')
            for parameter, reference in zip(model.parameters(), grads, strict=True):
                assert (parameter.grad is None) == (reference is None)
                if reference is not None:
                    assert torch.isfinite(parameter.grad).all()
                    difference += (parameter.grad - reference).square().sum()
                    denominator += reference.square().sum()
            relative = float((difference / denominator.clamp_min(1e-30)).sqrt())
            for key, value in expected.items():
                assert abs(actual[key] - value) <= .006 + .025 * abs(value), (key, actual[key], value)
            assert relative < .035, relative
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_norm_clip, error_if_nonfinite=True)
            optimizer.step()
            parameter_error = max(float((p.detach() - q).abs().max()) for p, q in zip(model.parameters(), expected_parameters, strict=True))
            assert parameter_error < 5e-5, parameter_error
            actual_adam = optimizer.state_dict()
            for key, value in expected_adam['state'].items():
                assert torch.equal(actual_adam['state'][key]['step'], value['step'])
            times = []
            for _ in range(8):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                assert graph.stage(model, batch, **kwargs)
                graph.run(model)
                torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
            row = dict(critic=critic, samples=count, capture_seconds=seconds,
                       adaptive_clip=None if bounds is None else bounds.metrics(),
                       shared_scratch_pool=True,
                       full_sliding_windows=args.long_window,
                       graph_token_capacity=graph.TOKENS,
                       graph_source_capacity=graph.SOURCES,
                       grad_relative_l2=relative, max_parameter_difference=parameter_error,
                       stage_and_backward_seconds=times, captures=graph.captures, replays=graph.replays,
                       allocated_gib=torch.cuda.memory_allocated()/2**30,
                       reserved_gib=torch.cuda.memory_reserved()/2**30)
            results.append(row)
            print(json.dumps(row), flush=True)
        # A replay with nonfinite targets must still stop before the Adam step.
        invalid = [replace(batch[0], **({'value_target': float('nan')} if critic else {'advantage': float('nan')})), *batch[1:]]
        original_parameters = [p.detach().clone() for p in model.parameters()]
        original_steps = [s['step'].clone() for s in optimizer.state.values()]
        optimizer.zero_grad(set_to_none=True)
        assert graph.stage(model, invalid, **kwargs)
        bad_metrics = graph.run(model)
        try:
            trainer._clip_and_materialize(model, bad_metrics)
        except RuntimeError as error:
            assert 'non-finite' in str(error)
        else:
            raise AssertionError('nonfinite graph gradients were accepted')
        assert all(torch.equal(p,q) for p,q in zip(model.parameters(),original_parameters,strict=True))
        assert all(torch.equal(s['step'],q) for s,q in zip(optimizer.state.values(),original_steps,strict=True))
        optimizer.zero_grad(set_to_none=True)
        print(json.dumps(dict(critic=critic, nonfinite_replay_rejected_before_adam=True)), flush=True)
        del graph, graphs
    suffix = 'long_window' if args.long_window else 'validation'
    path = args.output or f'docs/benchmarks/throughput_learner_graph_{suffix}_4090_20260913.json'
    Path(path).write_text(json.dumps(results, indent=2)+'\n')
    trainer.logger.close()


if __name__ == '__main__':
    main()
