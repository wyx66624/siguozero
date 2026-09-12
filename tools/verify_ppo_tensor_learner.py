"""CUDA packed-history forward/gradient validation and local timing."""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import json
from pathlib import Path
import tempfile
import time

import torch

from benchmark_ppo import diverse_states
from junqi.training.history_arrays import ArrayHistory
from junqi.training.models import GameValueTransformer
from junqi.training.packed_attention import forward, prepare
from junqi.training.settings import TrainingSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--low-precision-residual', action='store_true')
    parser.add_argument('--compile-mode', default='default')
    parser.add_argument('--static-tokens', type=int, default=0)
    parser.add_argument('--static-sequences', type=int, default=16)
    args = parser.parse_args()
    lock = open(Path(tempfile.gettempdir()) / 'siguozero-cuda-probe.lock', 'a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.manual_seed(619)
    torch.cuda.set_per_process_memory_fraction(.90)
    root = Path(__file__).resolve().parents[1]
    settings = TrainingSettings.from_yaml(root / 'configs/bootstrap.yaml', 'four_dark', model_scale='main')
    model = GameValueTransformer(replace(settings.model, ppo_tensor_learner=False,
        ppo_low_precision_residual=False, ppo_compile_mode='off')).cuda().train()
    torch.nn.init.normal_(model.value_head.weight, std=.05)
    bases = diverse_states(settings.mode, settings.model, 1001, 4)
    states = []
    for i, length in enumerate((7, 97, 501, 1001)):
        history = ArrayHistory(settings.mode, bases[i].records[:length], 1000, (i, 0))
        if i == 3:
            for j in range(3):
                history.append(bases[i].records[-1])
                states.append(replace(bases[i], records=history.view()))
        else:
            for end in range(max(1, length - 6), length + 1, 2):
                states.append(replace(bases[i], records=history.view()[:end]))
    packed, lengths = prepare(model, states)
    if args.static_tokens:
        raw, positions, cu, queries, maximum, mode = packed
        groups = args.static_sequences
        remaining = args.static_tokens - raw.shape[1]
        dummy = groups - len(lengths)
        if not dummy <= remaining <= dummy * maximum:
            raise ValueError('static token bucket cannot represent these histories plus dummy sequences')
        tails = [remaining // dummy + (i < remaining % dummy) for i in range(dummy)]
        extra_positions = torch.cat([torch.arange(n, device='cuda') for n in tails])[None, :]
        boundaries = cu.cpu().tolist()
        for length in tails:
            boundaries.append(boundaries[-1] + length)
        packed = (torch.cat((raw, raw.new_zeros((1, remaining, raw.shape[2]))), dim=1),
                  torch.cat((positions, extra_positions), dim=1),
                  torch.tensor(boundaries, device='cuda', dtype=torch.int32), queries, maximum, mode)
    result = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, cases=[], complete=False,
                  max_history=1001, independent_windows=len(lengths),
                  static_tokens=args.static_tokens, static_sequences=args.static_sequences,
                  compile_mode=args.compile_mode)
    path = Path(args.output)
    def save():
        path.write_text(json.dumps(result, indent=2) + '\n')
    expected, reference_grads = None, None
    compiled = torch.compile(lambda *data: forward(model, *data), dynamic=not bool(args.static_tokens), mode=args.compile_mode)
    for name in ('reference', 'packed', 'compiled'):
        if name != 'reference':
            model.config = replace(model.config, ppo_low_precision_residual=args.low_precision_residual)
        def run():
            model.zero_grad(set_to_none=True)
            if args.compile_mode == 'reduce-overhead':
                torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = (model(states, pack_sequences=True) if name == 'reference' else
                          model.value_head((forward if name == 'packed' else lambda m, *x: compiled(*x))(model, *packed)).squeeze(-1).float())
                output.square().mean().backward()
            return output.detach()
        torch.cuda.synchronize()
        started = time.perf_counter()
        if name == 'compiled' and args.compile_mode == 'reduce-overhead':
            # Validate a replay after warmup/capture, not only the first eager
            # execution used to build a CUDA Graph.
            run()
            run()
        observed = run()
        torch.cuda.synchronize()
        first = time.perf_counter() - started
        grads = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None}
        if expected is None:
            expected, reference_grads = observed.clone(), grads
            relative = maximum = 0.
        else:
            torch.testing.assert_close(expected, observed, atol=.02, rtol=.02)
            squared = sum(float((grads[k] - v).square().sum()) for k, v in reference_grads.items())
            norm = sum(float(v.square().sum()) for v in reference_grads.values())
            relative = (squared / norm) ** .5
            maximum = max(float((grads[k] - v).abs().max()) for k, v in reference_grads.items())
            assert relative < .025, relative
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            run()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
        result['cases'].append(dict(name=name, first_call_seconds=first, seconds=times,
            gradient_relative_l2_error=relative, gradient_max_absolute_error=maximum,
            forward_max_absolute_error=float((observed - expected).abs().max())))
        save()
        print(json.dumps(result['cases'][-1]), flush=True)
    if args.compile_mode == 'reduce-overhead':
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            run()
            torch.cuda.synchronize()
        result['graph_launches_in_profiled_replay'] = sum(e.count for e in profile.key_averages()
                                                         if 'cudaGraphLaunch' in e.key)
        result['compile_counters'] = {key: dict(value) for key, value in torch._dynamo.utils.counters.items()}
    result['complete'] = True
    save()


if __name__ == '__main__':
    main()
