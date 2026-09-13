"""Matched layout optimizer work from an immutable checkpoint, selected by PYTHONPATH."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import torch

import junqi.training.models as loaded_models
from junqi.training.models import PieceConditionedLayoutPointerDecoder
from junqi.training.settings import TrainingSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--microbatch', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=3)
    args = parser.parse_args()
    if min(args.batch, args.microbatch, args.iterations) < 1:
        parser.error('batch, microbatch and iterations must be positive')
    lock = open(Path(tempfile.gettempdir())/'siguozero-cuda-probe.lock', 'a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    path = Path(args.checkpoint)
    stat = path.stat()
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    config = TrainingSettings.from_yaml(args.config, 'four_dark', model_scale='main').model
    model = PieceConditionedLayoutPointerDecoder(config).cuda().train()
    reference = PieceConditionedLayoutPointerDecoder(config).cuda().eval().requires_grad_(False)
    reference.load_state_dict(payload['reference_layout'])
    rows = payload['trainer_state']['layout_buffer'][-args.batch:]
    assert len(rows) == args.batch
    choices = torch.tensor([x['position_indices'] for x in rows], device='cuda')
    modes = torch.tensor([{'four_dark':0,'double_open':1,'two_player':2}[x['mode']] for x in rows], device='cuda')
    rewards = torch.tensor([x['reward'] for x in rows], device='cuda')
    old_logs = torch.tensor([x['old_log_probs'] for x in rows], device='cuda')
    std = rewards.std(unbiased=False)
    advantages = torch.where(std >= 1e-4, (rewards-rewards.mean())/(std+1e-4), torch.zeros_like(rewards))
    timings = []
    for iteration in range(args.iterations + 1):
        model.load_state_dict(payload['layout'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(.9,.95), eps=1e-8,
                                      weight_decay=.05, fused=True)
        optimizer.load_state_dict(copy.deepcopy(payload['layout_optimizer']))
        for group in optimizer.param_groups:
            group['fused'], group['foreach'] = True, None
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        total = torch.zeros((), device='cuda')
        outputs = []
        for start in range(0, args.batch, args.microbatch):
            end = min(start+args.microbatch, args.batch)
            weight = (end-start)/args.batch
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logs, entropy = model.evaluate_layouts(choices[start:end], modes[start:end])
                with torch.no_grad():
                    ref_logs, _ = reference.evaluate_layouts(choices[start:end], modes[start:end])
                ratio = (logs.float()-old_logs[start:end]).clamp(-20,20).exp()
                adv = advantages[start:end,None]
                loss = (-torch.minimum(ratio*adv, ratio.clamp(.8,1.2)*adv).mean()
                        + .02*.5*(logs.float()-ref_logs.float()).square().mean() - .01*entropy.float().mean())
            (loss*weight).backward()
            total += loss.detach()*weight
            outputs.append(logs.detach().float())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        timings.append(dict(iteration=iteration, warmup=iteration==0, seconds=elapsed,
                            layouts_per_second=args.batch/elapsed, loss=float(total), grad_norm=float(norm),
                            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        print(json.dumps(timings[-1]), flush=True)
        if iteration == args.iterations:
            signature = dict(logs=torch.cat(outputs).cpu(),
                             gradients={k:p.grad.cpu() for k,p in model.named_parameters()
                                        if k in ('point_embedding.weight','query.0.weight','layers.0.attention.in_proj_weight')})
        del optimizer
    steady = timings[1:]
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(signature, output.with_suffix('.signature.pt'))
    source = Path(loaded_models.__file__)
    result = dict(checkpoint=str(path), checkpoint_update=payload['update'], batch=args.batch,
                  microbatch=args.microbatch, optimizer_steps_per_iteration=1, timings=timings,
                  mean_seconds=sum(x['seconds'] for x in steady)/len(steady),
                  source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  torch=torch.__version__, device=torch.cuda.get_device_name(),
                  checkpoint_stat_unchanged=(stat.st_size,stat.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns),
                  batch_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())
    assert result['checkpoint_stat_unchanged']
    output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
