"""Numerical checks and timings for fused AdamW and inference sampling graphs."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import tempfile
import time

import torch
import triton

from junqi.training.encoding import GameHistory
from junqi.training.models import GamePolicyTransformer
from junqi.training.modes import new_game
from junqi.training.ppo_sampling import sample_graph, sample_tensor
from junqi.training.settings import TrainingSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    lock = open(Path(tempfile.gettempdir()) / 'siguozero-cuda-probe.lock', 'a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.manual_seed(9134)
    root = Path(__file__).resolve().parents[1]
    config = TrainingSettings.from_yaml(root/'configs/bootstrap.yaml', 'four_dark', model_scale='main').model
    left = GamePolicyTransformer(config).cuda()
    right = copy.deepcopy(left)
    options = dict(lr=1e-4, betas=(.9,.95), weight_decay=.05)
    a = torch.optim.AdamW(left.parameters(), **options)
    b = torch.optim.AdamW(right.parameters(), fused=True, **options)
    for x,y in zip(left.parameters(),right.parameters(),strict=True):
        x.grad = torch.randn_like(x)
        y.grad = x.grad.clone()
    for _ in range(4):
        a.step()
        b.step()
    max_error = 0.
    for x,y in zip(left.parameters(),right.parameters(),strict=True):
        torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-6)
        max_error=max(max_error,float((x-y).detach().abs().max()))
        for name in ('step','exp_avg','exp_avg_sq'):
            torch.testing.assert_close(a.state[x][name],b.state[y][name],atol=2e-6,rtol=2e-6,check_device=False)
    timings={}
    for name,opt in (('original',a),('fused',b)):
        torch.cuda.synchronize()
        started=time.perf_counter()
        for _ in range(20): opt.step()
        torch.cuda.synchronize()
        timings[name]=(time.perf_counter()-started)/20
    del right,b,a
    left.zero_grad(set_to_none=True)
    left.eval()
    game=new_game('four_dark',seed=91)
    state=GameHistory.initialize(game,'four_dark').state_for(game)
    states=[state]*64
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        masks=left._sampling_legal_masks(states,129)
        contexts=torch.randn(64,config.temporal_dim,device='cuda')
        uniforms=torch.rand(64,2,device='cuda')
        uniforms[0]=0.
        uniforms[1]=1.-2**-24
        expected=sample_tensor(left,contexts,*masks,uniforms,1.)
        rng=torch.cuda.get_rng_state()
        actual=sample_graph(left,contexts,*masks,uniforms,1.).clone()
        assert torch.equal(rng,torch.cuda.get_rng_state())
        torch.testing.assert_close(expected,actual,atol=1e-5,rtol=1e-5)
        for row in actual[:,:2].long().cpu().tolist():
            assert tuple(row) in state.legal_actions
        uniforms=1.-uniforms.clamp_min(2**-24)
        expected2=sample_tensor(left,contexts,*masks,uniforms,.7)
        actual2=sample_graph(left,contexts,*masks,uniforms,.7).clone()
        torch.testing.assert_close(expected2,actual2,atol=1e-5,rtol=1e-5)
        eager_ms=triton.testing.do_bench(lambda:sample_tensor(left,contexts,*masks,uniforms,1.),warmup=50,rep=200)
        graph_ms=triton.testing.do_bench(lambda:sample_graph(left,contexts,*masks,uniforms,1.),warmup=50,rep=200)
    left.train()
    assert not left._ppo_sampling_graphs
    result=dict(complete=True,adamw_step_seconds=timings,adamw_max_parameter_error=max_error,
                sampling_eager_ms=eager_ms,sampling_graph_ms=graph_ms,
                sampling_matches_eager=True,capture_does_not_consume_rng=True,
                parameter_update_clears_graphs=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
