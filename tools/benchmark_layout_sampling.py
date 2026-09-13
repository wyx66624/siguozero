"""Time complete layout generation with the same weights and output count."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import time

import torch
import junqi.training.models as loaded_models
from junqi.training.models import PieceConditionedLayoutPointerDecoder
from junqi.training.settings import TrainingSettings


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=64)
    p.add_argument('--iterations',type=int,default=8)
    args=p.parse_args()
    if min(args.count,args.iterations)<1:
        p.error('count and iterations must be positive')
    lock=open(Path(tempfile.gettempdir())/'siguozero-cuda-probe.lock','a+b')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    path=Path(args.checkpoint)
    stat=path.stat()
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    settings=TrainingSettings.from_yaml(args.config,'four_dark',model_scale='main')
    model=PieceConditionedLayoutPointerDecoder(settings.model).cuda().eval()
    model.load_state_dict(payload['layout'])
    torch.cuda.manual_seed_all(4321)
    rows=[]
    for i in range(args.iterations+1):
        torch.cuda.synchronize()
        started=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            samples=model.sample_layouts(args.count,'four_dark')
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        assert all(sorted(s.position_indices)==list(range(25)) and len(s.setup)==25 for s in samples)
        rows.append(dict(iteration=i,warmup=i==0,seconds=elapsed,count=len(samples)))
    source=Path(loaded_models.__file__)
    result=dict(checkpoint=str(path),checkpoint_update=payload['update'],count=args.count,iterations=rows,
                mean_seconds=sum(x['seconds'] for x in rows[1:])/args.iterations,
                source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                graphs=len(getattr(model,'_layout_sampling_graphs',{})),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                all_layouts_valid=True,checkpoint_stat_unchanged=(path.stat().st_size,path.stat().st_mtime_ns)==(stat.st_size,stat.st_mtime_ns))
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('mean_seconds','count','graphs','peak_allocated_gib','all_layouts_valid')}))


if __name__=='__main__':
    main()
