"""Matched real-checkpoint weight swaps; read-only with respect to training.

Measures cache miss, RAM hit, and prefetched pinned upload separately. The
prefetch preparation time is reported, not silently counted as free compute.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import tempfile
import time
from types import SimpleNamespace

import torch

from junqi.training.arena import atomic_json, sha256_file
from junqi.training.distributed import DistributedContext
from junqi.training.encoding import GameHistory
from junqi.training.historical_opponents import HistoricalOpponents, packed_weights
from junqi.training.models import GamePolicyTransformer, PieceConditionedLayoutPointerDecoder
from junqi.training.modes import new_game
from junqi.training.settings import TrainingSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/local_4090_training.yaml')
    args = parser.parse_args()
    lock = open(Path(tempfile.gettempdir())/'siguozero-cuda-probe.lock', 'a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Production training must be paused; do not trust a GPU-utilization instant.
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if b'-m\x00junqi.training.cli\x00' in proc.read_bytes():
                raise RuntimeError('pause the production trainer before this GPU probe')
        except (FileNotFoundError, PermissionError):
            pass
    torch.set_num_threads(4)
    torch.manual_seed(1907)
    config = TrainingSettings.from_yaml(args.config, 'four_dark', model_scale='main')
    root = Path(args.directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    context = DistributedContext(0, 1, 0, torch.device('cuda'))
    policy = GamePolicyTransformer(config.model).eval()
    layout = PieceConditionedLayoutPointerDecoder(config.model).eval()
    template = HistoricalOpponents(config, root, context)
    directory = template.directory
    directory.mkdir(parents=True, exist_ok=True)
    entries, provenance = [], []
    for value in args.checkpoint:
        path = Path(value).resolve()
        before = path.stat()
        payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        policy.load_state_dict(payload['policy'])
        layout.load_state_dict(payload['layout'])
        p, pp = packed_weights(policy)
        l, lp = packed_weights(layout)
        target = directory/f"probe_u{payload['update']:09d}.pt"
        frozen = {k: payload[k] for k in ('format_version','mode','dead_rules_enabled','update')}
        frozen.update(config=config.serializable(), policy=p, layout=l, packed_weights={'policy':pp,'layout':lp})
        torch.save(frozen, target)
        entries.append(dict(id=target.stem,file=target.name,sha256=sha256_file(target),kind='early',
                            update=payload['update'],environment_plies=payload['trainer_state']['cumulative']['environment_plies'],
                            milestones=[],bytes=target.stat().st_size))
        provenance.append(dict(path=str(path),sha256=sha256_file(path),update=payload['update']))
        assert (before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
        del payload, frozen, p, l, pp, lp
    atomic_json(template.catalog_path,dict(contract=template.contract(),opponents=entries))
    del template, policy, layout
    gc.collect()
    # Identical legal observations for numerical verification on every version.
    rng = random.Random(11)
    states=[]
    for i in range(8):
        game=new_game('four_dark',seed=619+i,no_capture_draw_plies=config.no_capture_draw_plies)
        history=GameHistory.initialize(game,'four_dark',max_transitions=config.model.max_transitions)
        for _ in range(16):
            if game.is_terminal: break
            game.step(rng.choice(game.legal_actions())); history.append_after_step(game)
        states.append(history.state_for(game))
    actor=SimpleNamespace(amp_dtype=torch.bfloat16,max_batch_size=128)
    reference={}
    measurements={}
    for variant in ('uncached','ram','prefetched'):
        league=HistoricalOpponents(replace(config,historical_cache_gib=0 if variant=='uncached' else 2),root,context)
        league.progress=league.threshold
        if variant!='uncached':
            recipe=GamePolicyTransformer(config.model)
            league._configure_weight_cache(recipe)
            del recipe
        times, preparation, cold, pointers = [], [], [], None
        start_reads=0
        for index, entry in enumerate(entries*4):
            if variant=='prefetched':
                start=time.perf_counter()
                league.weight_cache.prefetch(entry)
                league.weight_cache._pending[1].result()
                preparation.append(time.perf_counter()-start)
            torch.cuda.synchronize()
            start=time.perf_counter()
            league.active_id=entry['id']
            league._load_active(actor)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            (cold if index<len(entries) else times).append(elapsed)
            current={k:v.data_ptr() for k,v in league.device_packs.items()}
            if pointers is None: pointers=current
            assert pointers==current
            if index<len(entries):
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=[x.cpu() for x in league.policy(states,[s.legal_actions for s in states])]
                if variant=='uncached': reference[entry['id']]=logits
                else:
                    for got,want in zip(logits,reference[entry['id']],strict=True):
                        torch.testing.assert_close(got,want,rtol=0,atol=0)
                league.clear_caches()
        measurements[variant]=dict(cold_load_seconds=cold,warm_switch_seconds=times,
            warm_switch_median_seconds=statistics.median(times),prefetch_preparation_seconds=preparation,
            metrics=league.metrics(),cuda_allocated_bytes=torch.cuda.memory_allocated())
        print(variant, measurements[variant]['warm_switch_median_seconds'],flush=True)
        league.release_replica(); league.close(); del league
        gc.collect(); torch.cuda.empty_cache()
    result=dict(kind='real_checkpoint_cache_and_weight_swap_diagnostic',production_training_modified=False,
                device=torch.cuda.get_device_name(),torch=torch.__version__,checkpoints=provenance,
                versions=len(entries),swaps_per_variant=len(entries)*4,identical_logits=True,
                measurements=measurements,limits=[
                    'Warm baseline includes CPU repacking and H2D; filesystem pages may already be cached.',
                    'Prefetch waits outside the timed switch; preparation is reported separately.',
                    'No optimizer, full game, evaluation or long-run ETA is measured here.'])
    atomic_json(Path(args.output),result)


if __name__=='__main__':
    main()
