"""Bounded fresh MAPPO training with the production architecture and schedule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/expand_mappo_training.yaml')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--batch', type=int, default=98304)
    parser.add_argument('--games', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--microbatch', type=int, default=32)
    parser.add_argument('--no-learner-graphs', action='store_true')
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('benchmark requires a new empty directory')
    if args.updates < 1:
        raise ValueError('benchmark updates must be positive')
    torch.set_num_threads(4)
    settings = TrainingSettings.from_yaml(args.config, 'four_dark', model_scale='main', overrides=dict(
        device='cuda', anchor_batch=args.batch, base_game_pool_size=args.games,
        actor_inference_batch=args.games, rollout_environment_workers=args.workers,
        policy_microbatch=args.microbatch, ppo_learner_cuda_graphs=not args.no_learner_graphs,
        historical_enabled=False, arena_enabled=False, checkpoint_policy='evaluation',
        inference_snapshot_every_updates=0))
    trainer = SelfPlayTrainer(settings, run_directory=directory)
    rows = []
    original = trainer.logger.log
    torch.cuda.synchronize()
    last = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def log(update, metrics):
        nonlocal last
        original(update, metrics)
        if 'rollout/environment_plies' not in metrics:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        rows.append(dict(update=update, seconds=now-last,
                         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                         peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                         metrics=dict(metrics)))
        (directory/'benchmark.json').write_text(json.dumps(dict(complete=False, updates=rows), indent=2))
        print(json.dumps(dict(probe_update=update, seconds=now-last,
                              steps=metrics['rollout/environment_plies'],
                              gpu_gib=rows[-1]['peak_reserved_gib'])), flush=True)
        last = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        if len(rows) >= args.updates:
            trainer.stop_requested = True

    trainer.logger.log = log
    trainer.train()
    (directory/'benchmark.json').write_text(json.dumps(dict(complete=True,
        settings=settings.serializable(), updates=rows), indent=2))


if __name__ == '__main__':
    main()
