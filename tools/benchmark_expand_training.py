"""Bounded fresh MAPPO training with the production architecture and schedule."""
from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import shutil
import time

import torch

from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/expand_mappo_training.yaml')
    parser.add_argument('--mode', choices=('four_dark', 'two_player', 'double_open'),
                        default='four_dark')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--init-from', help='initialize each isolated benchmark from the same trained weights')
    parser.add_argument('--resume-from', help='copy one complete checkpoint into an isolated benchmark')
    parser.add_argument('--resume-assets', help='run directory containing the immutable historical/champion models')
    parser.add_argument('--actor-graph-tokens', type=int, default=4096)
    parser.add_argument('--actor-graph-sequences', type=int, default=32)
    parser.add_argument('--profile-update', type=int, default=0,
                        help='profile this one-based benchmark round; exclude it from timing comparisons')
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--batch', type=int, default=98304)
    parser.add_argument('--games', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--microbatch', type=int, default=32)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--no-learner-graphs', action='store_true')
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('benchmark requires a new empty directory')
    if args.updates < 1:
        raise ValueError('benchmark updates must be positive')
    if args.torch_threads < 1:
        raise ValueError('torch threads must be positive')
    if args.resume_from and args.init_from:
        raise ValueError('choose either full resume or weights-only initialization')
    torch.set_num_threads(args.torch_threads)
    settings = TrainingSettings.from_yaml(args.config, args.mode, model_scale='main', overrides=dict(
        device='cuda', anchor_batch=args.batch, base_game_pool_size=args.games,
        actor_inference_batch=args.games, rollout_environment_workers=args.workers,
        policy_microbatch=args.microbatch, ppo_learner_cuda_graphs=not args.no_learner_graphs,
        ppo_actor_graph_tokens=args.actor_graph_tokens, ppo_actor_graph_sequences=args.actor_graph_sequences,
        **({} if args.resume_from else dict(historical_enabled=False, arena_enabled=False)),
        checkpoint_policy='evaluation',
        inference_snapshot_every_updates=0))
    if args.resume_from:
        checkpoint = settings.resolve_run_directory(directory) / 'checkpoints/latest.pt'
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.resume_from, checkpoint)
        assets = Path(args.resume_assets) if args.resume_assets else Path(args.resume_from).resolve().parent.parent
        for name in ('historical_opponents', 'model_selection'):
            if (assets / name).is_dir():
                shutil.copytree(assets / name, checkpoint.parent.parent / name)
    trainer = SelfPlayTrainer(settings, run_directory=directory, initialize_from=args.init_from,
                              expand_game_pool=bool(args.resume_from))
    # Disposable timing runs always restart from the supplied source checkpoint.
    # Production resumes still save an expanded pool before their first action.
    trainer.pool_expanded_on_resume = False
    rows = []
    profiler = cProfile.Profile() if args.profile_update else None
    original = trainer.logger.log
    torch.cuda.synchronize()
    last = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def log(update, metrics):
        nonlocal last
        original(update, metrics)
        if 'rollout/environment_plies' not in metrics:
            return
        if profiler and len(rows) + 1 == args.profile_update:
            profiler.disable()
            profiler.dump_stats(str(directory / 'profile.pstats'))
        torch.cuda.synchronize()
        now = time.perf_counter()
        rows.append(dict(update=update, seconds=now-last, profiled=len(rows)+1 == args.profile_update,
                         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                         peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                         actor_graph_shapes=len(getattr(trainer, '_policy_learner_graph', None) or {}),
                         actor_graph_captures=sum(g.captures for g in
                             (getattr(trainer, '_policy_learner_graph', None) or {}).values()),
                         metrics=dict(metrics)))
        (directory/'benchmark.json').write_text(json.dumps(dict(complete=False, updates=rows), indent=2))
        print(json.dumps(dict(probe_update=update, seconds=now-last,
                              steps=metrics['rollout/environment_plies'],
                              gpu_gib=rows[-1]['peak_reserved_gib'])), flush=True)
        last = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        if len(rows) >= args.updates:
            trainer.stop_requested = True
        elif profiler and len(rows) + 1 == args.profile_update:
            profiler.enable()

    trainer.logger.log = log
    if profiler and args.profile_update == 1:
        profiler.enable()
    trainer.train()
    (directory/'benchmark.json').write_text(json.dumps(dict(complete=True,
        settings=settings.serializable(), updates=rows), indent=2))


if __name__ == '__main__':
    main()
