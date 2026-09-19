"""Replay bounded CPU environment work from a read-only MAPPO checkpoint.

Use PYTHONPATH to select baseline/current source. The output digest checks the
actions, private/global rows and rewards on the same deterministic continuations.
No models or CUDA contexts are created, and the checkpoint is never modified.
"""
from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import json
from pathlib import Path
import pickle
import pstats
import random
import time

import torch

from junqi.game import JunqiGame
from junqi.training.futile_attack import restore_evidence
from junqi.training.modes import normalize_mode
from junqi.training.ppo_environment import _advance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--games', type=int, default=32)
    parser.add_argument('--plies-per-game', type=int, default=32)
    args = parser.parse_args()
    if min(args.games, args.plies_per_game) < 1:
        parser.error('games and plies-per-game must be positive')
    torch.set_num_threads(1)
    checkpoint = Path(args.checkpoint)
    before = checkpoint.stat()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    state = payload['trainer_state']
    pool = state.get('base_game_pool')
    if pool is None:
        pool = state['distributed']['rank_states'][0]['base_game_pool']
    mode, update = normalize_mode(pool['mode']), payload['update']
    games = []
    for slot in pool['slots'][:args.games]:
        raw = slot['game']
        fields = {key: value for key, value in raw.items()
                  if key not in ('config', 'pieces', 'futile_attack_evidence')}
        game = JunqiGame.from_position(raw['config'], raw['pieces'], **fields)
        restore_evidence(game, raw.get('futile_attack_evidence'))
        games.append(game)
    del payload, state, pool, raw, slot
    gc.collect()
    # Materialize legal moves before profiling, as the collector does at begin.
    for game in games:
        game.legal_actions()
    rng = random.Random(82914)
    rows = []
    profiler = cProfile.Profile()
    started = time.perf_counter()
    with profiler:
        for index, game in enumerate(games):
            for _ in range(args.plies_per_game):
                if game.is_terminal:
                    break
                action = rng.choice(game.legal_actions())
                result = _advance(index, game, action, mode,
                                  track_futile_attacks=True, centralized_critic=True)
                rows.append(result)
    elapsed = time.perf_counter() - started
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(output.with_suffix('.pstats')))
    digest = hashlib.sha256(pickle.dumps(rows, protocol=5)).hexdigest()
    report = dict(checkpoint=str(checkpoint.resolve()), checkpoint_update=update,
                  games=len(games), steps=len(rows), seconds=elapsed,
                  output_sha256=digest, source=str(Path(_advance.__code__.co_filename).resolve()),
                  checkpoint_unchanged=(before.st_size, before.st_mtime_ns) ==
                      (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns))
    assert report['checkpoint_unchanged']
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)
    pstats.Stats(profiler).strip_dirs().sort_stats('cumulative').print_stats(35)


if __name__ == '__main__':
    main()
