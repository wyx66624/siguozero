"""Persistent CPU processes for PPO rules and player-view observations.

Workers own rules-only games, never models or CUDA contexts. Per-step messages
contain actions and new observation rows, not whole histories. The parent keeps
immutable training histories and synchronizes complete games at rollout bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import io
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import time
import traceback

import numpy as np

from ..game import GameConfig, JunqiGame
from .packed_observation import observation_rows


@dataclass(frozen=True, slots=True)
class EnvironmentState:
    player: int | None
    legal_actions: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class EnvironmentStep:
    index: int
    state: EnvironmentState
    records: np.ndarray
    terminal: bool
    reward: float
    rewards: tuple[float, ...]
    next_team_sign: int


def environment_state(game):
    return EnvironmentState(game.current_player, tuple(game.legal_actions()) if not game.is_terminal else ())


def _advance(index, game, action, mode):
    player = game.current_player
    if player is None:
        raise RuntimeError('terminal game remained in PPO environment')
    team = game.team_of(player)
    game.step(action)
    rows = observation_rows(game, mode)
    terminal = game.is_terminal
    rewards = tuple(game.rewards()) if terminal else ()
    return EnvironmentStep(index, environment_state(game), rows, terminal,
                           rewards[player] if terminal else 0., rewards,
                           1 if terminal or game.team_of(game.current_player) == team else -1)


@lru_cache(maxsize=2)
def _shared_boards(variant):
    # The rules engine's clone() already shares these read-only geometries.
    template = JunqiGame.__new__(JunqiGame)
    template._initialize_empty(GameConfig(variant=variant))
    return template._boards


def _restore_game(fields):
    game = JunqiGame.__new__(JunqiGame)
    game.__dict__.update(fields)
    game._boards = _shared_boards(game.config.variant)
    return game


class _EnvironmentPickler(pickle.Pickler):
    def reducer_override(self, value):
        if type(value) is JunqiGame:
            # Do not send four immutable board graphs for every game/reset.
            # Preserve every mutable rule field, including private knowledge.
            return _restore_game, ({key: item for key, item in vars(value).items() if key != '_boards'},)
        return NotImplemented


def _serialize(value):
    buffer = io.BytesIO()
    _EnvironmentPickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    return buffer.getvalue()


def _send(stream, value):
    payload = _serialize(value)
    remaining = memoryview(struct.pack('!Q', len(payload)) + payload)
    while remaining:
        written = stream.write(remaining)
        if not written:
            raise BrokenPipeError('PPO environment worker pipe closed while writing')
        remaining = remaining[written:]
    stream.flush()


def _read_exact(stream, count):
    chunks = bytearray()
    while len(chunks) < count:
        piece = stream.read(count - len(chunks))
        if not piece:
            raise EOFError('PPO environment worker pipe closed')
        chunks.extend(piece)
    return chunks


def _receive(stream):
    size, = struct.unpack('!Q', _read_exact(stream, 8))
    return pickle.loads(_read_exact(stream, size))


def _worker():
    games, mode = {}, None
    try:
        while True:
            command, payload = _receive(sys.stdin.buffer)
            if command == 'close':
                return
            if command == 'initialize':
                mode, replacements = payload
                games = dict(replacements)
                value = [(index, environment_state(game)) for index, game in games.items()]
            elif command == 'step':
                replacements, actions = payload
                games.update(replacements)
                started = time.perf_counter()
                value = ([_advance(index, games[index], action, mode) for index, action in actions],
                         time.perf_counter() - started)
            elif command == 'snapshot':
                games.update(payload)
                value = list(games.items())
            else:
                raise ValueError(f'unknown PPO environment command: {command}')
            _send(sys.stdout.buffer, (True, value))
    except BaseException:
        try:
            _send(sys.stdout.buffer, (False, traceback.format_exc()))
        finally:
            sys.exit(1)


class ParallelPPOEnvironment:
    """One outstanding step wave; receive it before touching parent histories."""

    def __init__(self, workers: int):
        if type(workers) is not int or workers < 1:
            raise ValueError('PPO environment workers must be a positive integer')
        self.workers = workers
        self.processes = []
        self.pending = {}
        self.active_workers = []
        self.states = []
        self.worker_seconds = 0.

    def _start(self, count):
        if self.processes:
            return
        env = os.environ.copy()
        source = str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH'] = source + os.pathsep + env.get('PYTHONPATH', '')
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            env[key] = '1'
        try:
            for _ in range(min(self.workers, count)):
                self.processes.append(subprocess.Popen(
                    [sys.executable, '-c', 'from junqi.training.ppo_environment import _worker; _worker()'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env=env, bufsize=0,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0))
        except BaseException:
            self.close()
            raise

    def _result(self, worker):
        process = self.processes[worker]
        try:
            success, value = _receive(process.stdout)
        except (EOFError, BrokenPipeError) as error:
            raise RuntimeError(f'PPO environment worker {worker} exited unexpectedly') from error
        if not success:
            raise RuntimeError(f'PPO environment worker {worker} failed:\n{value}')
        return value

    def begin(self, pool):
        if self.active_workers:
            raise RuntimeError('cannot replace a PPO pool during an outstanding step')
        self._start(len(pool.slots))
        self.pending.clear()
        self.states = [None] * len(pool.slots)
        for worker, process in enumerate(self.processes):
            games = [(i, slot.game) for i, slot in enumerate(pool.slots) if i % len(self.processes) == worker]
            _send(process.stdin, ('initialize', (pool.mode, games)))
        for worker in range(len(self.processes)):
            for index, state in self._result(worker):
                self.states[index] = state

    def replace(self, index, game):
        self.pending[index] = game
        self.states[index] = environment_state(game)

    def submit(self, indices, actions):
        if self.active_workers:
            raise RuntimeError('a PPO environment step is already outstanding')
        grouped = [[] for _ in self.processes]
        for index, action in zip(indices, actions, strict=True):
            grouped[index % len(grouped)].append((index, action))
        for worker, items in enumerate(grouped):
            if not items:
                continue
            replacements = [(i, self.pending.pop(i)) for i, _ in items if i in self.pending]
            _send(self.processes[worker].stdin, ('step', (replacements, items)))
            self.active_workers.append(worker)

    def receive(self):
        results = []
        for worker in self.active_workers:
            rows, seconds = self._result(worker)
            self.worker_seconds += seconds
            results.extend(rows)
        self.active_workers.clear()
        results.sort(key=lambda item: item.index)
        for result in results:
            self.states[result.index] = result.state
        return results

    def synchronize(self, pool):
        if self.active_workers:
            raise RuntimeError('cannot checkpoint an outstanding PPO environment step')
        for worker, process in enumerate(self.processes):
            replacements = [(i, game) for i, game in self.pending.items() if i % len(self.processes) == worker]
            _send(process.stdin, ('snapshot', replacements))
        snapshots = []
        for worker in range(len(self.processes)):
            snapshots.extend(self._result(worker))
        if sorted(index for index, _ in snapshots) != list(range(len(pool.slots))):
            raise RuntimeError('PPO environment snapshot omitted or duplicated a game')
        for index, game in snapshots:
            pool.slots[index].game = game
        self.pending.clear()

    def close(self):
        for process in self.processes:
            try:
                if process.poll() is None:
                    _send(process.stdin, ('close', None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            finally:
                for stream in (process.stdin, process.stdout, process.stderr):
                    stream.close()
        self.processes.clear()
        self.active_workers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
