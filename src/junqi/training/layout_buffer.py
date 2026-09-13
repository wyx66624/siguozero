"""Bounded, age-limited layout replay without retaining board setup objects."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .models import layout_sample_from_trace
from .modes import TrainingMode, normalize_mode
from .rollout import LayoutOutcome


_MODES = (TrainingMode.FOUR_DARK, TrainingMode.DOUBLE_OPEN, TrainingMode.TWO_PLAYER)
_DTYPE = np.dtype([
    ("positions", "u1", (25,)), ("logs", "<f4", (25,)),
    ("reward", "<f8"), ("seat", "u1"), ("version", "<i8"), ("mode", "u1"),
])


@dataclass(frozen=True, slots=True)
class LayoutReplayTrace:
    position_indices: tuple[int, ...]
    old_log_probs: tuple[float, ...]
    mode: TrainingMode

    @property
    def setup(self):
        # Only game reconstruction/debugging needs a PlayerSetup, never learning.
        return layout_sample_from_trace(self.mode, self.position_indices, self.old_log_probs).setup


class LayoutOutcomeBuffer(Sequence):
    """FIFO ring; admission/expiry depend on age, never on game outcome.

    Checkpoints retain the existing list-of-records contract. Legacy queues are
    filtered before reconstructing samples, and all discarded counts are exposed.
    """

    def __init__(self, capacity: int, max_age: int):
        if capacity < 2 or max_age < 0:
            raise ValueError("layout buffer needs capacity >= 2 and age >= 0")
        self.capacity, self.max_age = capacity, max_age
        self._data = np.empty(capacity, dtype=_DTYPE)
        self._head = self._size = 0
        self.current_version = 0
        self.minimum_version = 0
        self.totals = dict(enqueued=0, consumed=0, expired=0, overflow=0)
        self._reported = dict(self.totals)

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError(index)
        row = self._data[(self._head + index) % self.capacity]
        return LayoutOutcome(
            sample=LayoutReplayTrace(tuple(map(int, row["positions"])),
                                     tuple(map(float, row["logs"])), _MODES[int(row["mode"])]),
            reward=float(row["reward"]), seat=int(row["seat"]),
            behavior_version=int(row["version"]),
        )

    def _ordered(self):
        return self._data[(self._head + np.arange(self._size)) % self.capacity]

    @property
    def cutoff(self):
        return max(self.minimum_version, self.current_version - self.max_age) if self.max_age else self.minimum_version

    def advance(self, version: int, *, minimum_version: int = 0):
        self.current_version = int(version)
        self.minimum_version = max(self.minimum_version, int(minimum_version))
        if self._size:
            rows = self._ordered()
            keep = rows["version"] >= self.cutoff
            expired = int((~keep).sum())
            if expired:
                rows = rows[keep]
                self._data[:len(rows)] = rows
                self._head, self._size = 0, len(rows)
                self.totals["expired"] += expired

    def append(self, outcome: LayoutOutcome):
        self.append_record(dict(mode=outcome.sample.mode, position_indices=outcome.sample.position_indices,
                                old_log_probs=outcome.sample.old_log_probs, reward=outcome.reward,
                                seat=outcome.seat, behavior_version=outcome.behavior_version))

    def append_record(self, row):
        version = int(row["behavior_version"])
        self.totals["enqueued"] += 1
        if version < self.cutoff:
            self.totals["expired"] += 1
            return
        positions, logs = row["position_indices"], row["old_log_probs"]
        if len(positions) != 25 or len(logs) != 25 or sorted(positions) != list(range(25)):
            raise ValueError("layout replay must contain a permutation of 25 positions")
        if not np.isfinite(logs).all() or not np.isfinite(row["reward"]):
            raise ValueError("layout replay contains nonfinite values")
        mode = normalize_mode(row["mode"])
        seat = int(row["seat"])
        if not 0 <= seat < (2 if mode is TrainingMode.TWO_PLAYER else 4) or version < 0:
            raise ValueError("invalid layout seat or behavior version")
        if self._size == self.capacity:
            self._head = (self._head + 1) % self.capacity
            self._size -= 1
            self.totals["overflow"] += 1
        index = (self._head + self._size) % self.capacity
        self._data[index] = (positions, logs, float(row["reward"]), seat, version, _MODES.index(mode))
        self._size += 1

    def extend(self, outcomes: Iterable[LayoutOutcome]):
        for outcome in outcomes:
            self.append(outcome)

    def take(self, count: int):
        count = min(count, self._size)
        outcomes = self[:count]
        self._head = (self._head + count) % self.capacity
        self._size -= count
        self.totals["consumed"] += count
        return outcomes

    def records(self):
        return [dict(mode=item.sample.mode.value, position_indices=item.sample.position_indices,
                     old_log_probs=item.sample.old_log_probs, reward=item.reward, seat=item.seat,
                     behavior_version=item.behavior_version) for item in self]

    def restore(self, rows, *, version: int, minimum_version: int = 0, totals=None):
        self._head = self._size = 0
        if totals is not None:
            self.totals = {key: int(totals.get(key, 0)) for key in self.totals}
        else:
            self.totals = dict(enqueued=len(rows), consumed=0, expired=0, overflow=0)
        self._reported = dict(self.totals)
        self.advance(version, minimum_version=minimum_version)
        initial_enqueued = self.totals["enqueued"]
        for row in rows:
            self.append_record(row)
        # Restoring is not new production. Expiry/overflow are still reported.
        self.totals["enqueued"] = initial_enqueued

    def metrics(self, consumed=()):
        result = {f"layout/{key}": self.totals[key] - self._reported[key] for key in self.totals}
        result.update({f"layout/total_{key}": value for key, value in self.totals.items()})
        self._reported = dict(self.totals)
        result.update({"layout/buffer_remaining": len(self), "layout/buffer_capacity": self.capacity,
                       "layout/max_behavior_age": self.max_age, "layout/buffer_storage_bytes": self._data.nbytes})
        if consumed:
            ages = np.asarray([self.current_version - x.behavior_version for x in consumed])
            result.update({"layout/sample_age_mean": float(ages.mean()),
                           "layout/sample_age_p95": float(np.percentile(ages, 95)),
                           "layout/sample_age_max": int(ages.max())})
        if self._size:
            result["layout/buffer_age_max"] = int(self.current_version - self._ordered()["version"].min())
        return result
