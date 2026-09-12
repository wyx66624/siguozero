"""Append-only player observations, shared by immutable indexed PPO samples.

Only the live context window is owned by ArrayHistory. Older blocks survive
while a rollout sample references them, then are reclaimed with that sample.
No referee identities or hidden information enter these arrays.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from itertools import count
from functools import lru_cache

import numpy as np

from .encoding import ActionFeatures, StateTokenRecord
from .modes import TrainingMode, mode_spec


_IDENTITIES = count()
BLOCK_ROWS = 256


def new_history_identity() -> int:
    return next(_IDENTITIES)


@lru_cache(maxsize=3)
def _coordinate_codes(mode: TrainingMode) -> dict:
    from .encoding import action_point_coordinates
    return {action_point_coordinates(code, mode): code for code in range(mode_spec(mode).point_count)}


def record_array(record: StateTokenRecord, mode: TrainingMode, dead_rules: bool,
                 out: np.ndarray | None = None) -> np.ndarray:
    """Validate once, when an observed record enters the numerical history."""
    points = mode_spec(mode).point_count
    if len(record.board_codes) != points:
        raise ValueError("history board length does not match its mode")
    width = points + (75 if dead_rules else 0)
    row = np.empty(width + 10, dtype=np.int16) if out is None else out
    row.fill(0)
    row[:points] = record.board_codes
    if dead_rules:
        bits = record.known_casualty_bits
        expected = 25 if mode is TrainingMode.TWO_PLAYER else 50 if mode is TrainingMode.DOUBLE_OPEN else 75
        if bits is None:
            raise ValueError("dead-rule model requires casualty history")
        if len(bits) != expected:
            raise ValueError("history casualty length does not match its mode")
        if any(value not in (0, 1) for value in bits):
            raise ValueError("history casualty values must be binary")
        if mode is TrainingMode.FOUR_DARK:
            row[points:points + 75] = bits
        elif mode is TrainingMode.DOUBLE_OPEN:
            row[points:points + 25] = bits[:25]
            row[points + 50:points + 75] = bits[25:]
        else:
            row[points + 25:points + 50] = bits
    elif record.known_casualty_bits is not None:
        raise ValueError("non-dead-rule model must not receive casualty history")
    if record.action is not None:
        row[width:width + 5] = record.action.as_vector(mode)
    # Scalars precede no tensor construction; int16 represents all fields exactly.
    row[width + 5:] = (record.action is not None, record.no_interaction_plies,
                      record.active_mask, record.revealed_mask, record.current_player)
    return row


class HistoryArrayView(Sequence[StateTokenRecord]):
    """A snapshot of a pinned initial observation and a bounded context window."""

    __slots__ = ("mode", "dead_rules", "identity", "window_start", "chunks", "_length")

    def __init__(self, mode: TrainingMode, dead_rules: bool, identity: tuple[int, int],
                 window_start: int, chunks: tuple[np.ndarray, ...]) -> None:
        self.mode, self.dead_rules, self.identity = mode, dead_rules, identity
        self.window_start, self.chunks = window_start, chunks
        for chunk in chunks:
            chunk.setflags(write=False)
        self._length = sum(len(chunk) for chunk in chunks)

    @property
    def cache_key(self) -> tuple[tuple[int, int], int, int]:
        return self.identity, self.window_start, self._length

    def __len__(self) -> int:
        return self._length

    def copy_rows(self, destination: np.ndarray, *, start: int = 0) -> None:
        """Copy a numerical slice, without visiting records or creating tensors."""
        if start < 0 or start + len(destination) > len(self):
            raise IndexError("history array slice exceeds snapshot")
        written = 0
        for chunk in self.chunks:
            if start >= len(chunk):
                start -= len(chunk)
                continue
            take = min(len(chunk) - start, len(destination) - written)
            destination[written:written + take] = chunk[start:start + take]
            written += take
            if written == len(destination):
                break
            start = 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if start == 0 and step == 1 and stop > 0:
                chunks = []
                for chunk in self.chunks:
                    take = min(stop, len(chunk))
                    chunks.append(chunk[:take])
                    stop -= take
                    if not stop:
                        break
                return HistoryArrayView(self.mode, self.dead_rules, self.identity,
                                        self.window_start, tuple(chunks))
            return tuple(self[position] for position in range(start, stop, step))
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("history record index out of range")
        for chunk in self.chunks:
            if index < len(chunk):
                row = chunk[index]
                break
            index -= len(chunk)
        points = mode_spec(self.mode).point_count
        width = points + (75 if self.dead_rules else 0)
        bits = None
        if self.dead_rules:
            raw = row[points:width]
            if self.mode is TrainingMode.DOUBLE_OPEN:
                raw = np.concatenate((raw[:25], raw[50:]))
            elif self.mode is TrainingMode.TWO_PLAYER:
                raw = raw[25:50]
            bits = tuple(map(int, raw))
        action = None
        if row[width + 5]:
            # Serialization/debugging only. Numerical model paths use copy_rows.
            coordinates = _coordinate_codes(self.mode)
            action = ActionFeatures(coordinates[tuple(row[width:width + 2])],
                                    coordinates[tuple(row[width + 2:width + 4])], int(row[width + 4]))
        return StateTokenRecord(tuple(map(int, row[:points])), bits, action,
                                *map(int, row[width + 6:width + 10]))

    def __iter__(self) -> Iterator[StateTokenRecord]:
        return (self[index] for index in range(len(self)))


class ArrayHistory:
    """Preallocated observation blocks; writes never modify an existing row."""

    def __init__(self, mode: TrainingMode, records: Sequence[StateTokenRecord],
                 max_transitions: int, identity: tuple[int, int]) -> None:
        self.mode, self.identity, self.max_transitions = mode, identity, max_transitions
        self.dead_rules = records[0].known_casualty_bits is not None
        self.initial = record_array(records[0], mode, self.dead_rules)[None, :]
        self.blocks: deque[tuple[int, np.ndarray]] = deque()
        self.end = 0
        for record in records[1:]:
            self.append(record)

    def append(self, record: StateTokenRecord) -> None:
        self.append_row(record_array(record, self.mode, self.dead_rules))

    def append_row(self, row: np.ndarray) -> None:
        """Copy one worker-produced observation; published rows stay immutable."""
        if row.dtype != np.int16 or row.shape != self.initial.shape[1:]:
            raise ValueError('invalid packed history observation row')
        position = self.end % BLOCK_ROWS
        if position == 0:
            self.blocks.append((self.end + 1, np.empty((BLOCK_ROWS, self.initial.shape[1]), dtype=np.int16)))
        self.blocks[-1][1][position] = row
        self.end += 1
        start = max(1, self.end - self.max_transitions + 1)
        while self.blocks and self.blocks[0][0] + BLOCK_ROWS <= start:
            self.blocks.popleft()

    def view(self) -> HistoryArrayView:
        start = max(1, self.end - self.max_transitions + 1)
        chunks = [self.initial]
        for first, block in self.blocks:
            left, right = max(start, first), min(self.end + 1, first + BLOCK_ROWS)
            if left < right:
                chunks.append(block[left - first:right - first])
        return HistoryArrayView(self.mode, self.dead_rules, self.identity, start, tuple(chunks))
