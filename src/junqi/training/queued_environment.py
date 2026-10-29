"""Bounded lookahead with completion-order pipe draining and stable game order."""
from collections import deque
import select

from .ppo_environment import ParallelPPOEnvironment, _send


class QueuedPPOEnvironment(ParallelPPOEnvironment):
    supports_lookahead = True

    def __init__(self, workers):
        super().__init__(workers)
        self._tickets = {}
        self._worker_tickets = []
        self._inflight_indices = set()
        self._next_ticket = 0

    def begin(self, *args, **kwargs):
        if self._tickets:
            raise RuntimeError('cannot initialize an outstanding environment wave')
        super().begin(*args, **kwargs)
        self._worker_tickets = [deque() for _ in self.processes]

    def submit(self, indices, actions):
        indices = tuple(indices)
        if len(self._tickets) >= 2:
            raise RuntimeError('environment lookahead is bounded to two waves')
        if len(set(indices)) != len(indices) or self._inflight_indices.intersection(indices):
            raise RuntimeError('a game already has an unconsumed step')
        grouped = [[] for _ in self.processes]
        assignment = getattr(self, 'worker_assignment', None)
        for index, action in zip(indices, actions, strict=True):
            worker = assignment[index] if assignment is not None else index % len(grouped)
            grouped[worker].append((index, action))
        ticket = self._next_ticket
        self._next_ticket += 1
        waiting = set()
        for worker, items in enumerate(grouped):
            if not items:
                continue
            replacements = [(i, self.pending.pop(i)) for i, _ in items if i in self.pending]
            _send(self.processes[worker].stdin, ('step', (replacements, items)))
            self._worker_tickets[worker].append(ticket)
            waiting.add(worker)
        self._tickets[ticket] = [waiting, [], indices]
        self._inflight_indices.update(indices)
        self.active_workers = [-1]
        return ticket

    def receive(self, ticket=None):
        if ticket is None:
            ticket = next(iter(self._tickets))
        waiting, results, indices = self._tickets[ticket]
        while waiting:
            streams = {p.stdout: i for i, p in enumerate(self.processes)
                       if self._worker_tickets[i]}
            ready, _, _ = select.select(list(streams), [], [])
            for stream in ready:
                worker = streams[stream]
                result_ticket = self._worker_tickets[worker].popleft()
                rows, seconds = self._result(worker)
                # Copy before the worker can reuse a slot. Consumer histories
                # keep owning independent arrays, including out-of-order replies.
                if self.shared_observations is not None:
                    rows = self.shared_observations.unpack(rows)
                self.worker_seconds += seconds
                entry = self._tickets[result_ticket]
                entry[0].remove(worker)
                entry[1].extend(rows)
        del self._tickets[ticket]
        self._inflight_indices.difference_update(indices)
        self.active_workers = [-1] if self._tickets else []
        results.sort(key=lambda row: row.index)
        for row in results:
            self.states[row.index] = row.state
        return results

    def close(self):
        # Fully drain bounded replies before close commands. This also avoids
        # workers blocking on stdout while the parent waits for their exit.
        try:
            while self._tickets:
                self.receive()
        finally:
            super().close()
