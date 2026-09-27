"""Reconstruct exploration trends from persisted training metrics."""
from __future__ import annotations

from collections import OrderedDict, deque
import gzip
import json
import math
from pathlib import Path

ENTROPY_WINDOW_UPDATES = 20


def normalized_action_entropy(metrics):
    """Combine phase moments before dividing; forced moves have zero weight."""
    masses, fractions = [], []
    for phase in ("opening", "other"):
        mass = metrics.get(f"policy/{phase}_entropy_ratio_mass")
        fraction = metrics.get(f"policy/{phase}_entropy_fraction")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
               for v in (mass, fraction)):
            return None
        masses.append(mass)
        fractions.append(fraction)
    count = sum(fractions)
    if count <= 0:
        return None
    ratio = sum(masses) / count
    return min(1., ratio) if ratio <= 1. + 1e-6 else None


def entropy_history(records, window_updates=ENTROPY_WINDOW_UPDATES, *,
                    ratio_reader=normalized_action_entropy):
    """Average per-update entropy by environment plies, not optimizer epochs.

    Records have already been restricted to the monitor's valid resume lineage.
    Missing observations occupy their place in the window, without becoming zero.
    Existing metrics are the persistent source, so page/service restarts are safe.
    """
    window = deque(maxlen=window_updates)
    history = []
    mean = None
    for row in records:
        ratio = ratio_reader(row)
        steps = row.get("rollout/environment_plies")
        weight = (steps if ratio is not None and isinstance(steps, (int, float))
                  and math.isfinite(steps) and steps > 0 else 0)
        window.append({"update": row.get("update"), "ratio": ratio, "weight": weight})
        covered = [item for item in window if item["weight"] > 0]
        total = sum(item["weight"] for item in covered)
        mean = sum(item["ratio"] * item["weight"] for item in covered) / total if total else None
        # The monitor reads a bounded log tail. Do not draw its warm-up prefix
        # as a full-window historical average when older rows were not loaded.
        history.append({"update": row.get("update"), "ratio": ratio,
                        "moving_average": mean if len(window) == window_updates else None})
    covered = [item for item in window if item["weight"] > 0]
    return {"window_updates": window_updates, "available_updates": len(window),
            "covered_updates": len(covered),
            "start_update": window[0]["update"] if window else None,
            "end_update": window[-1]["update"] if window else None,
            "environment_plies": sum(item["weight"] for item in covered),
            "latest_ratio": history[-1]["ratio"] if history else None,
            "moving_average": mean,
            "history": history}


def source_entropy_observation(row):
    """Marginalize one saved complete joint distribution over legal sources.

    Only post-update, learnable-policy probes qualify. Source 0 follows the
    model's action encoding, including the pass action; count legal categories,
    not all pieces on the board. Forced sources have no normalized observation.
    """
    if row.get("learnable") is not True:
        return None
    actions = row.get("legal_actions")
    after = row.get("after")
    values = after.get("legal_probabilities") if isinstance(after, dict) else None
    if not isinstance(actions, list) or not isinstance(values, list) or not actions or len(actions) != len(values):
        return None
    sources, seen = {}, set()
    for action, probability in zip(actions, values):
        if (not isinstance(action, (list, tuple)) or len(action) != 2
                or any(type(x) is not int or x < 0 for x in action)
                or tuple(action) in seen or type(probability) not in (int, float)
                or not math.isfinite(probability) or probability < 0):
            return None
        seen.add(tuple(action))
        sources[action[0]] = sources.get(action[0], 0.) + probability
    total = math.fsum(sources.values())
    if not math.isclose(total, 1., abs_tol=1e-4):
        return None
    entropy = -math.fsum((p / total) * math.log(p / total) for p in sources.values() if p > 0)
    ratio = entropy / math.log(len(sources)) if len(sources) > 1 else None
    return {"entropy": max(0., entropy), "ratio": min(1., max(0., ratio)) if ratio is not None else None}


class SourceEntropyStore:
    """Read cached CPU-only tactical probes without another policy forward.

    These probes are stratified by tactical category, not a uniform sample of
    every training state. Keep their provenance explicit in the dashboard.
    File fingerprints invalidate cached updates after overwrite/resume.
    """
    def __init__(self):
        self.cache = OrderedDict()

    def _read(self, path, update, *, not_before=0.):
        try:
            info = path.stat()
            if info.st_mtime < not_before:
                return None
            stamp = (info.st_mtime_ns, info.st_size)
            cached = self.cache.get(path)
            if cached is not None and cached[0] == stamp:
                self.cache.move_to_end(path)
                return cached[1]
            moments = {"samples": 0, "eligible": 0, "ratio_sum": 0., "entropy_sum": 0.}
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if not isinstance(row, dict) or row.get("update") != update:
                        continue
                    observation = source_entropy_observation(row)
                    if observation is None:
                        continue
                    moments["samples"] += 1
                    moments["entropy_sum"] += observation["entropy"]
                    if observation["ratio"] is not None:
                        moments["eligible"] += 1
                        moments["ratio_sum"] += observation["ratio"]
            self.cache[path] = (stamp, moments)
            self.cache.move_to_end(path)
            while len(self.cache) > 512:
                self.cache.popitem(last=False)
            return moments
        except (OSError, EOFError, ValueError, UnicodeError):
            # A missing/pruned/incomplete archive is unknown, never zero entropy.
            return None

    def snapshot(self, run, records, *, world_size=1, session_started=None,
                 resumed_update=None, window_updates=ENTROPY_WINDOW_UPDATES):
        observations = []
        for row in records:
            update = row.get("update")
            parts = []
            if type(update) is int and update >= 0:
                not_before = (session_started - 2 if session_started is not None
                              and resumed_update is not None and update > resumed_update else 0.)
                for rank in range(world_size):
                    path = Path(run) / "tactical_diagnostics/details" / f"update_{update:09d}.rank{rank:03d}.jsonl.gz"
                    parts.append(self._read(path, update, not_before=not_before))
            complete = bool(parts) and all(part is not None for part in parts)
            samples = sum(p["samples"] for p in parts) if complete else 0
            eligible = sum(p["eligible"] for p in parts) if complete else 0
            observations.append({"update": update,
                "rollout/environment_plies": row.get("rollout/environment_plies"),
                "ratio": sum(p["ratio_sum"] for p in parts) / eligible if eligible else None,
                "raw_entropy": sum(p["entropy_sum"] for p in parts) / samples if samples else None,
                "samples": samples, "eligible": eligible})
        result = entropy_history(observations, window_updates, ratio_reader=lambda row: row["ratio"])
        recent = observations[-window_updates:]
        latest = observations[-1] if observations else {}
        result.update(measurement="tactical_probe_after", phase="after",
            probe_samples=sum(row["samples"] for row in recent),
            eligible_samples=sum(row["eligible"] for row in recent),
            latest_probe_samples=latest.get("samples", 0),
            latest_eligible_samples=latest.get("eligible", 0),
            latest_raw_entropy=latest.get("raw_entropy"))
        return result
