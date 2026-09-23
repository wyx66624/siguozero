"""Reconstruct exploration trends from persisted training metrics."""
from __future__ import annotations

from collections import deque
import math

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


def entropy_history(records, window_updates=ENTROPY_WINDOW_UPDATES):
    """Average per-update entropy by environment plies, not optimizer epochs.

    Records have already been restricted to the monitor's valid resume lineage.
    Missing observations occupy their place in the window, without becoming zero.
    Existing metrics are the persistent source, so page/service restarts are safe.
    """
    window = deque(maxlen=window_updates)
    history = []
    mean = None
    for row in records:
        ratio = normalized_action_entropy(row)
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
