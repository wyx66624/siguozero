"""Learning-signal diagnostics from existing rollout scalars, without inference."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from .ppo import PPOSample


def ppo_signal_sums(samples: Sequence[PPOSample]) -> dict[str, float]:
    """Additive moments, to be summed across ranks before computing diagnostics."""
    advantages = [sample.advantage for sample in samples]
    targets = [sample.value_target for sample in samples]
    errors = [sample.value_target - sample.old_value for sample in samples]
    return {
        "count": float(len(samples)),
        "advantage_sum": math.fsum(advantages),
        "advantage_square_sum": math.fsum(value * value for value in advantages),
        "advantage_nonzero": float(sum(value != 0 for value in advantages)),
        "value_sum": math.fsum(sample.old_value for sample in samples),
        "target_sum": math.fsum(targets),
        "target_square_sum": math.fsum(value * value for value in targets),
        "target_nonzero": float(sum(value != 0 for value in targets)),
        "error_sum": math.fsum(errors),
        "error_square_sum": math.fsum(value * value for value in errors),
    }


def ppo_signal_metrics(sums: Mapping[str, float]) -> dict[str, float]:
    """Compare frozen rollout values with GAE targets, before normalization.

    Explained variance is a training diagnostic, not held-out playing strength.
    Constant targets have undefined EV even when their prediction error is zero.
    """
    count = sums["count"]
    if count <= 0:
        raise ValueError("PPO signal diagnostics require at least one global sample")

    def variance(prefix: str) -> float:
        return max(0.0, sums[f"{prefix}_square_sum"] / count
                   - (sums[f"{prefix}_sum"] / count) ** 2)

    target_variance = variance("target")
    defined = target_variance > 1e-12
    result = {
        "ppo/rollout_samples": count,
        "ppo/raw_advantage_mean": sums["advantage_sum"] / count,
        "ppo/raw_advantage_std": math.sqrt(variance("advantage")),
        "ppo/raw_nonzero_advantage_fraction": sums["advantage_nonzero"] / count,
        "ppo/no_advantage_signal": float(sums["advantage_nonzero"] == 0),
        "critic/rollout_value_mean": sums["value_sum"] / count,
        "critic/rollout_target_mean": sums["target_sum"] / count,
        "critic/rollout_target_std": math.sqrt(target_variance),
        "critic/rollout_nonzero_target_fraction": sums["target_nonzero"] / count,
        "critic/rollout_value_mse": sums["error_square_sum"] / count,
        "critic/rollout_explained_variance_defined": float(defined),
    }
    if defined:
        result["critic/rollout_explained_variance"] = 1 - variance("error") / target_variance
    return result
