"""ACPO advantage-aware PPO clipping with an optional legacy schedule.

The upper-bound erf modulation follows ACPO (arXiv:2510.00690, eqs. 5-6).
An explicit delta is additive; otherwise legacy base-scaled widening applies.
The progress schedule, absolute-advantage bounds and optional unclipped tail are
project-specific extensions, not its reuse curriculum.
Advantages are the detached, globally normalized PPO rollout advantages; never
renormalize inside a microbatch or include frozen seats/padding in their moments.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


DEFAULT_CLIP_SCHEDULE = (
    (0.0, 0.25, 0.30),
    (1 / 3, 0.20, 0.25),
    (0.5, 0.18, 0.20),
    (0.75, 0.15, 0.15),
    (1.0, 0.10, 0.10),
)


@dataclass(frozen=True)
class PolicyClip:
    lower: float
    upper: float
    bonus: float = 0.0
    minimum: float | None = 0.1
    maximum: float | None = 0.45
    progress: float = 0.0
    absolute_advantage: bool = False
    unclipped_advantage_threshold: float = 0.0
    progress_scheduled: bool = False

    @classmethod
    def fixed(cls, epsilon):
        return cls(epsilon, epsilon, minimum=epsilon, maximum=epsilon)

    def metrics(self):
        high = self.upper + self.bonus
        metrics = {"policy/clip_progress": self.progress,
                "policy/clip_progress_scheduled": float(self.progress_scheduled),
                "policy/clip_hard_bounds_enabled": float(self.minimum is not None or self.maximum is not None),
                "policy/clip_lower": self.lower,
                "policy/clip_upper_base": self.upper,
                "policy/clip_delta": self.bonus,
                "policy/clip_upper_max": high if self.maximum is None else min(self.maximum, high),
                "policy/clip_absolute_advantage": float(self.absolute_advantage),
                "policy/unclipped_advantage_threshold": self.unclipped_advantage_threshold}
        if self.minimum is not None:
            metrics["policy/clip_minimum"] = self.minimum
        if self.maximum is not None:
            metrics["policy/clip_maximum"] = self.maximum
        return metrics


def scheduled_policy_clip(settings, environment_plies: int) -> PolicyClip:
    if not settings.ppo_adaptive_clip:
        return PolicyClip.fixed(settings.clip_epsilon)
    if settings.ppo_clip_mode == "acpo":
        return PolicyClip(settings.ppo_clip_lower, settings.ppo_clip_upper,
                          settings.ppo_clip_delta, minimum=None, maximum=None)
    progress = min(1.0, max(0.0, environment_plies / settings.target_environment_plies))
    schedule = settings.ppo_clip_schedule
    lower, upper = schedule[-1][1:]
    for left, right in zip(schedule, schedule[1:]):
        if progress <= right[0]:
            fraction = (progress - left[0]) / (right[0] - left[0])
            lower = left[1] + fraction * (right[1] - left[1])
            upper = left[2] + fraction * (right[2] - left[2])
            break
    minimum, maximum = settings.ppo_clip_minimum, settings.ppo_clip_maximum
    bonus = (upper * settings.ppo_clip_advantage_scale
             if settings.ppo_clip_delta is None else settings.ppo_clip_delta)
    return PolicyClip(min(maximum, max(minimum, lower)),
                      min(maximum, max(minimum, upper)),
                      bonus, minimum, maximum, progress,
                      settings.ppo_clip_absolute_advantage, settings.ppo_unclipped_advantage_threshold,
                      progress_scheduled=True)


def advantage_clip_upper(advantage, upper, bonus, minimum, maximum):
    # No gradients through the bound; all operands stay on the learner device.
    percentile = 0.5 * (1 + torch.erf(advantage.detach() / math.sqrt(2)))
    bound = upper + bonus * percentile
    return bound if minimum is None and maximum is None else bound.clamp(min=minimum, max=maximum)


def advantage_clip_bounds(advantage, lower, upper, bonus, minimum, maximum, absolute=False):
    """Use signed ACPO upper bounds, or the legacy absolute-advantage extension."""
    magnitude = advantage.detach().abs()
    enabled = torch.as_tensor(absolute, device=advantage.device) > 0
    signal = torch.where(enabled, magnitude, advantage.detach())
    high = advantage_clip_upper(signal, upper, bonus, minimum, maximum)
    low = torch.where(enabled, high.clamp_min(lower), torch.zeros_like(high) + lower)
    return low, high


def unclipped_advantage_mask(advantage, threshold):
    return (threshold > 0) & (advantage.detach().abs() >= threshold)


def clipped_surrogate(ratio, advantage, lower, upper, unclipped_threshold=0.0):
    clipped = ratio.clamp_min(1 - lower).minimum(1 + upper)
    objective = torch.minimum(ratio * advantage, clipped * advantage)
    protected = unclipped_advantage_mask(advantage, unclipped_threshold)
    objective = torch.where(protected, ratio * advantage, objective)
    outside = ((ratio < 1 - lower) | (ratio > 1 + upper)) & ~protected
    return objective, outside.float()
