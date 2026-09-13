"""Bounded, progress-scheduled advantage-aware PPO clipping.

The upper-bound erf modulation follows ACPO (arXiv:2510.00690, eqs. 5-6).
The progress schedule is a project-specific extension, not its reuse curriculum.
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
    minimum: float = 0.1
    maximum: float = 0.45
    progress: float = 0.0

    @classmethod
    def fixed(cls, epsilon):
        return cls(epsilon, epsilon, minimum=epsilon, maximum=epsilon)

    def metrics(self):
        return {"policy/clip_progress": self.progress,
                "policy/clip_lower": self.lower,
                "policy/clip_upper_base": self.upper,
                "policy/clip_upper_max": min(self.maximum, self.upper + self.bonus),
                "policy/clip_minimum": self.minimum,
                "policy/clip_maximum": self.maximum}


def scheduled_policy_clip(settings, environment_plies: int) -> PolicyClip:
    if not settings.ppo_adaptive_clip:
        return PolicyClip.fixed(settings.clip_epsilon)
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
    return PolicyClip(min(maximum, max(minimum, lower)),
                      min(maximum, max(minimum, upper)),
                      upper * settings.ppo_clip_advantage_scale, minimum, maximum, progress)


def advantage_clip_upper(advantage, upper, bonus, minimum, maximum):
    # No gradients through the bound; all operands stay on the learner device.
    percentile = 0.5 * (1 + torch.erf(advantage.detach() / math.sqrt(2)))
    return (upper + bonus * percentile).clamp(min=minimum, max=maximum)


def clipped_surrogate(ratio, advantage, lower, upper):
    clipped = ratio.clamp_min(1 - lower).minimum(1 + upper)
    objective = torch.minimum(ratio * advantage, clipped * advantage)
    outside = (ratio < 1 - lower) | (ratio > 1 + upper)
    return objective, outside.float()
