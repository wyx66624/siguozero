"""Bounded PPO feedback, observed once after a complete learner update."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class BoundedLearningRate:
    minimum: float
    maximum: float
    target_kl: float
    high_kl_multiple: float = 1.5
    increase_factor: float = 1.03
    decrease_factor: float = 0.8
    stable_updates: int = 5
    cooldown_updates: int = 5
    ema_decay: float = 0.8
    recovery_clip_fraction: float = 0.1
    scale: float = 1.0
    kl_ema: float | None = None
    stable_count: int = 0
    cooldown: int = 0
    observations: int = 0
    reason: str = "initial"

    def rate(self, scheduled: float) -> float:
        if not math.isfinite(scheduled) or scheduled <= 0:
            raise ValueError("scheduled learning rate must be finite and positive")
        base = max(self.minimum, scheduled)
        ceiling = min(self.maximum, base)
        # Clamp the stored multiplier too: hitting a bound must not accumulate
        # hidden increases/decreases that would delay a later direction change.
        self.scale = min(ceiling / base, max(self.minimum / base, self.scale))
        return min(ceiling, max(self.minimum, base * self.scale))

    def observe(self, *, scheduled: float, kl: float | None,
                clip_fraction: float | None, early_stopped: bool,
                allow_increase: bool = True) -> tuple[float, int]:
        before = self.rate(scheduled)
        self.reason = "hold"
        if kl is None or clip_fraction is None:
            self.stable_count = 0
            self.reason = "no_samples"
            return before, 0
        if (not math.isfinite(kl) or kl < 0 or not math.isfinite(clip_fraction)
                or not 0 <= clip_fraction <= 1):
            raise ValueError("learning rate observations must be finite KL and clip fraction")
        self.observations += 1
        self.kl_ema = (kl if self.kl_ema is None else
                       self.ema_decay * self.kl_ema + (1 - self.ema_decay) * kl)
        cooling = self.cooldown > 0
        self.cooldown = max(0, self.cooldown - 1)
        high = self.target_kl * self.high_kl_multiple
        if kl > high:
            # A measured violation can lower LR even during recovery cooldown.
            self.scale *= self.decrease_factor
            self.stable_count = 0
            self.reason = "high_kl"
        elif early_stopped or clip_fraction > self.recovery_clip_fraction:
            self.stable_count = 0
            self.reason = "early_stop" if early_stopped else "clip_guard"
        elif cooling or not allow_increase:
            self.stable_count = 0
            self.reason = "cooldown" if cooling else "warmup"
        elif kl < self.target_kl * .5 and self.kl_ema < self.target_kl * .5:
            self.stable_count += 1
            self.reason = "collecting_stable"
            if self.stable_count >= self.stable_updates:
                self.scale *= self.increase_factor
                self.stable_count = 0
                self.reason = "low_kl"
        else:
            self.stable_count = 0
        after = self.rate(scheduled)
        direction = (after > before) - (after < before)
        if direction:
            self.cooldown = self.cooldown_updates
        elif self.reason in ("low_kl", "high_kl"):
            self.reason = "upper_bound" if self.reason == "low_kl" else "lower_bound"
        return after, direction

    def state_dict(self) -> dict:
        return {"version": 1, "scale": self.scale, "kl_ema": self.kl_ema,
                "stable_count": self.stable_count, "cooldown": self.cooldown,
                "observations": self.observations}

    def load_state_dict(self, state: dict | None, *, legacy_scale: float,
                        scheduled: float) -> None:
        state = state or {}
        scale = state.get("scale", legacy_scale)
        ema = state.get("kl_ema")
        counts = [state.get(key, 0) for key in ("stable_count", "cooldown", "observations")]
        if (state.get("version", 1) != 1 or not isinstance(scale, (int, float))
                or isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0
                or (ema is not None and (not isinstance(ema, (int, float))
                    or isinstance(ema, bool) or not math.isfinite(ema) or ema < 0))
                or any(type(value) is not int or value < 0 for value in counts)):
            raise ValueError("invalid learning rate controller checkpoint")
        self.scale, self.kl_ema = float(scale), ema
        self.stable_count = min(counts[0], self.stable_updates - 1)
        self.cooldown = min(counts[1], self.cooldown_updates)
        self.observations = counts[2]
        self.rate(scheduled)  # new configured bounds apply immediately on resume


def diagnostic_sequences(samples, *, limit: int, sequence_length: int, seed: int):
    """Sample bounded history chains without advancing the training RNG."""
    import random
    from .ppo import sequence_training_batches
    owned = [sample for sample in samples if sample.learnable]
    if len(owned) <= limit:
        return owned
    chains = sequence_training_batches(owned, sequences_per_batch=1,
                                      max_samples_per_sequence=sequence_length)
    random.Random(seed).shuffle(chains)
    selected = []
    for chain in chains:
        remaining = limit - len(selected)
        selected.extend(chain[:remaining])
        if len(selected) == limit:
            break
    return selected
