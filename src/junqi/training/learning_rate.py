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
    kl_band_lower: float | None = None
    kl_band_upper: float | None = None
    epoch_completion: bool = False
    minimum_completed_epochs: int = 2
    increase_kl_threshold: float = .09
    decrease_kl_threshold: float = .10
    scale: float = 1.0
    kl_ema: float | None = None
    stable_count: int = 0
    stable_direction: int = 0
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

    def observe(self, *, scheduled: float, kl: float | None, early_stopped: bool,
                allow_increase: bool = True,
                full_epochs_completed: bool | None = None,
                completed_epochs: float | None = None) -> tuple[float, int]:
        before = self.rate(scheduled)
        self.reason = "hold"
        if kl is None:
            self.stable_count = 0
            self.stable_direction = 0
            self.reason = "no_samples"
            return before, 0
        if not math.isfinite(kl) or kl < 0:
            raise ValueError("learning rate observations must be finite nonnegative KL")
        self.observations += 1
        self.kl_ema = (kl if self.kl_ema is None else
                       self.ema_decay * self.kl_ema + (1 - self.ema_decay) * kl)
        cooling = self.cooldown > 0
        self.cooldown = max(0, self.cooldown - 1)
        high = self.target_kl * self.high_kl_multiple
        if self.epoch_completion:
            # One decision per update; old streak/cooldown rules do not delay it.
            self.stable_count = self.stable_direction = self.cooldown = 0
            if completed_epochs is None or completed_epochs <= 0:
                self.reason = "no_samples"
            elif (completed_epochs <= self.minimum_completed_epochs
                  and kl > self.decrease_kl_threshold):
                self.scale *= self.decrease_factor
                self.reason = "early_high_kl"
            elif full_epochs_completed and kl < self.increase_kl_threshold and allow_increase:
                self.scale *= self.increase_factor
                self.reason = "completed_low_kl"
        elif self.kl_band_lower is not None:
            # Consecutive end-of-update probe means provide symmetric evidence.
            # Cooldown observations cannot count toward the next adjustment.
            trend = (1 if kl < self.kl_band_lower else
                     -1 if kl > self.kl_band_upper else 0)
            if cooling or trend == 0 or (trend > 0 and not allow_increase):
                self.stable_count = self.stable_direction = 0
                self.reason = "cooldown" if cooling else "hold" if trend == 0 else "warmup"
            else:
                self.stable_count = self.stable_count + 1 if trend == self.stable_direction else 1
                self.stable_direction = trend
                self.reason = "collecting_low_kl" if trend > 0 else "collecting_high_kl"
                if self.stable_count >= self.stable_updates:
                    self.scale *= self.increase_factor if trend > 0 else self.decrease_factor
                    self.stable_count = self.stable_direction = 0
                    self.reason = "low_kl" if trend > 0 else "high_kl"
        elif kl > high:
            # A measured violation can lower LR even during recovery cooldown.
            self.scale *= self.decrease_factor
            self.stable_count = 0
            self.reason = "high_kl"
        elif early_stopped:
            self.stable_count = 0
            self.reason = "early_stop"
        elif full_epochs_completed is not None:
            # Completion feedback replaces the low-KL/stability recovery rule.
            # A full pass budget is itself the evidence for another small rise;
            # measured KL violations above retain priority.
            self.stable_count = 0
            if full_epochs_completed and allow_increase:
                self.scale *= self.increase_factor
                self.reason = "completed_epochs"
            else:
                self.reason = "incomplete_epochs" if not full_epochs_completed else "warmup"
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
        if direction and not self.epoch_completion:
            self.cooldown = self.cooldown_updates
        elif not direction and self.reason in ("low_kl", "completed_epochs", "high_kl",
                                                "completed_low_kl", "early_high_kl"):
            self.reason = "upper_bound" if self.reason in ("low_kl", "completed_epochs", "completed_low_kl") else "lower_bound"
        return after, direction

    def state_dict(self) -> dict:
        return {"version": 3, "scale": self.scale, "kl_ema": self.kl_ema,
                "epoch_completion": self.epoch_completion,
                "stable_count": self.stable_count, "cooldown": self.cooldown,
                "stable_direction": self.stable_direction,
                "kl_band": ([self.kl_band_lower, self.kl_band_upper]
                            if self.kl_band_lower is not None else None),
                "observations": self.observations}

    def load_state_dict(self, state: dict | None, *, legacy_scale: float,
                        scheduled: float) -> None:
        state = state or {}
        scale = state.get("scale", legacy_scale)
        ema = state.get("kl_ema")
        counts = [state.get(key, 0) for key in ("stable_count", "cooldown", "observations")]
        trend = state.get("stable_direction", 0)
        if (state.get("version", 1) not in (1, 2, 3) or not isinstance(scale, (int, float))
                or isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0
                or (ema is not None and (not isinstance(ema, (int, float))
                    or isinstance(ema, bool) or not math.isfinite(ema) or ema < 0))
                or any(type(value) is not int or value < 0 for value in counts)
                or type(trend) is not int or trend not in (-1, 0, 1)):
            raise ValueError("invalid learning rate controller checkpoint")
        self.scale, self.kl_ema = float(scale), ema
        self.stable_count = min(counts[0], self.stable_updates - 1)
        self.stable_direction = trend
        self.cooldown = min(counts[1], self.cooldown_updates)
        self.observations = counts[2]
        band = [self.kl_band_lower, self.kl_band_upper] if self.kl_band_lower is not None else None
        if state.get("kl_band") != band:
            # A different band's streak is not evidence under the new bounds.
            # Within the same mean-KL mode, keep the EMA and remaining cooldown:
            # a threshold edit must not cancel observation after an LR change.
            self.stable_count = self.stable_direction = 0
            if state.get("kl_band") is None or band is None:
                self.kl_ema = None
                self.cooldown = 0
        self.rate(scheduled)  # new configured bounds apply immediately on resume
        if self.epoch_completion != bool(state.get("epoch_completion", False)):
            self.stable_count = self.stable_direction = self.cooldown = 0


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
