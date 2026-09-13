"""Bounded entropy feedback and phase statistics, reusing learner entropies."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import torch
from torch import Tensor

from .encoding import PolicyState


@dataclass
class AdaptiveEntropyCoefficient:
    initial: float
    minimum: float
    maximum: float
    target_ratio: float
    adaptation_rate: float
    ema_decay: float
    enabled: bool = True
    coefficient: float = field(init=False)
    entropy_ema: float | None = field(default=None, init=False)
    updates: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.coefficient = self.initial

    def observe(self, ratio: float | None) -> None:
        """Update once per completed learner update, never once per microbatch."""
        if not self.enabled or ratio is None:
            return
        if not math.isfinite(ratio):
            raise ValueError("observed normalized entropy must be finite")
        ratio = min(1.0, max(0.0, ratio))
        self.entropy_ema = (ratio if self.entropy_ema is None else
                            self.ema_decay * self.entropy_ema + (1 - self.ema_decay) * ratio)
        # Positive target error increases the entropy reward. Log-space updates
        # preserve positivity; configured bounds limit both feedback directions.
        self.coefficient = min(self.maximum, max(self.minimum,
            self.coefficient * math.exp(self.adaptation_rate * (self.target_ratio - self.entropy_ema))))
        self.updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "enabled": self.enabled, "coefficient": self.coefficient,
                "entropy_ema": self.entropy_ema, "updates": self.updates}

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        # Old checkpoints start at the configured initial coefficient. Disabling
        # feedback explicitly restores fixed-coefficient training.
        if not state or not self.enabled or state.get("enabled") is False:
            return
        coefficient, ema, updates = state.get("coefficient"), state.get("entropy_ema"), state.get("updates")
        if (state.get("version") != 1 or type(coefficient) not in (float, int)
                or not math.isfinite(coefficient) or coefficient <= 0
                or (ema is not None and (type(ema) not in (float, int) or not math.isfinite(ema) or not 0 <= ema <= 1))
                or type(updates) is not int or updates < 0):
            raise ValueError("invalid adaptive entropy checkpoint state")
        self.coefficient = min(self.maximum, max(self.minimum, float(coefficient)))
        self.entropy_ema, self.updates = ema, updates


def policy_entropy_bonus(
    states: Sequence[PolicyState], entropies: Tensor, *, coefficient: float,
    opening_coefficient: float | None = None, opening_plies: int = 0,
    metadata: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Keep the raw entropy objective; normalize only the feedback statistic.

    Moments are means over the entire input batch, not means of each phase.
    Existing sample-weighted microbatch/rank reductions can then combine them
    before the final phase division. Forced single-action turns are ineligible.
    """
    entropies = entropies.float()
    if opening_coefficient is None:
        bonus = coefficient * entropies.mean()
        return bonus, {"loss/policy_entropy": -bonus.detach()}
    # One compact metadata upload (8 bytes/action), with no logits transfer or
    # extra forward pass. History keeps an initial token plus bounded actions;
    # the caller caps opening_plies at the retained transition capacity.
    if metadata is None:
        metadata = entropies.new_tensor([
            (math.log(len(state.legal_actions)) if len(state.legal_actions) > 1 else 0.,
             float(len(state.records) <= opening_plies)) for state in states
        ])
    max_entropy, opening = metadata.unbind(-1)
    eligible = (max_entropy > 0).to(entropies.dtype)
    # log(2) < 1: clamp only the zero denominator, without distorting binary choices.
    normalized = entropies / torch.where(max_entropy > 0, max_entropy, torch.ones_like(max_entropy)) * eligible
    weights = coefficient + (opening_coefficient - coefficient) * opening
    bonus = (weights * entropies).mean()
    metrics = {"loss/policy_entropy": -bonus.detach()}
    for name, mask in (("opening", opening * eligible), ("other", (1 - opening) * eligible)):
        metrics[f"policy/{name}_entropy_ratio_mass"] = (normalized * mask).mean().detach()
        metrics[f"policy/{name}_entropy_mass"] = (entropies * mask).mean().detach()
        metrics[f"policy/{name}_entropy_fraction"] = mask.mean().detach()
    return bonus, metrics


def phase_entropy_ratios(metrics: dict[str, float]) -> dict[str, float | None]:
    """Form ratios only after reducing all sample-weighted moments."""
    result: dict[str, float | None] = {}
    for phase in ("opening", "other"):
        fraction = metrics.get(f"policy/{phase}_entropy_fraction", 0.)
        ratio = metrics.get(f"policy/{phase}_entropy_ratio_mass", 0.) / fraction if fraction > 0 else None
        result[phase] = ratio
        if ratio is not None:
            metrics[f"policy/{phase}_entropy_ratio"] = ratio
            metrics[f"policy/{phase}_entropy"] = metrics[f"policy/{phase}_entropy_mass"] / fraction
    return result
