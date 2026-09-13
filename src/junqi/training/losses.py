"""Terminal-only root GRPO and layout policy losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .models import GamePolicyTransformer, PieceConditionedLayoutPointerDecoder
from .rollout import LayoutOutcome, PolicyGroup
from .entropy import policy_entropy_bonus


@dataclass(slots=True)
class LossOutput:
    loss: Tensor
    metrics: dict[str, float | Tensor]


def policy_grpo_loss(
    policy: nn.Module,
    reference: GamePolicyTransformer,
    groups: Sequence[PolicyGroup],
    *,
    clip_epsilon: float,
    kl_coefficient: float,
    entropy_coefficient: float,
    opening_entropy_coefficient: float | None = None,
    entropy_opening_plies: int = 0,
) -> LossOutput:
    """Root-only clipped GRPO; suffix actions never receive the root advantage."""

    if not groups:
        raise ValueError("policy GRPO requires at least one complete anchor group")
    states = [group.state for group in groups]
    all_actions = [state.legal_actions for state in states]
    # Calling the module entry point is required for DistributedDataParallel to
    # install its forward/backward synchronization hooks.
    current_logs = policy(states, all_actions)
    device = current_logs[0].device
    with torch.no_grad():
        reference_logs = reference.log_probs_for_action_groups(states, all_actions)

    candidate_current: list[Tensor] = []
    candidate_old: list[Tensor] = []
    advantages: list[Tensor] = []
    kls: list[Tensor] = []
    entropies: list[Tensor] = []
    for group, current, ref in zip(
        groups, current_logs, reference_logs, strict=True
    ):
        action_to_index = {
            action: index for index, action in enumerate(group.state.legal_actions)
        }
        indices = torch.tensor(
            [action_to_index[action] for action in group.candidate_actions],
            dtype=torch.long,
            device=device,
        )
        candidate_current.append(current[indices])
        candidate_old.append(
            torch.tensor(group.old_log_probs, dtype=torch.float32, device=device)
        )
        advantages.append(
            torch.tensor(group.advantages, dtype=torch.float32, device=device)
        )
        probabilities = current.exp()
        kls.append((probabilities * (current - ref)).sum())
        entropies.append(-(probabilities * current).sum())

    current = torch.cat(candidate_current).float()
    old = torch.cat(candidate_old)
    advantage = torch.cat(advantages)
    log_ratio = (current - old).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    kl = torch.stack(kls).mean().float()
    entropy = torch.stack(entropies).mean().float()
    entropy_bonus, exploration_metrics = policy_entropy_bonus(
        states, torch.stack(entropies), coefficient=entropy_coefficient,
        opening_coefficient=opening_entropy_coefficient, opening_plies=entropy_opening_plies)
    total = policy_loss + kl_coefficient * kl - entropy_bonus
    clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean()
    nonzero_advantage = (advantage.abs() > 0).float().mean()
    return LossOutput(
        total,
        {
            "loss/policy_grpo": float(policy_loss.detach()),
            "loss/policy_total": float(total.detach()),
            "policy/kl_reference": float(kl.detach()),
            "policy/entropy": float(entropy.detach()),
            "policy/clip_fraction": float(clip_fraction.detach()),
            "policy/nonzero_advantage_fraction": float(
                nonzero_advantage.detach()
            ),
            "policy/importance_ratio_mean": float(ratio.detach().mean()),
            "policy/importance_ratio_max": float(ratio.detach().max()),
            **dict(zip(exploration_metrics, torch.stack(list(exploration_metrics.values())).cpu().tolist(), strict=True)),
        },
    )


@dataclass(slots=True)
class LayoutTrainingBatch:
    choices: Tensor
    mode_ids: Tensor
    old_logs: Tensor
    advantages: Tensor
    reward_mean: Tensor
    reward_std: Tensor

    def slice(self, start: int, stop: int):
        return LayoutTrainingBatch(self.choices[start:stop], self.mode_ids[start:stop],
                                   self.old_logs[start:stop], self.advantages[start:stop],
                                   self.reward_mean, self.reward_std)


def prepare_layout_batch(outcomes: Sequence[LayoutOutcome], *, device, advantage_epsilon=1e-4):
    """Transfer once and normalize over the full optimizer batch, not each tile."""
    if len(outcomes) < 2:
        raise ValueError("layout GRPO requires at least two terminal outcomes")
    choices = torch.tensor([x.sample.position_indices for x in outcomes], dtype=torch.long, device=device)
    modes = {"four_dark": 0, "double_open": 1, "two_player": 2}
    mode_ids = torch.tensor([modes[x.sample.mode.value] for x in outcomes], dtype=torch.long, device=device)
    rewards = torch.tensor([x.reward for x in outcomes], dtype=torch.float32, device=device)
    mean, std = rewards.mean(), rewards.std(unbiased=False)
    advantages = torch.where(std >= advantage_epsilon, (rewards - mean) / (std + advantage_epsilon),
                             torch.zeros_like(rewards))
    old_logs = torch.tensor([x.sample.old_log_probs for x in outcomes], dtype=torch.float32, device=device)
    return LayoutTrainingBatch(choices, mode_ids, old_logs, advantages, mean, std)


def layout_grpo_loss(
    layout: PieceConditionedLayoutPointerDecoder,
    reference: PieceConditionedLayoutPointerDecoder,
    outcomes: Sequence[LayoutOutcome],
    *,
    clip_epsilon: float,
    kl_coefficient: float,
    entropy_coefficient: float,
    advantage_epsilon: float = 1e-4,
    prepared: LayoutTrainingBatch | None = None,
    tensor_metrics: bool = False,
) -> LossOutput:
    """Clipped per-pointer loss using only complete-game terminal outcomes."""

    batch = prepared if prepared is not None else prepare_layout_batch(
        outcomes, device=layout.device, advantage_epsilon=advantage_epsilon)
    current_logs, entropies = layout.evaluate_layouts(batch.choices, batch.mode_ids)
    with torch.no_grad():
        reference_logs, _ = reference.evaluate_layouts(batch.choices, batch.mode_ids)
    log_ratio = (current_logs.float() - batch.old_logs).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    expanded_advantage = batch.advantages.unsqueeze(1)
    unclipped = ratio * expanded_advantage
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * expanded_advantage
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    # A sampled quadratic trust-region term is used in this first implementation;
    # all source trajectories still come from the behavior layout distribution.
    sampled_kl = 0.5 * (current_logs.float() - reference_logs.float()).square().mean()
    entropy = entropies.float().mean()
    total = policy_loss + kl_coefficient * sampled_kl - entropy_coefficient * entropy
    metrics = {
        "loss/layout_grpo": policy_loss.detach(), "loss/layout_total": total.detach(),
        "layout/sampled_trust_region": sampled_kl.detach(), "layout/entropy": entropy.detach(),
        "layout/reward_mean": batch.reward_mean, "layout/reward_std": batch.reward_std,
        "layout/clip_fraction": ((ratio - 1.0).abs() > clip_epsilon).float().mean().detach(),
        "layout/zero_advantage_batch": (batch.reward_std < advantage_epsilon).float(),
        "layout/nonzero_advantage_fraction": (batch.advantages != 0).float().mean(),
    }
    if not tensor_metrics:
        metrics = dict(zip(metrics, torch.stack(list(metrics.values())).cpu().tolist(), strict=True))
    return LossOutput(total, metrics)
