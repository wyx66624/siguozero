"""Terminal-only root GRPO and layout policy losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from .models import GamePolicyTransformer, PieceConditionedLayoutPointerDecoder
from .rollout import LayoutOutcome, PolicyGroup


@dataclass(slots=True)
class LossOutput:
    loss: Tensor
    metrics: dict[str, float]


def policy_grpo_loss(
    policy: GamePolicyTransformer,
    reference: GamePolicyTransformer,
    groups: Sequence[PolicyGroup],
    *,
    clip_epsilon: float,
    kl_coefficient: float,
    entropy_coefficient: float,
) -> LossOutput:
    """Root-only clipped GRPO; suffix actions never receive the root advantage."""

    if not groups:
        raise ValueError("policy GRPO requires at least one complete anchor group")
    states = [group.state for group in groups]
    all_actions = [state.legal_actions for state in states]
    current_logs = policy.log_probs_for_action_groups(states, all_actions)
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
            device=policy.device,
        )
        candidate_current.append(current[indices])
        candidate_old.append(
            torch.tensor(group.old_log_probs, dtype=torch.float32, device=policy.device)
        )
        advantages.append(
            torch.tensor(group.advantages, dtype=torch.float32, device=policy.device)
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
    total = policy_loss + kl_coefficient * kl - entropy_coefficient * entropy
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
        },
    )


def layout_grpo_loss(
    layout: PieceConditionedLayoutPointerDecoder,
    reference: PieceConditionedLayoutPointerDecoder,
    outcomes: Sequence[LayoutOutcome],
    *,
    clip_epsilon: float,
    kl_coefficient: float,
    entropy_coefficient: float,
    advantage_epsilon: float = 1e-4,
) -> LossOutput:
    """Clipped per-pointer loss using only complete-game terminal outcomes."""

    if len(outcomes) < 2:
        raise ValueError("layout GRPO requires at least two terminal outcomes")
    device = layout.device
    choices = torch.tensor(
        [item.sample.position_indices for item in outcomes],
        dtype=torch.long,
        device=device,
    )
    mode_ids = torch.tensor(
        [
            {"four_dark": 0, "double_open": 1, "two_player": 2}[
                item.sample.mode.value
            ]
            for item in outcomes
        ],
        dtype=torch.long,
        device=device,
    )
    rewards = torch.tensor(
        [item.reward for item in outcomes], dtype=torch.float32, device=device
    )
    reward_std = rewards.std(unbiased=False)
    if float(reward_std) < advantage_epsilon:
        advantages = torch.zeros_like(rewards)
    else:
        advantages = (rewards - rewards.mean()) / (
            reward_std + advantage_epsilon
        )
    old_logs = torch.tensor(
        [item.sample.old_log_probs for item in outcomes],
        dtype=torch.float32,
        device=device,
    )
    current_logs, entropies = layout.evaluate_layouts(choices, mode_ids)
    with torch.no_grad():
        reference_logs, _ = reference.evaluate_layouts(choices, mode_ids)
    log_ratio = (current_logs.float() - old_logs).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    expanded_advantage = advantages.unsqueeze(1)
    unclipped = ratio * expanded_advantage
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * expanded_advantage
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    # A sampled quadratic trust-region term is used in this first implementation;
    # all source trajectories still come from the behavior layout distribution.
    sampled_kl = 0.5 * (current_logs.float() - reference_logs.float()).square().mean()
    entropy = entropies.float().mean()
    total = policy_loss + kl_coefficient * sampled_kl - entropy_coefficient * entropy
    return LossOutput(
        total,
        {
            "loss/layout_grpo": float(policy_loss.detach()),
            "loss/layout_total": float(total.detach()),
            "layout/sampled_trust_region": float(sampled_kl.detach()),
            "layout/entropy": float(entropy.detach()),
            "layout/reward_mean": float(rewards.mean()),
            "layout/reward_std": float(reward_std),
            "layout/clip_fraction": float(
                ((ratio - 1.0).abs() > clip_epsilon).float().mean()
            ),
        },
    )
