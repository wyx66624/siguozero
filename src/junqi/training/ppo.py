"""Single-trajectory self-play PPO with a separate recurrent value network.

Values and rewards are in the current actor's team perspective.  A transition
to the other team changes the sign of both the TD bootstrap and GAE tail.
The actual next actor is used, so eliminated seats cannot corrupt the sign.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Sequence

import torch
from torch import nn

from .accelerator import empty_cache, is_out_of_memory
from .encoding import PolicyState, history_prefix_groups
from .losses import LossOutput
from .models import GameValueTransformer, PieceConditionedLayoutPointerDecoder
from .rollout import BaseGamePool, FrozenPolicyActor, LayoutOutcome, RolloutMetrics


@dataclass(frozen=True, slots=True)
class PPOSample:
    state: PolicyState
    action: tuple[int, int]
    old_log_prob: float
    old_value: float
    advantage: float
    value_target: float
    behavior_version: int


@dataclass(frozen=True, slots=True)
class PPOTransition:
    state: PolicyState
    action: tuple[int, int]
    old_log_prob: float
    old_value: float
    reward: float
    terminal: bool
    next_team_sign: int


def generalized_advantages(
    transitions: Sequence[PPOTransition],
    *,
    bootstrap_value: float,
    discount: float,
    gae_lambda: float,
    behavior_version: int,
) -> list[PPOSample]:
    """GAE across one environment's chronological transitions, including resets.

    Terminal transitions zero the bootstrap and trace.  At a collector boundary
    the last nonterminal transition bootstraps from a frozen critic; reaching
    the batch size is never treated as a draw or as a terminal state.
    """

    if not 0 < discount <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("invalid GAE discount or lambda")
    result: list[PPOSample] = []
    next_value, tail = float(bootstrap_value), 0.0
    for item in reversed(transitions):
        if item.next_team_sign not in (-1, 1):
            raise ValueError("team sign must be -1 or 1")
        factor = 0.0 if item.terminal else discount * item.next_team_sign
        delta = item.reward + factor * next_value - item.old_value
        advantage = delta + factor * gae_lambda * tail
        result.append(PPOSample(
            state=item.state, action=item.action,
            old_log_prob=item.old_log_prob, old_value=item.old_value,
            advantage=advantage, value_target=advantage + item.old_value,
            behavior_version=behavior_version,
        ))
        next_value, tail = item.old_value, advantage
    result.reverse()
    return result


def normalize_advantages(
    samples: Sequence[PPOSample], *, mean: float, std: float, epsilon: float
) -> list[PPOSample]:
    # Preserve a constant nonzero learning signal for a tiny/degenerate batch.
    if std < epsilon:
        return list(samples)
    return [replace(item, advantage=(item.advantage - mean) / (std + epsilon))
            for item in samples]


class FrozenValueActor:
    """Batched critic inference; caches live only during frozen collection."""

    def __init__(
        self, critic: GameValueTransformer, *, amp_dtype: torch.dtype | None,
        max_batch_size: int,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("critic batch size must be positive")
        self.critic = critic.eval()
        self.amp_dtype = amp_dtype
        self.max_batch_size = max_batch_size
        self.critic.start_inference_board_cache()

    def values(self, states: Sequence[PolicyState]) -> list[float]:
        values: list[float] = []
        device_type = self.critic.device.type
        with torch.inference_mode(), torch.autocast(
            device_type=device_type, dtype=self.amp_dtype or torch.float32,
            enabled=self.amp_dtype is not None and device_type in ("cuda", "npu"),
        ):
            start = 0
            while start < len(states):
                size = min(self.max_batch_size, len(states) - start)
                try:
                    batch = self.critic(states[start:start + size]).cpu().tolist()
                except RuntimeError as error:
                    if not is_out_of_memory(error, device_type) or size <= 1:
                        raise
                    self.max_batch_size = max(1, size // 2)
                    empty_cache(self.critic.device)
                    continue
                if not all(math.isfinite(value) for value in batch):
                    raise FloatingPointError("critic produced non-finite rollout values")
                values.extend(batch)
                start += size
        return values


def collect_ppo_samples(
    pool: BaseGamePool, actor: FrozenPolicyActor, critic: FrozenValueActor,
    layout: PieceConditionedLayoutPointerDecoder, *, count: int,
    behavior_version: int, discount: float, gae_lambda: float,
) -> tuple[list[PPOSample], list[LayoutOutcome], RolloutMetrics]:
    """Take exactly count real self-play actions, without cloned continuations."""

    if count <= 0:
        raise ValueError("PPO transition count must be positive")
    started = time.perf_counter()
    pool.fill(layout, behavior_version)
    traces: list[list[PPOTransition]] = [[] for _ in pool.slots]
    outcomes: list[LayoutOutcome] = []
    metrics = RolloutMetrics()
    collected = 0
    while collected < count:
        indices = range(min(len(pool.slots), count - collected))
        states = [pool.slots[index].history.state_for(pool.slots[index].game)
                  for index in indices]
        step_started = time.perf_counter()
        actions, logs = actor.sample(states, count=1, return_log_probs=True)
        metrics.actor_inference_seconds += time.perf_counter() - step_started
        step_started = time.perf_counter()
        values = critic.values(states)
        metrics.critic_inference_seconds += time.perf_counter() - step_started
        step_started = time.perf_counter()
        for index, state, sampled, log, value in zip(
            indices, states, actions, logs, values, strict=True
        ):
            slot = pool.slots[index]
            player = slot.game.current_player
            if player is None:
                raise RuntimeError("terminal game remained in PPO pool")
            team = slot.game.team_of(player)
            slot.game.step(sampled[0])
            metrics.record_environment_steps()
            slot.history.append_after_step(slot.game)
            terminal = slot.game.is_terminal
            reward = slot.game.rewards()[player] if terminal else 0.0
            next_player = slot.game.current_player
            sign = (1 if terminal or slot.game.team_of(next_player) == team else -1)
            traces[index].append(PPOTransition(
                state=state, action=sampled[0], old_log_prob=float(log[0]),
                old_value=value, reward=reward, terminal=terminal,
                next_team_sign=sign,
            ))
            if terminal:
                metrics.base_games_completed += 1
                rewards = slot.game.rewards()
                metrics.wins += int(rewards[0] > 0)
                metrics.draws += int(rewards[0] == 0)
                metrics.losses += int(rewards[0] < 0)
                outcomes.extend(LayoutOutcome(
                    sample=sample, reward=rewards[seat], seat=seat,
                    behavior_version=slot.layout_behavior_version,
                ) for seat, sample in enumerate(slot.layouts))
                pool.slots[index] = pool._new_slot(layout, behavior_version)
            collected += 1
        metrics.environment_step_seconds += time.perf_counter() - step_started

    bootstrap_indices = [i for i, trace in enumerate(traces)
                         if trace and not trace[-1].terminal]
    step_started = time.perf_counter()
    bootstrap_values = critic.values([
        pool.slots[i].history.state_for(pool.slots[i].game) for i in bootstrap_indices
    ])
    metrics.critic_inference_seconds += time.perf_counter() - step_started
    bootstrap = dict(zip(bootstrap_indices, bootstrap_values, strict=True))
    samples: list[PPOSample] = []
    for index, trace in enumerate(traces):
        samples.extend(generalized_advantages(
            trace, bootstrap_value=bootstrap.get(index, 0.0), discount=discount,
            gae_lambda=gae_lambda, behavior_version=behavior_version,
        ))
    metrics.policy_samples = count
    metrics.wall_seconds = time.perf_counter() - started
    return samples, outcomes, metrics


def sequence_training_batches(
    samples: Sequence[PPOSample], *, sequences_per_batch: int,
    max_samples_per_sequence: int,
) -> list[list[PPOSample]]:
    """Batch whole prefix chains, with bounded action-head activation memory."""
    if sequences_per_batch <= 0 or max_samples_per_sequence <= 0:
        raise ValueError("sequence batch limits must be positive")
    chains = []
    for members in history_prefix_groups([sample.state for sample in samples]):
        chronological = sorted(members, key=lambda i: len(samples[i].state.records))
        for start in range(0, len(chronological), max_samples_per_sequence):
            chains.append([samples[i] for i in chronological[start:start + max_samples_per_sequence]])
    chains.sort(key=lambda chain: len(chain[-1].state.records))
    return [
        [sample for chain in chains[start:start + sequences_per_batch] for sample in chain]
        for start in range(0, len(chains), sequences_per_batch)
    ]


def policy_ppo_loss(
    policy: nn.Module, samples: Sequence[PPOSample], *, clip_epsilon: float,
    entropy_coefficient: float,
    sequence_training: bool = False,
) -> LossOutput:
    if not samples:
        raise ValueError("PPO requires a nonempty batch")
    states = [sample.state for sample in samples]
    options = {"pack_sequences": True} if sequence_training else {}
    logs = policy(states, [state.legal_actions for state in states], **options)
    current = torch.stack([
        row[sample.state.legal_actions.index(sample.action)]
        for sample, row in zip(samples, logs, strict=True)
    ]).float()
    old = current.new_tensor([sample.old_log_prob for sample in samples])
    advantage = current.new_tensor([sample.advantage for sample in samples])
    log_ratio = (current - old).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    objective = torch.minimum(
        ratio * advantage,
        ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage,
    )
    entropy = torch.stack([-(row.exp() * row).sum() for row in logs]).mean()
    policy_loss = -objective.mean()
    loss = policy_loss - entropy_coefficient * entropy
    return LossOutput(loss, {
        "loss/policy_ppo": float(policy_loss.detach()),
        "loss/policy_total": float(loss.detach()),
        "policy/approx_kl_old": float((ratio - 1 - log_ratio).mean().detach()),
        "policy/entropy": float(entropy.detach()),
        "policy/clip_fraction": float(((ratio - 1).abs() > clip_epsilon).float().mean()),
        "policy/importance_ratio_mean": float(ratio.mean().detach()),
        "policy/nonzero_advantage_fraction": float((advantage != 0).float().mean()),
    })


def critic_ppo_loss(
    critic: nn.Module, samples: Sequence[PPOSample], *, clip_epsilon: float,
    value_coefficient: float,
    sequence_training: bool = False,
) -> LossOutput:
    if not samples:
        raise ValueError("critic requires a nonempty batch")
    options = {"pack_sequences": True} if sequence_training else {}
    values = critic([sample.state for sample in samples], **options).float()
    old = values.new_tensor([sample.old_value for sample in samples])
    targets = values.new_tensor([sample.value_target for sample in samples])
    clipped = old + (values - old).clamp(-clip_epsilon, clip_epsilon)
    mse = torch.maximum((values - targets).square(), (clipped - targets).square()).mean()
    loss = 0.5 * value_coefficient * mse
    return LossOutput(loss, {
        "loss/critic_total": float(loss.detach()),
        "critic/value_mse": float((values - targets).square().mean().detach()),
        "critic/value_mean": float(values.mean().detach()),
        "critic/target_mean": float(targets.mean()),
    })
