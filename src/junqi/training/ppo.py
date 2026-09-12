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
        deferred: bool = False,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("critic batch size must be positive")
        self.critic = critic.eval()
        self.amp_dtype = amp_dtype
        self.max_batch_size = max_batch_size
        self.deferred = deferred
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

    def values_packed(self, states: Sequence[PolicyState]) -> list[float]:
        """Evaluate every frozen value using exact shared causal prefixes.

        No value influences action sampling. Window starts remain distinct, so
        this also handles resets and arbitrary sliding windows without stale KV.
        """
        if not states:
            return []
        self.critic.clear_inference_board_cache()
        groups = history_prefix_groups(states)
        result = [0.] * len(states)
        start = 0
        device_type = self.critic.device.type
        with torch.inference_mode(), torch.autocast(
            device_type=device_type, dtype=self.amp_dtype or torch.float32,
            enabled=self.amp_dtype is not None and device_type in ('cuda', 'npu'),
        ):
            while start < len(groups):
                size = min(self.max_batch_size, len(groups) - start)
                indices = [i for group in groups[start:start + size] for i in group]
                try:
                    values = self.critic([states[i] for i in indices], pack_sequences=True).cpu().tolist()
                except RuntimeError as error:
                    if not is_out_of_memory(error, device_type) or size <= 1:
                        raise
                    self.max_batch_size = max(1, size // 2)
                    empty_cache(self.critic.device)
                    continue
                if not all(math.isfinite(value) for value in values):
                    raise FloatingPointError('critic produced non-finite rollout values')
                for index, value in zip(indices, values, strict=True):
                    result[index] = value
                start += size
        return result


def _finalize_rollout_values(traces, bootstrap_states, critic, actor):
    if not critic.deferred:
        return critic.values(bootstrap_states)
    # Policy caches are no longer needed, allowing a large teacher-forced
    # critic batch without keeping two complete per-seat KV arenas resident.
    actor.policy._rollout_cache_snapshot = actor.policy.board_encoding_metrics()
    actor.policy.clear_inference_board_cache()
    states = [item.state for trace in traces for item in trace]
    values = critic.values_packed([*states, *bootstrap_states])
    offset = 0
    for trace in traces:
        for index, transition in enumerate(trace):
            trace[index] = replace(transition, old_value=values[offset])
            offset += 1
    return values[offset:]


def collect_ppo_samples(
    pool: BaseGamePool, actor: FrozenPolicyActor, critic: FrozenValueActor,
    layout: PieceConditionedLayoutPointerDecoder, *, count: int,
    behavior_version: int, discount: float, gae_lambda: float,
    environment_workers: int = 1, environment=None,
    pipeline_groups: int = 1,
) -> tuple[list[PPOSample], list[LayoutOutcome], RolloutMetrics]:
    """Take exactly count real self-play actions, without cloned continuations."""

    if count <= 0:
        raise ValueError("PPO transition count must be positive")
    if type(environment_workers) is not int or environment_workers <= 0:
        raise ValueError('PPO environment workers must be a positive integer')
    if type(pipeline_groups) is not int or pipeline_groups <= 0:
        raise ValueError('PPO pipeline groups must be a positive integer')
    if pipeline_groups > 1 and (not critic.deferred or not actor.policy.config.ppo_array_history):
        raise ValueError('pipelined PPO requires deferred values and array histories')
    if pipeline_groups > 1 and environment is None and environment_workers < 2:
        raise ValueError('pipelined PPO requires at least two environment workers')
    if environment is not None or environment_workers > 1:
        from .ppo_environment import ParallelPPOEnvironment
        owned = environment is None
        environment = environment or ParallelPPOEnvironment(environment_workers)
        try:
            if pipeline_groups > 1:
                from .ppo_pipeline import collect_pipelined
                return collect_pipelined(pool, actor, critic, layout, count=count,
                    behavior_version=behavior_version, discount=discount, gae_lambda=gae_lambda,
                    environment=environment, groups=pipeline_groups)
            return _collect_parallel_ppo_samples(
                pool, actor, critic, layout, count=count, behavior_version=behavior_version,
                discount=discount, gae_lambda=gae_lambda, environment=environment)
        finally:
            if owned:
                environment.close()
    started = time.perf_counter()
    pool.fill(layout, behavior_version)
    array_history = actor.policy.config.ppo_array_history
    if array_history:
        for slot in pool.slots:
            slot.history.enable_array_storage()
        capacity = len(pool.slots) * len(pool.slots[0].history.players)
        actor.policy.start_ppo_inference_cache(capacity=capacity, behavior_version=behavior_version)
        if not critic.deferred:
            critic.critic.start_ppo_inference_cache(capacity=capacity, behavior_version=behavior_version)
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
        values = [0.] * len(states) if critic.deferred else critic.values(states)
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
                if array_history:
                    game_identity = state.records.identity[0]
                    for model in (actor.policy, critic.critic):
                        if model._fixed_kv_store is not None:
                            model._fixed_kv_store.release_game(game_identity)
                pool.slots[index] = pool._new_slot(layout, behavior_version)
                if array_history:
                    pool.slots[index].history.enable_array_storage()
            collected += 1
        metrics.environment_step_seconds += time.perf_counter() - step_started

    bootstrap_indices = [i for i, trace in enumerate(traces)
                         if trace and not trace[-1].terminal]
    step_started = time.perf_counter()
    bootstrap_values = _finalize_rollout_values(traces, [
        pool.slots[i].history.state_for(pool.slots[i].game) for i in bootstrap_indices
    ], critic, actor)
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


def _collect_parallel_ppo_samples(pool, actor, critic, layout, *, count,
                                  behavior_version, discount, gae_lambda, environment):
    started = time.perf_counter()
    pool.fill(layout, behavior_version)
    array_history = actor.policy.config.ppo_array_history
    if array_history:
        for slot in pool.slots:
            slot.history.enable_array_storage()
        capacity = len(pool.slots) * len(pool.slots[0].history.players)
        actor.policy.start_ppo_inference_cache(capacity=capacity, behavior_version=behavior_version)
        if not critic.deferred:
            critic.critic.start_ppo_inference_cache(capacity=capacity, behavior_version=behavior_version)
    metrics = RolloutMetrics()
    boundary = time.perf_counter()
    environment.begin(pool)
    metrics.environment_sync_seconds += time.perf_counter() - boundary
    metrics.environment_workers = len(environment.processes)
    initial_worker_seconds = environment.worker_seconds
    traces = [[] for _ in pool.slots]
    outcomes = []
    collected = 0

    def states_for(indices):
        return [pool.slots[i].history.players[environment.states[i].player].as_policy_state(
            environment.states[i].legal_actions) for i in indices]

    while collected < count:
        indices = list(range(min(len(pool.slots), count - collected)))
        states = states_for(indices)
        boundary = time.perf_counter()
        actions, logs = actor.sample(states, count=1, return_log_probs=True)
        metrics.actor_inference_seconds += time.perf_counter() - boundary
        boundary = time.perf_counter()
        environment.submit(indices, [sampled[0] for sampled in actions])
        metrics.environment_step_seconds += time.perf_counter() - boundary
        # Workers advance different games while the GPU evaluates the immutable
        # pre-action observations. No policy update or next action is overlapped.
        boundary = time.perf_counter()
        values = [0.] * len(states) if critic.deferred else critic.values(states)
        metrics.critic_inference_seconds += time.perf_counter() - boundary
        boundary = time.perf_counter()
        responses = environment.receive()
        if [row.index for row in responses] != indices:
            raise RuntimeError('PPO workers returned an incomplete step wave')
        for index, state, sampled, log, value, row in zip(indices, states, actions, logs, values, responses, strict=True):
            slot = pool.slots[index]
            slot.history.append_encoded_rows(row.records)
            metrics.record_environment_steps()
            traces[index].append(PPOTransition(
                state=state, action=sampled[0], old_log_prob=float(log[0]), old_value=value,
                reward=row.reward, terminal=row.terminal, next_team_sign=row.next_team_sign))
            if row.terminal:
                metrics.base_games_completed += 1
                metrics.wins += int(row.rewards[0] > 0)
                metrics.draws += int(row.rewards[0] == 0)
                metrics.losses += int(row.rewards[0] < 0)
                outcomes.extend(LayoutOutcome(sample=sample, reward=row.rewards[seat], seat=seat,
                                              behavior_version=slot.layout_behavior_version)
                                for seat, sample in enumerate(slot.layouts))
                if array_history:
                    for model in (actor.policy, critic.critic):
                        if model._fixed_kv_store is not None:
                            model._fixed_kv_store.release_game(state.records.identity[0])
                slot = pool.slots[index] = pool._new_slot(layout, behavior_version)
                if array_history:
                    slot.history.enable_array_storage()
                environment.replace(index, slot.game)
            collected += 1
        metrics.environment_step_seconds += time.perf_counter() - boundary
    bootstrap_indices = [i for i, trace in enumerate(traces) if trace and not trace[-1].terminal]
    boundary = time.perf_counter()
    bootstrap = dict(zip(bootstrap_indices, _finalize_rollout_values(
        traces, states_for(bootstrap_indices), critic, actor), strict=True))
    metrics.critic_inference_seconds += time.perf_counter() - boundary
    boundary = time.perf_counter()
    environment.synchronize(pool)
    metrics.environment_sync_seconds += time.perf_counter() - boundary
    samples = [sample for i, trace in enumerate(traces) for sample in generalized_advantages(
        trace, bootstrap_value=bootstrap.get(i, 0.), discount=discount,
        gae_lambda=gae_lambda, behavior_version=behavior_version)]
    metrics.environment_worker_seconds = environment.worker_seconds - initial_worker_seconds
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
    defer_metrics: bool = False,
) -> LossOutput:
    if not samples:
        raise ValueError("PPO requires a nonempty batch")
    states = [sample.state for sample in samples]
    options = {"pack_sequences": True} if sequence_training else {}
    if hasattr(getattr(policy, "module", policy), "ppo_statistics"):
        current, entropies = policy(states, [state.legal_actions for state in states],
                                    ppo_actions=[sample.action for sample in samples], **options)
        entropy = entropies.mean()
    else:
        # Keep the public loss usable with small external/test policy modules.
        logs = policy(states, [state.legal_actions for state in states], **options)
        current = torch.stack([
            row[sample.state.legal_actions.index(sample.action)]
            for sample, row in zip(samples, logs, strict=True)
        ]).float()
        entropy = torch.stack([-(row.exp() * row).sum() for row in logs]).mean()
    old = current.new_tensor([sample.old_log_prob for sample in samples])
    advantage = current.new_tensor([sample.advantage for sample in samples])
    log_ratio = (current - old).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    objective = torch.minimum(
        ratio * advantage,
        ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage,
    )
    policy_loss = -objective.mean()
    loss = policy_loss - entropy_coefficient * entropy
    names = ("loss/policy_ppo", "loss/policy_total", "policy/approx_kl_old", "policy/entropy",
             "policy/clip_fraction", "policy/importance_ratio_mean", "policy/nonzero_advantage_fraction")
    values = torch.stack((policy_loss, loss, (ratio - 1 - log_ratio).mean(), entropy,
                          ((ratio - 1).abs() > clip_epsilon).float().mean(), ratio.mean(),
                          (advantage != 0).float().mean())).detach()
    values = values.unbind() if defer_metrics else values.cpu().tolist()
    return LossOutput(loss, dict(zip(names, values, strict=True)))


def critic_ppo_loss(
    critic: nn.Module, samples: Sequence[PPOSample], *, clip_epsilon: float,
    value_coefficient: float,
    sequence_training: bool = False,
    defer_metrics: bool = False,
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
    names = ("loss/critic_total", "critic/value_mse", "critic/value_mean", "critic/target_mean")
    metrics = torch.stack((loss, (values - targets).square().mean(), values.mean(), targets.mean())).detach()
    metrics = metrics.unbind() if defer_metrics else metrics.cpu().tolist()
    return LossOutput(loss, dict(zip(names, metrics, strict=True)))
