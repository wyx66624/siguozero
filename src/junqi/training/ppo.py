"""Single-trajectory self-play PPO with a separate recurrent value network.

Team outcomes change sign at an opposing actor. The draw penalty is common to
all seats and never changes sign. Separate GAE components preserve both rules,
including unfinished collector boundaries and eliminated seats.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Sequence

import torch
from torch import nn

from .accelerator import empty_cache, is_out_of_memory, supports_pinned_memory
from .encoding import PolicyState, history_prefix_groups
from .losses import LossOutput
from .models import GameValueTransformer, PieceConditionedLayoutPointerDecoder
from .rollout import BaseGamePool, FrozenPolicyActor, LayoutOutcome, RolloutMetrics
from .rewards import DEFAULT_DRAW_REWARD, terminal_utility
from .entropy import policy_entropy_bonus


@dataclass(frozen=True, slots=True)
class PPOSample:
    state: PolicyState
    action: tuple[int, int]
    old_log_prob: float
    old_value: float
    advantage: float
    value_target: float
    behavior_version: int
    old_draw_value: float = 0.0
    draw_value_target: float = 0.0
    learnable: bool = True
    value_state: PolicyState | None = None


@dataclass(frozen=True, slots=True)
class PPOTransition:
    state: PolicyState
    action: tuple[int, int]
    old_log_prob: float
    old_value: float
    reward: float
    terminal: bool
    next_team_sign: int
    old_draw_value: float = 0.0
    terminal_draw: bool = False
    learnable: bool = True
    value_state: PolicyState | None = None


def generalized_advantages(
    transitions: Sequence[PPOTransition],
    *,
    bootstrap_value: float,
    discount: float,
    gae_lambda: float,
    behavior_version: int,
    bootstrap_draw_value: float = 0.0,
) -> list[PPOSample]:
    """GAE across one environment's chronological transitions, including resets.

    Terminal transitions zero the bootstrap and trace.  At a collector boundary
    the last nonterminal transition bootstraps from a frozen critic; reaching
    the batch size is never treated as a draw or as a terminal state.
    """

    if not 0 < discount <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("invalid GAE discount or lambda")
    result: list[PPOSample] = []
    next_outcome = float(bootstrap_value) - float(bootstrap_draw_value)
    next_draw = float(bootstrap_draw_value)
    outcome_tail = draw_tail = 0.0
    for item in reversed(transitions):
        if item.next_team_sign not in (-1, 1):
            raise ValueError("team sign must be -1 or 1")
        if item.terminal_draw and not item.terminal:
            raise ValueError("only a terminal transition can be a draw")
        common_reward = item.reward if item.terminal_draw else 0.0
        old_outcome = item.old_value - item.old_draw_value
        common_factor = 0.0 if item.terminal else discount
        factor = common_factor * item.next_team_sign
        outcome_delta = item.reward - common_reward + factor * next_outcome - old_outcome
        draw_delta = common_reward + common_factor * next_draw - item.old_draw_value
        outcome_advantage = outcome_delta + factor * gae_lambda * outcome_tail
        draw_advantage = draw_delta + common_factor * gae_lambda * draw_tail
        advantage = outcome_advantage + draw_advantage
        result.append(PPOSample(
            state=item.state, action=item.action,
            old_log_prob=item.old_log_prob, old_value=item.old_value,
            advantage=advantage, value_target=advantage + item.old_value,
            behavior_version=behavior_version,
            old_draw_value=item.old_draw_value,
            draw_value_target=draw_advantage + item.old_draw_value,
            learnable=item.learnable, value_state=item.value_state,
        ))
        next_outcome, outcome_tail = old_outcome, outcome_advantage
        next_draw, draw_tail = item.old_draw_value, draw_advantage
    result.reverse()
    return result


def normalize_advantages(
    samples: Sequence[PPOSample], *, mean: float, std: float, epsilon: float
) -> list[PPOSample]:
    # Preserve a constant nonzero learning signal for a tiny/degenerate batch.
    if std < epsilon:
        return list(samples)
    return [replace(item, advantage=(item.advantage - mean) / (std + epsilon)) if item.learnable else item
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
        return [total for total, _draw in self.estimates(states)]

    def estimates(self, states: Sequence[PolicyState]) -> list[tuple[float, float]]:
        """Return total and shared-draw values from the same encoder pass."""
        values = []
        device_type = self.critic.device.type
        with torch.inference_mode(), torch.autocast(
            device_type=device_type, dtype=self.amp_dtype or torch.float32,
            enabled=self.amp_dtype is not None and device_type in ("cuda", "npu"),
        ):
            start = 0
            while start < len(states):
                size = min(self.max_batch_size, len(states) - start)
                try:
                    batch = self.critic(states[start:start + size], return_components=True).cpu().tolist()
                except RuntimeError as error:
                    if not is_out_of_memory(error, device_type) or size <= 1:
                        raise
                    self.max_batch_size = max(1, size // 2)
                    empty_cache(self.critic.device)
                    continue
                if not all(math.isfinite(value) for pair in batch for value in pair):
                    raise FloatingPointError("critic produced non-finite rollout values")
                values.extend(tuple(pair) for pair in batch)
                start += size
        return values

    def values_packed(self, states: Sequence[PolicyState]) -> list[float]:
        return [total for total, _draw in self.estimates_packed(states)]

    def estimates_packed(self, states: Sequence[PolicyState]) -> list[tuple[float, float]]:
        """Evaluate every frozen value using exact shared causal prefixes.

        No value influences action sampling. Window starts remain distinct, so
        this also handles resets and arbitrary sliding windows without stale KV.
        """
        if not states:
            return []
        self.critic.clear_inference_board_cache()
        groups = history_prefix_groups(states)
        result = [(0., 0.)] * len(states)
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
                    values = self.critic([states[i] for i in indices], pack_sequences=True,
                                         return_components=True).cpu().tolist()
                except RuntimeError as error:
                    if not is_out_of_memory(error, device_type) or size <= 1:
                        raise
                    self.max_batch_size = max(1, size // 2)
                    empty_cache(self.critic.device)
                    continue
                if not all(math.isfinite(value) for pair in values for value in pair):
                    raise FloatingPointError('critic produced non-finite rollout values')
                for index, value in zip(indices, values, strict=True):
                    result[index] = tuple(value)
                start += size
        return result


def _finalize_rollout_values(traces, bootstrap_states, critic, actor):
    if not critic.deferred:
        return critic.estimates(bootstrap_states)
    # Policy caches are no longer needed, allowing a large teacher-forced
    # critic batch without keeping two complete per-seat KV arenas resident.
    actor.policy._rollout_cache_snapshot = actor.policy.board_encoding_metrics()
    actor.policy.clear_inference_board_cache()
    states = [item.value_state or item.state for trace in traces for item in trace]
    values = critic.estimates_packed([*states, *bootstrap_states])
    offset = 0
    for trace in traces:
        for index, transition in enumerate(trace):
            trace[index] = replace(transition, old_value=values[offset][0], old_draw_value=values[offset][1])
            offset += 1
    return values[offset:]


def collect_ppo_samples(
    pool: BaseGamePool, actor: FrozenPolicyActor, critic: FrozenValueActor,
    layout: PieceConditionedLayoutPointerDecoder, *, count: int,
    behavior_version: int, discount: float, gae_lambda: float,
    environment_workers: int = 1, environment=None,
    pipeline_groups: int = 1,
    draw_reward: float = DEFAULT_DRAW_REWARD,
) -> tuple[list[PPOSample], list[LayoutOutcome], RolloutMetrics]:
    """Take exactly count real self-play actions, without cloned continuations."""

    if count <= 0:
        raise ValueError("PPO transition count must be positive")
    if type(environment_workers) is not int or environment_workers <= 0:
        raise ValueError('PPO environment workers must be a positive integer')
    if type(pipeline_groups) is not int or pipeline_groups <= 0:
        raise ValueError('PPO pipeline groups must be a positive integer')
    if pool.historical is not None and pool.historical.active and pipeline_groups < 2:
        raise ValueError('historical opponents require the pipelined collector')
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
                    environment=environment, groups=pipeline_groups, draw_reward=draw_reward)
            return _collect_parallel_ppo_samples(
                pool, actor, critic, layout, count=count, behavior_version=behavior_version,
                discount=discount, gae_lambda=gae_lambda, environment=environment, draw_reward=draw_reward)
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
        values = [(0., 0.)] * len(states) if critic.deferred else critic.estimates(states)
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
            outcome = slot.game.rewards()[player] if terminal else 0.0
            reward = terminal_utility(outcome, draw_reward=draw_reward) if terminal else 0.0
            next_player = slot.game.current_player
            sign = (1 if terminal or slot.game.team_of(next_player) == team else -1)
            traces[index].append(PPOTransition(
                state=state, action=sampled[0], old_log_prob=float(log[0]),
                old_value=value[0], old_draw_value=value[1], reward=reward, terminal=terminal,
                terminal_draw=terminal and outcome == 0,
                next_team_sign=sign,
            ))
            if terminal:
                metrics.base_games_completed += 1
                rewards = slot.game.rewards()
                metrics.wins += int(rewards[0] > 0)
                metrics.draws += int(rewards[0] == 0)
                metrics.losses += int(rewards[0] < 0)
                outcomes.extend(LayoutOutcome(
                    sample=sample, reward=terminal_utility(rewards[seat], draw_reward=draw_reward), seat=seat,
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
            trace, bootstrap_value=bootstrap.get(index, (0., 0.))[0], discount=discount,
            bootstrap_draw_value=bootstrap.get(index, (0., 0.))[1],
            gae_lambda=gae_lambda, behavior_version=behavior_version,
        ))
    metrics.policy_samples = count
    metrics.wall_seconds = time.perf_counter() - started
    return samples, outcomes, metrics


def _collect_parallel_ppo_samples(pool, actor, critic, layout, *, count,
                                  behavior_version, discount, gae_lambda, environment, draw_reward):
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
        values = [(0., 0.)] * len(states) if critic.deferred else critic.estimates(states)
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
                state=state, action=sampled[0], old_log_prob=float(log[0]), old_value=value[0], old_draw_value=value[1],
                reward=terminal_utility(row.reward, draw_reward=draw_reward) if row.terminal else 0.,
                terminal_draw=row.terminal and row.reward == 0,
                terminal=row.terminal, next_team_sign=row.next_team_sign))
            if row.terminal:
                metrics.base_games_completed += 1
                metrics.wins += int(row.rewards[0] > 0)
                metrics.draws += int(row.rewards[0] == 0)
                metrics.losses += int(row.rewards[0] < 0)
                outcomes.extend(LayoutOutcome(sample=sample, reward=terminal_utility(row.rewards[seat], draw_reward=draw_reward), seat=seat,
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
        trace, bootstrap_value=bootstrap.get(i, (0., 0.))[0], discount=discount,
        bootstrap_draw_value=bootstrap.get(i, (0., 0.))[1],
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


def _upload_targets(rows, reference):
    """One asynchronous upload for immutable PPO scalars, not one per column."""
    pin = supports_pinned_memory(reference.device)
    host = torch.empty((len(rows), len(rows[0])), dtype=torch.float32, pin_memory=pin)
    host.numpy()[:] = rows
    return host.to(reference.device, non_blocking=pin)


def policy_ppo_loss(
    policy: nn.Module, samples: Sequence[PPOSample], *, clip_epsilon: float,
    entropy_coefficient: float,
    sequence_training: bool = False,
    defer_metrics: bool = False,
    opening_entropy_coefficient: float | None = None,
    entropy_opening_plies: int = 0,
    policy_clip=None,
) -> LossOutput:
    if not samples:
        raise ValueError("PPO requires a nonempty batch")
    # Frozen opponents supply transitions for GAE, never policy gradients or
    # entropy regularization. A dummy zero backward preserves DDP collectives.
    owned = [s for s in samples if s.learnable]
    empty = not owned
    samples = owned or samples[:1]
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
        entropies = torch.stack([-(row.exp() * row).sum() for row in logs])
        entropy = entropies.mean()
    rows = [(sample.old_log_prob, sample.advantage,
             math.log(len(sample.state.legal_actions)) if len(sample.state.legal_actions) > 1 else 0.,
             float(len(sample.state.records) <= entropy_opening_plies)) for sample in samples]
    metadata = _upload_targets(rows, current)
    old, advantage = metadata[:, 0], metadata[:, 1]
    log_ratio = (current - old).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    from .clipping import PolicyClip, advantage_clip_upper, clipped_surrogate
    bounds = policy_clip or PolicyClip.fixed(clip_epsilon)
    upper = advantage_clip_upper(advantage, bounds.upper, bounds.bonus, bounds.minimum, bounds.maximum)
    objective, outside = clipped_surrogate(ratio, advantage, bounds.lower, upper)
    policy_loss = -objective.mean()
    entropy_bonus, exploration_metrics = policy_entropy_bonus(
        states, entropies, coefficient=entropy_coefficient,
        opening_coefficient=opening_entropy_coefficient, opening_plies=entropy_opening_plies,
        metadata=metadata[:, 2:])
    loss = policy_loss - entropy_bonus
    if empty:
        loss = loss * 0
    names = ("loss/policy_ppo", "loss/policy_total", "policy/approx_kl_old", "policy/entropy",
             "policy/clip_fraction", "policy/importance_ratio_mean", "policy/nonzero_advantage_fraction")
    values = torch.stack((policy_loss, loss, (ratio - 1 - log_ratio).mean(), entropy,
                          outside.mean(), ratio.mean(),
                          (advantage != 0).float().mean(), *exploration_metrics.values(), upper.mean())).detach()
    names = (*names, *exploration_metrics, "policy/clip_upper_mean")
    if empty:
        values = values * 0
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
    has_components = hasattr(getattr(critic, "module", critic), "draw_value_head")
    predictions = critic([sample.value_state or sample.state for sample in samples],
                         **({"return_components": True} if has_components else {}), **options).float()
    values, common = predictions.unbind(-1) if has_components else (predictions, torch.zeros_like(predictions))
    metadata = _upload_targets([(s.old_value, s.value_target, s.old_draw_value, s.draw_value_target)
                                for s in samples], values)
    old, targets, old_common, common_targets = metadata.unbind(-1)
    outcome, old_outcome, outcome_targets = values - common, old - old_common, targets - common_targets
    clipped = old_outcome + (outcome - old_outcome).clamp(-clip_epsilon, clip_epsilon)
    common_clipped = old_common + (common - old_common).clamp(-clip_epsilon, clip_epsilon)
    outcome_mse = torch.maximum((outcome - outcome_targets).square(), (clipped - outcome_targets).square()).mean()
    common_mse = torch.maximum((common - common_targets).square(), (common_clipped - common_targets).square()).mean()
    mse = outcome_mse + common_mse
    loss = 0.5 * value_coefficient * mse
    names = ("loss/critic_total", "critic/value_mse", "critic/value_mean", "critic/target_mean",
             "critic/draw_value_mse", "critic/draw_value_mean", "critic/draw_target_mean")
    metrics = torch.stack((loss, (values - targets).square().mean(), values.mean(), targets.mean(),
                           (common - common_targets).square().mean(), common.mean(), common_targets.mean())).detach()
    metrics = metrics.unbind() if defer_metrics else metrics.cpu().tolist()
    return LossOutput(loss, dict(zip(names, metrics, strict=True)))
