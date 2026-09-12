"""Overlap independent game groups with one frozen policy and one CPU wave."""
from __future__ import annotations

import time

from .ppo import PPOTransition, _finalize_rollout_values, generalized_advantages
from .rollout import LayoutOutcome, RolloutMetrics


def collect_pipelined(pool, actor, critic, layout, *, count, behavior_version,
                      discount, gae_lambda, environment, groups=2):
    if not critic.deferred or not actor.policy.config.ppo_array_history:
        raise ValueError('pipelined PPO requires deferred values and array histories')
    if groups < 2:
        raise ValueError('pipelined PPO needs at least two scheduling groups')
    started = time.perf_counter()
    pool.fill(layout, behavior_version)
    for slot in pool.slots:
        slot.history.enable_array_storage()
    actor.policy.start_ppo_inference_cache(
        capacity=len(pool.slots) * len(pool.slots[0].history.players), behavior_version=behavior_version)
    metrics = RolloutMetrics()
    boundary = time.perf_counter()
    environment.begin(pool)
    metrics.environment_sync_seconds += time.perf_counter() - boundary
    metrics.environment_workers = len(environment.processes)
    initial_worker_seconds = environment.worker_seconds
    traces, outcomes = [[] for _ in pool.slots], []
    size = max(1, (len(pool.slots) + groups - 1) // groups)
    waves = []
    for offset in range(0, count, len(pool.slots)):
        indices = list(range(min(len(pool.slots), count - offset)))
        waves.extend(indices[i:i + size] for i in range(0, len(indices), size))

    def states_for(indices):
        return [pool.slots[i].history.players[environment.states[i].player].as_policy_state(
            environment.states[i].legal_actions) for i in indices]

    def sample(indices):
        states = states_for(indices)
        boundary = time.perf_counter()
        actions, logs = actor.sample(states, count=1, return_log_probs=True)
        metrics.actor_inference_seconds += time.perf_counter() - boundary
        return states, actions, logs

    ready = sample(waves[0])
    for wave_index, indices in enumerate(waves):
        states, actions, logs = ready
        boundary = time.perf_counter()
        environment.submit(indices, [a[0] for a in actions])
        metrics.environment_step_seconds += time.perf_counter() - boundary
        next_indices = waves[wave_index + 1] if wave_index + 1 < len(waves) else None
        # The next group must not observe an unfinished step in its own game.
        can_overlap = next_indices is not None and not set(indices).intersection(next_indices)
        pending = sample(next_indices) if can_overlap else None
        boundary = time.perf_counter()
        responses = environment.receive()
        if [row.index for row in responses] != indices:
            raise RuntimeError('PPO workers returned an incomplete pipeline wave')
        for index, state, action, log, row in zip(indices, states, actions, logs, responses, strict=True):
            slot = pool.slots[index]
            slot.history.append_encoded_rows(row.records)
            metrics.record_environment_steps()
            traces[index].append(PPOTransition(state, action[0], float(log[0]), 0.,
                                               row.reward, row.terminal, row.next_team_sign))
            if row.terminal:
                metrics.base_games_completed += 1
                metrics.wins += int(row.rewards[0] > 0)
                metrics.draws += int(row.rewards[0] == 0)
                metrics.losses += int(row.rewards[0] < 0)
                outcomes.extend(LayoutOutcome(sample=s, reward=row.rewards[seat], seat=seat,
                                              behavior_version=slot.layout_behavior_version)
                                for seat, s in enumerate(slot.layouts))
                if actor.policy._fixed_kv_store is not None:
                    actor.policy._fixed_kv_store.release_game(state.records.identity[0])
                slot = pool.slots[index] = pool._new_slot(layout, behavior_version)
                slot.history.enable_array_storage()
                environment.replace(index, slot.game)
        metrics.environment_step_seconds += time.perf_counter() - boundary
        if next_indices is not None:
            ready = pending if can_overlap else sample(next_indices)
    bootstrap_indices = [i for i, trace in enumerate(traces) if trace and not trace[-1].terminal]
    boundary = time.perf_counter()
    bootstrap = dict(zip(bootstrap_indices, _finalize_rollout_values(
        traces, states_for(bootstrap_indices), critic, actor), strict=True))
    metrics.critic_inference_seconds += time.perf_counter() - boundary
    boundary = time.perf_counter()
    environment.synchronize(pool)
    metrics.environment_sync_seconds += time.perf_counter() - boundary
    samples = [s for i, trace in enumerate(traces) for s in generalized_advantages(
        trace, bootstrap_value=bootstrap.get(i, 0.), discount=discount,
        gae_lambda=gae_lambda, behavior_version=behavior_version)]
    metrics.environment_worker_seconds = environment.worker_seconds - initial_worker_seconds
    metrics.policy_samples = count
    metrics.wall_seconds = time.perf_counter() - started
    return samples, outcomes, metrics
