"""Overlap game groups with batched current/historical policies and CPU rules."""
from __future__ import annotations

import time

from .ppo import PPOTransition, _finalize_rollout_values, generalized_advantages
from .rollout import LayoutOutcome, RolloutMetrics
from .rewards import DEFAULT_DRAW_REWARD, terminal_utility


def collect_pipelined(pool, actor, critic, layout, *, count, behavior_version,
                      discount, gae_lambda, environment, groups=2, draw_reward=DEFAULT_DRAW_REWARD):
    if not critic.deferred or not actor.policy.config.ppo_array_history:
        raise ValueError('pipelined PPO requires deferred values and array histories')
    if groups < 2:
        raise ValueError('pipelined PPO needs at least two scheduling groups')
    started = time.perf_counter()
    historical = pool.historical
    if historical is not None:
        historical.begin_rollout(pool, actor, behavior_version)
    pool.fill(layout, behavior_version)
    for slot in pool.slots:
        slot.history.enable_array_storage()
    actor.policy.start_ppo_inference_cache(
        capacity=len(pool.slots) * len(pool.slots[0].history.players), behavior_version=behavior_version)
    if historical is not None:
        historical.attach_cache(actor, behavior_version)
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
        owned = [pool.slots[i].learner_owns(environment.states[i].player) for i in indices]
        # Current seats, including current opponents, learn from their own view.
        # Frozen turns bridge the trajectory using the anchor's private view.
        values = [state if learnable else pool.slots[i].history.players[pool.slots[i].primary_seat].as_value_state()
                  for i, state, learnable in zip(indices, states, owned, strict=True)]
        boundary = time.perf_counter()
        actions, logs = [None] * len(states), [None] * len(states)
        for learner in (True, False):
            positions = [p for p, own in enumerate(owned) if own == learner]
            if not positions:
                continue
            source = actor if learner else historical.actor
            batch_actions, batch_logs = source.sample([states[p] for p in positions], count=1, return_log_probs=True)
            for p, action, log in zip(positions, batch_actions, batch_logs, strict=True):
                actions[p], logs[p] = action, log
            if not learner:
                for p in positions:
                    slot, seat = pool.slots[indices[p]], environment.states[indices[p]].player
                    if seat % 2 == slot.learner_team:
                        historical.teammate_plies += 1
                    else:
                        historical.opponent_plies += 1
        metrics.actor_inference_seconds += time.perf_counter() - boundary
        return states, actions, logs, owned, values

    ready = sample(waves[0])
    for wave_index, indices in enumerate(waves):
        states, actions, logs, owned, values = ready
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
        for index, state, action, log, learnable, value_state, row in zip(indices, states, actions, logs, owned, values, responses, strict=True):
            slot = pool.slots[index]
            slot.history.append_encoded_rows(row.records)
            metrics.record_environment_steps()
            historical_game = slot.opponent_id is not None or slot.teammate_id is not None
            value_owner = slot.value_seat(state.records.identity[1]) if historical_game else None
            reward = row.rewards[value_owner] if historical_game and row.terminal else row.reward
            sign = row.next_team_sign
            if historical_game and not row.terminal:
                sign = 1 if value_owner % 2 == slot.value_seat(row.state.player) % 2 else -1
            traces[index].append(PPOTransition(state, action[0], float(log[0]), 0.,
                terminal_utility(reward, draw_reward=draw_reward) if row.terminal else 0.,
                row.terminal, sign,
                terminal_draw=row.terminal and reward == 0, learnable=learnable,
                value_state=value_state if historical_game else None))
            if row.terminal:
                metrics.base_games_completed += 1
                metrics.wins += int(row.rewards[0] > 0)
                metrics.draws += int(row.rewards[0] == 0)
                metrics.losses += int(row.rewards[0] < 0)
                outcomes.extend(LayoutOutcome(sample=s, reward=terminal_utility(row.rewards[seat], draw_reward=draw_reward), seat=seat,
                                              behavior_version=slot.layout_behavior_version)
                                for seat, s in enumerate(slot.layouts) if slot.learner_owns(seat))
                if historical is not None:
                    historical.record_result(slot, row.rewards[slot.learner_team])
                if actor.policy._fixed_kv_store is not None:
                    actor.policy._fixed_kv_store.release_game(state.records.identity[0])
                slot = pool.slots[index] = pool._new_slot(layout, behavior_version)
                slot.history.enable_array_storage()
                environment.replace(index, slot.game)
        metrics.environment_step_seconds += time.perf_counter() - boundary
        if next_indices is not None:
            ready = pending if can_overlap else sample(next_indices)
    bootstrap_indices = [i for i, trace in enumerate(traces) if trace and not trace[-1].terminal]
    bootstrap_states = [pool.slots[i].history.players[pool.slots[i].value_seat(
        environment.states[i].player)].as_value_state() for i in bootstrap_indices]
    # Both views own the same allocation: release the frozen view before
    # releasing the learner view and allocating the deferred critic batch.
    if historical is not None:
        historical.clear_caches()
    boundary = time.perf_counter()
    bootstrap = dict(zip(bootstrap_indices, _finalize_rollout_values(
        traces, bootstrap_states, critic, actor), strict=True))
    metrics.critic_inference_seconds += time.perf_counter() - boundary
    boundary = time.perf_counter()
    environment.synchronize(pool)
    metrics.environment_sync_seconds += time.perf_counter() - boundary
    samples = [s for i, trace in enumerate(traces) for s in generalized_advantages(
        trace, bootstrap_value=bootstrap.get(i, (0., 0.))[0], discount=discount,
        bootstrap_draw_value=bootstrap.get(i, (0., 0.))[1],
        gae_lambda=gae_lambda, behavior_version=behavior_version)]
    metrics.environment_worker_seconds = environment.worker_seconds - initial_worker_seconds
    metrics.policy_samples = sum(sample.learnable for sample in samples)
    metrics.wall_seconds = time.perf_counter() - started
    return samples, outcomes, metrics
