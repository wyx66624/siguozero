"""MAPPO with private actors and a separate, scalar, full-information critic.

There is one real transition per environment step. Values/GAE are evaluated
from both fixed team perspectives with the same scalar network. They must not
be obtained by negation: a drawn game penalizes both teams equally.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import gc
import time
from typing import Sequence

from .encoding import PolicyState
from .ppo import PPOSample
from .rewards import DEFAULT_DRAW_REWARD, flag_capture_utility, terminal_utility
from .rollout import LayoutOutcome, RolloutMetrics
from .tactical_rewards import BOMB_REWARD_FIELDS


@dataclass(frozen=True, slots=True)
class MAPPOTransition:
    state: PolicyState
    action: tuple[int, int]
    old_log_prob: float
    value_states: tuple[PolicyState, PolicyState]
    rewards: tuple[float, float]
    terminal: bool
    value_team: int
    learnable: bool = True
    old_values: tuple[float, float] = (0.0, 0.0)


def team_generalized_advantages(
    transitions: Sequence[MAPPOTransition], *, bootstrap_values: tuple[float, float],
    discount: float, gae_lambda: float, behavior_version: int,
) -> list[PPOSample]:
    """Scalar GAE per fixed team, including frozen turns, eliminations and resets.

    Team 0 always refers to seats 0/2 and team 1 to seats 1/3. Neither skipped
    turns nor a captured flag change those identities. Only game termination
    cuts the trace; collector truncation uses the frozen global bootstrap.
    """
    if not 0 < discount <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("invalid MAPPO discount or lambda")
    next_values, tails = list(bootstrap_values), [0.0, 0.0]
    result = []
    for item in reversed(transitions):
        if item.value_team not in (0, 1):
            raise ValueError("MAPPO value team must be 0 or 1")
        factor = 0.0 if item.terminal else discount
        advantages = [item.rewards[team] + factor * next_values[team] - item.old_values[team]
                      + factor * gae_lambda * tails[team] for team in (0, 1)]
        team = item.value_team
        result.append(PPOSample(
            state=item.state, action=item.action, old_log_prob=item.old_log_prob,
            old_value=item.old_values[team], advantage=advantages[team],
            value_target=item.old_values[team] + advantages[team],
            behavior_version=behavior_version, learnable=item.learnable,
            value_state=item.value_states[team],
        ))
        next_values, tails = list(item.old_values), advantages
    return list(reversed(result))


def collect_mappo_samples(
    pool, actor, critic, layout, *, count, behavior_version, discount, gae_lambda,
    environment_workers=1, environment=None, pipeline_groups=1,
    draw_reward=DEFAULT_DRAW_REWARD, flag_capture_reward=0.0, futile_attack_penalty=0.0,
    bomb_commander_reward=0.0, bomb_army_commander_reward=0.0, bomb_division_commander_reward=0.0,
):
    """Collect private actions and global values without changing step budgets."""
    if type(count) is not int or count <= 0:
        raise ValueError("MAPPO transition count must be positive")
    if type(pipeline_groups) is not int or pipeline_groups <= 0:
        raise ValueError("MAPPO pipeline groups must be positive")
    if not pool.centralized_critic:
        raise ValueError("MAPPO requires a pool with separate global histories")
    if not critic.deferred or not actor.policy.config.ppo_array_history:
        raise ValueError("MAPPO requires deferred values and array histories")
    if hasattr(critic.critic, "draw_value_head"):
        raise ValueError("MAPPO requires a single V head")
    from .ppo_environment import ParallelPPOEnvironment
    owned = environment is None
    environment = environment or ParallelPPOEnvironment(environment_workers)
    # Collection creates many acyclic history views and IPC tuples. Reference
    # counts still reclaim them immediately; defer only cyclic scans until the
    # temporary traces have gone away, rather than repeatedly scanning a growing
    # batch. Restore the caller's GC state even after a worker/inference failure.
    collect_cycles = gc.isenabled()
    result = None
    if collect_cycles:
        gc.disable()
    try:
        result = _collect(pool, actor, critic, layout, count=count, behavior_version=behavior_version,
                          discount=discount, gae_lambda=gae_lambda, environment=environment,
                          groups=pipeline_groups, draw_reward=draw_reward,
                          flag_capture_reward=flag_capture_reward, futile_attack_penalty=futile_attack_penalty,
                          bomb_rewards=dict(zip(BOMB_REWARD_FIELDS, (
                              bomb_commander_reward, bomb_army_commander_reward, bomb_division_commander_reward), strict=True)))
        return result
    finally:
        try:
            if owned:
                environment.close()
        finally:
            if collect_cycles:
                boundary = time.perf_counter()
                try:
                    gc.collect()
                finally:
                    gc.enable()
                if result is not None:
                    result[2].wall_seconds += time.perf_counter() - boundary


def _collect(pool, actor, critic, layout, *, count, behavior_version, discount, gae_lambda,
             environment, groups, draw_reward, flag_capture_reward, futile_attack_penalty, bomb_rewards):
    started = time.perf_counter()
    historical = pool.historical
    if historical is not None:
        historical.begin_rollout(pool, actor, behavior_version)
    pool.fill(layout, behavior_version)

    def prepare(slot):
        if slot.critic_history is None:
            raise ValueError("MAPPO game is missing its global history")
        slot.history.enable_array_storage()
        slot.critic_history.enable_array_storage()

    for slot in pool.slots:
        prepare(slot)
    actor.policy.start_ppo_inference_cache(
        capacity=len(pool.slots) * len(pool.slots[0].history.players), behavior_version=behavior_version)
    if historical is not None:
        historical.attach_cache(actor, behavior_version)
    metrics = RolloutMetrics()
    boundary = time.perf_counter()
    environment.begin(pool, track_futile_attacks=futile_attack_penalty > 0, centralized_critic=True)
    metrics.environment_sync_seconds += time.perf_counter() - boundary
    metrics.environment_workers = len(environment.processes)
    initial_worker_seconds = environment.worker_seconds
    traces, outcomes = [[] for _ in pool.slots], []
    size = max(1, (len(pool.slots) + groups - 1) // groups)
    waves = []
    for offset in range(0, count, len(pool.slots)):
        indices = list(range(min(len(pool.slots), count - offset)))
        waves.extend(indices[i:i + size] for i in range(0, len(indices), size))

    def sample(indices):
        seats = [environment.states[i].player for i in indices]
        states = [pool.slots[i].history.players[seat].as_policy_state(environment.states[i].legal_actions)
                  for i, seat in zip(indices, seats, strict=True)]
        owned = [pool.slots[i].learner_owns(seat) for i, seat in zip(indices, seats, strict=True)]
        globals_ = [tuple(pool.slots[i].critic_history.value_state_for(team) for team in (0, 1)) for i in indices]
        teams = [pool.slots[i].value_seat(seat) % 2 for i, seat in zip(indices, seats, strict=True)]
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
                    slot = pool.slots[indices[p]]
                    if seats[p] % 2 == slot.learner_team:
                        historical.teammate_plies += 1
                    else:
                        historical.opponent_plies += 1
        metrics.actor_inference_seconds += time.perf_counter() - boundary
        return states, actions, logs, owned, globals_, teams

    ready = sample(waves[0])
    for wave_index, indices in enumerate(waves):
        states, actions, logs, owned, globals_, teams = ready
        boundary = time.perf_counter()
        environment.submit(indices, [a[0] for a in actions])
        metrics.environment_step_seconds += time.perf_counter() - boundary
        next_indices = waves[wave_index + 1] if wave_index + 1 < len(waves) else None
        can_overlap = next_indices is not None and not set(indices).intersection(next_indices)
        pending = sample(next_indices) if can_overlap else None
        boundary = time.perf_counter()
        responses = environment.receive()
        if [row.index for row in responses] != indices:
            raise RuntimeError("MAPPO workers returned an incomplete wave")
        for index, state, action, log, learnable, global_states, team, row in zip(
            indices, states, actions, logs, owned, globals_, teams, responses, strict=True,
        ):
            slot = pool.slots[index]
            if row.centralized_records is None:
                raise RuntimeError("MAPPO worker omitted centralized observations")
            slot.history.append_encoded_rows(row.records)
            slot.critic_history.append_encoded_rows(row.centralized_records)
            metrics.record_environment_steps()
            rewards = [(terminal_utility(row.rewards[t], draw_reward=draw_reward) if row.terminal else 0.0)
                       for t in (0, 1)]
            captures = [flag_capture_utility(slot.game, row.flag_captured_owner,
                                            player=t, coefficient=flag_capture_reward) for t in (0, 1)]
            rewards = [reward + capture for reward, capture in zip(rewards, captures, strict=True)]
            metrics.record_flag_capture(row.flag_captured_owner, captures[team], terminal=row.terminal)
            if row.bombed_kind is not None:
                bomb_reward = bomb_rewards[row.bombed_kind]
                # Credit the bomb's team, including defensive explosions and
                # frozen teammates. Explosion rewards are not zero-sum.
                rewards[slot.game.team_of(row.bomb_owner)] += bomb_reward
                metrics.record_bomb(row.bombed_kind, bomb_reward)
            # This non-zero-sum penalty belongs only to the acting learner's
            # team. Do not turn an opponent's futile move into a positive bonus.
            rewards[team] += metrics.record_futile_attack(row.futile_attack, futile_attack_penalty,
                                                         learnable=learnable)
            traces[index].append(MAPPOTransition(
                state=state, action=action[0], old_log_prob=float(log[0]),
                value_states=global_states, rewards=tuple(rewards), terminal=row.terminal,
                value_team=team, learnable=learnable,
            ))
            if row.terminal:
                metrics.base_games_completed += 1
                metrics.wins += int(row.rewards[0] > 0)
                metrics.draws += int(row.rewards[0] == 0)
                metrics.losses += int(row.rewards[0] < 0)
                outcomes.extend(LayoutOutcome(
                    sample=s, reward=terminal_utility(row.rewards[seat], draw_reward=draw_reward), seat=seat,
                    behavior_version=slot.layout_behavior_version,
                ) for seat, s in enumerate(slot.layouts) if slot.learner_owns(seat))
                if historical is not None:
                    historical.record_result(slot, row.rewards[slot.learner_team])
                if actor.policy._fixed_kv_store is not None:
                    actor.policy._fixed_kv_store.release_game(state.records.identity[0])
                slot = pool.slots[index] = pool._new_slot(layout, behavior_version)
                prepare(slot)
                environment.replace(index, slot.game)
        metrics.environment_step_seconds += time.perf_counter() - boundary
        if next_indices is not None:
            ready = pending if can_overlap else sample(next_indices)

    bootstrap_indices = [i for i, trace in enumerate(traces) if trace and not trace[-1].terminal]
    bootstrap_states = [pool.slots[i].critic_history.value_state_for(team)
                        for i in bootstrap_indices for team in (0, 1)]
    if historical is not None:
        historical.clear_caches()
    actor.policy._rollout_cache_snapshot = actor.policy.board_encoding_metrics()
    actor.policy.clear_inference_board_cache()
    boundary = time.perf_counter()
    value_states = [state for trace in traces for item in trace for state in item.value_states]
    values = critic.values_packed([*value_states, *bootstrap_states])
    offset = 0
    for trace in traces:
        for index, item in enumerate(trace):
            trace[index] = replace(item, old_values=(values[offset], values[offset + 1]))
            offset += 2
    bootstrap = {i: (values[offset + 2 * n], values[offset + 2 * n + 1])
                 for n, i in enumerate(bootstrap_indices)}
    metrics.critic_inference_seconds += time.perf_counter() - boundary
    boundary = time.perf_counter()
    environment.synchronize(pool)
    metrics.environment_sync_seconds += time.perf_counter() - boundary
    samples = [sample for i, trace in enumerate(traces) for sample in team_generalized_advantages(
        trace, bootstrap_values=bootstrap.get(i, (0.0, 0.0)), discount=discount,
        gae_lambda=gae_lambda, behavior_version=behavior_version)]
    metrics.environment_worker_seconds = environment.worker_seconds - initial_worker_seconds
    metrics.policy_samples = sum(sample.learnable for sample in samples)
    metrics.wall_seconds = time.perf_counter() - started
    return samples, outcomes, metrics
