"""MAPPO with private actors and a separate, scalar, full-information critic.

There is one real actor transition per environment step. Personal MAPPO uses
each player's rotated value view and its own decision-time GAE. The legacy
fixed-team mode remains available for explicit checkpoint compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import gc
import time
from typing import Sequence

from .encoding import PolicyState
from .ppo import PPOSample, ValueTarget
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
    training_diagnostic: dict | None = None


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
        other = 1 - team
        result.append(PPOSample(
            state=item.state, action=item.action, old_log_prob=item.old_log_prob,
            old_value=item.old_values[team], advantage=advantages[team],
            value_target=item.old_values[team] + advantages[team],
            behavior_version=behavior_version, learnable=item.learnable,
            training_diagnostic=item.training_diagnostic,
            value_state=item.value_states[team],
            other_team_value=ValueTarget(item.value_states[other], item.old_values[other],
                                         item.old_values[other] + advantages[other]),
        ))
        next_values, tails = list(item.old_values), advantages
    return list(reversed(result))


def critic_training_samples(samples: Sequence[PPOSample]) -> list[PPOSample]:
    """Supervise every value perspective used to bootstrap GAE.

    The acting team's sample alone leaves all opponent-to-move values without
    labels, even though those values bootstrap its returns. Keep these extra
    labels out of the actor batch and out of environment-step accounting.
    """
    values = []
    for sample in samples:
        extras = sample.additional_values or ((sample.other_team_value,)
                                              if sample.other_team_value is not None else ())
        if (sample.value_state is None or not extras
                or not sample.value_state.centralized or any(not v.state.centralized for v in extras)):
            raise ValueError("MAPPO critic requires both centralized team targets")
        values.append(replace(sample, state=sample.value_state, value_state=None, other_team_value=None,
                              additional_values=(), advantage=sample.value_target - sample.old_value))
        values.extend(replace(sample, state=other.state, value_state=None, other_team_value=None,
                    additional_values=(), value_player=other.player, old_value=other.old_value,
                    value_target=other.target, advantage=other.target - other.old_value) for other in extras)
    return values


@dataclass(frozen=True, slots=True)
class PlayerMAPPOTransition:
    state: PolicyState
    action: tuple[int, int]
    old_log_prob: float
    value_states: tuple[PolicyState, ...]
    rewards: tuple[float, ...]
    terminal: bool
    value_player: int
    learnable: bool = True
    old_values: tuple[float, ...] = ()
    training_diagnostic: dict | None = None
    # Eliminated seats crossed after this real move, before the next actor.
    # These advance each player's Actor/Critic trace, not the action count.
    virtual_turns: tuple[int, ...] = ()


def virtual_player_turns(*, player: int, next_player: int | None,
                        player_count: int, turn_step: int) -> tuple[int, ...]:
    """Recover skipped dead seats from the rules engine's resolved next actor.

    A game.step includes automatic no-move eliminations. All skipped seats use
    its final post-step board, already present in the four critic histories.
    """
    if (player_count not in (2, 4) or turn_step not in (-1, 1)
            or type(player) is not int or not 0 <= player < player_count):
        raise ValueError("invalid personal turn order")
    if next_player is None:
        return ()  # Whole-game termination has no following virtual turns.
    if (type(next_player) is not int or not 0 <= next_player < player_count
            or next_player == player):
        raise ValueError("invalid next player for personal turn order")
    skipped = []
    seat = (player + turn_step) % player_count
    while seat != next_player:
        skipped.append(seat)
        seat = (seat + turn_step) % player_count
    return tuple(skipped)


def player_generalized_advantages(transitions: Sequence[PlayerMAPPOTransition], *,
        bootstrap_values: tuple[float, ...], discount: float, gae_lambda: float,
        behavior_version: int) -> list[PPOSample]:
    """Personal Actor GAE and Critic lambda returns, including dead turns.

    Between two decisions of player i, lambda is 1: rewards accumulate and
    intermediate values telescope. Lambda also applies when
    the rules pass i's eliminated seat, retaining its personal turn clock.
    Thus A_i = R_interval + gamma**d * V_i(next)
    - V_i(now) + gamma**d * lambda * A_i(next). Discount remains per real ply.
    Every Critic perspective uses V_old + its own raw trace, including off-turn
    and eliminated-player views. Only the real actor's trace enters PPO; its
    later normalization must not change the four value targets.
    """
    players = len(bootstrap_values)
    if players not in (2, 4) or not 0 < discount <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("invalid per-player MAPPO GAE settings")
    next_values, tails = list(bootstrap_values), [0.] * players
    next_actor = None
    result = []
    for item in reversed(transitions):
        actor = item.value_player
        if (type(actor) is not int or not 0 <= actor < players
                or len(item.value_states) != players or len(item.old_values) != players
                or len(item.rewards) != players):
            raise ValueError("per-player MAPPO requires aligned player values and rewards")
        if (len(set(item.virtual_turns)) != len(item.virtual_turns)
                or any(type(p) is not int or not 0 <= p < players
                       or p in (actor, next_actor) for p in item.virtual_turns)):
            raise ValueError("virtual turns must identify skipped eliminated players")
        factor = 0. if item.terminal else discount
        deltas = [item.rewards[p] + factor * next_values[p] - item.old_values[p]
                  for p in range(players)]
        advantages = [deltas[p] + factor * (
            gae_lambda if next_actor == p or p in item.virtual_turns else 1.) * tails[p]
            for p in range(players)]
        if item.training_diagnostic is not None:
            item.training_diagnostic.update(raw_advantage=advantages[actor],
                actor_gae_clock="player_turns_including_eliminated",
                critic_gae_clock="player_turns_including_eliminated",
                virtual_player_turns=list(item.virtual_turns))
        result.append(PPOSample(state=item.state, action=item.action, old_log_prob=item.old_log_prob,
            old_value=item.old_values[actor], advantage=advantages[actor],
            value_target=item.old_values[actor] + advantages[actor], behavior_version=behavior_version,
            learnable=item.learnable, value_state=item.value_states[actor], value_player=actor,
            training_diagnostic=item.training_diagnostic,
            additional_values=tuple(ValueTarget(item.value_states[p], item.old_values[p],
                item.old_values[p] + advantages[p], p) for p in range(players) if p != actor)))
        next_values, tails, next_actor = list(item.old_values), advantages, actor
    return list(reversed(result))


def player_step_rewards(row, *, player: int, player_count: int, draw_reward: float,
                        flag_capture_reward: float, bomb_rewards, futile_reward: float,
                        reward_scope: str = "player", known_nonflag_hq_reward: float = 0.) -> tuple[float, ...]:
    """Same terminal result for allies; configurable ownership of event rewards."""
    if reward_scope not in ("player", "team"):
        raise ValueError("MAPPO reward scope must be player or team")
    rewards = [terminal_utility(row.rewards[p], draw_reward=draw_reward) if row.terminal else 0.
               for p in range(player_count)]
    def credit(owner, amount):
        recipients = (owner,) if reward_scope == "player" else tuple(
            p for p in range(player_count) if p % 2 == owner % 2)
        for p in recipients:
            rewards[p] += amount
    if row.flag_captured_owner is not None:
        credit(player, flag_capture_reward)
        credit(row.flag_captured_owner, -flag_capture_reward)
    if row.bombed_kind is not None:
        credit(row.bomb_owner, bomb_rewards[row.bombed_kind])
    credit(player, futile_reward)
    # A personal action cost; do not punish the ally or reward the opponents.
    rewards[player] += known_nonflag_hq_reward
    return tuple(rewards)


def collect_mappo_samples(
    pool, actor, critic, layout, *, count, behavior_version, discount, gae_lambda,
    environment_workers=1, environment=None, pipeline_groups=1,
    draw_reward=DEFAULT_DRAW_REWARD, flag_capture_reward=0.0, futile_attack_penalty=0.0,
    bomb_commander_reward=0.0, bomb_army_commander_reward=0.0, bomb_division_commander_reward=0.0,
    known_nonflag_hq_penalty=0.0,
    advantage_mode="team_steps", reward_scope="team",
    training_audit=None,
):
    """Collect private actions and global values without changing step budgets."""
    if type(count) is not int or count <= 0:
        raise ValueError("MAPPO transition count must be positive")
    if type(pipeline_groups) is not int or pipeline_groups <= 0:
        raise ValueError("MAPPO pipeline groups must be positive")
    if advantage_mode not in ("team_steps", "player_decisions") or reward_scope not in ("team", "player"):
        raise ValueError("invalid MAPPO advantage mode or reward scope")
    if advantage_mode == "team_steps" and reward_scope != "team":
        raise ValueError("personal rewards require player decision advantages")
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
                          known_nonflag_hq_penalty=known_nonflag_hq_penalty,
                          advantage_mode=advantage_mode, reward_scope=reward_scope,
                          training_audit=training_audit,
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
             environment, groups, draw_reward, flag_capture_reward, futile_attack_penalty, bomb_rewards,
             advantage_mode, reward_scope, known_nonflag_hq_penalty, training_audit=None):
    started = time.perf_counter()
    historical = pool.historical
    if historical is not None:
        historical.begin_rollout(pool, actor, behavior_version)
    pool.fill(layout, behavior_version)
    personal = advantage_mode == "player_decisions"
    perspectives = len(pool.slots[0].history.players) if personal else 2

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
    if training_audit is not None:
        training_audit.begin(pool)
    boundary = time.perf_counter()
    environment.begin(pool, track_futile_attacks=futile_attack_penalty > 0, centralized_critic=True,
                      **({"track_known_nonflag_hq": True} if known_nonflag_hq_penalty else {}),
                      **({"tactical_indices": training_audit.indices} if training_audit is not None else {}))
    metrics.environment_sync_seconds += time.perf_counter() - boundary
    metrics.environment_workers = len(environment.processes)
    initial_worker_seconds = environment.worker_seconds
    traces, outcomes = [[] for _ in pool.slots], []
    size = max(1, (len(pool.slots) + groups - 1) // groups)
    waves = []
    for offset in range(0, count, len(pool.slots)):
        indices = list(range(min(len(pool.slots), count - offset)))
        waves.extend(indices[i:i + size] for i in range(0, len(indices), size))

    def sample(indices, *, defer_result=False):
        seats = [environment.states[i].player for i in indices]
        states = [pool.slots[i].history.players[seat].as_policy_state(environment.states[i].legal_actions)
                  for i, seat in zip(indices, seats, strict=True)]
        owned = [pool.slots[i].learner_owns(seat) for i, seat in zip(indices, seats, strict=True)]
        globals_ = [tuple(pool.slots[i].critic_history.value_state_for(p) for p in range(perspectives)) for i in indices]
        owners = (seats if personal else
                  [pool.slots[i].value_seat(seat) % 2 for i, seat in zip(indices, seats, strict=True)])
        boundary = time.perf_counter()
        actions, logs = [None] * len(states), [None] * len(states)
        pending_results = []
        for learner in (True, False):
            positions = [p for p, own in enumerate(owned) if own == learner]
            if not positions:
                continue
            source = actor if learner else historical.actor
            sampler = getattr(source, 'sample_async', source.sample) if defer_result else source.sample
            sampled = sampler([states[p] for p in positions], count=1, return_log_probs=True)
            if callable(getattr(sampled, 'result', None)):
                pending_results.append((positions, sampled))
                metrics.actor_deferred_batches += 1
            else:
                batch_actions, batch_logs = sampled
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
        return (states, actions, logs, owned, globals_, owners), pending_results

    def finish_sample(prepared):
        wave, pending_results = prepared
        boundary = time.perf_counter()
        for positions, pending_result in pending_results:
            batch_actions, batch_logs = pending_result.result()
            for p, action, log in zip(positions, batch_actions, batch_logs, strict=True):
                wave[1][p], wave[2][p] = action, log
        if pending_results:
            seconds = time.perf_counter() - boundary
            metrics.actor_result_wait_seconds += seconds
            metrics.actor_inference_seconds += seconds
        return wave

    ready = finish_sample(sample(waves[0]))
    for wave_index, indices in enumerate(waves):
        states, actions, logs, owned, globals_, owners = ready
        boundary = time.perf_counter()
        environment.submit(indices, [a[0] for a in actions])
        metrics.environment_step_seconds += time.perf_counter() - boundary
        next_indices = waves[wave_index + 1] if wave_index + 1 < len(waves) else None
        can_overlap = next_indices is not None and not set(indices).intersection(next_indices)
        # Submit next-wave inference before receiving/assembling the previous
        # wave. Its pinned action copy precedes any layout work on this stream;
        # randomness and game ordering remain identical to synchronous sampling.
        pending = sample(next_indices, defer_result=True) if can_overlap else None
        boundary = time.perf_counter()
        responses = environment.receive()
        if [row.index for row in responses] != indices:
            raise RuntimeError("MAPPO workers returned an incomplete wave")
        for index, state, action, log, learnable, global_states, owner, row in zip(
            indices, states, actions, logs, owned, globals_, owners, responses, strict=True,
        ):
            slot = pool.slots[index]
            if row.centralized_records is None:
                raise RuntimeError("MAPPO worker omitted centralized observations")
            metrics.record_environment_steps()
            futile_reward = metrics.record_futile_attack(row.futile_attack, futile_attack_penalty,
                                                         learnable=learnable)
            hq_reward = metrics.record_known_nonflag_hq(row.known_nonflag_hq, known_nonflag_hq_penalty,
                                                       learnable=learnable)
            captures = [flag_capture_utility(slot.game, row.flag_captured_owner,
                                            player=t, coefficient=flag_capture_reward) for t in (0, 1)]
            metrics.record_flag_capture(row.flag_captured_owner, captures[owner % 2], terminal=row.terminal)
            if row.bombed_kind is not None:
                bomb_reward = bomb_rewards[row.bombed_kind]
                metrics.record_bomb(row.bombed_kind, bomb_reward)
            if personal:
                rewards = player_step_rewards(row, player=owner, player_count=perspectives,
                    draw_reward=draw_reward, flag_capture_reward=flag_capture_reward,
                    bomb_rewards=bomb_rewards, futile_reward=futile_reward, reward_scope=reward_scope,
                    known_nonflag_hq_reward=hq_reward)
                transition = PlayerMAPPOTransition(state, action[0], float(log[0]), global_states,
                    rewards, row.terminal, owner, learnable,
                    virtual_turns=virtual_player_turns(player=owner, next_player=row.state.player,
                        player_count=perspectives, turn_step=slot.game.config.turn_step))
            else:
                rewards = [(terminal_utility(row.rewards[t], draw_reward=draw_reward) if row.terminal else 0.)
                           + captures[t] for t in (0, 1)]
                if row.bombed_kind is not None:
                    rewards[slot.game.team_of(row.bomb_owner)] += bomb_reward
                rewards[owner] += futile_reward + hq_reward
                transition = MAPPOTransition(state, action[0], float(log[0]), global_states,
                                             tuple(rewards), row.terminal, owner, learnable)
            if training_audit is not None:
                training_audit.total_decisions += 1
            if training_audit is not None and index in training_audit.indices:
                diagnostic = training_audit.record(index, row, state, learnable=learnable,
                    behavior_version=behavior_version, rewards=rewards, reward_terms=dict(
                        terminal=list(row.rewards) if row.terminal else None,
                        draw_reward=draw_reward, flag_captured_owner=row.flag_captured_owner,
                        flag_capture_coefficient=flag_capture_reward,
                        bomb_owner=row.bomb_owner, bombed_kind=row.bombed_kind,
                        bomb_reward=bomb_rewards.get(row.bombed_kind, 0.),
                        futile_attack=row.futile_attack, futile_attack_reward=futile_reward,
                        known_nonflag_hq=row.known_nonflag_hq, known_nonflag_hq_reward=hq_reward))
                if diagnostic is not None:
                    transition = replace(transition, training_diagnostic=diagnostic)
            slot.history.append_encoded_rows(row.records)
            slot.critic_history.append_encoded_rows(row.centralized_records)
            traces[index].append(transition)
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
                if training_audit is not None:
                    training_audit.new_game(index, slot)
                environment.replace(index, slot.game)
        metrics.environment_step_seconds += time.perf_counter() - boundary
        if next_indices is not None:
            ready = finish_sample(pending if can_overlap else sample(next_indices))

    bootstrap_indices = [i for i, trace in enumerate(traces) if trace and not trace[-1].terminal]
    bootstrap_states = [pool.slots[i].critic_history.value_state_for(p)
                        for i in bootstrap_indices for p in range(perspectives)]
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
            trace[index] = replace(item, old_values=tuple(values[offset:offset + perspectives]))
            offset += perspectives
    bootstrap = {i: tuple(values[offset + perspectives*n:offset + perspectives*(n+1)])
                 for n, i in enumerate(bootstrap_indices)}
    if training_audit is not None:
        training_audit.bootstrap(bootstrap)
    metrics.critic_inference_seconds += time.perf_counter() - boundary
    boundary = time.perf_counter()
    environment.synchronize(pool)
    metrics.environment_sync_seconds += time.perf_counter() - boundary
    advantage_function = player_generalized_advantages if personal else team_generalized_advantages
    samples = [sample for i, trace in enumerate(traces) for sample in advantage_function(
        trace, bootstrap_values=bootstrap.get(i, (0.,) * perspectives), discount=discount,
        gae_lambda=gae_lambda, behavior_version=behavior_version)]
    metrics.environment_worker_seconds = environment.worker_seconds - initial_worker_seconds
    metrics.policy_samples = sum(sample.learnable for sample in samples)
    metrics.wall_seconds = time.perf_counter() - started
    return samples, outcomes, metrics
