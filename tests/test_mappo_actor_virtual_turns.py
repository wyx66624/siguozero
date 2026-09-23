"""Shared dead-seat Actor/Critic clock, lambda returns, and versioned replay."""
import copy
from dataclasses import replace
from functools import lru_cache
from unittest.mock import patch

import pytest
import torch

from junqi import ArmPoint, GameConfig, GameVariant, InformationMode, JunqiGame, Piece, PieceType
from junqi.training.mappo import (PlayerMAPPOTransition, collect_mappo_samples,
    critic_training_samples, player_generalized_advantages, virtual_player_turns)
from junqi.training.ppo import critic_ppo_loss, normalize_advantages
from junqi.training.tactical_logging import MAPPOTrainingAudit, TacticalLogSettings
from test_mappo import histories
from test_mappo_tactical_logging import report_tool
from test_mappo_tactical_rewards import MAPPOTacticalRewardTests


def death_trace(cycles, *, terminal=True, nonzero_values=False):
    game, private, central = histories()
    state = private.state_for(game)
    views = tuple(central.value_state_for(p) for p in range(4))
    # Seat 3 acts, seat 2 captures its flag, then the other three keep playing.
    actors = [3, 2] + [1, 0, 2] * cycles
    trace = []
    for t, actor in enumerate(actors):
        rewards = [0.] * 4
        if t == 1:
            rewards[3] = -.2
        if terminal and t == len(actors)-1:
            rewards = [-1., 1., -1., 1.]
        following = None if t == len(actors)-1 else actors[t+1]
        trace.append(PlayerMAPPOTransition(state, (0, 0), -1., views, tuple(rewards),
            terminal and t == len(actors)-1, actor, learnable=actor != 2,
            old_values=tuple(.03*t-.1*p for p in range(4)) if nonzero_values else (0.,)*4,
            virtual_turns=virtual_player_turns(player=actor, next_player=following,
                                              player_count=4, turn_step=-1)))
    return trace


def calculate(trace, *, gamma=1., lam=.95, bootstrap=(0.,)*4):
    return player_generalized_advantages(trace, bootstrap_values=bootstrap,
        discount=gamma, gae_lambda=lam, behavior_version=7)


def reference_returns(trace, *, gamma=1., lam=.95, bootstrap=(0.,)*4, virtual=True):
    """Independent interval-return definition, without summing TD residuals."""
    @lru_cache(None)
    def target(t, player):
        reward = 0.
        for j in range(t, len(trace)):
            row = trace[j]
            reward += gamma**(j-t)*row.rewards[player]
            if row.terminal:
                return reward
            weight = gamma**(j+1-t)
            if j+1 == len(trace):
                return reward + weight*bootstrap[player]
            if trace[j+1].value_player == player or (virtual and player in row.virtual_turns):
                return reward + weight*((1-lam)*trace[j+1].old_values[player]
                                        + lam*target(j+1, player))
        raise AssertionError('missing interval boundary')
    return [[target(t, p) for p in range(4)] for t in range(len(trace))]


@pytest.mark.parametrize('actor,following,expected', [
    (0, 3, ()), (0, 2, (3,)), (1, 2, (0, 3)), (2, 1, ()), (1, None, ())])
def test_virtual_clock_follows_original_seats_even_when_predecessor_is_dead(actor, following, expected):
    assert virtual_player_turns(player=actor, next_player=following,
                                player_count=4, turn_step=-1) == expected
    assert virtual_player_turns(player=0, next_player=1, player_count=2, turn_step=1) == ()


@pytest.mark.parametrize('gamma,lam', [(1., .95), (.9, .7), (1., 0.), (1., 1.)])
@pytest.mark.parametrize('cycles', [1, 3, 8])
def test_terminal_credit_decays_once_per_virtual_turn_without_weakening_flag_loss(gamma, lam, cycles):
    trace = death_trace(cycles)
    samples = calculate(trace, gamma=gamma, lam=lam)
    # Independent closed form: the immediate loss is on the next real move;
    # only the remote terminal reward crosses all `cycles` virtual turns.
    expected = -.2*gamma + gamma**(len(trace)-1)*lam**cycles
    assert samples[0].advantage == pytest.approx(expected)
    assert samples[0].value_target == pytest.approx(expected)
    removed_loss = list(trace)
    removed_loss[1] = replace(trace[1], rewards=(0.,)*4)
    assert calculate(removed_loss, gamma=gamma, lam=lam)[0].advantage-samples[0].advantage == pytest.approx(.2*gamma)
    assert len(samples) == len(trace)
    assert [s.learnable for s in samples] == [t.learnable for t in trace]


def test_virtual_boundary_values_train_dead_views_and_leave_surviving_views_unchanged():
    trace = death_trace(3, nonzero_values=True)
    old = calculate([replace(t, virtual_turns=()) for t in trace])
    new = calculate(trace)
    assert new[0].advantage != old[0].advantage
    # These players survive throughout: their individual Actor signals match.
    assert [s.advantage for s in new[1:]] == [s.advantage for s in old[1:]]
    normalized = normalize_advantages(new, mean=.3, std=.8, epsilon=1e-8)
    old_values = critic_training_samples(old)
    new_values = critic_training_samples(normalized)
    expected = reference_returns(trace)
    assert len(old_values) == len(new_values) == 4*len(trace)
    assert any(before.value_target != after.value_target for before, after in zip(old_values, new_values, strict=True))
    for index, (before, after) in enumerate(zip(old_values, new_values, strict=True)):
        assert before.state is after.state
        assert before.old_value == after.old_value and before.value_player == after.value_player
        assert after.value_target == pytest.approx(expected[index//4][after.value_player])
        assert after.advantage == pytest.approx(after.value_target-after.old_value)
        if after.value_player != 3:
            assert before.value_target == after.value_target

    # The first virtual turn is after seat 0's move, at state index 4.
    changed = list(trace)
    changed[4] = replace(trace[4], old_values=(*trace[4].old_values[:3], trace[4].old_values[3]+.5))
    modified = calculate(changed)[0]
    assert modified.advantage-new[0].advantage == pytest.approx(.5*(1-.95))
    assert modified.value_target-new[0].value_target == pytest.approx(.5*(1-.95))


@pytest.mark.parametrize('gamma,lam', [(1., .95), (.9, .7), (1., 0.), (1., 1.)])
def test_all_four_targets_match_interval_returns_with_truncation_and_game_resets(gamma, lam):
    trace = death_trace(3, nonzero_values=True) + death_trace(2, terminal=False, nonzero_values=True)
    bootstrap = (.1, .2, -.3, .4)
    samples = calculate(trace, gamma=gamma, lam=lam, bootstrap=bootstrap)
    expected = reference_returns(trace, gamma=gamma, lam=lam, bootstrap=bootstrap)
    for index, sample in enumerate(samples):
        values = {sample.value_player: sample.value_target}
        values.update({v.player: v.target for v in sample.additional_values})
        assert [values[p] for p in range(4)] == pytest.approx(expected[index])
        assert sample.advantage == pytest.approx(expected[index][sample.value_player]-sample.old_value)


def test_existing_critic_loss_and_gradient_consume_the_new_target():
    trace = death_trace(8)
    trace[0] = replace(trace[0], old_values=(0., 0., 0., .6))
    old = calculate([replace(t, virtual_turns=()) for t in trace])
    new = calculate(trace)
    assert new[0].value_target < .6 < old[0].value_target

    class ConstantCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(.6))

        def forward(self, states):
            return self.value.expand(len(states))

    critic = ConstantCritic()
    results = []
    for samples in (old, new):
        batch = critic_training_samples([samples[0]])[:1]
        critic.zero_grad()
        loss = critic_ppo_loss(critic, batch, clip_epsilon=.2, value_coefficient=.5)
        loss.loss.backward()
        results.append((loss.loss.item(), critic.value.grad.item()))
    assert results[0][1] < 0 < results[1][1]  # The new target now lowers this value.


def test_truncation_bootstraps_and_terminal_reset_does_not_leak():
    truncated = death_trace(3, terminal=False)
    truncated_sample = calculate(truncated, bootstrap=(0., 0., 0., .6))[0]
    assert truncated_sample.advantage == pytest.approx(-.2+.95**3*.6)
    assert truncated_sample.value_target == pytest.approx(-.2+.95**3*.6)
    complete = death_trace(2)
    restarted = [replace(t, rewards=(999.,)*4) for t in death_trace(1)]
    assert calculate(complete+restarted)[0].advantage == calculate(complete)[0].advantage
    immediate = [complete[0], replace(complete[1], terminal=True, rewards=(1., -1., 1., -1.2))]
    sample = calculate(immediate, bootstrap=(999.,)*4)[0]
    assert sample.advantage == sample.value_target == -1.2


def flag_capture_position(*, predecessor_dead=False):
    pieces = {ArmPoint(p, 6, 4): Piece(p, PieceType.FLAG) for p in range(4)}
    pieces.update({ArmPoint(p, 2, 3): Piece(p, PieceType.ENGINEER) for p in range(4)
                   if not (predecessor_dead and p == 0)})
    start, end = ArmPoint(3, 5, 4), ArmPoint(3, 6, 4)
    pieces[start] = Piece(2, PieceType.ENGINEER)
    game = JunqiGame.from_position(GameConfig(variant=GameVariant.FOUR_PLAYER,
        information_mode=InformationMode.FOUR_DARK, max_plies=8), pieces, current_player=3,
        # Even with unused passes, seat 0 is eliminated when its
        # turn arrives. Its slot and the already-dead seat 3 are both crossed.
        passes_remaining=(4, 4, 4, 4))
    game.enable_inventory_tracking()
    capture = (game.board_for(2).encode(start), game.board_for(2).encode(end))
    return game, capture


@pytest.mark.parametrize('workers,groups', [(1, 1), (2, 2)])
@pytest.mark.parametrize('predecessor_dead', [False, True])
def test_real_collection_and_replay_use_one_clock_with_legacy_archive_compatibility(workers, groups, predecessor_dead):
    torch.set_num_threads(1)
    game, capture = flag_capture_position(predecessor_dead=predecessor_dead)
    cycles = 3 if predecessor_dead else 2
    helper = MAPPOTacticalRewardTests()
    actor, critic, layout, pool = helper.models_and_pool([(copy.deepcopy(game), capture) for _ in range(4)])
    audit = MAPPOTrainingAudit(TacticalLogSettings(enabled=True, environment_lanes=1, probe_samples=0), update=1)

    def sample(states, *, count, return_log_probs):
        moves = [capture if len(s.records) == 2 else (0, 0) for s in states]
        assert all(a in s.legal_actions for a, s in zip(moves, states, strict=True))
        return [[a] for a in moves], [[-1.]]*len(moves)

    with patch.object(actor, 'sample', side_effect=sample), patch.object(
            critic, 'values_packed', side_effect=lambda states: [0.]*len(states)), patch(
            'junqi.training.mappo.player_generalized_advantages', wraps=player_generalized_advantages) as gae:
        samples, _, metrics = collect_mappo_samples(pool, actor, critic, layout,
            count=32, behavior_version=0, discount=1., gae_lambda=.95,
            draw_reward=-.15, flag_capture_reward=.2, environment_workers=workers,
            pipeline_groups=groups, advantage_mode='player_decisions', reward_scope='player',
            training_audit=audit)
    assert metrics.environment_plies == metrics.policy_samples == 32
    assert len(critic_training_samples(samples)) == 128
    dead_actions = [s for s in samples if s.value_player == 3]
    assert len(dead_actions) == 4  # No fabricated Actor samples after death.
    for s in dead_actions:
        assert s.advantage == pytest.approx(-.2-.15*.95**cycles)
        assert s.value_target == pytest.approx(s.old_value+s.advantage)

    normalized = normalize_advantages(samples, mean=.2, std=.5, epsilon=1e-8)
    objective = dict(advantage_mode='player_decisions', reward_scope='player', discount=1., gae_lambda=.95,
        draw_reward=-.15, flag_capture_reward=.2, futile_attack_penalty=0.,
        bomb_commander_reward=0., bomb_army_commander_reward=0., bomb_division_commander_reward=0.)
    audit.bind(normalized, normalization=dict(mean=.2, std=.5, epsilon=1e-8), objective=objective)
    rows = [s.training_diagnostic for s in normalized if s.training_diagnostic is not None]
    assert [r['virtual_player_turns'] for r in rows].count([0, 3] if predecessor_dead else [3]) == cycles
    dead_row = next(r for r in rows if r['value_player'] == 3)
    assert dead_row['raw_advantage'] == pytest.approx(-.2-.15*.95**cycles)
    assert dead_row['raw_advantage'] == pytest.approx(dead_row['value_target']-dead_row['old_value'])
    assert dead_row['critic_gae_clock'] == dead_row['actor_gae_clock'] == 'player_turns_including_eliminated'
    payload = dict(objective=objective, mode='four_dark', update=1,
        games=[g for g in audit.games.values() if g['steps']], normalization=audit.normalization)
    assert report_tool.verify_replay(payload)['verified_decisions'] == 8
    # Both old real-decision archives and the intermediate Actor-only version
    # must retain their own Critic targets when verified by the new code.
    trace = next(call.args[0] for call in gae.call_args_list
                 if call.args[0][0].training_diagnostic is not None)
    previous_targets = reference_returns(trace, virtual=False)
    for actor_only in (False, True):
        legacy = copy.deepcopy(payload)
        for saved in legacy['games']:
            for index, step in enumerate(saved['steps']):
                row = step['row']
                row['value_target'] = previous_targets[index][row['value_player']]
                for value in row['critic_values']:
                    value['target'] = previous_targets[index][value['player']]
                if actor_only:
                    row['critic_gae_clock'] = 'player_decisions'
                else:
                    for field in ('actor_gae_clock', 'critic_gae_clock', 'virtual_player_turns'):
                        del row[field]
                    row['raw_advantage'] = row['value_target']-row['old_value']
                    row['normalized_advantage'] = (row['raw_advantage']-.2)/(.5+1e-8)
        assert report_tool.verify_replay(legacy)['verified_decisions'] == 8
