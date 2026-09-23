"""Independent per-player returns, event ownership, visibility and resume."""
import copy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from junqi.training.mappo import (PlayerMAPPOTransition, collect_mappo_samples,
    critic_training_samples, player_generalized_advantages, player_step_rewards)
from junqi.training.objective_migration import reconcile_player_advantages
from junqi.training.ppo import normalize_advantages
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer
from test_mappo import histories
import test_mappo_tactical_rewards as tactical_helpers
from test_mappo_tactical_rewards import bomb_position, REWARDS


def trace_fixture():
    game, private, central = histories()
    state = private.state_for(game)
    views = tuple(central.value_state_for(p) for p in range(4))
    # Skipped/eliminated players, repeated allies and a reset are deliberate.
    actors = [0, 1, 2, 3, 0, 2, 0, 2, 0, 1]
    rewards = [(0.,)*4 for _ in actors]
    rewards[2] = (0., 0., .03, 0.)
    rewards[3] = (-.2, 0., 0., .2)
    rewards[7] = (1., -1., 1., -1.)
    rewards[8] = (-.15,)*4
    return [PlayerMAPPOTransition(state, state.legal_actions[0], -1., views, rewards[i],
        i in (7,8), actor, i != 2, tuple(.07*i - .13*p for p in range(4)))
        for i,actor in enumerate(actors)]


@pytest.mark.parametrize('gamma,lam', [(1.,.95),(.9,.7),(1.,1.),(1.,0.)])
def test_global_recursion_matches_independent_personal_decision_intervals(gamma,lam):
    trace=trace_fixture()
    bootstrap=(.3,-.2,.7,-.5)
    @lru_cache(None)
    def reference(t,p):
        reward=0.
        for j in range(t,len(trace)):
            item=trace[j]
            reward += gamma**(j-t)*item.rewards[p]
            if item.terminal:
                return reward-trace[t].old_values[p]
            if j+1==len(trace):
                return reward+gamma**(j+1-t)*bootstrap[p]-trace[t].old_values[p]
            if trace[j+1].value_player==p:
                weight=gamma**(j+1-t)
                return reward+weight*trace[j+1].old_values[p]-trace[t].old_values[p]+weight*lam*reference(j+1,p)
        raise AssertionError('missing interval boundary')
    samples=player_generalized_advantages(trace,bootstrap_values=bootstrap,
        discount=gamma,gae_lambda=lam,behavior_version=21)
    normalized=normalize_advantages(samples,mean=.2,std=.3,epsilon=1e-8)
    values=critic_training_samples(normalized)
    assert len(samples)==len(trace) and len(values)==4*len(trace)
    for t,(sample,item) in enumerate(zip(samples,trace,strict=True)):
        assert sample.state is item.state and not sample.state.centralized
        assert sample.value_player==item.value_player
        assert sample.advantage==pytest.approx(reference(t,item.value_player))
        assert sample.learnable==item.learnable
        assert sample.value_state is item.value_states[item.value_player]
        assert normalized[t].additional_values is sample.additional_values
        targets={s.value_player:s for s in values[4*t:4*t+4]}
        assert targets.keys()=={0,1,2,3}
        for p,s in targets.items():
            assert s.state is item.value_states[p] and s.state.centralized
            assert s.value_target==pytest.approx(item.old_values[p]+reference(t,p))
            assert s.advantage==pytest.approx(reference(t,p))
            assert not s.additional_values and s.other_team_value is None


def test_intermediate_off_turn_values_cancel_from_the_players_decision_advantage():
    trace=trace_fixture()[:8]
    changed=[replace(t,old_values=tuple(v if p==t.value_player else v+1000.
                     for p,v in enumerate(t.old_values))) for t in trace]
    kwargs=dict(bootstrap_values=(0.,)*4,discount=1.,gae_lambda=.95,behavior_version=0)
    old=player_generalized_advantages(trace,**kwargs)
    new=player_generalized_advantages(changed,**kwargs)
    assert [s.advantage for s in new]==pytest.approx([s.advantage for s in old],abs=1e-10)
    # 0 and 2 share a win but have their own baseline and reward histories.
    assert old[0].advantage != old[2].advantage


@pytest.mark.parametrize('actor',range(4))
@pytest.mark.parametrize('defensive',[False,True])
def test_bomb_bonus_has_one_personal_owner_with_shared_terminal_result(actor,defensive):
    owner=(actor+1)%4 if defensive else actor
    row=SimpleNamespace(terminal=True,rewards=(1.,-1.,1.,-1.),
        flag_captured_owner=None,bomb_owner=owner,bombed_kind='commander')
    values=player_step_rewards(row,player=actor,player_count=4,draw_reward=-.15,
        flag_capture_reward=.2,bomb_rewards={'commander':.03},futile_reward=-.01)
    expected=[1.,-1.,1.,-1.]
    expected[owner]+=.03
    expected[actor]-=.01
    assert values==pytest.approx(expected)
    assert values[(owner+2)%4]==pytest.approx(expected[(owner+2)%4])


def test_personal_flag_rewards_and_draw_are_additive_and_team_scope_is_explicit():
    row=SimpleNamespace(terminal=True,rewards=(0.,)*4,
                       flag_captured_owner=3,bombed_kind=None,bomb_owner=None)
    args=dict(player=0,player_count=4,draw_reward=-.15,flag_capture_reward=.2,
              bomb_rewards={},futile_reward=0.)
    assert player_step_rewards(row,**args)==pytest.approx((.05,-.15,-.15,-.35))
    assert player_step_rewards(row,**args,reward_scope='team')==pytest.approx((.05,-.35,.05,-.35))


@pytest.mark.parametrize('workers,groups',[(1,1),(2,2)])
def test_real_collection_uses_actual_frozen_seat_and_four_value_targets(workers,groups):
    torch.set_num_threads(1)
    helper=tactical_helpers.MAPPOTacticalRewardTests()
    positions=[bomb_position(actor=p) for p in range(4)]
    actor,critic,layout,pool=helper.models_and_pool(positions)
    for slot in pool.slots:
        slot.opponent_id=slot.teammate_id='old'
        slot.learner_team=slot.learner_seat=0
    pool.historical=SimpleNamespace(actor=actor,opponent_plies=0,teammate_plies=0,
        begin_rollout=lambda *_:None,attach_cache=lambda *_:None,clear_caches=lambda:None)
    with patch.object(actor,'sample',side_effect=helper.scripted([p[1] for p in positions])):
        samples,_,metrics=collect_mappo_samples(pool,actor,critic,layout,count=4,behavior_version=0,
            discount=1.,gae_lambda=.95,environment_workers=workers,pipeline_groups=groups,
            advantage_mode='player_decisions',reward_scope='player',**REWARDS)
    assert metrics.environment_plies==4 and metrics.policy_samples==1
    assert [s.value_player for s in samples]==[0,1,2,3]
    assert [s.learnable for s in samples]==[True,False,False,False]
    for p,s in enumerate(samples):
        assert s.value_target==pytest.approx(.03,abs=1e-6)
        assert s.state.records[-1].current_player==0
        assert s.value_state.records[-1].current_player==0
        assert all(v.target==pytest.approx(0.,abs=1e-6) for v in s.additional_values)
        assert {v.player for v in s.additional_values}==set(range(4))-{p}


def test_personal_critic_covers_four_views_with_unchanged_adam_step_budget():
    samples=player_generalized_advantages(trace_fixture(),bootstrap_values=(0.,)*4,
        discount=1.,gae_lambda=.95,behavior_version=0)
    trainer=SelfPlayTrainer.__new__(SelfPlayTrainer)
    trainer.settings=SimpleNamespace(algorithm='mappo',critic_epochs=3,policy_epochs=3,
        ppo_minibatch_samples=3,ppo_max_samples_per_sequence=256,
        early_stop_kl_threshold=.1,lr_kl_band_lower=.08)
    trainer.distributed=SimpleNamespace(sum_metrics=lambda s:s)
    trainer._sequence_training_enabled=lambda:True
    seen=[]
    def update(batch,*,epochs):
        assert len(batch)<=12 and epochs==1
        seen.extend(batch)
        return {'optimizer/critic_steps':1.}
    trainer._update_critic_batch=update
    result=trainer._update_critic(samples)
    assert result['optimizer/critic_epochs']==3
    assert result['optimizer/critic_steps']==12  # ceil(10/3)*3
    assert result['optimizer/critic_value_samples']==40
    assert result['optimizer/critic_auxiliary_value_samples']==30
    assert result['optimizer/critic_perspectives_per_transition']==4
    assert len(seen)==120 and {s.value_player for s in seen}==set(range(4))


def test_personal_contract_requires_explicit_adoption_and_preserves_saved_state():
    settings=SimpleNamespace(algorithm='mappo',mappo_advantage_mode='player_decisions',mappo_reward_scope='player')
    payload={'update':42,'config':{},'trainer_state':{'cumulative':{'environment_plies':1234},
             'base_game_pool':{'sentinel':[1,2,3]}},'policy':{'w':torch.tensor([3.])}}
    with pytest.raises(RuntimeError,match='adopt-player-advantages'):
        reconcile_player_advantages(payload,settings)
    reconcile_player_advantages(payload,settings,adopt=True)
    assert payload['policy']['w'].item()==3
    assert payload['trainer_state']['base_game_pool']=={'sentinel':[1,2,3]}
    assert payload['trainer_state']['cumulative']=={'environment_plies':1234}
    once=copy.deepcopy(payload)
    reconcile_player_advantages(payload,settings)
    assert payload==once
    assert payload['trainer_state']['player_advantage_migration']['from_advantage_mode']=='team_steps'


def test_player_metrics_count_only_their_own_learnable_actions():
    samples=player_generalized_advantages(trace_fixture(),bootstrap_values=(0.,)*4,
        discount=1.,gae_lambda=.95,behavior_version=0)
    trainer=SelfPlayTrainer.__new__(SelfPlayTrainer)
    trainer.settings=SimpleNamespace(algorithm='mappo',mappo_advantage_mode='player_decisions',
                                    advantage_epsilon=1e-8)
    trainer.distributed=SimpleNamespace(sum_metrics=lambda s:s)
    normalized=trainer._normalize_ppo_batch(samples)
    assert sum(trainer.ppo_rollout_metrics[f'ppo/player_{p}/samples'] for p in range(4))==9
    for p in range(4):
        own=[s.advantage for s in samples if s.learnable and s.value_player==p]
        assert trainer.ppo_rollout_metrics[f'ppo/player_{p}/raw_advantage_mean']==pytest.approx(sum(own)/len(own))
    # Statistics and global scaling must not collapse allies' distinct values.
    assert normalized[0].advantage != normalized[2].advantage
    assert len(critic_training_samples(normalized))==40


def test_expand_config_and_pinned_initial_turn_identify_relative_player():
    settings=TrainingSettings.from_yaml(Path(__file__).parents[1]/'configs/expand_mappo_training.yaml',
                                        'four_dark',model_scale='main')
    assert (settings.mappo_advantage_mode,settings.mappo_reward_scope)==('player_decisions','player')
    game,private,central=histories()
    assert game.config.first_player==0
    for player in range(4):
        relative=game._relative_player_order(player).index(0)
        assert private.players[player].records[0].current_player==relative
        assert central.value_state_for(player).records[0].current_player==relative
