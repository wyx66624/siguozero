"""Public flag-defense decisions, personal GAE credit and full-state adoption."""
import copy
from dataclasses import replace
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pytest
import torch

from junqi import ArmPoint, GameConfig, GameVariant, InformationMode, JunqiGame, Piece, PieceType
from junqi.training.encoding import GameHistory
from junqi.training.distributed import DistributedContext
from junqi.training.mappo import collect_mappo_samples, player_step_rewards
from junqi.training.mappo_encoding import CentralizedHistory
from junqi.training.metrics import MetricLogger
from junqi.training.models import GameCentralValueTransformer, GamePolicyTransformer, PieceConditionedLayoutPointerDecoder
from junqi.training.modes import TrainingMode
from junqi.training.objective_migration import reconcile_flag_defense_objective
from junqi.training.ppo import FrozenValueActor, normalize_advantages
from junqi.training.ppo_environment import _advance
from junqi.training.rollout import BaseGamePool, FrozenPolicyActor, RolloutMetrics
from junqi.training.settings import TrainingSettings
from junqi.training.tactical_rewards import flag_defense_context
from junqi.training.tactical_logging import MAPPOTrainingAudit, TacticalLogSettings
from junqi.training.trainer import SelfPlayTrainer
from test_mappo_tactical_rewards import assert_tree_equal
from test_mappo_tactical_logging import report_tool

CONFIG = Path(__file__).parents[1] / 'configs/expand_mappo_training.yaml'
MODE = TrainingMode.TWO_PLAYER
REWARDS = dict(flag_defense_missed_penalty=.1, flag_defense_success_reward=.05)


def position(*, actor=0, defender=PieceType.COMPANY_COMMANDER, enemy=PieceType.ENGINEER,
             known=True, extra=None, max_plies=None, moved=True):
    other = 1 - actor
    target, source = ArmPoint(actor, 5, 4), ArmPoint(actor, 5, 3)
    pieces = {ArmPoint(p, 6, 4): Piece(p, PieceType.FLAG) for p in range(2)}
    pieces.update({ArmPoint(p, 2, 1): Piece(p, PieceType.COMMANDER) for p in range(2)})
    pieces.update({source: Piece(actor, defender), target: Piece(other, enemy, has_moved=moved)})
    pieces.update(extra or {})
    identities = [{}, {}]
    if known:
        identities[actor][target] = enemy
    game = JunqiGame.from_position(GameConfig(variant=GameVariant.TWO_PLAYER,
        information_mode=InformationMode.DARK, max_plies=max_plies), pieces,
        current_player=actor, known_identities=identities)
    game.enable_inventory_tracking()
    board = game.board_for(actor)
    attack = (board.encode(source), board.encode(target))
    quiet = (board.encode(ArmPoint(actor, 2, 1)), board.encode(ArmPoint(actor, 3, 1)))
    assert (pieces[target].owner == actor or attack in game.legal_actions()) and quiet in game.legal_actions()
    return game, attack, quiet


@pytest.fixture(autouse=True)
def one_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


@pytest.mark.parametrize('actor', [0, 1])
@pytest.mark.parametrize('defender', [PieceType.COMPANY_COMMANDER, PieceType.ENGINEER, PieceType.BOMB])
def test_any_rank_capture_or_trade_defends_flag(actor, defender):
    game, attack, quiet = position(actor=actor, defender=defender)
    assert attack in flag_defense_context(game)[1]
    row = _advance(0, game, attack, MODE, track_flag_defense=True)
    assert row.flag_defense == 'success'
    assert not row.terminal
    # No second bonus for a later unrelated move after the threat is removed.
    row = _advance(0, game, game.legal_actions()[0], MODE, track_flag_defense=True)
    assert row.flag_defense is None


def test_ignoring_a_defensible_threat_is_missed_and_no_legal_removal_is_exempt():
    game, _, quiet = position()
    assert _advance(0, game, quiet, MODE, track_flag_defense=True).flag_defense == 'missed'
    game, _, quiet = position(enemy=PieceType.ARMY_COMMANDER)
    assert not flag_defense_context(game)[1]
    assert _advance(0, game, quiet, MODE, track_flag_defense=True).flag_defense is None


def test_failed_weak_attack_is_missed_when_another_piece_can_defend():
    game, attack, _ = position(enemy=PieceType.ARMY_COMMANDER,
        extra={ArmPoint(0, 4, 4): Piece(0, PieceType.BOMB)})
    assert flag_defense_context(game)[1] and attack not in flag_defense_context(game)[1]
    assert _advance(0, game, attack, MODE, track_flag_defense=True).flag_defense == 'missed'


def test_hidden_identity_cannot_change_missed_penalty_or_actor_inputs():
    observations, contexts, rows = [], [], []
    target, reserve = ArmPoint(0, 5, 4), ArmPoint(1, 4, 1)
    for enemy, other in ((PieceType.ENGINEER, PieceType.ARMY_COMMANDER),
                         (PieceType.ARMY_COMMANDER, PieceType.ENGINEER)):
        game, _, quiet = position(enemy=enemy, known=False,
            extra={reserve: Piece(1, other, has_moved=True)})
        observations.append(game.observe(0))
        contexts.append(flag_defense_context(game))
        rows.append(_advance(0, game, quiet, MODE, track_flag_defense=True))
    assert observations[0] == observations[1]
    assert contexts[0] == contexts[1]
    assert rows[0].flag_defense is rows[1].flag_defense is None
    np.testing.assert_array_equal(rows[0].records[0], rows[1].records[0])


def test_own_piece_in_front_of_flag_is_not_an_enemy_threat():
    game, _, quiet = position(extra={ArmPoint(0, 5, 4): Piece(0, PieceType.ENGINEER)}, known=False)
    assert not flag_defense_context(game)[0]
    assert _advance(0, game, quiet, MODE, track_flag_defense=True).flag_defense is None


def test_unknown_mobile_threat_can_be_removed_with_bomb_without_rank_information():
    game, attack, _ = position(defender=PieceType.BOMB, known=False)
    assert attack in flag_defense_context(game)[1]
    assert _advance(0, game, attack, MODE, track_flag_defense=True).flag_defense == 'success'


def test_multiple_threats_require_full_relief_for_success():
    # A second moved engineer can attack the flag through its side road.
    second = ArmPoint(0, 6, 3)
    game, attack, _ = position(extra={second: Piece(1, PieceType.ENGINEER, has_moved=True)})
    assert len(flag_defense_context(game)[0]) == 2
    assert _advance(0, game, attack, MODE, track_flag_defense=True).flag_defense is None


def test_immediate_win_is_success_even_without_capturing_flag_attacker():
    game, _, _ = position(extra={ArmPoint(1, 5, 4): Piece(0, PieceType.ENGINEER)})
    board = game.board_for(0)
    win = (board.encode(ArmPoint(1, 5, 4)), board.encode(ArmPoint(1, 6, 4)))
    row = _advance(0, game, win, MODE, track_flag_defense=True)
    assert row.terminal and row.flag_defense == 'success'
    assert player_step_rewards(row, player=0, player_count=2, draw_reward=-.15,
        flag_capture_reward=0, bomb_rewards={}, futile_reward=0, flag_defense_reward=.05) == (1.05, -1.)


@pytest.mark.parametrize('workers,groups', [(1, 1), (2, 1), (2, 2)])
def test_real_worker_collection_gaes_bomb_stacking_and_personal_credit(workers, groups, tmp_path):
    settings = TrainingSettings.from_yaml(CONFIG, MODE, tiny=True)
    model = replace(settings.model, temporal_layers=2, token_fusion_hidden_dim=64,
                    ppo_fixed_kv=False, ppo_cuda_graphs=False, history_window_stride=3)
    actor = FrozenPolicyActor(GamePolicyTransformer(model).eval(), max_batch_size=4)
    critic = FrozenValueActor(GameCentralValueTransformer(model).eval(), amp_dtype=None,
                              max_batch_size=4, deferred=True)
    layout = PieceConditionedLayoutPointerDecoder(model).eval()
    pool = BaseGamePool(MODE, pool_size=3, max_transitions=model.max_transitions,
                       max_game_plies=None, seed=44, centralized_critic=True,
                       history_window_stride=model.history_window_stride)
    pool.fill(layout, 0)
    positions = [position(), position(), position(defender=PieceType.BOMB, enemy=PieceType.ARMY_COMMANDER)]
    moves = []
    for i, (slot, (game, attack, quiet)) in enumerate(zip(pool.slots, positions, strict=True)):
        slot.game = game
        slot.history = GameHistory.initialize(game, MODE, max_transitions=model.max_transitions,
            history_window_stride=model.history_window_stride)
        slot.critic_history = CentralizedHistory.initialize(game, MODE, max_transitions=model.max_transitions,
            history_window_stride=model.history_window_stride)
        moves.append(quiet if i == 1 else attack)
    move_iter = iter(moves)
    audit = MAPPOTrainingAudit(TacticalLogSettings(enabled=True, environment_lanes=3, probe_samples=0), update=1)
    with patch.object(actor, 'sample', side_effect=lambda states, **_: ([[next(move_iter)] for _ in states], [[-1.]] * len(states))), \
         patch.object(critic, 'values_packed', side_effect=lambda states: [0.] * len(states)):
        samples, _, metrics = collect_mappo_samples(pool, actor, critic, layout,
            count=3, behavior_version=0, discount=1., gae_lambda=.95,
            environment_workers=workers, pipeline_groups=groups,
            advantage_mode='player_decisions', reward_scope='player',
            training_audit=audit,
            bomb_commander_reward=.03, bomb_army_commander_reward=.01,
            bomb_division_commander_reward=.003, **REWARDS)
    assert [s.value_target for s in samples] == pytest.approx([.05, -.1, .06])
    assert [s.advantage for s in samples] == pytest.approx([.05, -.1, .06])
    assert all(s.additional_values[0].target == 0. for s in samples)
    assert (metrics.flag_defense_successes, metrics.flag_defense_missed) == (2, 1)
    assert metrics.flag_defense_reward_sum == pytest.approx(0.)
    assert metrics.bomb_reward_sum == pytest.approx(.01)
    assert metrics.environment_plies == 3
    assert all(not s.state.centralized for s in samples)
    raw = [sample.advantage for sample in samples]
    mean, std = float(np.mean(raw)), float(np.std(raw))
    normalized = normalize_advantages(samples, mean=mean, std=std, epsilon=1e-4)
    audit.bind(normalized, normalization=dict(mean=mean, std=std, epsilon=1e-4), objective=dict(
        mode=MODE.value, advantage_mode='player_decisions', reward_scope='player', discount=1.,
        gae_lambda=.95, draw_reward=-.15, flag_capture_reward=0., futile_attack_penalty=0.,
        bomb_commander_reward=.03, bomb_army_commander_reward=.01, bomb_division_commander_reward=.003, **REWARDS))
    audit.finish()
    audit.write(tmp_path, environment_plies=3)
    archive = next((tmp_path/'tactical_diagnostics/details').glob('*.pt.gz'))
    with gzip.open(archive, 'rb') as stream:
        payload = torch.load(stream, weights_only=False)
    assert report_tool.verify_replay(payload)['verified_decisions'] == 3


def test_frozen_actions_do_not_receive_guard_shaping_and_metrics_aggregate():
    metrics = RolloutMetrics()
    assert metrics.record_flag_defense('success', .1, .05, learnable=False) == 0.
    assert metrics.record_flag_defense('missed', .1, .05, learnable=False) == 0.
    assert metrics.flag_defense_successes == metrics.flag_defense_missed == 0
    metrics.record_flag_defense('success', .1, .05)
    metrics.record_flag_defense('missed', .1, .05)
    trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
    trainer.distributed = DistributedContext(0, 1, 0, torch.device('cpu'))
    result = trainer._aggregate_rollout_metrics(metrics).as_dict()
    assert result['rollout/flag_defense_reward_sum'] == -.05


def test_full_checkpoint_adoption_preserves_weights_optimizers_rng_games_and_resumes(tmp_path):
    settings = TrainingSettings.from_yaml(CONFIG, MODE, tiny=True, overrides=dict(
        device='cpu', target_environment_plies=64, anchor_batch=4, base_game_pool_size=2,
        policy_microbatch=2, adaptive_learning_rate=False, inference_snapshot_every_updates=0))
    with patch.object(MetricLogger, 'start_resource_monitor'):
        old = SelfPlayTrainer(settings, run_directory=tmp_path)
        old.train()
        path = old.checkpoints.latest_path
        before = torch.load(path, weights_only=False)
        target = replace(settings, **REWARDS)
        with pytest.raises(RuntimeError, match='adopt-flag-defense-rewards'):
            reconcile_flag_defense_objective(copy.deepcopy(before), target)
        trainer = SelfPlayTrainer(target, run_directory=tmp_path, adopt_flag_defense_rewards=True)
        try:
            after = torch.load(path, weights_only=False)
            check = unittest.TestCase()
            for key in ('policy', 'critic', 'layout', 'reference_layout', 'policy_optimizer',
                        'critic_optimizer', 'layout_optimizer', 'rng_state'):
                assert_tree_equal(check, after[key], before[key])
            for key in ('base_game_pool', 'cumulative', 'layout_buffer', 'adaptive_learning_rate',
                        'adaptive_entropy', 'historical_opponents', 'model_selection'):
                if key in before['trainer_state']:
                    assert_tree_equal(check, after['trainer_state'][key], before['trainer_state'][key])
            assert after['reason'] == 'flag_defense_objective_migration'
            assert trainer.flag_defense_objective_migration['to_rewards'] == REWARDS
            saved = json.loads((trainer.run_directory/'resolved_config.json').read_text())
            assert saved['flag_defense_missed_penalty'] == .1
        finally:
            trainer.historical.close()
            trainer.logger.close()
        resumed = SelfPlayTrainer(target, run_directory=tmp_path)
        try:
            assert resumed.update == before['update']
            assert resumed.flag_defense_objective_migration['to_rewards'] == REWARDS
        finally:
            resumed.historical.close()
            resumed.logger.close()


@pytest.mark.parametrize('field', list(REWARDS))
def test_guard_config_validation_and_default_scope(field):
    settings = TrainingSettings.from_yaml(CONFIG, MODE, tiny=True)
    for invalid in (-.1, True, float('nan'), float('inf'), '.1'):
        with pytest.raises(ValueError, match=field):
            replace(settings, **{field: invalid}).validate()
    for changed in ({'algorithm': 'ppo', 'mappo_advantage_mode': 'team_steps', 'mappo_reward_scope': 'team'},
                    {'mode': TrainingMode.FOUR_DARK}):
        with pytest.raises(ValueError, match='requires two-player MAPPO'):
            replace(settings, **{field: .1}, **changed).validate()
