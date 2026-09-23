"""Public pre-action HQ knowledge, personal costs, worker paths and resume."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from junqi import ArmPoint, GameConfig, InformationMode, JunqiGame, Piece, PieceType as K
from junqi.training.cli import build_parser
from junqi.training.encoding import GameHistory
from junqi.training.mappo import collect_mappo_samples, player_step_rewards
from junqi.training.mappo_encoding import CentralizedHistory
from junqi.training.metrics import MetricLogger
from junqi.training.models import GameCentralValueTransformer, GamePolicyTransformer, ModelConfig, PieceConditionedLayoutPointerDecoder
from junqi.training.modes import TrainingMode
from junqi.training.objective_migration import reconcile_known_nonflag_hq_objective
from junqi.training.ppo import FrozenValueActor
from junqi.training.ppo_environment import _advance
from junqi.training.rollout import BaseGamePool, FrozenPolicyActor, RolloutMetrics
from junqi.training.settings import TrainingSettings
from junqi.training.tactical_rewards import NONFLAG_HQ_PENALIZED_KINDS, known_nonflag_hq_entry
from junqi.training.trainer import SelfPlayTrainer

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / 'configs/expand_mappo_training.yaml'
MODE = TrainingMode.FOUR_DARK


def position(kind=K.ARMY_COMMANDER, *, actor=0, known=True, target=K.MINE, target_flag=False,
             own_arm=False, terminal=False, inferred=False):
    owner = actor if own_arm else (actor + 1) % 4
    flags = {ArmPoint(p, 6, 2): Piece(p, K.FLAG) for p in range(4)}
    pieces = {**flags, **{ArmPoint(p, 2, 3): Piece(p, K.ENGINEER) for p in range(4)}}
    for p in range(4):
        if p != owner or not known:
            pieces[ArmPoint(p, 3, 3)] = Piece(p, K.COMMANDER)
    source = ArmPoint(owner, 5, 2 if target_flag else 4)
    end = ArmPoint(owner, 6, 2 if target_flag else 4)
    pieces[source] = Piece(actor, kind)
    if not target_flag and target is not None:
        pieces[end] = Piece(owner, target)
    if kind is K.COMMANDER:
        pieces.pop(ArmPoint(actor, 3, 3), None)
    if target is K.COMMANDER:
        pieces.pop(ArmPoint(owner, 3, 3), None)
    known_identities = [{} for _ in range(4)]
    if inferred:
        known_identities[actor][ArmPoint(owner, 6, 2)] = K.FLAG
    game = JunqiGame.from_position(GameConfig(information_mode=InformationMode.FOUR_DARK,
        max_plies=1 if terminal else None), pieces, current_player=actor, known_identities=known_identities)
    game.enable_inventory_tracking()
    board = game.board_for(actor)
    action = (board.encode(source), board.encode(end))
    assert action in game.legal_actions()
    return game, action


@pytest.mark.parametrize('actor', range(4))
@pytest.mark.parametrize('kind', sorted(NONFLAG_HQ_PENALIZED_KINDS, key=lambda k: k.value))
def test_all_requested_ranks_all_seats_failed_attacks_and_known_empty_hq(actor, kind):
    for target in (K.MINE, K.PLATOON_COMMANDER, None):
        game, action = position(kind, actor=actor, target=target)
        assert known_nonflag_hq_entry(game, action)
        row = _advance(0, game, action, MODE, track_known_nonflag_hq=True)
        assert row.known_nonflag_hq
        assert row.flag_captured_owner is None
        assert not _advance(0, game, (0, 0), MODE, track_known_nonflag_hq=True).known_nonflag_hq


@pytest.mark.parametrize('kind', [K.PLATOON_COMMANDER, K.COMPANY_COMMANDER, K.BATTALION_COMMANDER])
def test_smaller_ranks_not_penalized(kind):
    game, action = position(kind)
    assert not known_nonflag_hq_entry(game, action)


def test_reported_game_army_attack_is_penalized_using_pre_action_public_flag():
    fixture = json.loads((ROOT / 'tests/fixtures/known_nonflag_hq_entry.json').read_text())
    board0 = JunqiGame.new_four_player(seed=0).board_for(0)
    pieces = {board0.decode(code): Piece(owner, kind) for code, owner, kind in fixture['pieces']}
    game = JunqiGame.from_position(GameConfig(information_mode=InformationMode.FOUR_DARK), pieces)
    game.enable_inventory_tracking()
    for ply, pair in enumerate(fixture['actions'], 1):
        actor = game.current_player
        action = tuple(pair) if pair == [0, 0] else tuple(game.board_for(actor).encode(board0.decode(c)) for c in pair)
        if ply == fixture['penalized_ply']:
            assert actor == fixture['actor'] and game.revealed_flags[2]
            obs = game.observe(actor, include_legal_masks=False)
            true_flag = game.board_for(actor).encode(board0.decode(fixture['true_flag']))
            false_hq = game.board_for(actor).encode(board0.decode(fixture['false_hq']))
            assert obs.points[true_flag].kind is K.FLAG
            assert obs.points[false_hq].kind is None  # Mine became known only after this attack.
            assert known_nonflag_hq_entry(game, action)
            row = _advance(0, game, action, MODE, track_known_nonflag_hq=True)
            assert row.known_nonflag_hq and row.flag_captured_owner is None
            assert player_step_rewards(row, player=actor, player_count=4, draw_reward=-.15,
                flag_capture_reward=.2, bomb_rewards={}, futile_reward=0.,
                known_nonflag_hq_reward=-.2) == (0., -.2, 0., 0.)
        else:
            game.step(action)


def test_unknown_flag_true_flag_inferred_flag_own_empty_hq_and_post_action_reveal():
    for target in (K.MINE, K.BOMB, K.PLATOON_COMMANDER, None):
        game, action = position(known=False, target=target)
        assert not game.revealed_flags[1]
        assert not known_nonflag_hq_entry(game, action)
        assert not _advance(0, game, action, MODE, track_known_nonflag_hq=True).known_nonflag_hq
    game, action = position(target_flag=True)
    assert not known_nonflag_hq_entry(game, action)
    assert not _advance(0, game, action, MODE, track_known_nonflag_hq=True).known_nonflag_hq
    game, action = position(known=False, inferred=True)
    assert not game.revealed_flags[1]
    assert known_nonflag_hq_entry(game, action)
    game, action = position(own_arm=True, known=False, target=None)
    assert known_nonflag_hq_entry(game, action)
    assert not known_nonflag_hq_entry(game, (0, 0))
    # Capturing a commander in the false HQ reveals the other flag only afterwards.
    game, action = position(K.BOMB, known=False, target=K.COMMANDER)
    assert not game.revealed_flags[1]
    assert not _advance(0, game, action, MODE, track_known_nonflag_hq=True).known_nonflag_hq
    assert game.revealed_flags[1]


def tree_equal(left, right):
    if isinstance(right, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(right, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in right:
            tree_equal(left[key], right[key])
    elif isinstance(right, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('workers,groups', [(1, 1), (2, 1), (2, 2)])
def test_real_collectors_personal_gae_critic_targets_terminal_stacking_and_metrics(workers, groups):
    torch.set_num_threads(1)
    model = replace(ModelConfig.tiny(), temporal_layers=2, token_fusion_hidden_dim=64,
                    ppo_fixed_kv=False, ppo_cuda_graphs=False)
    actor = FrozenPolicyActor(GamePolicyTransformer(model).eval(), max_batch_size=4)
    critic = FrozenValueActor(GameCentralValueTransformer(model).eval(), amp_dtype=None,
                              max_batch_size=4, deferred=True)
    layout = PieceConditionedLayoutPointerDecoder(model).eval()
    pool = BaseGamePool(MODE, pool_size=4, max_transitions=model.max_transitions,
                       max_game_plies=None, seed=42, centralized_critic=True)
    pool.fill(layout, 0)
    positions = [position(actor=p, terminal=True) for p in range(4)]
    for slot, (game, _) in zip(pool.slots, positions, strict=True):
        slot.game = game
        slot.history = GameHistory.initialize(game, MODE, max_transitions=model.max_transitions)
        slot.critic_history = CentralizedHistory.initialize(game, MODE, max_transitions=model.max_transitions)
    actions = [a for _, a in positions]
    def scripted(states, *, count, return_log_probs):
        return ([[next(a for a in actions if a in s.legal_actions)] for s in states], [[-1.]] * len(states))
    with patch.object(actor, 'sample', side_effect=scripted):
        samples, outcomes, metrics = collect_mappo_samples(pool, actor, critic, layout,
            count=4, behavior_version=0, discount=1., gae_lambda=.95,
            environment_workers=workers, pipeline_groups=groups, advantage_mode='player_decisions',
            reward_scope='player', draw_reward=-.15, known_nonflag_hq_penalty=.2)
    assert len(samples) == 4 and metrics.known_nonflag_hq_entries == 4
    assert metrics.known_nonflag_hq_reward_sum == pytest.approx(-.8)
    for sample in samples:
        assert sample.value_target == pytest.approx(-.35)
        assert sample.advantage == pytest.approx(-.35 - sample.old_value)
        assert all(v.target == pytest.approx(-.15) for v in sample.additional_values)
    assert all(outcome.reward == -.15 for outcome in outcomes)  # Layout objective is unchanged.


def test_reward_ownership_disabled_frozen_and_reward_scope_does_not_charge_ally():
    row = SimpleNamespace(terminal=False, flag_captured_owner=None, bombed_kind=None)
    for scope in ('player', 'team'):
        assert player_step_rewards(row, player=1, player_count=4, draw_reward=-.15,
            flag_capture_reward=.2, bomb_rewards={}, futile_reward=0., reward_scope=scope,
            known_nonflag_hq_reward=-.2) == (0., -.2, 0., 0.)
    metrics = RolloutMetrics()
    assert metrics.record_known_nonflag_hq(True, .2, learnable=False) == 0.
    assert metrics.record_known_nonflag_hq(True, 0.) == 0.
    assert metrics.record_known_nonflag_hq(False, .2) == 0.
    assert metrics.known_nonflag_hq_entries == 0
    game, action = position()
    assert not _advance(0, game, action, MODE).known_nonflag_hq


def test_config_and_cli_require_explicit_adoption_and_fixed_amount():
    settings = TrainingSettings.from_yaml(CONFIG, MODE)
    assert settings.known_nonflag_hq_penalty == .2
    assert TrainingSettings.__dataclass_fields__['known_nonflag_hq_penalty'].default == 0.
    for invalid in (-.2, True, float('nan'), float('inf'), '.2'):
        with pytest.raises(ValueError, match='known_nonflag_hq_penalty'):
            replace(settings, known_nonflag_hq_penalty=invalid).validate()
    with pytest.raises(ValueError, match='known_nonflag_hq_penalty requires MAPPO'):
        replace(settings, algorithm='ppo', mappo_advantage_mode='team_steps', mappo_reward_scope='team').validate()
    parsed = build_parser().parse_args(['--mode', 'four_dark', '--config', str(CONFIG),
                                       '--adopt-known-nonflag-hq-penalty'])
    assert parsed.adopt_known_nonflag_hq_penalty
    for progress in (0, 1_500_000_000, 3_000_000_000):
        payload = {'update': 7, 'config': {}, 'trainer_state': {'cumulative': {'environment_plies': progress}}}
        with pytest.raises(RuntimeError, match='adopt-known-nonflag-hq-penalty'):
            reconcile_known_nonflag_hq_objective(payload, settings)
        reconcile_known_nonflag_hq_objective(payload, settings, adopt=True)
        assert payload['config']['known_nonflag_hq_penalty'] == .2
        assert payload['trainer_state']['known_nonflag_hq_objective_migration']['schedule'] == 'constant'
        before = copy.deepcopy(payload)
        reconcile_known_nonflag_hq_objective(payload, settings)
        tree_equal(payload, before)


def test_full_checkpoint_adoption_preserves_state_and_normal_resume(tmp_path):
    torch.set_num_threads(1)
    old_settings = TrainingSettings.from_yaml(CONFIG, MODE, tiny=True, overrides={
        'device': 'cpu', 'target_environment_plies': 32, 'anchor_batch': 8, 'policy_microbatch': 2,
        'adaptive_learning_rate': False, 'inference_snapshot_every_updates': 0})
    with patch.object(MetricLogger, 'start_resource_monitor'):
        old = SelfPlayTrainer(old_settings, run_directory=tmp_path)
        old.train()
        path = old.checkpoints.latest_path
        before = torch.load(path, weights_only=False)
        before['config'].pop('known_nonflag_hq_penalty')
        torch.save(before, path)
        target = replace(old_settings, known_nonflag_hq_penalty=.2, total_updates=2)
        with pytest.raises(RuntimeError, match='adopt-known-nonflag-hq-penalty'):
            SelfPlayTrainer(target, run_directory=tmp_path)
        trainer = SelfPlayTrainer(target, run_directory=tmp_path, adopt_known_nonflag_hq_penalty=True)
        try:
            migrated = torch.load(path, weights_only=False)
            for key in ('policy', 'critic', 'layout', 'reference_layout', 'policy_optimizer',
                        'critic_optimizer', 'layout_optimizer', 'rng_state'):
                tree_equal(migrated[key], before[key])
            for key in ('base_game_pool', 'layout_buffer', 'cumulative', 'futile_attack_penalty',
                        'adaptive_learning_rate', 'adaptive_entropy', 'historical_opponents', 'model_selection'):
                if key in before['trainer_state']:
                    tree_equal(migrated['trainer_state'][key], before['trainer_state'][key])
            assert migrated['reason'] == 'known_nonflag_hq_objective_migration'
            assert trainer.known_nonflag_hq_objective_migration['to_penalty'] == .2
            aggregated = trainer._aggregate_rollout_metrics(RolloutMetrics(
                known_nonflag_hq_entries=3, known_nonflag_hq_reward_sum=-.6))
            assert aggregated.as_dict()['rollout/known_nonflag_hq_reward_sum'] == pytest.approx(-.6)
            with patch('junqi.training.trainer.collect_mappo_samples', wraps=collect_mappo_samples) as collector:
                trainer.train()
            assert collector.call_args.kwargs['known_nonflag_hq_penalty'] == .2
            metrics = [json.loads(s) for s in (trainer.run_directory / 'metrics.jsonl').read_text().splitlines()]
            assert metrics[-1]['training/known_nonflag_hq_penalty'] == .2
        finally:
            trainer.historical.close()
            trainer.logger.close()
        resumed = SelfPlayTrainer(target, run_directory=tmp_path)
        try:
            assert resumed.update == 2 and resumed.known_nonflag_hq_objective_migration['to_penalty'] == .2
        finally:
            resumed.historical.close()
            resumed.logger.close()
