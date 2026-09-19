"""Real bomb/futile-attack events, worker reward ownership and resume contracts."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from junqi import ArmPoint, GameConfig, GameVariant, InformationMode, JunqiGame, Piece, PieceType
from junqi.training.encoding import GameHistory
from junqi.training.futile_attack import EVIDENCE_ATTRIBUTE, FutileAttackPenalty, step_with_evidence
from junqi.training.mappo import collect_mappo_samples
from junqi.training.mappo_encoding import CentralizedHistory
from junqi.training.metrics import MetricLogger
from junqi.training.models import GameCentralValueTransformer, GamePolicyTransformer, ModelConfig, PieceConditionedLayoutPointerDecoder
from junqi.training.modes import TrainingMode
from junqi.training.objective_migration import reconcile_bomb_objective
from junqi.training.ppo import FrozenValueActor
from junqi.training.ppo_environment import _advance
from junqi.training.rollout import BaseGamePool, FrozenPolicyActor, RolloutMetrics
from junqi.training.settings import TrainingSettings
from junqi.training.tactical_rewards import BOMB_REWARD_FIELDS, halfway_reward_scale
from junqi.training.trainer import SelfPlayTrainer


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/expand_mappo_training.yaml"
MODE = TrainingMode.FOUR_DARK
REWARDS = dict(zip(BOMB_REWARD_FIELDS.values(), (.03, .01, .003), strict=True))


def move(game, start, end):
    board = game.board_for(game.current_player)
    result = (board.encode(start), board.encode(end))
    assert result in game.legal_actions()
    return result


def bomb_position(kind=PieceType.COMMANDER, *, actor=0, defensive=False, draw=False):
    other = (actor + 1) % 4
    pieces = {ArmPoint(seat, 6, 4): Piece(seat, PieceType.FLAG) for seat in range(4)}
    pieces.update({ArmPoint(seat, 2, 3): Piece(seat, PieceType.ENGINEER) for seat in range(4)})
    start, end = ArmPoint(other, 4, 1), ArmPoint(other, 5, 1)
    pieces[start] = Piece(actor, kind if defensive else PieceType.BOMB)
    pieces[end] = Piece(other, PieceType.BOMB if defensive else kind)
    game = JunqiGame.from_position(GameConfig(variant=GameVariant.FOUR_PLAYER,
        information_mode=InformationMode.FOUR_DARK, max_plies=1 if draw else None), pieces, current_player=actor)
    game.enable_inventory_tracking()
    return game, move(game, start, end)


def futile_position(*, second=PieceType.PLATOON_COMMANDER, first=PieceType.ARMY_COMMANDER,
                    target=PieceType.COMMANDER, probe=True):
    game, _ = bomb_position(target)
    pieces = dict(game.pieces)
    start, end, follow = ArmPoint(1, 4, 1), ArmPoint(1, 5, 1), ArmPoint(1, 4, 2)
    pieces[start] = Piece(0, first)
    pieces[follow] = Piece(0, second)
    game = JunqiGame.from_position(game.config, pieces, current_player=0)
    game.enable_inventory_tracking()
    if probe:
        assert not step_with_evidence(game, move(game, start, end), enabled=True)[1]
        while game.current_player != 0:
            step_with_evidence(game, (0, 0), enabled=True)
    return game, move(game, follow, end)


def assert_tree_equal(test, left, right):
    if isinstance(right, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(right, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(right, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in right:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(right, (list, tuple)):
        test.assertEqual(type(left), type(right))
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right, strict=True):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class MAPPOTacticalRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def models_and_pool(self, positions):
        config = replace(ModelConfig.tiny(), temporal_layers=2, token_fusion_hidden_dim=64,
                         ppo_fixed_kv=False, ppo_cuda_graphs=False)
        actor = FrozenPolicyActor(GamePolicyTransformer(config).eval(), max_batch_size=4)
        critic = FrozenValueActor(GameCentralValueTransformer(config).eval(), amp_dtype=None,
                                  max_batch_size=4, deferred=True)
        layout = PieceConditionedLayoutPointerDecoder(config).eval()
        pool = BaseGamePool(MODE, pool_size=4, max_transitions=config.max_transitions,
                            max_game_plies=None, seed=44, centralized_critic=True)
        pool.fill(layout, 0)
        for slot, (game, _) in zip(pool.slots, positions, strict=True):
            slot.game = game
            slot.history = GameHistory.initialize(game, MODE, max_transitions=config.max_transitions)
            slot.critic_history = CentralizedHistory.initialize(game, MODE, max_transitions=config.max_transitions)
        return actor, critic, layout, pool

    def scripted(self, actions):
        def sample(states, *, count, return_log_probs):
            return ([[next((a for a in actions if a in s.legal_actions), (0, 0))] for s in states],
                    [[-1.]] * len(states))
        return sample

    def test_bombs_actual_casualties_both_directions_all_seats_and_no_repeat(self):
        for seat in range(4):
            for defensive in (False, True):
                for kind in (*BOMB_REWARD_FIELDS, PieceType.BRIGADE_COMMANDER, PieceType.BOMB, PieceType.MINE):
                    with self.subTest(seat=seat, defensive=defensive, kind=kind):
                        if defensive and kind is PieceType.MINE:
                            continue  # Mines cannot initiate an attack.
                        game, action = bomb_position(kind, actor=seat, defensive=defensive)
                        row = _advance(0, game, action, MODE)
                        if kind in BOMB_REWARD_FIELDS:
                            self.assertEqual(row.bombed_kind, kind.value)
                            self.assertEqual(row.bomb_owner, (seat + 1) % 4 if defensive else seat)
                        else:
                            self.assertIsNone(row.bombed_kind)
                            self.assertIsNone(row.bomb_owner)
                        self.assertFalse(row.terminal)
                        self.assertEqual(row.reward, 0.)
                        self.assertIsNone(_advance(0, game, (0, 0), MODE).bombed_kind)
        # Ordinary equal-rank trades are not bomb events.
        game, action = bomb_position()
        pieces = dict(game.pieces)
        pieces[ArmPoint(1, 4, 1)] = Piece(0, PieceType.COMMANDER)
        game = JunqiGame.from_position(game.config, pieces, current_player=0)
        game.enable_inventory_tracking()
        self.assertIsNone(_advance(0, game, action, MODE).bombed_kind)

    def test_mappo_collectors_terminal_stacking_metrics_and_no_checkpoint_reissue(self):
        for workers, groups in ((1, 1), (2, 1), (2, 2)):
            for draw in (False, True):
                with self.subTest(workers=workers, groups=groups, draw=draw):
                    positions = [bomb_position(k, draw=draw) for k in (*BOMB_REWARD_FIELDS, PieceType.BRIGADE_COMMANDER)]
                    actor, critic, layout, pool = self.models_and_pool(positions)
                    options = dict(count=4, behavior_version=0, discount=1., gae_lambda=1.,
                                   environment_workers=workers, pipeline_groups=groups, **REWARDS)
                    with patch.object(actor, 'sample', side_effect=self.scripted([p[1] for p in positions])):
                        samples, outcomes, metrics = collect_mappo_samples(pool, actor, critic, layout, **options)
                    for sample, bonus in zip(samples, (.03, .01, .003, 0.), strict=True):
                        self.assertAlmostEqual(sample.value_target, bonus - (.15 if draw else 0.), places=6)
                        self.assertFalse(sample.state.centralized)
                    self.assertEqual((metrics.bombed_commanders, metrics.bombed_army_commanders,
                                      metrics.bombed_division_commanders), (1, 1, 1))
                    self.assertAlmostEqual(metrics.bomb_reward_sum, .043)
                    self.assertEqual(metrics.environment_plies, 4)
                    if draw:
                        self.assertEqual(len(outcomes), 16)
                        self.assertTrue(all(item.reward == -.15 for item in outcomes))
                    else:
                        pool.load_state_dict(pool.state_dict())
                        with patch.object(actor, 'sample', side_effect=self.scripted([(0, 0)])):
                            _, _, again = collect_mappo_samples(pool, actor, critic, layout, **options)
                        self.assertEqual(again.bomb_reward_sum, 0.)
                        self.assertEqual(again.bombed_commanders, 0)

    def test_bomb_casualty_reward_metadata_stays_out_of_actor_observations(self):
        observations, rows = [], []
        for kind in (PieceType.ARMY_COMMANDER, PieceType.DIVISION_COMMANDER):
            game, action = bomb_position(kind)
            before = game.observe(0)
            row = _advance(0, game, action, MODE)
            observations.append((before, game.observe(0)))
            rows.append(row)
        self.assertEqual(observations[0], observations[1])
        np.testing.assert_array_equal(rows[0].records[0], rows[1].records[0])
        self.assertEqual([row.bombed_kind for row in rows], ['army_commander', 'division_commander'])

    def test_frozen_teammate_and_defensive_bomb_credit_bomb_team_only(self):
        positions = [bomb_position(actor=0), bomb_position(actor=2),
                     bomb_position(actor=1), bomb_position(actor=1, defensive=True)]
        actor, critic, layout, pool = self.models_and_pool(positions)
        for slot in pool.slots:
            slot.opponent_id = slot.teammate_id = 'old'
            slot.learner_team = slot.learner_seat = 0
        pool.historical = SimpleNamespace(actor=actor, opponent_plies=0, teammate_plies=0,
            begin_rollout=lambda *_: None, attach_cache=lambda *_: None, clear_caches=lambda: None)
        with patch.object(actor, 'sample', side_effect=self.scripted([p[1] for p in positions])):
            samples, _, metrics = collect_mappo_samples(pool, actor, critic, layout,
                count=4, behavior_version=0, discount=1., gae_lambda=1., environment_workers=2,
                pipeline_groups=2, **REWARDS)
        for sample, expected in zip(samples, (.03, .03, 0., .03), strict=True):
            self.assertAlmostEqual(sample.value_target, expected, places=6)
        self.assertEqual([s.learnable for s in samples], [True, False, False, False])
        self.assertEqual(metrics.bombed_commanders, 4)

    def test_first_probe_engineer_and_blocking_moves_exempt_but_repeated_loss_penalized(self):
        # The identical hidden defender only triggers after public battle evidence.
        positions = [futile_position(), futile_position(probe=False),
                     futile_position(second=PieceType.ENGINEER, target=PieceType.MINE), futile_position()]
        game = positions[3][0]
        positions[3] = (game, move(game, ArmPoint(1, 4, 2), ArmPoint(1, 4, 1)))
        expected = (-.01, 0., 0., 0.)
        for (game, action), penalty in zip(positions, expected, strict=True):
            row = _advance(0, game, action, MODE, track_futile_attacks=True)
            self.assertEqual(row.futile_attack, penalty != 0)
        # Verify the negative reward actually enters MAPPO, including checkpointed evidence.
        positions = [futile_position() for _ in range(4)]
        actor, critic, layout, pool = self.models_and_pool(positions)
        pool.load_state_dict(pool.state_dict())
        for slot in pool.slots:
            self.assertIn(ArmPoint(1, 5, 1), getattr(slot.game, EVIDENCE_ATTRIBUTE)[0])
        with patch.object(actor, 'sample', side_effect=self.scripted([positions[0][1]])):
            samples, _, metrics = collect_mappo_samples(pool, actor, critic, layout,
                count=4, behavior_version=0, discount=1., gae_lambda=1., environment_workers=2,
                pipeline_groups=2, futile_attack_penalty=.01, **REWARDS)
        self.assertEqual(metrics.futile_attacks, 4)
        for sample in samples:
            self.assertAlmostEqual(sample.value_target, -.01, places=6)

    def test_decay_halfway_batch_split_ddp_and_config_scope(self):
        settings = TrainingSettings.from_yaml(CONFIG, MODE, tiny=True, overrides={
            'target_environment_plies': 200, 'futile_attack_penalty': .01, **REWARDS})
        trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
        trainer.settings, trainer.local_anchor_batch = settings, 64
        trainer.distributed = SimpleNamespace(world_size=1)
        trainer.futile_attack_controller = FutileAttackPenalty(.01, 200)
        for step, scale in ((0, 1.), (50, .5), (99, .01), (100, 0.), (150, 0.)):
            trainer.cumulative = {'environment_plies': step}
            self.assertAlmostEqual(halfway_reward_scale(step, 200), scale)
            self.assertAlmostEqual(trainer._futile_attack_coefficient(), .01 * scale)
            for key, initial in REWARDS.items():
                self.assertAlmostEqual(trainer._bomb_reward_coefficients()[key], initial * scale)
        trainer.cumulative = {'environment_plies': 99}
        self.assertEqual(trainer._ppo_collection_count(), 1)
        trainer.settings = replace(settings, futile_attack_penalty=0.)
        self.assertEqual(trainer._ppo_collection_count(), 1)  # Bombs alone also split.
        trainer.distributed.world_size = 2
        self.assertEqual(set(trainer._bomb_reward_coefficients().values()), {0.})
        self.assertEqual(trainer._futile_attack_coefficient(), 0.)
        for mode in (MODE, TrainingMode.DOUBLE_OPEN):
            configured = TrainingSettings.from_yaml(CONFIG, mode)
            self.assertEqual({key: getattr(configured, key) for key in REWARDS}, REWARDS)
        import yaml
        for config in ('local_4090_training.yaml', 'bootstrap.yaml'):
            legacy = yaml.safe_load((ROOT / 'configs' / config).read_text(encoding='utf-8'))
            self.assertTrue(all(legacy.get('ppo', {}).get(key, 0.) == 0. for key in REWARDS))
        for key in REWARDS:
            self.assertEqual(TrainingSettings.__dataclass_fields__[key].default, 0.)
        with self.assertRaisesRegex(ValueError, 'requires MAPPO'):
            replace(settings, algorithm='ppo').validate()
        for key in REWARDS:
            for invalid in (-.1, True, float('nan'), float('inf'), '.03'):
                with self.subTest(key=key, invalid=invalid), self.assertRaisesRegex(ValueError, key):
                    replace(settings, **{key: invalid}).validate()
        with self.assertRaisesRegex(ValueError, 'bomb_commander_reward'):
            replace(settings, target_environment_plies=None, futile_attack_penalty=0.).validate()

    def test_full_state_migration_live_coefficients_and_normal_resume(self):
        settings = TrainingSettings.from_yaml(CONFIG, MODE, tiny=True, overrides={
            'device': 'cpu', 'target_environment_plies': 32, 'anchor_batch': 8,
            'policy_microbatch': 2, 'adaptive_learning_rate': False, 'inference_snapshot_every_updates': 0})
        with tempfile.TemporaryDirectory() as directory, patch.object(MetricLogger, 'start_resource_monitor'):
            old = SelfPlayTrainer(settings, run_directory=directory)
            old.train()
            path = old.checkpoints.latest_path
            before = torch.load(path, weights_only=False)
            for key in REWARDS:
                before['config'].pop(key)  # Old checkpoints imply zero, even if raw YAML differs.
            torch.save(before, path)
            target = replace(settings, total_updates=3, **REWARDS)
            with self.assertRaisesRegex(RuntimeError, 'adopt-bomb-rewards'):
                SelfPlayTrainer(target, run_directory=directory)
            trainer = SelfPlayTrainer(target, run_directory=directory, adopt_bomb_rewards=True)
            try:
                migrated = torch.load(path, weights_only=False)
                for key in ('policy', 'critic', 'layout', 'reference_layout', 'policy_optimizer',
                            'critic_optimizer', 'layout_optimizer', 'rng_state'):
                    assert_tree_equal(self, migrated[key], before[key])
                for key in ('base_game_pool', 'layout_buffer', 'cumulative', 'futile_attack_penalty',
                            'adaptive_learning_rate', 'adaptive_entropy', 'model_selection', 'historical_opponents'):
                    if key in before['trainer_state']:
                        assert_tree_equal(self, migrated['trainer_state'][key], before['trainer_state'][key])
                self.assertEqual(trainer.update, 1)
                self.assertEqual(migrated['reason'], 'bomb_objective_migration')
                migration = trainer.bomb_objective_migration
                self.assertEqual(set(migration['from_rewards'].values()), {0.})
                self.assertEqual(migration['to_rewards'], REWARDS)
                self.assertAlmostEqual(trainer._bomb_reward_coefficients()['bomb_commander_reward'], .015)
                aggregated = trainer._aggregate_rollout_metrics(RolloutMetrics(bombed_commanders=2, bomb_reward_sum=.06))
                self.assertEqual(aggregated.as_dict()['rollout/bombed_commanders'], 2)
                self.assertAlmostEqual(aggregated.bomb_reward_sum, .06)
                trainer.train()
                metrics = [json.loads(line) for line in (trainer.run_directory / 'metrics.jsonl').read_text().splitlines()]
                steps = {row['update']: row for row in metrics if 'training/bomb_commander_reward' in row}
                self.assertAlmostEqual(steps[2]['training/bomb_commander_reward'], .015)
                self.assertEqual(steps[3]['training/bomb_commander_reward'], 0.)
            finally:
                trainer.historical.close()
                trainer.logger.close()
            resumed = SelfPlayTrainer(target, run_directory=directory)
            try:
                self.assertEqual(resumed.bomb_objective_migration, migration)
                self.assertEqual(resumed.update, 3)
                self.assertEqual(set(resumed._bomb_reward_coefficients().values()), {0.})
            finally:
                resumed.historical.close()
                resumed.logger.close()
            # A later schedule change also requires explicit adoption.
            payload = torch.load(path, weights_only=False)
            with self.assertRaisesRegex(RuntimeError, 'adopt-bomb-rewards'):
                reconcile_bomb_objective(copy.deepcopy(payload), replace(target, target_environment_plies=64))


if __name__ == '__main__':
    unittest.main()
