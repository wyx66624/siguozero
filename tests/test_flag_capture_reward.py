"""Actual flag events, team GAE, every PPO collector, and full-state migration."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from junqi import ArmPoint, GameConfig, GameVariant, InformationMode, JunqiGame, Piece, PieceType
from junqi.training.encoding import GameHistory
from junqi.training.metrics import MetricLogger
from junqi.training.models import GamePolicyTransformer, GameValueTransformer, ModelConfig, PieceConditionedLayoutPointerDecoder
from junqi.training.modes import TrainingMode
from junqi.training.ppo import FrozenValueActor, PPOTransition, collect_ppo_samples, generalized_advantages
from junqi.training.ppo_environment import _advance
from junqi.training.rewards import flag_capture_utility
from junqi.training.rollout import BaseGamePool, FrozenPolicyActor, RolloutMetrics
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer


ROOT = Path(__file__).parents[1]
MODE = TrainingMode.FOUR_DARK


def position(*, actor=0, final=False, draw=False, ordinary=False, bomb=False):
    target = (actor + 1) % 4
    active = [not (final and seat == (actor + 3) % 4) for seat in range(4)]
    pieces = {}
    for seat in range(4):
        if active[seat]:
            column = 4 if ordinary and seat == target else 2
            pieces[ArmPoint(seat, 6, column)] = Piece(seat, PieceType.FLAG)
            pieces[ArmPoint(seat, 2, 3)] = Piece(seat, PieceType.COMMANDER)
    pieces[ArmPoint(target, 5, 2)] = Piece(actor, PieceType.BOMB if bomb else PieceType.ENGINEER)
    game = JunqiGame.from_position(GameConfig(variant=GameVariant.FOUR_PLAYER,
        information_mode=InformationMode.FOUR_DARK, max_plies=1 if draw else None),
        pieces, current_player=actor, active_players=active, revealed_flags=[True] * 4)
    board = game.board_for(actor)
    action = (board.encode(ArmPoint(target, 5, 2)), board.encode(ArmPoint(target, 6, 2)))
    assert action in game.legal_actions()
    return game, action


def scripted_sample(states, *, count, return_log_probs):
    # Rotated player boards put the next physical arm at the same coordinates.
    action = (51, 56)
    return ([[action if action in state.legal_actions else state.legal_actions[0]] for state in states],
            [[-1.] for _ in states])


class FlagCaptureRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_events_all_seats_team_sharing_and_no_repeat(self):
        for actor in range(4):
            for bomb in (False, True):
                with self.subTest(actor=actor, bomb=bomb):
                    game, action = position(actor=actor, bomb=bomb)
                    row = _advance(0, game, action, MODE)
                    self.assertEqual(row.flag_captured_owner, (actor + 1) % 4)
                    self.assertFalse(row.terminal)
                    self.assertEqual(row.reward, 0.)  # Rule outcome stays unshaped.
                    for viewer in range(4):
                        expected = .1 if viewer % 2 == actor % 2 else -.1
                        self.assertEqual(flag_capture_utility(game, row.flag_captured_owner,
                            player=viewer, coefficient=.1), expected)
                    later = _advance(0, game, (0, 0), MODE)
                    self.assertIsNone(later.flag_captured_owner)
                    self.assertEqual(flag_capture_utility(game, None, player=actor, coefficient=.1), 0.)
        game, action = position(ordinary=True)
        self.assertIsNone(_advance(0, game, action, MODE).flag_captured_owner)

    def test_signed_capture_and_shared_draw_propagate_separately(self):
        game, _ = position()
        state = GameHistory.initialize(game, MODE).state_for(game)
        for terminal, draw, reward, expected in (
            (False, False, .1, [-.1, .1, -.1, .1]),
            (True, False, 1.1, [-1.1, 1.1, -1.1, 1.1]),
            (True, True, -.05, [-.25, -.05, -.25, -.05]),
        ):
            trace = [PPOTransition(state, state.legal_actions[0], -1., 0., 0., False, -1)
                     for _ in range(3)]
            trace.append(PPOTransition(state, state.legal_actions[0], -1., 0., reward, terminal, -1,
                terminal_draw=draw, flag_capture_reward=.1))
            samples = generalized_advantages(trace, bootstrap_value=0., discount=1.,
                                            gae_lambda=1., behavior_version=0)
            for sample, target in zip(samples, expected, strict=True):
                self.assertAlmostEqual(sample.value_target, target)
                self.assertAlmostEqual(sample.draw_value_target, -.15 if draw else 0.)
        # An eliminated seat may be skipped: consecutive allied value owners
        # retain the same negative reward, rather than flipping every turn.
        trace = [PPOTransition(state, (51, 56), -1., 0., 0., False, 1),
                 PPOTransition(state, (51, 56), -1., 0., -.1, False, -1, flag_capture_reward=-.1)]
        samples = generalized_advantages(trace, bootstrap_value=0., discount=1., gae_lambda=1., behavior_version=0)
        self.assertEqual([s.value_target for s in samples], [-.1, -.1])

    def models_and_pool(self, *, arrays=True):
        config = replace(ModelConfig.tiny(), ppo_array_history=arrays, ppo_fixed_kv=False)
        actor = FrozenPolicyActor(GamePolicyTransformer(config).eval(), max_batch_size=4)
        critic = GameValueTransformer(config).eval()  # Both value heads start at zero.
        layout = PieceConditionedLayoutPointerDecoder(config).eval()
        pool = BaseGamePool(MODE, pool_size=4, max_transitions=config.max_transitions, max_game_plies=None, seed=37)
        pool.fill(layout, 0)
        return actor, critic, layout, pool

    def test_all_collectors_capture_terminal_stacking_and_checkpoint_no_reissue(self):
        paths = ((1, False, 1, False), (1, True, 1, True), (2, False, 1, False),
                 (2, True, 1, True), (2, True, 2, True))
        for workers, deferred, groups, arrays in paths:
            for coefficient in (0., .1, .2):
                with self.subTest(workers=workers, deferred=deferred, groups=groups, coefficient=coefficient):
                    actor, model, layout, pool = self.models_and_pool(arrays=arrays)
                    for slot, options in zip(pool.slots, ({}, dict(final=True), dict(draw=True), dict(ordinary=True)), strict=True):
                        slot.game, action = position(**options)
                        self.assertEqual(action, (51, 56))
                        slot.history = GameHistory.initialize(slot.game, MODE, max_transitions=model.config.max_transitions)
                    critic = FrozenValueActor(model, amp_dtype=None, max_batch_size=4, deferred=deferred)
                    options = dict(count=4, behavior_version=0, discount=1., gae_lambda=1.,
                                   environment_workers=workers, pipeline_groups=groups, flag_capture_reward=coefficient)
                    with patch.object(actor, 'sample', side_effect=scripted_sample):
                        samples, outcomes, metrics = collect_ppo_samples(pool, actor, critic, layout, **options)
                    for sample, target in zip(samples, (coefficient, 1. + coefficient, -.15 + coefficient, 0.), strict=True):
                        self.assertAlmostEqual(sample.value_target, target)
                    self.assertAlmostEqual(samples[2].draw_value_target, -.15)
                    self.assertEqual((metrics.wins, metrics.draws, metrics.losses), (1, 1, 0))
                    self.assertEqual((metrics.flag_captures, metrics.nonterminal_flag_captures), (3, 1))
                    self.assertAlmostEqual(metrics.flag_capture_reward_abs_sum, 3 * coefficient)
                    self.assertEqual(len(outcomes), 8)
                    self.assertEqual({o.reward for o in outcomes}, {-1., 1., -.15})
                    restored = BaseGamePool(MODE, pool_size=4, max_transitions=model.config.max_transitions,
                                            max_game_plies=None, seed=37)
                    restored.load_state_dict(pool.state_dict())
                    with patch.object(actor, 'sample', side_effect=scripted_sample):
                        _, _, next_metrics = collect_ppo_samples(restored, actor, critic, layout, **options)
                    self.assertEqual(next_metrics.flag_captures, 0)
                    self.assertEqual(next_metrics.flag_capture_reward_abs_sum, 0.)

    def test_frozen_opponent_and_teammate_use_learner_value_perspective(self):
        actor, model, layout, pool = self.models_and_pool()
        # All four events are nonterminal. Frozen opponents taking either our
        # flag or our partner's flag are negative from anchor 0's value view.
        roles = ((3, 'old', None, -.1), (1, 'old', None, -.1),
                 (2, None, 'old', .1), (0, 'old', 'old', .1))
        for slot, (seat, opponent, teammate, _) in zip(pool.slots, roles, strict=True):
            slot.game, _ = position(actor=seat)
            slot.history = GameHistory.initialize(slot.game, MODE, max_transitions=model.config.max_transitions)
            slot.opponent_id, slot.teammate_id = opponent, teammate
            slot.learner_team, slot.learner_seat = 0, 0
        pool.historical = SimpleNamespace(active=True, actor=actor, opponent_plies=0, teammate_plies=0,
            begin_rollout=lambda *_: None, attach_cache=lambda *_: None, clear_caches=lambda: None)
        with patch.object(actor, 'sample', side_effect=scripted_sample):
            samples, _, metrics = collect_ppo_samples(pool, actor,
                FrozenValueActor(model, amp_dtype=None, max_batch_size=4, deferred=True), layout,
                count=4, behavior_version=0, discount=1., gae_lambda=1., environment_workers=2,
                pipeline_groups=2, flag_capture_reward=.1)
        for sample, (_, _, _, expected) in zip(samples, roles, strict=True):
            self.assertAlmostEqual(sample.value_target, expected)
            self.assertEqual(sample.value_state.records.identity[1], 0)
        self.assertEqual([s.learnable for s in samples], [False, False, False, True])
        self.assertEqual(metrics.flag_captures, 4)

    def test_config_defaults_validation_and_grpo_isolation(self):
        self.assertEqual(TrainingSettings.from_yaml(ROOT / 'configs/bootstrap.yaml', MODE, tiny=True).flag_capture_reward, 0.)
        for mode, expected in ((MODE, .2), (TrainingMode.TWO_PLAYER, 0.)):
            current = TrainingSettings.from_yaml(ROOT / 'configs/local_4090_training.yaml', mode, tiny=True)
            self.assertEqual(current.flag_capture_reward, expected)
        for value in (-.1, True, float('nan'), float('inf'), '.1'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'flag_capture_reward'):
                TrainingSettings.from_yaml(ROOT / 'configs/bootstrap.yaml', MODE, tiny=True,
                                           overrides={'flag_capture_reward': value})

    def assert_tree_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected, strict=True):
                self.assert_tree_equal(a, b)
        elif hasattr(expected, 'shape'):
            self.assertTrue((actual == expected).all())
        else:
            self.assertEqual(actual, expected)

    def test_full_checkpoint_migration_preserves_training_state(self):
        settings = TrainingSettings.from_yaml(ROOT / 'configs/bootstrap.yaml', MODE, tiny=True, overrides={
            'device': 'cpu', 'anchor_batch': 16, 'policy_microbatch': 2, 'base_game_pool_size': 4,
            'actor_inference_batch': 4, 'total_updates': 1, 'max_game_plies': 7, 'arena_enabled': False})
        with tempfile.TemporaryDirectory() as directory, patch.object(MetricLogger, 'start_resource_monitor'):
            old = SelfPlayTrainer(settings, run_directory=directory, auto_resume=False)
            aggregated = old._aggregate_rollout_metrics(RolloutMetrics(
                flag_captures=3, nonterminal_flag_captures=1, flag_capture_reward_abs_sum=.3))
            self.assertEqual((aggregated.flag_captures, aggregated.nonterminal_flag_captures), (3, 1))
            self.assertAlmostEqual(aggregated.flag_capture_reward_abs_sum, .3)
            old.train()
            path = old.checkpoints.latest_path
            before = torch.load(path, weights_only=False)
            # Missing field in a legacy checkpoint also means zero, even if raw
            # YAML happens to have been edited before the process saved.
            before['config'].pop('flag_capture_reward')
            torch.save(before, path)
            target = replace(settings, flag_capture_reward=.1, total_updates=2)
            with self.assertRaisesRegex(RuntimeError, 'adopt-flag-capture-reward'):
                SelfPlayTrainer(target, run_directory=directory)
            resumed = SelfPlayTrainer(target, run_directory=directory, adopt_flag_capture_reward=True)
            try:
                migrated = torch.load(path, weights_only=False)
                for key in ('policy', 'layout', 'reference_policy', 'reference_layout', 'critic',
                            'policy_optimizer', 'layout_optimizer', 'critic_optimizer', 'rng_state'):
                    self.assert_tree_equal(migrated[key], before[key])
                for key in ('cumulative', 'base_game_pool', 'layout_buffer', 'model_selection', 'historical_opponents'):
                    if key in before['trainer_state']:
                        self.assert_tree_equal(migrated['trainer_state'][key], before['trainer_state'][key])
                self.assertEqual(resumed.update, before['update'])
                self.assertEqual(migrated['config']['flag_capture_reward'], .1)
                migration = migrated['trainer_state']['flag_capture_objective_migration']
                self.assertEqual(migration['from_flag_capture_reward'], 0.)
                self.assertEqual(migration['update'], before['update'])
                self.assertEqual(migrated['reason'], 'flag_capture_objective_migration')
                resumed.train()
                self.assertEqual(resumed.update, 2)
                latest = json.loads((resumed.run_directory / 'latest_metrics.json').read_text())
                self.assertEqual(latest['training/flag_capture_reward'], .1)
            finally:
                resumed.logger.close()
            again = SelfPlayTrainer(target, run_directory=directory)
            try:
                self.assertEqual(again.flag_capture_objective_migration, migration)
                self.assertEqual(again.update, 2)
            finally:
                again.logger.close()
            # A later coefficient increase preserves the complete state too.
            before_increase = torch.load(path, weights_only=False)
            increased = SelfPlayTrainer(replace(target, flag_capture_reward=.2),
                run_directory=directory, adopt_flag_capture_reward=True)
            try:
                after_increase = torch.load(path, weights_only=False)
                for key in ('policy', 'layout', 'critic', 'policy_optimizer',
                            'layout_optimizer', 'critic_optimizer', 'rng_state'):
                    self.assert_tree_equal(after_increase[key], before_increase[key])
                self.assert_tree_equal(after_increase['trainer_state']['base_game_pool'],
                                       before_increase['trainer_state']['base_game_pool'])
                migration = increased.flag_capture_objective_migration
                self.assertEqual(migration['from_flag_capture_reward'], .1)
                self.assertEqual(migration['to_flag_capture_reward'], .2)
                self.assertEqual(increased.update, 2)
            finally:
                increased.logger.close()


if __name__ == '__main__':
    unittest.main()
