from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from junqi.training.arena import MatchSettings
from junqi.training.encoding import GameHistory, _record_from_observation
from junqi.training.models import GamePolicyTransformer, ModelConfig
from junqi.training.packed_observation import observation_rows
from junqi.training.modes import TrainingMode, new_game
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer


CONFIG = Path(__file__).parents[1] / 'configs' / 'bootstrap.yaml'


def settings(**overrides):
    return TrainingSettings.from_yaml(CONFIG, 'four_dark', tiny=True, overrides={
        'device': 'cpu', 'anchor_batch': 8, 'policy_microbatch': 2,
        'total_updates': 1, **overrides,
    })


class DrawRuleMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_counter_69_and_terminal_70_are_encoded_without_clamping_at_60(self):
        game = new_game('four_dark', seed=7)
        game.no_interaction_plies = 69
        history = GameHistory.initialize(game, TrainingMode.FOUR_DARK, max_transitions=16)
        state = history.state_for(game)
        policy = GamePolicyTransformer(ModelConfig.tiny())
        self.assertTrue(torch.isfinite(policy([state], [[state.legal_actions[0]]])[0]).all())
        # Test both observation paths used by serial and process-based PPO.
        for count in (60, 69, 70):
            game.no_interaction_plies = count
            observation = game.observe(0)
            record = _record_from_observation(observation, None)
            self.assertEqual(record.no_interaction_plies, count)
            row = observation_rows(game, TrainingMode.FOUR_DARK)[0]
            self.assertEqual(int(row[-8]), count)
        self.assertIsNone(MatchSettings().max_plies)

    def test_global_step_checkpoint_schedule_is_independent_of_evaluation_and_resume(self):
        trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
        trainer.settings = settings(checkpoint_every_updates=5,
                                    checkpoint_interval_environment_plies=25_000_000)
        trainer.update = 10
        trainer.last_checkpoint_environment_plies = 11_304_960
        trainer.cumulative = {'environment_plies': 24_999_999}
        self.assertFalse(trainer._periodic_checkpoint_due())
        trainer.cumulative['environment_plies'] = 25_067_520
        self.assertTrue(trainer._periodic_checkpoint_due())
        trainer.last_checkpoint_environment_plies = 25_067_520
        trainer.cumulative['environment_plies'] += 98_304
        self.assertFalse(trainer._periodic_checkpoint_due())
        trainer.cumulative['environment_plies'] = 50_036_736
        self.assertTrue(trainer._periodic_checkpoint_due())
        trainer.settings = replace(trainer.settings, checkpoint_interval_environment_plies=None)
        self.assertTrue(trainer._periodic_checkpoint_due())
        trainer.update = 11
        self.assertFalse(trainer._periodic_checkpoint_due())

    def test_v6_resume_preserves_actor_critic_adam_history_and_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = settings(max_game_plies=2000, no_capture_draw_plies=60)
            original = SelfPlayTrainer(legacy, run_directory=directory, auto_resume=False)
            original.train()
            path = original.checkpoints.latest_path
            payload = torch.load(path, weights_only=False)
            payload['format_version'] = 6
            for key, model in (('policy', original.policy), ('critic', original.critic)):
                weight = payload[key]['no_interaction_embedding.weight'][:61].clone()
                payload[key]['no_interaction_embedding.weight'] = weight
                names = [name for name, _ in model.named_parameters()]
                parameter_id = payload[key + '_optimizer']['param_groups'][0]['params'][
                    names.index('no_interaction_embedding.weight')]
                moments = payload[key + '_optimizer']['state'][parameter_id]
                for moment in ('exp_avg', 'exp_avg_sq'):
                    moments[moment] = moments[moment][:61].clone()
            payload['trainer_state']['base_game_pool'].pop('no_capture_draw_plies')
            torch.save(payload, path)
            expected = copy.deepcopy(payload['trainer_state']['cumulative'])
            modern = replace(legacy, max_game_plies=None, no_capture_draw_plies=70, total_updates=2)
            with self.assertRaisesRegex(ValueError, 'maximum game plies'):
                SelfPlayTrainer(modern, run_directory=directory, adopt_pass_rule=True)
            resumed = SelfPlayTrainer(modern, run_directory=directory, adopt_current_draw_rules=True, adopt_pass_rule=True)
            try:
                self.assertEqual(resumed.update, 1)
                self.assertEqual(resumed.cumulative, expected)
                self.assertEqual(resumed.policy_lr_scale, original.policy_lr_scale)
                for key, model, optimizer in (
                    ('policy', resumed.policy, resumed.policy_optimizer),
                    ('critic', resumed.critic, resumed.critic_optimizer),
                ):
                    parameter = model.no_interaction_embedding.weight
                    torch.testing.assert_close(parameter[:61], payload[key]['no_interaction_embedding.weight'], rtol=0, atol=0)
                    torch.testing.assert_close(parameter[61:], parameter[60:61].expand(10, -1), rtol=0, atol=0)
                    names = [name for name, _ in model.named_parameters()]
                    parameter_id = payload[key + '_optimizer']['param_groups'][0]['params'][names.index('no_interaction_embedding.weight')]
                    for moment in ('exp_avg', 'exp_avg_sq'):
                        saved = payload[key + '_optimizer']['state'][parameter_id][moment]
                        torch.testing.assert_close(optimizer.state[parameter][moment][:61], saved, rtol=0, atol=0)
                        self.assertEqual(optimizer.state[parameter][moment][61:].count_nonzero(), 0)
                for before, after in zip(original.pool.slots, resumed.pool.slots):
                    self.assertEqual(dict(before.game.pieces), dict(after.game.pieces))
                    self.assertEqual(before.game.no_interaction_plies, after.game.no_interaction_plies)
                    self.assertEqual(before.game.ply_count, after.game.ply_count)
                    self.assertEqual(before.history.state_dict(), after.history.state_dict())
                    self.assertIsNone(after.game.config.max_plies)
                    self.assertEqual(after.game.config.no_interaction_draw_plies, 70)
                resumed.train()  # An actual Adam update catches incompatible moment shapes.
                self.assertEqual(resumed.update, 2)
                saved = torch.load(path, weights_only=False)
                self.assertEqual(saved['format_version'], 9)
                self.assertEqual(saved['trainer_state']['draw_rule_migration']['update'], 1)
                self.assertEqual(saved['trainer_state']['last_checkpoint_environment_plies'], resumed.cumulative['environment_plies'])
            finally:
                resumed.logger.close()

    def test_training_saves_at_completed_global_step_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = SelfPlayTrainer(settings(total_updates=3, checkpoint_interval_environment_plies=12),
                                      run_directory=directory, auto_resume=False)
            saved = []
            real_save = trainer.save_checkpoint
            def record_save(*, reason, archive):
                saved.append((reason, trainer.update, trainer.cumulative['environment_plies']))
                return real_save(reason=reason, archive=archive)
            with patch.object(trainer, 'save_checkpoint', side_effect=record_save):
                trainer.train()
            self.assertEqual(saved, [('periodic', 2, 16), ('periodic', 3, 24),
                                     ('completed_or_stopped', 3, 24)])
            manifest = json.loads((trainer.checkpoints.directory / 'manifest.json').read_text())
            self.assertEqual(manifest['environment_plies'], 24)

    def test_selection_adoption_keeps_frozen_baseline_and_marks_new_rules(self):
        from junqi.training.distributed import DistributedContext
        from junqi.training.model_selection import ModelSelection
        from junqi.training.models import PieceConditionedLayoutPointerDecoder
        with tempfile.TemporaryDirectory() as directory:
            legacy = settings(arena_enabled=True, arena_games=4, arena_max_plies=2000,
                              no_capture_draw_plies=60, total_updates=100)
            context = DistributedContext(0, 1, 0, torch.device('cpu'))
            policy = GamePolicyTransformer(legacy.model)
            layout = PieceConditionedLayoutPointerDecoder(legacy.model)
            first = ModelSelection(legacy, Path(directory), context)
            first.initialize(policy, layout, update=0, cumulative={})
            prior_hash = first.state['best_sha256']
            saved = json.loads(first.state_path.read_text())
            saved['contract'].pop('no_capture_draw_plies')
            first.state_path.write_text(json.dumps(saved))
            current = ModelSelection(replace(legacy, arena_max_plies=None, no_capture_draw_plies=70), Path(directory), context)
            current.initialize(policy, layout, update=12, cumulative={'environment_plies': 100}, adopt_current_draw_rules=True)
            self.assertEqual(current.state['best_sha256'], prior_hash)
            self.assertEqual(current.state['baseline_update'], 0)
            self.assertEqual(current.state['contract']['no_capture_draw_plies'], 70)
            self.assertIsNone(current.state['contract']['max_plies'])
            self.assertEqual(current.state['statistical_family'], 'no_capture_70_unlimited_pass4')
            self.assertEqual(current.state['draw_rule_migration']['update'], 12)

    def test_invalid_draw_and_save_intervals_are_rejected(self):
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    settings(checkpoint_interval_environment_plies=invalid)
                with self.assertRaises(ValueError):
                    settings(no_capture_draw_plies=invalid)
