"""Champion succession, protocol migration, and uninterrupted latest-weight training."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from test_balanced_evaluation import balanced_result
from test_historical_opponents import settings as history_settings
from test_model_selection import training_settings, match_result, file_sha256
from junqi.training.distributed import DistributedContext
from junqi.training.model_selection import ModelSelection, atomic_json
from junqi.training.models import GamePolicyTransformer, PieceConditionedLayoutPointerDecoder
from junqi.training.settings import TrainingSettings


class ChampionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.context = DistributedContext(0, 1, 0, torch.device('cpu'))

    def selection(self, settings, *, update=0, steps=0, adopt=False):
        selection = ModelSelection(settings, self.root, self.context)
        policy = GamePolicyTransformer(settings.model)
        layout = PieceConditionedLayoutPointerDecoder(settings.model)
        selection.initialize(policy, layout, update=update, cumulative={'environment_plies': steps},
                             adopt_champion_evaluation=adopt)
        return selection, policy, layout

    def old_result(self, *, win=True):
        settings = replace(history_settings(), arena_observational_only=True,
                           arena_after_half_historical_only=True, arena_games=500,
                           historical_eval_total_games=500, arena_historical_teammate_fraction=.5)
        selection, policy, layout = self.selection(settings)
        with patch('junqi.training.model_selection.run_match', return_value=balanced_result(500, win=win)):
            report = selection.evaluate(policy, layout, update=10, cumulative={'environment_plies': 100})
        new = replace(settings, arena_champion_only=True, arena_observational_only=False,
                      arena_after_half_historical_only=False)
        return selection, report, new

    def test_migration_uses_latest_completed_win_preserves_history_and_is_idempotent(self):
        old, report, settings = self.old_result()
        before = old.state_dict()
        old_bytes = {n: (old.directory / n).read_bytes() for n in before['rounds']}
        with self.assertRaisesRegex(RuntimeError, '--adopt-champion-evaluation'):
            self.selection(settings, update=11, steps=110)
        self.assertEqual(json.loads(old.state_path.read_text()), before)
        new, _, _ = self.selection(settings, update=11, steps=110, adopt=True)
        self.assertEqual(new.state['best_update'], 10)
        self.assertEqual(file_sha256(new.best_path), report['candidate_sha256'])
        self.assertEqual(new.state['rounds'], before['rounds'])
        self.assertEqual(new.state[new.completed_key], before[new.completed_key])
        self.assertEqual(new._round_alpha(), old._round_alpha())
        migration = new.state['champion_evaluation_migrations'][0]
        self.assertEqual(migration['source_report'], before['rounds'][-1])
        self.assertGreater(migration['next_seed_base'], migration['previous_seed_base'])
        self.assertEqual(old_bytes, {n: (new.directory / n).read_bytes() for n in old_bytes})
        resumed, _, _ = self.selection(settings, update=11, steps=110, adopt=True)
        self.assertEqual(new.state, resumed.state)

    def test_migration_loss_keeps_incumbent(self):
        old, _, settings = self.old_result(win=False)
        resumed, _, _ = self.selection(settings, update=11, steps=110, adopt=True)
        self.assertEqual(resumed.state['best_sha256'], old.state['best_sha256'])
        self.assertEqual(resumed.state['champion_evaluation_migrations'][0]['source_score'], 0)

    def test_migration_requires_matching_opponent_provenance(self):
        old, report, settings = self.old_result()
        report['opponent_sha256'] = 'unrelated opponent'
        atomic_json(old.directory / old.state['rounds'][-1], report)
        resumed, _, _ = self.selection(settings, update=11, steps=110, adopt=True)
        self.assertEqual(resumed.state['best_sha256'], old.state['best_sha256'])
        self.assertNotIn('source_report', resumed.state['champion_evaluation_migrations'][0])

    def test_pending_round_and_other_contract_changes_cannot_be_migrated(self):
        old, _, settings = self.old_result()
        before = old.state_path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'finish the pending evaluation'):
            self.selection(settings, update=20, steps=200, adopt=True)
        with self.assertRaisesRegex(RuntimeError, 'schedule/budget changed'):
            self.selection(replace(settings, arena_seed=settings.arena_seed + 1), update=11, steps=110, adopt=True)
        self.assertEqual(old.state_path.read_bytes(), before)

    def test_every_round_challenges_incumbent_across_half_and_never_uses_panel(self):
        settings = replace(history_settings(), arena_champion_only=True,
                           arena_historical_teammate_fraction=.5, arena_games=500)
        selection, policy, layout = self.selection(settings)
        league = SimpleNamespace(active=True, directory=self.root,
            evaluation_panel=Mock(side_effect=AssertionError('champion challenges must not read a panel')),
            release_replica=Mock())
        split = match_result(100, 50, 100)
        tie = {**match_result(200, 100, 200), 'teammate_results': {'current': split, 'historical': split}}
        cases = [(100, balanced_result(500, win=True), 0, 10),
                 (200, tie, 10, 10),
                 (500, balanced_result(500), 10, 10),
                 (600, balanced_result(500, win=True), 10, 60),
                 (700, balanced_result(500), 60, 60)]
        for steps, result, opponent_update, best_update in cases:
            with self.subTest(steps=steps):
                # Changing latest weights must not be undone when the match loses.
                with torch.no_grad():
                    next(policy.parameters()).add_(.001)
                parameters = copy.deepcopy(policy.state_dict())
                rng = torch.get_rng_state().clone()
                with patch('junqi.training.model_selection.run_match', return_value=result) as match:
                    report = selection.evaluate(policy, layout, update=steps // 10,
                        cumulative={'environment_plies': steps}, historical=league)
                match.assert_called_once()
                self.assertEqual(match.call_args.args[2].pairs, 125)
                self.assertEqual(match.call_args.args[2].historical_teammate_fraction, .5)
                self.assertEqual(report['opponent_update'], opponent_update)
                self.assertEqual(report['best_update'], best_update)
                self.assertEqual(report['evaluation_type'], 'champion')
                self.assertNotIn('historical_panel', report)
                self.assertEqual(selection.state['best_update'], best_update)
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                for name, value in parameters.items():
                    torch.testing.assert_close(policy.state_dict()[name], value, rtol=0, atol=0)
        league.evaluation_panel.assert_not_called()
        latest = json.loads((self.root / 'historical_opponents/latest_evaluation.json').read_text())
        self.assertEqual((latest['evaluation_type'], latest['candidate_update']), ('champion', 70))
        self.assertEqual(latest['results'][0]['opponent_update'], 60)
        self.assertFalse(latest['promoted'])

    def test_champion_match_rejects_incomplete_teammate_groups(self):
        settings = training_settings(target_environment_plies=1000, arena_interval_environment_plies=100,
                                    arena_champion_only=True, arena_historical_teammate_fraction=.5)
        selection, policy, layout = self.selection(settings)
        with patch('junqi.training.model_selection.run_match', return_value=match_result()), \
                self.assertRaisesRegex(RuntimeError, 'missing balanced teammate'):
            selection.evaluate(policy, layout, update=1, cumulative={'environment_plies': 100})
        self.assertEqual(selection.state['best_update'], 0)
        self.assertEqual(selection.state['rounds'], [])

    def test_completed_match_reused_after_failed_state_commit(self):
        settings = training_settings(target_environment_plies=1000, arena_interval_environment_plies=100,
                                    arena_champion_only=True, arena_historical_teammate_fraction=.5)
        selection, policy, layout = self.selection(settings)
        def commit(path, value):
            if path == selection.state_path:
                raise OSError('interrupted state commit')
            atomic_json(path, value)
        with patch('junqi.training.model_selection.run_match', return_value=balanced_result(4, win=True)), \
                patch('junqi.training.model_selection.atomic_json', side_effect=commit), \
                self.assertRaisesRegex(RuntimeError, 'interrupted state commit'):
            selection.evaluate(policy, layout, update=1, cumulative={'environment_plies': 100})
        self.assertEqual(selection.state['best_update'], 0)
        resumed = ModelSelection(settings, self.root, self.context)
        resumed.initialize(policy, layout, update=1, cumulative={'environment_plies': 100})
        with patch('junqi.training.model_selection.run_match') as match:
            report = resumed.evaluate(policy, layout, update=1, cumulative={'environment_plies': 100})
        match.assert_not_called()
        self.assertEqual(report['opponent_update'], 0)
        self.assertEqual(resumed.state['best_update'], 1)
        self.assertEqual(len(resumed.state['rounds']), 1)

    def test_real_cpu_training_and_full_checkpoint_migration_keep_latest_weights(self):
        from junqi.training.trainer import SelfPlayTrainer
        old = training_settings(total_updates=2, anchor_batch=4, target_environment_plies=8,
            arena_interval_environment_plies=4, arena_observational_only=True, arena_historical_teammate_fraction=.5)
        trainer = SelfPlayTrainer(old, run_directory=self.root)
        trainer.train()
        original = torch.load(trainer.checkpoints.latest_path, weights_only=False)
        settings = replace(old, arena_observational_only=False, arena_champion_only=True)
        resumed = SelfPlayTrainer(settings, run_directory=self.root, adopt_champion_evaluation=True)
        self.addCleanup(resumed.logger.close)
        saved = torch.load(resumed.checkpoints.latest_path, weights_only=False)
        self.assertEqual(saved['reason'], 'champion_evaluation_migration')
        self.assertEqual(saved['update'], original['update'])
        self.assertEqual(saved['trainer_state']['base_game_pool'], original['trainer_state']['base_game_pool'])
        self.assertEqual(saved['trainer_state']['cumulative'], original['trainer_state']['cumulative'])
        for key in ('policy', 'layout', 'critic', 'reference_policy', 'reference_layout',
                    'policy_optimizer', 'layout_optimizer', 'critic_optimizer'):
            torch.testing.assert_close(saved[key], original[key], rtol=0, atol=0)
        with patch('junqi.training.model_selection.run_match') as match:
            resumed.train()
        match.assert_not_called()

    def test_real_cpu_champion_matches_before_and_after_half(self):
        from junqi.training.trainer import SelfPlayTrainer
        settings = training_settings(total_updates=3, anchor_batch=4, target_environment_plies=12,
            arena_interval_environment_plies=4, arena_champion_only=True, arena_historical_teammate_fraction=.5)
        trainer = SelfPlayTrainer(settings, run_directory=self.root)
        trainer.train()
        reports = [json.loads((trainer.model_selection.directory / name).read_text())
                   for name in trainer.model_selection.state['rounds']]
        self.assertEqual([r['decision'] for r in reports], ['retain_best'] * 3)
        self.assertEqual([r['draws'] for r in reports], [4] * 3)
        self.assertEqual([r['opponent_update'] for r in reports], [0] * 3)
        self.assertEqual(trainer.update, 3)

    def test_local_settings_modes_and_cli_adoption(self):
        from junqi.training.cli import main
        config = Path(__file__).parents[1] / 'configs/local_4090_training.yaml'
        for mode in ('four_dark', 'double_open'):
            settings = TrainingSettings.from_yaml(config, mode)
            self.assertTrue(settings.arena_champion_only)
            self.assertFalse(settings.arena_observational_only)
            self.assertFalse(settings.arena_after_half_historical_only)
            self.assertEqual(settings.arena_games, 500)
            for changes in ({'arena_champion_only': 'true'}, {'arena_observational_only': True},
                            {'arena_after_half_historical_only': True}):
                with self.assertRaises(ValueError):
                    replace(settings, **changes).validate()
        self.assertFalse(TrainingSettings.from_yaml(config, 'two_player').arena_champion_only)
        self.assertFalse(TrainingSettings.from_yaml(config, 'four_dark', tiny=True).arena_champion_only)
        with patch('junqi.training.cli.SelfPlayTrainer') as trainer:
            main(['--config', str(config), '--mode', 'four_dark', '--device', 'cpu', '--adopt-champion-evaluation'])
        self.assertTrue(trainer.call_args.kwargs['adopt_champion_evaluation'])
