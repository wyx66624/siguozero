from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_historical_opponents import settings
from junqi.training.distributed import DistributedContext
from junqi.training.historical_opponents import HistoricalOpponents
from junqi.training.models import GamePolicyTransformer, PieceConditionedLayoutPointerDecoder
from junqi.training.rollout import BaseGamePool, FrozenPolicyActor
from junqi.training.settings import TrainingSettings


class HistoricalWinRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = settings(historical_sampling="win_rate_distance", historical_uniform_fraction=0.,
                                 historical_checkpoint_start_fraction=.2)
        self.context = DistributedContext(0, 1, 0, torch.device("cpu"))
        self.policy = GamePolicyTransformer(self.settings.model).eval()
        self.layout = PieceConditionedLayoutPointerDecoder(self.settings.model).eval()
        self.league = self.make_league()
        for progress in (0, 100, 200):
            self.league.update_progress(self.policy, self.layout, environment_plies=progress, update=progress)
        self.ids = [e["id"] for e in self.league.entries]

    def make_league(self, config=None):
        league = HistoricalOpponents(config or self.settings, self.root, self.context)
        self.addCleanup(league.close)
        return league

    def result(self, identifier, reward, teammate=None):
        self.league.active_id = identifier
        self.league.record_result(SimpleNamespace(opponent_id=identifier, teammate_id=teammate), reward)

    def test_exact_distance_formula_prefers_even_match_over_both_extremes(self):
        self.league.win_rates = dict(zip(self.ids, (.01, .5, .99)))
        probabilities = self.league.probabilities()
        middle = 1 / (1 + 2 * math.exp(-4.9))
        self.assertAlmostEqual(probabilities[self.ids[1]], middle)
        self.assertAlmostEqual(probabilities[self.ids[0]], probabilities[self.ids[2]])
        self.assertGreater(probabilities[self.ids[1]], 100 * probabilities[self.ids[0]])
        self.assertAlmostEqual(sum(probabilities.values()), 1)
        self.assertTrue(all(p > 0 for p in probabilities.values()))
        # The old quality score and archive age cannot override this formula.
        self.league.q = dict(zip(self.ids, (1000., -1000., 1000.)))
        self.assertEqual(self.league.probabilities(), probabilities)

    def test_unseen_versions_start_neutral_and_every_saved_checkpoint_is_eligible(self):
        self.assertEqual(set(self.league.probabilities().values()), {1 / 3})
        self.league.archive_checkpoint(self.policy, self.layout, environment_plies=250, update=250)
        newest = self.league.entries[-1]["id"]
        self.assertEqual(len(self.league.probabilities()), 4)
        self.assertEqual(self.league.probabilities()[newest], .25)
        self.league.progress = 200
        self.assertNotIn(newest, self.league.probabilities())

    def test_recent_results_update_win_probability_and_draws_are_not_wins(self):
        identifier = self.ids[0]
        self.result(identifier, 1.)
        self.assertAlmostEqual(self.league.win_rates[identifier], .505)
        self.result(identifier, 0.)
        self.assertAlmostEqual(self.league.win_rates[identifier], .49995)
        self.assertEqual(self.league.win_rate_results[identifier], dict(wins=1, draws=1, losses=0))
        for _ in range(300):
            self.result(identifier, 1.)
        self.assertGreater(self.league.win_rates[identifier], .97)
        for _ in range(300):
            self.result(identifier, -1.)
        self.assertLess(self.league.win_rates[identifier], .05)
        self.assertEqual(self.league.quality_update_games, 602)

    def test_historical_teammates_do_not_change_opponent_estimate(self):
        identifier = self.ids[0]
        self.result(identifier, 1.)
        rates, results = copy.deepcopy((self.league.win_rates, self.league.win_rate_results))
        self.result(identifier, -1., teammate=identifier)
        self.league.record_result(SimpleNamespace(opponent_id=None, teammate_id=identifier), 1.)
        self.assertEqual(self.league.win_rates, rates)
        self.assertEqual(self.league.win_rate_results, results)
        self.assertEqual(self.league.quality_update_games, 1)

    def test_checkpoint_restores_rates_pending_choice_and_future_rng_exactly(self):
        pool = BaseGamePool("four_dark", pool_size=5, max_transitions=8, max_game_plies=8, seed=9)
        actor = FrozenPolicyActor(self.policy, max_batch_size=8)
        self.league.update_progress(self.policy, self.layout, environment_plies=500, update=500)
        self.league.begin_rollout(pool, actor, 500)
        for _ in range(20):
            self.league.assignment()
        for reward in (1., 1., 0., -1.):
            self.result(self.league.active_id, reward)
        self.league.update_progress(self.policy, self.layout, environment_plies=501, update=501)
        saved = self.league.state_dict()
        self.assertIsNotNone(saved["pending_cohort"])
        restored = self.make_league()
        restored.load_state_dict(saved)
        self.assertEqual(restored.state_dict(), saved)
        self.assertEqual(restored.probabilities(), self.league.probabilities())
        self.assertEqual([restored._choose_cohort() for _ in range(50)],
                         [self.league._choose_cohort() for _ in range(50)])

    def test_legacy_migration_is_limited_to_unused_sampler_and_preserves_state(self):
        saved = self.league.state_dict()
        for key in ("sampling_contract", "win_rates", "win_rate_results"):
            saved.pop(key)
        # Reproduce the local run's old 50% stage mixture and legacy score.
        saved["contract"]["stage_mix_fraction"] = .5
        saved["q"] = {self.ids[0]: -3.}
        restored = self.make_league()
        restored.load_state_dict(saved)
        self.assertEqual(restored.rng.getstate(), self.league.rng.getstate())
        self.assertEqual(restored.panel(), saved["entries"])
        self.assertEqual(restored.q, saved["q"])
        self.assertEqual(restored.win_rates, {})
        saved["contract"].pop("stage_mix_fraction")
        saved["started"] = 1
        with self.assertRaisesRegex(ValueError, "sampling changed"):
            self.make_league().load_state_dict(saved)

    def test_active_sampler_parameters_cannot_change_silently_on_resume(self):
        self.result(self.ids[0], 1.)
        saved = self.league.state_dict()
        for field, value in (("historical_win_rate_temperature", .2), ("historical_win_rate_ema_alpha", .1)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "sampling changed"):
                self.make_league(replace(self.settings, **{field: value})).load_state_dict(saved)
        saved["win_rates"][self.ids[0]] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid saved historical win rate"):
            self.make_league().load_state_dict(saved)

    def test_small_temperature_stays_finite_and_keeps_nonzero_support(self):
        self.league.settings = replace(self.settings, historical_win_rate_temperature=1e-300)
        self.league.win_rates = dict(zip(self.ids, (.01, .5, .99)))
        probabilities = self.league.probabilities()
        self.assertTrue(all(math.isfinite(p) and p > 0 for p in probabilities.values()))
        self.assertAlmostEqual(sum(probabilities.values()), 1)

    def test_status_exposes_sampling_estimate_and_its_clean_result_counts(self):
        self.result(self.ids[0], 1.)
        self.league.write_status(SimpleNamespace(slots=[]))
        status = json.loads((self.league.directory / "status.json").read_text())
        self.assertEqual(status["sampling"], self.league.sampling_contract())
        self.assertAlmostEqual(status["opponents"][0]["win_rate"], .505)
        self.assertEqual(status["opponents"][0]["win_rate_results"], dict(wins=1, draws=0, losses=0))
        self.assertEqual(status["opponents"][1]["win_rate_results"], {})

    def test_local_config_keeps_gates_and_role_fractions(self):
        config = TrainingSettings.from_yaml(Path(__file__).parents[1] / "configs/local_4090_training.yaml", "four_dark")
        self.assertEqual(config.historical_sampling, "win_rate_distance")
        self.assertEqual(config.historical_checkpoint_start_fraction, .2)
        self.assertEqual(config.historical_start_fraction, .5)
        self.assertEqual((config.historical_training_fraction, config.historical_teammate_fraction), (.2, .2))
        self.assertEqual((config.historical_uniform_fraction, config.historical_stage_mix_fraction), (0., 0.))

    def test_invalid_sampling_settings_are_rejected(self):
        for changes in (dict(historical_sampling="unknown"), dict(historical_win_rate_temperature=0),
                        dict(historical_win_rate_temperature=float("nan")), dict(historical_win_rate_ema_alpha=2),
                        dict(historical_win_rate_ema_alpha=True), dict(historical_stage_mix_fraction=.5),
                        dict(historical_uniform_fraction=.05)):
            options = dict(historical_sampling="win_rate_distance", historical_uniform_fraction=0.)
            options.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                settings(**options)

    def test_real_mixed_update_changes_learner_but_preserves_frozen_policy_and_layout(self):
        from junqi.training.trainer import SelfPlayTrainer
        config = replace(self.settings, total_updates=1, max_game_plies=8, anchor_batch=200,
                         base_game_pool_size=25, policy_microbatch=8,
                         historical_teammate_fraction=.2, historical_cohort_games=1000)
        trainer = SelfPlayTrainer(config, run_directory=self.root / "trainer")
        self.addCleanup(trainer.historical.close)
        self.addCleanup(trainer.logger.close)
        for progress in (100, 200, 500):
            trainer.historical.update_progress(trainer.policy, trainer.layout, environment_plies=progress, update=0)
        trainer.cumulative["environment_plies"] = 500
        learner_before = {k: v.clone() for k, v in trainer.policy.state_dict().items()}
        frozen_before = {}
        original_load = trainer.historical._load_active

        def remember(actor):
            original_load(actor)
            for name in ("policy", "layout"):
                frozen_before[name] = {k: v.clone() for k, v in getattr(trainer.historical, name).state_dict().items()}

        with patch.object(trainer.historical, "_load_active", side_effect=remember), \
             patch.object(trainer, "_maybe_evaluate_model", return_value=None):
            trainer.train()
        self.assertTrue(any(not torch.equal(v, learner_before[k]) for k, v in trainer.policy.state_dict().items()))
        self.assertGreater(trainer.historical.opponent_plies, 0)
        self.assertGreater(trainer.historical.teammate_plies, 0)
        self.assertLess(trainer.cumulative["policy_samples"], 200)
        for name in ("policy", "layout"):
            model = getattr(trainer.historical, name)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, frozen_before[name][key], rtol=0, atol=0)
        resumed = SelfPlayTrainer(config, run_directory=self.root / "trainer")
        self.addCleanup(resumed.logger.close)
        self.addCleanup(resumed.historical.close)
        self.assertEqual(resumed.historical.state_dict(), trainer.historical.state_dict())


if __name__ == "__main__":
    unittest.main()
