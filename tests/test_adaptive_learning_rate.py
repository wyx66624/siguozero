from dataclasses import replace
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from junqi.training.learning_rate import BoundedLearningRate, diagnostic_sequences
from junqi.training.ppo import PPOSample, policy_ppo_loss
from junqi.training.trainer import SelfPlayTrainer
from test_ppo import training_settings, states


def controller(**kwargs):
    return BoundedLearningRate(**{**dict(minimum=1e-6, maximum=2e-5, target_kl=.015,
                                        scale=.1), **kwargs})


class AdaptiveLearningRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_increase_requires_stability_and_cooldown_and_reports_real_changes(self):
        c = controller()
        for _ in range(4):
            rate, direction = c.observe(scheduled=1e-4, kl=.002, clip_fraction=.01, early_stopped=False)
            self.assertEqual(direction, 0)
            self.assertAlmostEqual(rate, 1e-5)
        rate, direction = c.observe(scheduled=1e-4, kl=.002, clip_fraction=.01, early_stopped=False)
        self.assertEqual(direction, 1)
        self.assertAlmostEqual(rate, 1.03e-5)
        for _ in range(5):
            _, direction = c.observe(scheduled=1e-4, kl=.002, clip_fraction=.01, early_stopped=False)
            self.assertEqual(direction, 0)
        self.assertEqual(c.stable_count, 0)

    def test_both_bounds_no_hidden_windup_and_reversal(self):
        c = controller(stable_updates=1, cooldown_updates=0, ema_decay=0)
        for _ in range(2000):
            rate, direction = c.observe(scheduled=1e-4, kl=.001, clip_fraction=0, early_stopped=False)
            self.assertLessEqual(rate, 2e-5)
        self.assertEqual(rate, 2e-5)
        self.assertEqual(direction, 0)
        self.assertEqual(c.reason, "upper_bound")
        rate, direction = c.observe(scheduled=1e-4, kl=.04, clip_fraction=.2, early_stopped=True)
        self.assertEqual(direction, -1)
        self.assertAlmostEqual(rate, 1.6e-5)
        for _ in range(2000):
            rate, direction = c.observe(scheduled=1e-4, kl=.04, clip_fraction=.2, early_stopped=True)
            self.assertGreaterEqual(rate, 1e-6)
        self.assertEqual(rate, 1e-6)
        self.assertEqual(direction, 0)
        self.assertEqual(c.reason, "lower_bound")
        rate, direction = c.observe(scheduled=1e-4, kl=.001, clip_fraction=0, early_stopped=False)
        self.assertEqual(direction, 1)
        self.assertAlmostEqual(rate, 1.03e-6)

    def test_local_spikes_missing_samples_and_warmup_cannot_trigger_recovery(self):
        c = controller(stable_updates=1)
        for options in (dict(early_stopped=True), dict(clip_fraction=.2),
                        dict(allow_increase=False), dict(kl=None)):
            rate, direction = c.observe(**{**dict(scheduled=1e-4, kl=.001,
                clip_fraction=.01, early_stopped=False), **options})
            self.assertEqual(direction, 0)
            self.assertAlmostEqual(rate, 1e-5)
        for value in (math.nan, math.inf, -.1):
            with self.assertRaises(ValueError):
                c.observe(scheduled=1e-4, kl=value, clip_fraction=0, early_stopped=False)

    def test_schedule_ceiling_and_exact_resume_of_feedback(self):
        c = controller()
        for _ in range(3):
            c.observe(scheduled=1e-4, kl=.002, clip_fraction=.01, early_stopped=False)
        clone = controller()
        clone.load_state_dict(c.state_dict(), legacy_scale=1., scheduled=1e-4)
        for kl in [.002] * 20 + [.04] * 20:
            opts = dict(scheduled=1e-4, kl=kl, clip_fraction=0, early_stopped=False)
            self.assertEqual(c.observe(**opts), clone.observe(**opts))
            self.assertEqual(c.state_dict(), clone.state_dict())
        c.scale = 1e20
        self.assertEqual(c.rate(5e-6), 5e-6)
        self.assertEqual(c.rate(1e-6), 1e-6)
        self.assertEqual(c.scale, 1.)
        clone.load_state_dict(None, legacy_scale=.1, scheduled=1e-4)
        self.assertAlmostEqual(clone.rate(1e-4), 1e-5)
        for key, value in (("scale", math.nan), ("kl_ema", math.inf), ("cooldown", -1)):
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                clone.load_state_dict({key: value}, legacy_scale=1., scheduled=1e-4)

    def test_settings_reject_invalid_bounds_and_unbounded_feedback(self):
        good = training_settings()
        for changes in ({"minimum_learning_rate": 0.}, {"maximum_learning_rate": 1e-9},
                        {"maximum_learning_rate": math.inf}, {"lr_increase_factor": 2.},
                        {"lr_decrease_factor": 0.}, {"lr_stable_updates": 0},
                        {"lr_cooldown_updates": -1}, {"lr_probe_samples": 0},
                        {"lr_ema_decay": 1.}, {"adaptive_learning_rate": "true"}):
            with self.assertRaises(ValueError, msg=str(changes)):
                replace(good, **changes).validate()

    def test_probe_is_bounded_excludes_frozen_actions_and_preserves_randomness(self):
        ss = states(12)
        samples = [PPOSample(s, s.legal_actions[0], -1., 0., 1., 0., 0,
                             learnable=i % 3 != 0) for i, s in enumerate(ss)]
        rng = random.getstate()
        probe = diagnostic_sequences(samples, limit=5, sequence_length=2, seed=44)
        self.assertEqual(len(probe), 5)
        self.assertTrue(all(x.learnable for x in probe))
        self.assertEqual(random.getstate(), rng)
        self.assertEqual([id(x) for x in probe], [id(x) for x in
            diagnostic_sequences(samples, limit=5, sequence_length=2, seed=44)])

    def test_final_probe_matches_frozen_policy_math_without_changing_weights_or_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = SelfPlayTrainer(training_settings(lr_probe_samples=8), run_directory=directory)
            try:
                ss = states(4)
                samples = []
                with torch.no_grad():
                    for s in ss:
                        log = trainer.policy([s], [s.legal_actions])[0]
                        samples.append(PPOSample(s, s.legal_actions[0], float(log[0]) + .2,
                                                 0., 1., 0., 0))
                before = {k: v.clone() for k, v in trainer.policy.state_dict().items()}
                trainer.policy.train()
                rng = torch.get_rng_state().clone()
                actual = trainer._final_policy_diagnostics(samples)
                self.assertAlmostEqual(actual["policy/final_kl"], math.exp(-.2) - 1 + .2, places=5)
                self.assertEqual(actual["policy/final_probe_samples"], 4)
                self.assertTrue(trainer.policy.training)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                self.assertTrue(all(p.grad is None for p in trainer.policy.parameters()))
                self.assertTrue(all(torch.equal(before[k], v) for k, v in trainer.policy.state_dict().items()))
                # Unequal per-rank sample counts require summing moments/counts,
                # not averaging each rank's already averaged KL.
                with patch.object(trainer.distributed.__class__, "sum_metrics",
                    side_effect=lambda values: {"count": values["count"] + 6,
                        "kl": values["kl"] + 6 * .04, "clip": values["clip"] + 6 * .2}):
                    merged = trainer._final_policy_diagnostics(samples)
                self.assertAlmostEqual(merged["policy/final_kl"],
                                       (actual["policy/final_kl"] * 4 + .04 * 6) / 10, places=6)
                self.assertEqual(merged["policy/final_probe_samples"], 10)
            finally:
                trainer.logger.close()

    def test_trainer_observes_once_per_update_and_saves_resume_state(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = training_settings(ppo_minibatch_samples=2, policy_epochs=2,
                maximum_learning_rate=2e-5, lr_probe_samples=8)
            trainer = SelfPlayTrainer(settings, run_directory=directory)
            trainer.train()
            self.assertEqual(trainer.lr_controller.observations, 1)
            saved = torch.load(trainer.checkpoints.latest_path, map_location="cpu", weights_only=False)
            self.assertIn("adaptive_learning_rate", saved["trainer_state"])
            metrics = json.loads((trainer.run_directory / "latest_metrics.json").read_text())
            self.assertLessEqual(metrics["optimizer/policy_lr_next"], 2e-5)
            self.assertGreaterEqual(metrics["optimizer/policy_lr_next"], 1e-6)
            self.assertIn("policy/final_kl", metrics)
            resumed = SelfPlayTrainer(settings, run_directory=directory)
            try:
                self.assertEqual(trainer.lr_controller.state_dict(), resumed.lr_controller.state_dict())
            finally:
                resumed.logger.close()
            saved["trainer_state"].pop("adaptive_learning_rate")
            saved["trainer_state"]["policy_lr_scale"] = .1
            torch.save(saved, trainer.checkpoints.latest_path)
            resumed = SelfPlayTrainer(settings, run_directory=directory)
            try:
                self.assertEqual(resumed.lr_controller.observations, 0)
                self.assertTrue(1e-6 <= resumed._learning_rate(2) <= 2e-5)
            finally:
                resumed.logger.close()

    def test_disabled_controller_has_no_probe_or_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = SelfPlayTrainer(training_settings(adaptive_learning_rate=False,
                maximum_learning_rate=2e-5), run_directory=directory)
            with patch.object(trainer, "_final_policy_diagnostics", side_effect=AssertionError("unused")):
                trainer.train()
            self.assertIsNone(trainer.lr_controller)
            self.assertEqual(trainer._learning_rate(1), 2e-5)


if __name__ == "__main__":
    unittest.main()
