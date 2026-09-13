"""Clipping signs, rollout normalization, scheduling and checkpoint recovery."""
import json
import math
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from junqi.training.clipping import (
    DEFAULT_CLIP_SCHEDULE, PolicyClip, advantage_clip_upper,
    clipped_surrogate, scheduled_policy_clip,
)
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer
from junqi.training.ppo import policy_ppo_loss
from test_ppo import training_settings
import test_learner_graph
from junqi.training.models import GamePolicyTransformer, ModelConfig, PolicyFeatures


class AdaptiveClippingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def settings(self, **kwargs):
        return training_settings(ppo_adaptive_clip=True, target_environment_plies=3_000_000_000, **kwargs)

    def test_milestones_interpolation_and_hard_bounds(self):
        settings = self.settings()
        for fraction, lower, upper in DEFAULT_CLIP_SCHEDULE:
            bounds = scheduled_policy_clip(settings, round(3_000_000_000 * fraction))
            self.assertAlmostEqual(bounds.lower, lower)
            self.assertAlmostEqual(bounds.upper, upper)
        midpoint = scheduled_policy_clip(settings, 500_000_000)
        self.assertAlmostEqual(midpoint.lower, .225)
        self.assertAlmostEqual(midpoint.upper, .275)
        previous = torch.full((5,), float('inf'))
        for steps in range(0, 3_100_000_000, 30_000_000):
            b = scheduled_policy_clip(settings, steps)
            actual = advantage_clip_upper(torch.tensor([-1e30, -1., 0., 1., 1e30]), b.upper, b.bonus, b.minimum, b.maximum)
            self.assertTrue(torch.all(actual <= previous))
            self.assertTrue(torch.all((actual >= .1) & (actual <= .45)))
            previous = actual
        self.assertEqual(scheduled_policy_clip(settings, -1).progress, 0.)
        self.assertEqual(scheduled_policy_clip(settings, 6_000_000_000).progress, 1.)

    def test_positive_advantage_keeps_gradient_longer_negative_uses_lower_bound(self):
        advantage = torch.tensor([3., .01, -2., -2.], requires_grad=True)
        ratio = torch.tensor([1.4, 1.4, .65, 1.5], requires_grad=True)
        upper = advantage_clip_upper(advantage, .3, .15, .1, .45)
        self.assertFalse(upper.requires_grad)
        objective, fraction = clipped_surrogate(ratio, advantage, .25, upper)
        (-objective.sum()).backward()
        torch.testing.assert_close(ratio.grad, torch.tensor([-3., 0., 0., 2.]))
        torch.testing.assert_close(fraction, torch.tensor([0., 1., 1., 1.]))
        self.assertGreater(upper[0], upper[1])

    def test_disabled_path_matches_original_ppo_objective_and_gradients(self):
        ratio = torch.tensor([.6, .9, 1.1, 1.6], requires_grad=True)
        advantage = torch.tensor([-2., 3., -.1, 1.])
        b = scheduled_policy_clip(replace(self.settings(), ppo_adaptive_clip=False), 100)
        upper = advantage_clip_upper(advantage, b.upper, b.bonus, b.minimum, b.maximum)
        actual, _ = clipped_surrogate(ratio, advantage, b.lower, upper)
        expected = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), ratio, retain_graph=True)[0],
                                   torch.autograd.grad(expected.sum(), ratio)[0], atol=0, rtol=0)

    def test_invalid_schedules_and_nonfinite_bounds_are_rejected(self):
        settings = self.settings()
        for overrides in [dict(ppo_adaptive_clip='true'), dict(ppo_clip_maximum=1.),
                          dict(ppo_clip_minimum=math.nan), dict(ppo_clip_advantage_scale=math.inf),
                          dict(ppo_clip_schedule=((0., .2, .3), (1., .3, .3))),
                          dict(ppo_clip_schedule=((0., .2, .3), (0., .1, .2), (1., .1, .1))),
                          dict(ppo_clip_schedule=((0., .2, .3),)), dict(target_environment_plies=None)]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(settings, **overrides).validate()

    def test_graph_padding_matches_public_adaptive_loss_and_gradients_at_each_stage(self):
        import copy
        fixture = test_learner_graph.LearnerGraphTests()
        for progress in (0, 1_500_000_000, 3_000_000_000):
            torch.manual_seed(92)
            model = GamePolicyTransformer(ModelConfig.tiny())
            reference = copy.deepcopy(model)
            samples = fixture.batch()
            contexts = torch.randn(8, model.config.temporal_dim)
            graph = fixture.graph(model, False)
            graph.callable = lambda *args: contexts
            bounds = scheduled_policy_clip(self.settings(), progress)
            graph.stage(model, samples, clip=.2, coefficient=.01, opening_coefficient=.04,
                        opening_plies=1, policy_clip=bounds)
            loss, metrics = graph._loss(model)
            with patch.object(reference, '_encode_full', return_value=PolicyFeatures(
                    contexts[:3], torch.ones(3, 129, dtype=torch.bool))):
                expected = policy_ppo_loss(reference, samples, clip_epsilon=.2, entropy_coefficient=.01,
                    opening_entropy_coefficient=.04, entropy_opening_plies=1, sequence_training=True,
                    defer_metrics=True, policy_clip=bounds)
            torch.testing.assert_close(metrics, torch.stack(list(expected.metrics.values())), rtol=2e-6, atol=2e-7)
            loss.backward(); expected.loss.backward()
            for p, q in zip(model.parameters(), reference.parameters(), strict=True):
                if q.grad is not None:
                    torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=3e-7)

    def test_graceful_stop_resume_uses_nonmilestone_checkpoint_and_repairs_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.settings(checkpoint_interval_environment_plies=10_000_000)
            trainer = SelfPlayTrainer(settings, run_directory=directory)
            trainer.train()
            self.assertEqual(trainer.update, 1)
            self.assertEqual(trainer.cumulative['environment_plies'], 8)
            path = trainer.run_directory
            progress = json.loads((path/'training_progress.json').read_text())
            self.assertEqual(progress['cumulative']['environment_plies'], 8)
            manifest = json.loads((path/'checkpoints/manifest.json').read_text())
            self.assertEqual(manifest['environment_plies'], 8)
            self.assertEqual(manifest['reason'], 'completed_or_stopped')
            # Simulate a later log plus a crash between checkpoint and sidecar writes.
            (path/'latest_metrics.json').write_text(json.dumps({'update': 100, 'cumulative/environment_plies': 800}))
            (path/'checkpoints/manifest.json').write_text('{}')
            resumed = SelfPlayTrainer(settings, run_directory=directory)
            try:
                self.assertEqual(resumed.update, 1)
                self.assertEqual(resumed.cumulative, trainer.cumulative)
                self.assertEqual(resumed._policy_clip(), trainer._policy_clip())
                self.assertEqual(json.loads((path/'training_progress.json').read_text())['update'], 1)
                self.assertEqual(json.loads((path/'checkpoints/manifest.json').read_text())['environment_plies'], 8)
                self.assertEqual(resumed.policy_optimizer.state_dict()['state'].keys(), trainer.policy_optimizer.state_dict()['state'].keys())
                resumed.last_checkpoint_environment_plies = 8
                for steps, due in ((9_999_999, False), (10_000_000, True), (10_092_303, True)):
                    resumed.cumulative['environment_plies'] = steps
                    self.assertEqual(resumed._periodic_checkpoint_due(), due)
                resumed.last_checkpoint_environment_plies = 10_092_303
                resumed.cumulative['environment_plies'] = 19_999_999
                self.assertFalse(resumed._periodic_checkpoint_due())
                resumed.cumulative['environment_plies'] = 20_000_000
                self.assertTrue(resumed._periodic_checkpoint_due())
            finally:
                resumed.logger.close()

    def test_local_schedule_enabled_without_altering_bootstrap_or_grpo(self):
        root = Path(__file__).resolve().parents[1]
        local = TrainingSettings.from_yaml(root/'configs/local_4090_training.yaml', 'four_dark')
        self.assertTrue(local.ppo_adaptive_clip)
        self.assertEqual(local.checkpoint_interval_environment_plies, 10_000_000)
        self.assertFalse(TrainingSettings.from_yaml(root/'configs/bootstrap.yaml', 'four_dark').ppo_adaptive_clip)
        self.assertFalse(TrainingSettings.from_yaml(root/'configs/local_4090_training.yaml', 'two_player').ppo_adaptive_clip)
