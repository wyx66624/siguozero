"""Shared entropy feedback, actual gradients, and checkpoint continuity."""
from dataclasses import replace
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from junqi.training.entropy import AdaptiveEntropyCoefficient
from junqi.training.ppo import policy_ppo_loss
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer
from test_ppo import training_settings
from test_source_entropy_bonus import FactorizedPolicy, SourceEntropyBonusTests


def controller(**kwargs):
    return AdaptiveEntropyCoefficient(**dict(initial=.0075, minimum=.005, maximum=.015,
        target_ratio=.45, adaptation_rate=.05, ema_decay=.95, enabled=True,
        deadband=.03, max_relative_change=.02) | kwargs)


def shared_settings(**kwargs):
    return replace(training_settings(), **dict(adaptive_entropy=True, entropy_control_mode='joint_source',
        entropy_feedback='source',
        entropy_coefficient=.0075, source_entropy_coefficient=.0075, layout_entropy_coefficient=.01,
        entropy_minimum=.005, entropy_maximum=.015, entropy_opening_maximum=.015,
        entropy_target_ratio=.45, entropy_opening_target_ratio=.45, entropy_target_schedule=(),
        entropy_deadband=.03, entropy_adaptation_rate=.05, entropy_ema_decay=.95,
        entropy_max_relative_change=.02) | kwargs)


class JointEntropyControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ema_bidirectional_deadband_bounds_and_step_limit(self):
        c = controller()
        c.observe(.4)
        self.assertAlmostEqual(c.coefficient, .0075 * math.exp(.05 * .05))
        old = c.coefficient
        c.observe(.6)
        self.assertAlmostEqual(c.entropy_ema, .41)
        self.assertGreater(c.coefficient, old)  # EMA, not this noisy single sample.
        for ratio in (.42, .45, .48):
            stable = controller()
            stable.observe(ratio)
            self.assertEqual(stable.coefficient, .0075)
        high = controller()
        high.observe(.5)
        self.assertAlmostEqual(high.coefficient, .0075 * math.exp(-.05 * .05))
        for _ in range(1000):
            c.observe(0.)
            high.observe(1.)
        self.assertEqual(c.coefficient, .015)
        self.assertEqual(high.coefficient, .005)
        for ratio in (0., 1.):
            limited = controller(adaptation_rate=1.)
            limited.observe(ratio)
            self.assertAlmostEqual(limited.coefficient / .0075, 1.02 if ratio == 0 else .98)
        saved = c.state_dict()
        c.observe(None)
        self.assertEqual(saved, c.state_dict())

    def test_final_probe_controls_once_and_actual_loss_keeps_two_to_one(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = SelfPlayTrainer(shared_settings(), run_directory=directory)
            try:
                policy = FactorizedPolicy()
                trainer.policy = trainer.policy_parallel = policy
                trainer.policy_optimizer = torch.optim.SGD(policy.parameters(), lr=0.)
                sample = SourceEntropyBonusTests().sample()
                c = trainer.entropy_controllers['other']
                self.assertIs(c, trainer.entropy_controllers['opening'])
                for ratio in (.2, .9):
                    beta, updates = c.coefficient, c.updates
                    with patch.object(trainer, '_final_policy_diagnostics', return_value={
                            'policy/final_probe_samples': 1., 'policy/final_source_entropy_ratio': ratio,
                            'policy/final_entropy_ratio': 1 - ratio,
                            'timing/lr_probe_seconds': 0.}) as probe:
                        metrics = trainer._update_policy([sample])
                    probe.assert_called_once()
                    self.assertEqual(c.updates, updates + 1)
                    expected_ema = .2 if updates == 0 else .95 * .2 + .05 * .9
                    self.assertAlmostEqual(c.entropy_ema, expected_ema)
                    self.assertEqual(metrics['policy/entropy_feedback_ratio'], ratio)
                    self.assertEqual(metrics['policy/source_entropy_coefficient'], beta)
                    self.assertEqual(metrics['policy/opening_entropy_coefficient'], beta)
                    self.assertEqual(metrics['policy/effective_source_entropy_coefficient'], 2 * beta)
                    expected_loss = -2 * beta * metrics['policy/source_entropy'] - beta * metrics['policy/conditional_destination_entropy']
                    self.assertAlmostEqual(metrics['loss/policy_entropy'], expected_loss, places=7)
                    self.assertEqual(metrics['policy/source_entropy_coefficient_next'], c.coefficient)
                before = c.state_dict()
                with patch.object(trainer, '_update_policy_batch', return_value={
                        'optimizer/policy_steps': 3., 'policy/final_probe_samples': 8.,
                        'policy/final_source_entropy_ratio': .8}), \
                        patch.object(trainer, '_final_policy_diagnostics') as probe:
                    trainer._update_policy([sample])
                probe.assert_not_called()
                self.assertEqual(c.updates, before['updates'] + 1)
                before = c.state_dict()
                with patch.object(trainer, '_update_policy_batch', return_value={
                        'optimizer/policy_steps': 1., 'policy/final_probe_samples': 8.,
                        'policy/final_entropy_ratio': .1}):
                    trainer._update_policy([sample])  # All forced moves: no ratio.
                self.assertEqual(before, c.state_dict())
                with patch.object(trainer, '_update_policy_batch', return_value={'optimizer/policy_steps': 0.}), \
                        patch.object(trainer, '_final_policy_diagnostics') as probe:
                    trainer._update_policy([sample])
                probe.assert_not_called()
                self.assertEqual(before, c.state_dict())
            finally:
                trainer.logger.close()

    def test_resume_preserves_ema_beta_and_learning_state(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = shared_settings()
            trainer = SelfPlayTrainer(settings, run_directory=directory)
            trainer.train()
            checkpoint = torch.load(trainer.checkpoints.latest_path, map_location='cpu', weights_only=False)
            saved = trainer.entropy_controllers['other'].state_dict()
            trainer.logger.close()
            resumed = SelfPlayTrainer(settings, run_directory=directory)
            try:
                self.assertIs(resumed.entropy_controllers['opening'], resumed.entropy_controllers['other'])
                self.assertEqual(resumed.entropy_controllers['other'].state_dict(), saved)
                self.assertEqual(resumed.update, checkpoint['update'])
                self.assertEqual(resumed.cumulative, checkpoint['trainer_state']['cumulative'])
                for name in ('policy', 'critic', 'layout', 'reference_layout', 'policy_optimizer',
                             'critic_optimizer', 'layout_optimizer'):
                    torch.testing.assert_close(getattr(resumed, name).state_dict(), checkpoint[name], rtol=0, atol=0)
                self.assertEqual(resumed.pool.state_dict(), checkpoint['trainer_state']['base_game_pool'])
                self.assertEqual(random.getstate(), checkpoint['rng_state']['python'])
                torch.testing.assert_close(torch.get_rng_state(), checkpoint['rng_state']['torch_cpu'], rtol=0, atol=0)
                clone = controller()
                clone.load_state_dict(saved)
                for ratio in (.1, .9, .6):
                    clone.observe(ratio)
                    resumed.entropy_controllers['other'].observe(ratio)
                self.assertEqual(clone.state_dict(), resumed.entropy_controllers['other'].state_dict())
            finally:
                resumed.logger.close()

    def test_fixed_checkpoint_migrates_without_reusing_phase_ema(self):
        with tempfile.TemporaryDirectory() as directory:
            fixed = replace(shared_settings(), adaptive_entropy=False, entropy_control_mode='phase',
                            entropy_feedback='joint')
            trainer = SelfPlayTrainer(fixed, run_directory=directory)
            trainer.train()
            old_update, old_counts = trainer.update, dict(trainer.cumulative)
            trainer.logger.close()
            resumed = SelfPlayTrainer(shared_settings(), run_directory=directory)
            try:
                self.assertEqual(resumed.update, old_update)
                self.assertEqual(resumed.cumulative, old_counts)
                c = resumed.entropy_controllers['other']
                self.assertEqual(c.coefficient, .0075)
                self.assertIsNone(c.entropy_ema)
                self.assertEqual(c.updates, 0)
            finally:
                resumed.logger.close()

    def test_configuration_and_rejection_of_incompatible_modes(self):
        config = Path(__file__).parents[1] / 'configs/expand_mappo_training.yaml'
        settings = TrainingSettings.from_yaml(config, 'two_player')
        self.assertEqual(settings.entropy_control_mode, 'joint_source')
        self.assertEqual(settings.entropy_feedback, 'source')
        self.assertEqual(settings.entropy_target_ratio, .50)
        other = TrainingSettings.from_yaml(config, 'four_dark')
        self.assertFalse(other.adaptive_entropy)
        self.assertEqual(other.entropy_coefficient, .005)
        for changes in ({'entropy_max_relative_change': 1.}, {'entropy_deadband': -1.},
                        {'entropy_deadband': True}, {'entropy_control_mode': 'typo'},
                        {'entropy_feedback': 'typo'}, {'entropy_control_mode': 'phase'},
                        {'source_entropy_coefficient': .01}, {'entropy_opening_target_ratio': .45}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(settings, **changes).validate()

    def test_source_probe_normalizes_each_state_and_excludes_forced_and_frozen(self):
        class KnownPolicy(torch.nn.Module):
            def forward(self, states, actions, **kwargs):
                probabilities = {2: [.1, .9], 3: [.8, .1, .1],
                                 4: [.1, .2, .3, .4], 5: [.2] * 5}
                return [torch.tensor(probabilities[len(row)]).log() for row in actions]

        def entropy(probabilities):
            return -sum(p * math.log(p) for p in probabilities)

        base = SourceEntropyBonusTests().sample()
        choices = (((0, 1), (1, 2), (1, 3)),
                   ((0, 1), (0, 2)),  # Forced source, but destination remains optional.
                   ((0, 1), (1, 2), (2, 3), (3, 4)),
                   ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5)))
        samples = [replace(base, state=replace(base.state, legal_actions=actions), action=actions[0])
                   for actions in choices]
        samples.append(replace(samples[0], learnable=False))
        expected_sum = (entropy([.8, .2]) / math.log(2)
                        + entropy([.1, .2, .3, .4]) / math.log(4)
                        + entropy([.4, .4, .2]) / math.log(3))
        with tempfile.TemporaryDirectory() as directory:
            trainer = SelfPlayTrainer(shared_settings(), run_directory=directory)
            try:
                trainer.policy = KnownPolicy()
                for size in (1, 2, 3, 8):
                    trainer.effective_policy_microbatch = size
                    trainer._kl_probe_cache = None
                    with patch.object(trainer, '_sequence_training_enabled', return_value=False):
                        metrics = trainer._final_policy_diagnostics(samples)
                    self.assertEqual(metrics['policy/final_probe_samples'], 4)
                    self.assertAlmostEqual(metrics['policy/final_source_entropy_samples'], 3, places=6)
                    self.assertAlmostEqual(metrics['policy/final_source_entropy_ratio'], expected_sum / 3, places=6)
                # A second rank has two eligible states averaging .1; weight by
                # their count, not equally by ranks or by microbatches.
                def add_rank(sums):
                    return sums | {'source_ratio': sums['source_ratio'] + .2,
                                   'source_count': sums['source_count'] + 2,
                                   'count': sums['count'] + 2}
                with patch.object(type(trainer.distributed), 'sum_metrics', side_effect=add_rank):
                    metrics = trainer._final_policy_diagnostics(samples)
                self.assertAlmostEqual(metrics['policy/final_source_entropy_ratio'], (expected_sum + .2) / 5, places=6)
                for subset in ([samples[1]], [samples[-1]]):
                    metrics = trainer._final_policy_diagnostics(subset)
                    self.assertEqual(metrics['policy/final_source_entropy_samples'], 0)
                    self.assertNotIn('policy/final_source_entropy_ratio', metrics)
                metrics = policy_ppo_loss(trainer.policy, [samples[-1]], clip_epsilon=.2,
                    entropy_coefficient=0., source_entropy_diagnostics=True).metrics
                self.assertEqual(metrics['policy/source_entropy_ratio_mass'], 0.)
                self.assertEqual(metrics['policy/source_entropy_fraction'], 0.)
            finally:
                trainer.logger.close()

    def test_switching_feedback_preserves_beta_but_resets_incompatible_ema(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = SelfPlayTrainer(shared_settings(entropy_feedback='joint'), run_directory=directory)
            trainer.train()
            path = trainer.checkpoints.latest_path
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            trainer.logger.close()
            # Missing feedback field represents the earlier joint-only checkpoints.
            checkpoint['config'].pop('entropy_feedback')
            for state in checkpoint['trainer_state']['adaptive_entropy'].values():
                state.update(coefficient=.011, entropy_ema=.3, updates=7)
            torch.save(checkpoint, path)
            resumed = SelfPlayTrainer(shared_settings(), run_directory=directory)
            try:
                c = resumed.entropy_controllers['other']
                self.assertEqual(c.coefficient, .011)
                self.assertIsNone(c.entropy_ema)
                self.assertEqual(c.updates, 0)
                self.assertEqual(resumed.update, checkpoint['update'])
                self.assertEqual(resumed.cumulative, checkpoint['trainer_state']['cumulative'])
                for name in ('policy', 'critic', 'layout', 'reference_layout', 'policy_optimizer',
                             'critic_optimizer', 'layout_optimizer'):
                    torch.testing.assert_close(getattr(resumed, name).state_dict(), checkpoint[name], rtol=0, atol=0)
                self.assertEqual(resumed.pool.state_dict(), checkpoint['trainer_state']['base_game_pool'])
                self.assertEqual(random.getstate(), checkpoint['rng_state']['python'])
                torch.testing.assert_close(torch.get_rng_state(), checkpoint['rng_state']['torch_cpu'], rtol=0, atol=0)
                c.observe(.7)
                self.assertEqual(c.entropy_ema, .7)
                self.assertLess(c.coefficient, .011)
            finally:
                resumed.logger.close()
