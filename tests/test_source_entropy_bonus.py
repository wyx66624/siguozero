"""Exact source exploration, padding/gradient parity, and safe coefficient resume."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from junqi.training.learner_graph import POLICY_NAMES, PackedLearnerGraph
from junqi.training.models import GamePolicyTransformer, ModelConfig, PolicyFeatures
from junqi.training.modes import TrainingMode
from junqi.training.ppo import PPOSample, policy_ppo_loss
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer
import test_learner_graph as graph_tests
from test_ppo import states, training_settings


class FactorizedPolicy(torch.nn.Module):
    """Two sources with unequal destination counts and independent logits."""
    def __init__(self):
        super().__init__()
        self.source = torch.nn.Parameter(torch.tensor([3., 0.]))
        self.destination = torch.nn.Parameter(torch.tensor([1., 0., -1.]))

    def forward(self, states, actions, **kwargs):
        left = self.source.log_softmax(0)
        right = self.destination.log_softmax(0)
        joint = torch.cat((left[0:1], left[1] + right))
        return [joint for _ in states]


class SourceEntropyBonusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sample(self):
        state = replace(states(1)[0], legal_actions=((0, 1), (1, 2), (1, 3), (1, 4)))
        return PPOSample(state, (0, 1), 0., 0., 0., 0., 0)

    def test_bonus_identity_and_separate_gradient_effects(self):
        policy = FactorizedPolicy()
        sample = self.sample()
        logs = policy.source.log_softmax(0)
        destinations = policy.destination.log_softmax(0)
        source_entropy = -(logs.exp() * logs).sum()
        conditional_entropy = logs.exp()[1] * -(destinations.exp() * destinations).sum()
        output = policy_ppo_loss(policy, [sample], clip_epsilon=.2,
                                entropy_coefficient=.01, source_entropy_coefficient=.005)
        expected = -.015 * source_entropy - .01 * conditional_entropy
        torch.testing.assert_close(output.loss, expected)
        grads = torch.autograd.grad(output.loss, tuple(policy.parameters()), retain_graph=True)
        wanted = torch.autograd.grad(expected, tuple(policy.parameters()), retain_graph=True)
        for actual, target in zip(grads, wanted, strict=True):
            torch.testing.assert_close(actual, target)
        baseline = policy_ppo_loss(policy, [sample], clip_epsilon=.2, entropy_coefficient=.01)
        baseline_grads = torch.autograd.grad(baseline.loss, tuple(policy.parameters()))
        # Extra marginal entropy raises the rare source; no extra gradient in
        # its independent destination logits, even with unequal branch widths.
        torch.testing.assert_close(grads[1], baseline_grads[1], atol=1e-9, rtol=1e-5)
        self.assertLess(float(grads[0][1] - baseline_grads[0][1]), 0.)
        self.assertAlmostEqual(output.metrics['policy/source_entropy'], float(source_entropy.detach()), places=6)
        self.assertAlmostEqual(output.metrics['policy/conditional_destination_entropy'],
                               float(conditional_entropy.detach()), places=6)
        self.assertAlmostEqual(output.metrics['loss/policy_entropy'],
                               output.metrics['loss/policy_joint_entropy'] +
                               output.metrics['loss/policy_source_entropy'], places=7)

    def test_frozen_seat_receives_no_added_entropy_gradient(self):
        policy = FactorizedPolicy()
        output = policy_ppo_loss(policy, [replace(self.sample(), learnable=False)],
                                clip_epsilon=.2, entropy_coefficient=.01, source_entropy_coefficient=.005)
        output.loss.backward()
        for parameter in policy.parameters():
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))
        self.assertTrue(all(value == 0 for value in output.metrics.values()))

    def test_requested_split_matches_loss_and_both_head_gradients(self):
        config = Path(__file__).parents[1] / 'configs/expand_mappo_training.yaml'
        settings = TrainingSettings.from_yaml(config, TrainingMode.FOUR_DARK)
        policy = FactorizedPolicy()
        source_logs = policy.source.log_softmax(0)
        destination_logs = policy.destination.log_softmax(0)
        source_entropy = -(source_logs.exp() * source_logs).sum()
        # The current source probability remains differentiable in E_s[H_d].
        destination_entropy = source_logs.exp()[1] * -(destination_logs.exp() * destination_logs).sum()
        expected = -.01 * source_entropy - .005 * destination_entropy
        actual = policy_ppo_loss(policy, [self.sample()], clip_epsilon=.2,
                                entropy_coefficient=settings.entropy_coefficient,
                                source_entropy_coefficient=settings.source_entropy_coefficient)
        torch.testing.assert_close(actual.loss, expected)
        gradients = torch.autograd.grad(actual.loss, tuple(policy.parameters()), retain_graph=True)
        expected_gradients = torch.autograd.grad(expected, tuple(policy.parameters()))
        for gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
            torch.testing.assert_close(gradient, expected_gradient)

    def test_layout_entropy_can_remain_fixed_when_movement_entropy_changes(self):
        from junqi.training.rollout import LayoutOutcome
        for configured, expected in ((None, .005), (.01, .01)):
            with self.subTest(layout_coefficient=configured), tempfile.TemporaryDirectory() as directory:
                settings = replace(training_settings(), entropy_coefficient=.005,
                                   layout_entropy_coefficient=configured)
                trainer = SelfPlayTrainer(settings, run_directory=directory)
                try:
                    with torch.no_grad():
                        samples = trainer.layout.sample_layouts(2, settings.mode)
                    trainer.layout_buffer.extend(LayoutOutcome(sample, 0., seat, 0)
                                                 for seat, sample in enumerate(samples))
                    metrics = trainer._update_layout()
                    self.assertEqual(metrics['optimizer/layout_steps'], 1.)
                    self.assertEqual(metrics['layout/entropy_coefficient'], expected)
                    self.assertAlmostEqual(metrics['loss/layout_grpo'], 0., places=7)
                    self.assertAlmostEqual(metrics['loss/layout_total'],
                        settings.kl_coefficient * metrics['layout/sampled_trust_region']
                        - expected * metrics['layout/entropy'], places=6)
                finally:
                    trainer.logger.close()

    def test_graph_and_eager_match_with_forced_source_padding_and_nonzero_bonus(self):
        fixture = graph_tests.LearnerGraphTests()
        samples = fixture.batch()
        # A single source can still have several destinations: H(source)=0.
        state = replace(samples[1].state, legal_actions=((1, 2), (1, 3)))
        samples[1] = replace(samples[1], state=state, action=(1, 2))
        torch.manual_seed(271)
        model = GamePolicyTransformer(ModelConfig.tiny())
        reference = copy.deepcopy(model)
        contexts = torch.randn(8, model.config.temporal_dim)
        features = PolicyFeatures(contexts[:3], torch.ones(3, 129, dtype=torch.bool))
        graph = fixture.graph(model, False)
        graph.callable = lambda *args: contexts
        self.assertTrue(graph.stage(model, samples, clip=.2, coefficient=.01,
                                   source_coefficient=.005, opening_coefficient=.01, opening_plies=16))
        loss, metrics = graph._loss(model)
        with patch.object(reference, '_encode_full', return_value=features):
            output = policy_ppo_loss(reference, samples, clip_epsilon=.2, entropy_coefficient=.01,
                                    source_entropy_coefficient=.005, opening_entropy_coefficient=.01,
                                    entropy_opening_plies=16, sequence_training=True, defer_metrics=True)
            _, _, source_entropies = reference.ppo_statistics(
                [s.state for s in samples], [s.action for s in samples],
                pack_sequences=True, include_source_entropy=True)
        torch.testing.assert_close(source_entropies[:2], torch.zeros(2), rtol=0, atol=0)
        torch.testing.assert_close(loss, output.loss, rtol=2e-6, atol=2e-7)
        self.assertEqual(tuple(output.metrics), POLICY_NAMES)
        torch.testing.assert_close(metrics, torch.stack(list(output.metrics.values())), rtol=2e-6, atol=2e-7)
        loss.backward()
        output.loss.backward()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            self.assertEqual(actual.grad is None, expected.grad is None)
            if expected.grad is not None:
                self.assertTrue(torch.isfinite(actual.grad).all())
                torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=3e-7)

    def test_resume_adopts_bonus_without_resetting_learning_state(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = training_settings()
            trainer = SelfPlayTrainer(settings, run_directory=directory)
            trainer.train()
            saved = torch.load(trainer.checkpoints.latest_path, map_location='cpu', weights_only=False)
            trainer.logger.close()
            enabled = replace(settings, adaptive_entropy=False, entropy_coefficient=.005,
                              entropy_minimum=.005, entropy_maximum=.005, entropy_opening_maximum=.005,
                              source_entropy_coefficient=.005)
            resumed = SelfPlayTrainer(enabled, run_directory=directory)
            try:
                self.assertEqual(resumed.update, saved['update'])
                self.assertEqual(resumed.cumulative, saved['trainer_state']['cumulative'])
                for name in ('policy', 'critic', 'layout', 'reference_layout',
                             'policy_optimizer', 'critic_optimizer', 'layout_optimizer'):
                    torch.testing.assert_close(getattr(resumed, name).state_dict(), saved[name], rtol=0, atol=0)
                self.assertEqual(random.getstate(), saved['rng_state']['python'])
                torch.testing.assert_close(torch.get_rng_state(), saved['rng_state']['torch_cpu'], rtol=0, atol=0)
                # Compare serialized pool contents, including unfinished games and its RNG.
                self.assertEqual(resumed.pool.state_dict(), saved['trainer_state']['base_game_pool'])
                resolved = json.loads((resumed.run_directory / 'resolved_config.json').read_text())
                self.assertEqual(resolved['source_entropy_coefficient'], .005)
                self.assertEqual(resolved['entropy_coefficient'], .005)
                resumed.policy = FactorizedPolicy()
                resumed.policy_parallel = resumed.policy
                resumed.policy_optimizer = torch.optim.SGD(resumed.policy.parameters(), lr=0.)
                metrics = resumed._update_policy([self.sample()])
                self.assertEqual(metrics['policy/source_entropy_coefficient'], .005)
                self.assertAlmostEqual(metrics['loss/policy_total'],
                                       -.005 * metrics['policy/entropy'] - .005 * metrics['policy/source_entropy'], places=7)
            finally:
                resumed.logger.close()

    def test_only_expand_four_dark_enables_extra_entropy_and_validation(self):
        config = Path(__file__).parents[1] / 'configs/expand_mappo_training.yaml'
        settings = TrainingSettings.from_yaml(config, TrainingMode.FOUR_DARK)
        self.assertEqual(settings.entropy_coefficient, .005)
        self.assertEqual(settings.entropy_minimum, .005)
        self.assertEqual(settings.entropy_maximum, .005)
        self.assertEqual(settings.entropy_opening_maximum, .005)
        self.assertEqual(settings.source_entropy_coefficient, .005)
        self.assertEqual(settings.layout_entropy_coefficient, .01)
        self.assertFalse(settings.adaptive_entropy)
        self.assertEqual(training_settings().source_entropy_coefficient, 0.)
        self.assertIsNone(training_settings().layout_entropy_coefficient)
        for value in (-.005, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(settings, source_entropy_coefficient=value).validate()
            with self.subTest(layout=value), self.assertRaises(ValueError):
                replace(settings, layout_entropy_coefficient=value).validate()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA capture verification')
    def test_cuda_replay_reads_new_coefficient_with_finite_gradients(self):
        model = GamePolicyTransformer(ModelConfig.tiny()).cuda()
        graph = PackedLearnerGraph(model, critic=False, amp_dtype=torch.bfloat16, tokens=64, sources=256)
        graph.SEQUENCES, graph.SAMPLES, graph.ACTIONS = 8, 8, 2048
        samples = graph_tests.LearnerGraphTests().batch()
        contexts = torch.randn(8, model.config.temporal_dim, device='cuda')
        options = dict(clip=.2, coefficient=.005, opening_coefficient=.005, opening_plies=16)
        # Real CUDA capture/replay of the action heads, entropy and backward;
        # fixed temporal contexts keep this independent of compiler warmup.
        with patch.object(model, '_packed_temporal_forward', lambda *args: contexts):
            self.assertTrue(graph.stage(model, samples, source_coefficient=0., **options))
            before = {k: float(v) for k, v in graph.run(model).items()}
            self.assertTrue(graph.stage(model, samples, source_coefficient=.005, **options))
            after = {k: float(v) for k, v in graph.run(model).items()}
        self.assertEqual(graph.captures, 1)
        self.assertEqual(graph.replays, 2)
        self.assertAlmostEqual(after['loss/policy_total'] - before['loss/policy_total'],
                               -.005 * after['policy/source_entropy'], places=6)
        self.assertGreater(after['policy/source_entropy'], 0.)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
