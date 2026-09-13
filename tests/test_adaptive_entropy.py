from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from junqi.training.entropy import AdaptiveEntropyCoefficient, phase_entropy_ratios, policy_entropy_bonus
from junqi.training.ppo import PPOSample, policy_ppo_loss
from junqi.training.losses import policy_grpo_loss
from junqi.training.rollout import PolicyGroup
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer
from test_ppo import states, training_settings


def controller(**options):
    return AdaptiveEntropyCoefficient(**{**dict(initial=.01, minimum=.005, maximum=.02,
        target_ratio=.6, adaptation_rate=.1, ema_decay=.9), **options})


class TablePolicy(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, states, actions):
        return [self.logits[i, :len(state.legal_actions)].log_softmax(0) for i, state in enumerate(states)]


class AdaptiveEntropyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_feedback_direction_bounds_missing_observations_and_resume(self):
        c = controller()
        c.observe(.2)
        self.assertGreater(c.coefficient, .01)
        previous = c.state_dict()
        c.observe(None)
        self.assertEqual(c.state_dict(), previous)
        clone = controller()
        clone.load_state_dict(c.state_dict())
        for _ in range(120):
            c.observe(.1)
            clone.observe(.1)
        self.assertEqual(c.state_dict(), clone.state_dict())
        self.assertEqual(c.coefficient, .02)
        for _ in range(200):
            c.observe(.95)
        self.assertEqual(c.coefficient, .005)
        disabled = controller(enabled=False)
        disabled.load_state_dict(clone.state_dict())
        disabled.observe(.1)
        self.assertEqual(disabled.coefficient, .01)
        with self.assertRaisesRegex(ValueError, "finite"):
            c.observe(float("nan"))
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            clone.load_state_dict({**clone.state_dict(), "coefficient": float("nan")})

    def test_normalization_for_binary_multiaction_and_forced_moves(self):
        state = states(1)[0]
        batch = [replace(state, legal_actions=state.legal_actions[:n]) for n in (2, 4, 1)]
        hs = torch.tensor([math.log(2), math.log(4), 0.], requires_grad=True)
        _, values = policy_entropy_bonus(batch, hs, coefficient=.01, opening_coefficient=.02, opening_plies=16)
        metrics = {k: float(v) for k, v in values.items()}
        ratios = phase_entropy_ratios(metrics)
        self.assertAlmostEqual(ratios["opening"], 1.)
        self.assertIsNone(ratios["other"])
        self.assertAlmostEqual(metrics["policy/opening_entropy_fraction"], 2 / 3)

    def test_moments_combine_unequal_microbatches_before_phase_division(self):
        s = states(1)[0]
        batch = [replace(s, legal_actions=s.legal_actions[:2], records=s.records * length)
                 for length in (1, 2, 20, 25, 27)]
        entropies = torch.tensor([.1, .2, .3, .4, .5])
        opts = dict(coefficient=.01, opening_coefficient=.02, opening_plies=16)
        full_bonus, full_stats = policy_entropy_bonus(batch, entropies, **opts)
        combined, combined_bonus = {}, 0
        for start, stop in ((0, 1), (1, 4), (4, 5)):
            bonus, stats = policy_entropy_bonus(batch[start:stop], entropies[start:stop], **opts)
            weight = (stop - start) / len(batch)
            combined_bonus += float(bonus) * weight
            for key, value in stats.items():
                combined[key] = combined.get(key, 0.) + float(value) * weight
        self.assertAlmostEqual(combined_bonus, float(full_bonus), places=8)
        for key, value in full_stats.items():
            self.assertAlmostEqual(combined[key], float(value), places=7)
        self.assertAlmostEqual(phase_entropy_ratios(combined)["opening"], .15 / math.log(2), places=6)

    def test_ppo_entropy_gradient_increases_alternative_action_probability(self):
        s = replace(states(1)[0], legal_actions=((0, 1), (1, 2)))
        policy = TablePolicy([[5., 0.]])
        sample = PPOSample(s, (0, 1), 0., 0., 0., 0., 0)
        optimizer = torch.optim.SGD(policy.parameters(), lr=5.)
        before = float(policy.logits.softmax(-1)[0, 1].detach())
        output = policy_ppo_loss(policy, [sample], clip_epsilon=.2, entropy_coefficient=.01,
                                 opening_entropy_coefficient=.02, entropy_opening_plies=16)
        self.assertEqual(output.metrics["loss/policy_ppo"], 0.)
        self.assertLess(output.metrics["loss/policy_entropy"], 0.)
        output.loss.backward()
        optimizer.step()
        self.assertGreater(float(policy.logits.softmax(-1)[0, 1].detach()), before)

    def test_phase_coefficients_apply_to_gradient_and_frozen_seats_are_excluded(self):
        s = replace(states(1)[0], legal_actions=((0, 1), (1, 2)))
        later = replace(s, records=s.records * 20)
        policy = TablePolicy([[4., 0.], [4., 0.]])
        samples = [PPOSample(x, (0, 1), 0., 0., 0., 0., 0) for x in (s, later)]
        out = policy_ppo_loss(policy, samples, clip_epsilon=.2, entropy_coefficient=.005,
                              opening_entropy_coefficient=.02, entropy_opening_plies=16)
        out.loss.backward()
        self.assertAlmostEqual(float(policy.logits.grad[0, 0] / policy.logits.grad[1, 0]), 4., places=5)
        policy.zero_grad(set_to_none=True)
        out = policy_ppo_loss(policy, [replace(samples[0], learnable=False)], clip_epsilon=.2,
                              entropy_coefficient=.005, opening_entropy_coefficient=.02,
                              entropy_opening_plies=16)
        out.loss.backward()
        self.assertEqual(float(policy.logits.grad.abs().sum()), 0.)
        self.assertTrue(all(value == 0 for value in out.metrics.values()))

    def test_disabled_entropy_preserves_fixed_loss(self):
        s = replace(states(1)[0], legal_actions=((0, 1), (1, 2)))
        policy = TablePolicy([[4., 0.]])
        sample = PPOSample(s, (0, 1), 0., 0., 0., 0., 0)
        output = policy_ppo_loss(policy, [sample], clip_epsilon=.2, entropy_coefficient=.01)
        logs = policy.logits.log_softmax(-1)
        expected = .01 * (logs.exp() * logs).sum()
        self.assertTrue(torch.equal(output.loss, expected))

    def test_trainer_updates_once_and_restores_controller_with_old_checkpoint_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = training_settings(ppo_minibatch_samples=2, policy_epochs=2)
            trainer = SelfPlayTrainer(settings, run_directory=directory)
            trainer.train()
            self.assertEqual(trainer.entropy_controllers["opening"].updates, 1)
            self.assertEqual(trainer.entropy_controllers["other"].updates, 0)
            path = trainer.checkpoints.latest_path
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertIn("adaptive_entropy", saved["trainer_state"])
            metrics = json.loads((trainer.run_directory / "latest_metrics.json").read_text())
            self.assertIn("policy/opening_entropy_ratio", metrics)
            self.assertIn("loss/policy_entropy", metrics)
            self.assertEqual(metrics["policy/opening_entropy_coefficient"], .01)
            resumed = SelfPlayTrainer(settings, run_directory=directory)
            try:
                self.assertEqual(resumed.entropy_controllers["opening"].state_dict(),
                                 trainer.entropy_controllers["opening"].state_dict())
            finally:
                resumed.logger.close()
            saved["trainer_state"].pop("adaptive_entropy")
            torch.save(saved, path)
            legacy = SelfPlayTrainer(settings, run_directory=directory)
            try:
                self.assertEqual(legacy.entropy_controllers["opening"].coefficient, .01)
                self.assertEqual(legacy.entropy_controllers["opening"].updates, 0)
            finally:
                legacy.logger.close()

    def test_configuration_rejects_invalid_controller_settings(self):
        valid = training_settings()
        for values in ({"adaptive_entropy": "true"}, {"entropy_coefficient": 0.},
                       {"entropy_target_ratio": 1.1}, {"entropy_ema_decay": 1.},
                       {"entropy_adaptation_rate": float("nan")}, {"entropy_opening_plies": True},
                       {"entropy_minimum": .03}):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "entropy"):
                replace(valid, **values).validate()
        replace(valid, adaptive_entropy=False, entropy_coefficient=0.).validate()

    def test_opening_ceiling_is_independent_and_old_configs_inherit_shared_limit(self):
        config = Path(__file__).parents[1] / 'configs/bootstrap.yaml'
        legacy = TrainingSettings.from_yaml(config, 'four_dark', tiny=True)
        self.assertIsNone(legacy.entropy_opening_maximum)
        local = TrainingSettings.from_yaml(config.parent / 'local_4090_training.yaml', 'four_dark', tiny=True)
        self.assertEqual(local.entropy_opening_maximum, .04)
        self.assertEqual(local.entropy_maximum, .02)
        for maximum in (True, '0.04', float('nan'), float('inf'), 0., .004, .009):
            with self.subTest(maximum=maximum), self.assertRaisesRegex(ValueError, 'entropy'):
                replace(training_settings(), entropy_opening_maximum=maximum).validate()

    def test_opening_ceiling_resume_preserves_state_and_increases_actual_opening_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = training_settings()
            trainer = SelfPlayTrainer(settings, run_directory=directory)
            trainer.train()
            opening = trainer.entropy_controllers['opening']
            opening.coefficient, opening.entropy_ema = .02, .36
            old_state = opening.state_dict()
            other_state = trainer.entropy_controllers['other'].state_dict()
            trainer.save_checkpoint(reason='opening-ceiling-test', archive=False)
            weights = {key: value.clone() for key, value in trainer.policy.state_dict().items()}
            trainer.logger.close()
            resumed = SelfPlayTrainer(replace(settings, entropy_opening_maximum=.04), run_directory=directory)
            try:
                self.assertEqual(resumed.update, trainer.update)
                self.assertEqual(resumed.cumulative, trainer.cumulative)
                self.assertEqual(resumed.entropy_controllers['opening'].state_dict(), old_state)
                self.assertEqual(resumed.entropy_controllers['other'].state_dict(), other_state)
                self.assertEqual(resumed.entropy_controllers['opening'].maximum, .04)
                self.assertEqual(resumed.entropy_controllers['other'].maximum, .02)
                for key, value in resumed.policy.state_dict().items():
                    torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
                resumed.entropy_controllers['opening'].observe(.36)
                self.assertGreater(resumed.entropy_controllers['opening'].coefficient, .02)
                for _ in range(100):
                    resumed.entropy_controllers['opening'].observe(.36)
                self.assertEqual(resumed.entropy_controllers['opening'].coefficient, .04)
            finally:
                resumed.logger.close()
        s = replace(states(1)[0], legal_actions=((0, 1), (1, 2)))
        later = replace(s, records=s.records * 20)
        samples = [PPOSample(x, (0, 1), 0., 0., 0., 0., 0) for x in (s, later)]
        grads = []
        for opening_max in (.02, .04):
            policy = TablePolicy([[4., 0.], [4., 0.]])
            loss = policy_ppo_loss(policy, samples, clip_epsilon=.2, entropy_coefficient=.012,
                                  opening_entropy_coefficient=opening_max, entropy_opening_plies=16)
            loss.loss.backward()
            grads.append(policy.logits.grad.clone())
        torch.testing.assert_close(grads[1][0], grads[0][0] * 2)
        torch.testing.assert_close(grads[1][1], grads[0][1], rtol=0, atol=0)

    def test_grpo_uses_the_same_phase_entropy_bonus(self):
        s = replace(states(1)[0], legal_actions=((0, 1), (1, 2)))
        class Reference(TablePolicy):
            def log_probs_for_action_groups(self, states, actions):
                return self(states, actions)
        policy, reference = TablePolicy([[5., 0.]]), Reference([[5., 0.]])
        group = PolicyGroup(s, ((0, 1),) * 4, (0.,) * 4, ((0., 0.),) * 4,
                            (0.,) * 4, (0.,) * 4, 8, 0)
        out = policy_grpo_loss(policy, reference, [group], clip_epsilon=.2,
            kl_coefficient=.02, entropy_coefficient=.005,
            opening_entropy_coefficient=.02, entropy_opening_plies=16)
        self.assertAlmostEqual(float(out.loss.detach()), -.02 * out.metrics["policy/entropy"], places=7)
        out.loss.backward()
        self.assertLess(float(policy.logits.grad[0, 1]), 0.)

    def test_saturated_history_is_not_reclassified_as_opening(self):
        from junqi.training.encoding import GameHistory
        from junqi.training.modes import new_game
        game = new_game("four_dark", seed=12)
        history = GameHistory.initialize(game, "four_dark", max_transitions=3)
        for _ in range(5):
            game.step(game.legal_actions()[0])
            history.append_after_step(game)
        state = history.state_for(game)
        _, stats = policy_entropy_bonus([state], torch.tensor([1.]), coefficient=.01,
            opening_coefficient=.02, opening_plies=3)
        self.assertEqual(float(stats["policy/opening_entropy_fraction"]), 0.)
        self.assertEqual(float(stats["policy/other_entropy_fraction"]), 1.)


if __name__ == "__main__":
    unittest.main()
