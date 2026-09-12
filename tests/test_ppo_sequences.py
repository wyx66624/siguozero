from __future__ import annotations

import copy
from dataclasses import replace
import random
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("PyTorch training extra is not installed") from error

from junqi.training.encoding import GameHistory, history_prefix_groups
from junqi.training.models import GamePolicyTransformer, GameValueTransformer, ModelConfig
from junqi.training.modes import TrainingMode, new_game
from junqi.training.ppo import PPOSample, critic_ppo_loss, policy_ppo_loss, sequence_training_batches
from junqi.training.settings import TrainingSettings, equivalent_ppo_decisions
from junqi.training.trainer import SelfPlayTrainer

CONFIG = Path(__file__).parents[1] / "configs/bootstrap.yaml"


def trajectory_states(mode, *, seed=41, max_transitions=32, dead_rules=True):
    game = new_game(mode, seed=seed, max_plies=100, dead_rules_enabled=dead_rules)
    history = GameHistory.initialize(game, mode, max_transitions=max_transitions)
    rng = random.Random(seed)
    states = []
    for _ in range(25):
        if game.is_terminal:
            break
        if game.current_player == 0:
            states.append(history.state_for(game))
        game.step(rng.choice(game.legal_actions()))
        history.append_after_step(game)
    return states


class SequencePPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_outputs_and_gradients_for_policy_and_critic(self):
        for mode in (TrainingMode.FOUR_DARK, TrainingMode.DOUBLE_OPEN):
            for dead_rules in (False, True):
                with self.subTest(mode=mode, dead_rules=dead_rules):
                    batch = (trajectory_states(mode, dead_rules=dead_rules)
                             + trajectory_states(mode, seed=42, dead_rules=dead_rules)[:3])
                    batch = [*reversed(batch), batch[1]]
                    config = replace(ModelConfig.tiny(dead_rules_enabled=dead_rules),
                                     max_transitions=32, activation_checkpointing=True)
                    torch.manual_seed(42)
                    for kind in (GamePolicyTransformer, GameValueTransformer):
                        original = kind(config).train()
                        if kind is GameValueTransformer:
                            # A zero value head would mask backbone gradient errors.
                            torch.nn.init.normal_(original.value_head.weight, std=.03)
                        packed = copy.deepcopy(original)
                        with torch.no_grad():
                            if kind is GamePolicyTransformer:
                                old = original(batch, [state.legal_actions for state in batch])
                                old_logs = [float(row[0]) for row in old]
                                outputs = packed(batch, [state.legal_actions for state in batch], pack_sequences=True)
                                for left, right in zip(old, outputs, strict=True):
                                    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)
                            else:
                                old_logs = [-1.] * len(batch)
                                torch.testing.assert_close(original(batch), packed(batch, pack_sequences=True),
                                                           atol=2e-6, rtol=2e-6)
                        samples = [PPOSample(state, state.legal_actions[0], log, 0,
                                             .8 if i % 2 else -.4, .3 if i % 2 else -.2, 0)
                                   for i, (state, log) in enumerate(zip(batch, old_logs, strict=True))]
                        loss_function = policy_ppo_loss if kind is GamePolicyTransformer else critic_ppo_loss
                        options = ({"entropy_coefficient": .01} if kind is GamePolicyTransformer
                                   else {"value_coefficient": .5})
                        left = loss_function(original, samples, clip_epsilon=.2, **options).loss
                        right = loss_function(packed, samples, clip_epsilon=.2,
                                              sequence_training=True, **options).loss
                        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)
                        left.backward()
                        right.backward()
                        for (name, parameter), (_, other) in zip(
                            original.named_parameters(), packed.named_parameters(), strict=True
                        ):
                            self.assertEqual(parameter.grad is None, other.grad is None, name)
                            if parameter.grad is not None:
                                torch.testing.assert_close(parameter.grad, other.grad, atol=3e-6, rtol=2e-4,
                                                           msg=lambda message: f"{kind.__name__}.{name}: {message}")

    def test_divergent_public_history_and_sliding_windows_are_not_merged(self):
        batch = trajectory_states(TrainingMode.FOUR_DARK)
        longest = batch[-1]
        records = list(longest.records)
        records[5] = replace(records[5], no_interaction_plies=records[5].no_interaction_plies + 1)
        divergent = replace(longest, records=tuple(records))
        rolled = replace(longest, records=(longest.records[0], *longest.records[2:]))
        candidates = [batch[1], longest, divergent, rolled, batch[0], longest]
        groups = history_prefix_groups(candidates)
        self.assertEqual(len(groups), 3)
        self.assertEqual(sorted(index for group in groups for index in group), list(range(6)))
        for group in groups:
            parent = candidates[group[0]]
            for index in group:
                state = candidates[index]
                self.assertEqual(parent.records[:len(state.records)], state.records)
        model = GamePolicyTransformer(replace(ModelConfig.tiny(), max_transitions=32)).eval()
        with torch.no_grad():
            independent = model(candidates, [state.legal_actions for state in candidates])
            packed = model(candidates, [state.legal_actions for state in candidates], pack_sequences=True)
        for left, right in zip(independent, packed, strict=True):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)

    def test_sequence_batches_cover_every_decision_once_and_preserve_weighted_gradient(self):
        batch = trajectory_states(TrainingMode.FOUR_DARK) + trajectory_states(TrainingMode.FOUR_DARK, seed=43)
        config = replace(ModelConfig.tiny(), max_transitions=32)
        original = GameValueTransformer(config)
        torch.nn.init.normal_(original.value_head.weight, std=.03)
        packed = copy.deepcopy(original)
        samples = [PPOSample(state, state.legal_actions[0], -1, 0, 1, i / 100, 0)
                   for i, state in enumerate(batch)]
        chunks = sequence_training_batches(samples, sequences_per_batch=2, max_samples_per_sequence=3)
        self.assertCountEqual([id(item) for chunk in chunks for item in chunk], [id(item) for item in samples])
        self.assertTrue(all(len(chunk) <= 6 for chunk in chunks))
        critic_ppo_loss(original, samples, clip_epsilon=.2, value_coefficient=.5).loss.backward()
        for chunk in chunks:
            output = critic_ppo_loss(packed, chunk, clip_epsilon=.2, value_coefficient=.5,
                                     sequence_training=True)
            (output.loss * len(chunk) / len(samples)).backward()
        for left, right in zip(original.parameters(), packed.parameters(), strict=True):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, atol=3e-6, rtol=2e-4)

    def test_destination_distribution_reuse_matches_unshared_action_head(self):
        batch = trajectory_states(TrainingMode.FOUR_DARK)[:2]
        model = GamePolicyTransformer(replace(ModelConfig.tiny(), max_transitions=32)).eval()
        actions = [[*state.legal_actions, state.legal_actions[0]] for state in batch]
        with torch.no_grad():
            features = model.encode(batch)
            sources = model._source_log_probs(features, batch, 1.)
            actual = model(batch, actions)
            for row, group in enumerate(actions):
                origin = torch.tensor([action[0] for action in group])
                target = torch.tensor([action[1] for action in group])
                destinations = model._destination_log_probs(
                    features, batch, torch.full_like(origin, row), origin, 1.)
                expected = sources[row, origin] + destinations[torch.arange(len(group)), target]
                torch.testing.assert_close(actual[row], expected)

    def test_larger_rollout_keeps_multiple_optimizer_steps_and_exact_last_batch(self):
        settings = TrainingSettings.from_yaml(CONFIG, TrainingMode.FOUR_DARK, tiny=True, overrides={
            "device": "cpu", "anchor_batch": 8, "policy_microbatch": 2,
            "ppo_minibatch_samples": 3, "policy_epochs": 2, "critic_epochs": 2,
            "total_updates": 3, "target_environment_plies": 13, "arena_enabled": False,
            "early_stop_kl_multiple": 1e8, "early_stop_clip_fraction": 1.,
        })
        with tempfile.TemporaryDirectory() as directory:
            trainer = SelfPlayTrainer(settings, run_directory=directory)
            with patch.object(trainer.policy_optimizer, "step", wraps=trainer.policy_optimizer.step) as policy_step, \
                 patch.object(trainer.critic_optimizer, "step", wraps=trainer.critic_optimizer.step) as critic_step:
                trainer.train()
            self.assertEqual(trainer.cumulative["environment_plies"], 13)
            self.assertEqual(trainer.update, 2)
            # ceil(8/3)*2 epochs + ceil(5/3)*2 epochs
            self.assertEqual(policy_step.call_count, 10)
            self.assertEqual(critic_step.call_count, 10)

    def test_old_grpo_budget_is_planning_metadata_with_real_ppo_target(self):
        self.assertEqual(equivalent_ppo_decisions(3_000_000_000, 334.5), 1_121_077)
        settings = TrainingSettings.from_yaml(CONFIG, TrainingMode.FOUR_DARK, overrides={
            "anchor_batch": 4096, "grpo_equivalent_plies": 3_000_000_000,
        })
        self.assertEqual(settings.target_environment_plies, 1_121_077)
        self.assertEqual(settings.total_updates, 274)
        self.assertEqual(settings.warmup_updates, 63)
        self.assertEqual(settings.grpo_equivalent_plies, 3_000_000_000)
        with self.assertRaisesRegex(ValueError, "choose an environment"):
            TrainingSettings.from_yaml(CONFIG, TrainingMode.FOUR_DARK, overrides={
                "grpo_equivalent_plies": 30, "target_environment_plies": 30,
            })
        with self.assertRaises(ValueError):
            equivalent_ppo_decisions(30, float("nan"))

    def test_four_player_real_budget_extends_stop_and_lr_horizon(self):
        for mode in (TrainingMode.FOUR_DARK, TrainingMode.DOUBLE_OPEN):
            with self.subTest(mode=mode):
                settings = TrainingSettings.from_yaml(CONFIG, mode, model_scale="main")
                self.assertEqual(settings.target_environment_plies, 3_000_000_000)
                self.assertIsNone(settings.grpo_equivalent_plies)
                self.assertEqual(settings.total_updates, 244_141)
                self.assertEqual(settings.warmup_updates, 2000)
                self.assertEqual(3_000_000_000 - 244_140 * settings.anchor_batch, 7680)
                schedule = SimpleNamespace(settings=settings, policy_lr_scale=1.0)
                # The former 200K limit must neither stop training nor exhaust LR.
                self.assertGreater(SelfPlayTrainer._learning_rate(schedule, 200_000),
                                   2 * settings.minimum_learning_rate)
                self.assertAlmostEqual(SelfPlayTrainer._learning_rate(schedule, settings.total_updates),
                                       settings.minimum_learning_rate)
        two_player = TrainingSettings.from_yaml(CONFIG, TrainingMode.TWO_PLAYER)
        self.assertIsNone(two_player.target_environment_plies)
        self.assertEqual(two_player.total_updates, 200_000)

    def test_explicit_real_target_derives_updates_and_collects_exact_tail(self):
        settings = TrainingSettings.from_yaml(CONFIG, TrainingMode.FOUR_DARK, overrides={
            "target_environment_plies": 13, "anchor_batch": 8, "policy_microbatch": 2,
            "device": "cpu", "base_game_pool_size": 2, "max_game_plies": 4,
            "policy_epochs": 1, "critic_epochs": 1, "arena_enabled": False,
            "warmup_updates": 1, "layout_update_interval": 1, "layout_outcomes_per_update": 2,
        })
        self.assertEqual(settings.total_updates, 2)
        settings = replace(settings, model=ModelConfig.tiny())
        with tempfile.TemporaryDirectory() as directory:
            trainer = SelfPlayTrainer(settings, run_directory=directory)
            trainer.train()
            self.assertEqual(trainer.update, 2)
            self.assertEqual(trainer.cumulative["environment_plies"], 13)

    def test_real_budget_preserves_bounded_smoke_and_explicit_update_limit(self):
        smoke = TrainingSettings.from_yaml(CONFIG, TrainingMode.FOUR_DARK, tiny=True)
        self.assertEqual(smoke.total_updates, 1)
        self.assertIsNone(smoke.target_environment_plies)
        # The launcher supplies a real target even when --smoke-test is forwarded.
        smoke = TrainingSettings.from_yaml(CONFIG, TrainingMode.FOUR_DARK, tiny=True,
                                          overrides={"target_environment_plies": 3_000_000_000})
        self.assertEqual(smoke.total_updates, 1)
        self.assertEqual(smoke.warmup_updates, 1)
        bounded = TrainingSettings.from_yaml(CONFIG, TrainingMode.FOUR_DARK,
                                            overrides={"total_updates": 7})
        self.assertEqual(bounded.total_updates, 7)
        self.assertEqual(bounded.target_environment_plies, 3_000_000_000)


if __name__ == "__main__":
    unittest.main()
