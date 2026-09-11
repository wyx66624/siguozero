from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("PyTorch training extra is not installed") from error

from junqi.training.encoding import GameHistory
from junqi.training.models import GamePolicyTransformer, GameValueTransformer, ModelConfig
from junqi.training.modes import TrainingMode, new_game
from junqi.training.ppo import (
    FrozenValueActor, PPOSample, PPOTransition, collect_ppo_samples,
    critic_ppo_loss, generalized_advantages, policy_ppo_loss,
)
from junqi.training.rollout import FrozenPolicyActor
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer


CONFIG = Path(__file__).parents[1] / "configs" / "bootstrap.yaml"


def states(count=2, mode=TrainingMode.FOUR_DARK):
    result = []
    for seed in range(count):
        game = new_game(mode, seed=seed)
        result.append(GameHistory.initialize(game, mode, max_transitions=16).state_for(game))
    return result


def training_settings(mode=TrainingMode.FOUR_DARK, **overrides):
    return TrainingSettings.from_yaml(
        CONFIG, mode, tiny=True,
        overrides={"device": "cpu", "anchor_batch": 8, "policy_microbatch": 2,
                   "total_updates": 1, **overrides},
    )


class PPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gae_team_sign_terminal_reset_and_nonterminal_bootstrap(self):
        state = states(1)[0]
        def transition(value, reward=0, terminal=False, sign=-1):
            return PPOTransition(state, state.legal_actions[0], -1, value,
                                 reward, terminal, sign)
        samples = generalized_advantages(
            [transition(.2), transition(.3), transition(-.4, 1, True)],
            bootstrap_value=999, discount=1, gae_lambda=1, behavior_version=7,
        )
        for sample, target, advantage in zip(samples, [1, -1, 1], [.8, -1.3, 1.4]):
            self.assertAlmostEqual(sample.value_target, target)
            self.assertAlmostEqual(sample.advantage, advantage)
            self.assertEqual(sample.behavior_version, 7)
        # An eliminated seat can leave an ally next to act: do not blindly negate.
        allied = generalized_advantages(
            [transition(0, sign=1), transition(0, 1, True)],
            bootstrap_value=999, discount=1, gae_lambda=.95, behavior_version=0,
        )
        self.assertAlmostEqual(allied[0].value_target, .95)
        truncated = generalized_advantages(
            [transition(.25)], bootstrap_value=.4, discount=.9,
            gae_lambda=.95, behavior_version=0,
        )
        self.assertAlmostEqual(truncated[0].value_target, -.36)
        reset = generalized_advantages(
            [transition(0, 1, True), transition(0, 0, True)],
            bootstrap_value=999, discount=1, gae_lambda=1, behavior_version=0,
        )
        self.assertEqual([sample.value_target for sample in reset], [1, 0])

    def test_critic_is_independent_and_both_networks_learn(self):
        config = ModelConfig.tiny()
        policy = GamePolicyTransformer(config)
        critic = GameValueTransformer(config)
        critic.initialize_from_policy(policy)
        batch = states()
        self.assertEqual(critic(batch).shape, (2,))
        self.assertTrue(torch.equal(critic(batch), torch.zeros(2)))
        self.assertFalse(hasattr(critic, "source_query"))
        self.assertNotEqual(policy.board_encoder.board_token.data_ptr(),
                            critic.board_encoder.board_token.data_ptr())
        with torch.no_grad():
            logs = policy(batch, [[state.legal_actions[0]] for state in batch])
        samples = [PPOSample(state, state.legal_actions[0], float(log[0]), 0,
                             -1 if i == 0 else 1, -1 if i == 0 else 1, 0)
                   for i, (state, log) in enumerate(zip(batch, logs))]
        policy_before = policy.source_query[0].weight.detach().clone()
        critic_before = critic.value_head.weight.detach().clone()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=.001)
        policy_ppo_loss(policy, samples, clip_epsilon=.2, entropy_coefficient=.01).loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in critic.parameters()))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=.001)
        for _ in range(2):
            critic_optimizer.zero_grad(set_to_none=True)
            loss = critic_ppo_loss(critic, samples, clip_epsilon=.2, value_coefficient=.5).loss
            loss.backward()
            critic_optimizer.step()
        self.assertFalse(torch.equal(policy_before, policy.source_query[0].weight))
        self.assertFalse(torch.equal(critic_before, critic.value_head.weight))
        self.assertGreater(float(critic.board_encoder.board_token.grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in policy.parameters()))

    def test_clipped_policy_and_value_losses(self):
        state = replace(states(1)[0], legal_actions=((0, 1), (1, 2)))
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.log(torch.tensor([.8, .2])))
            def forward(self, states, actions):
                return [self.logits.log_softmax(0) for _ in states]
        sample = PPOSample(state, (0, 1), float(torch.log(torch.tensor(.4))), 0, 1, 2, 0)
        policy = Policy()
        output = policy_ppo_loss(policy, [sample], clip_epsilon=.2, entropy_coefficient=0)
        self.assertAlmostEqual(float(output.loss.detach()), -1.2, places=6)
        output.loss.backward()
        self.assertAlmostEqual(float(policy.logits.grad.abs().sum()), 0)
        class Value(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(1.0))
            def forward(self, states):
                return self.value.expand(len(states))
        value = Value()
        loss = critic_ppo_loss(value, [sample], clip_epsilon=.2, value_coefficient=.5).loss
        self.assertAlmostEqual(float(loss.detach()), .81, places=6)
        loss.backward()
        self.assertEqual(float(value.value.grad), 0)

    def test_collection_uses_real_actions_without_monte_carlo_clones(self):
        from junqi import JunqiGame
        for mode in (TrainingMode.FOUR_DARK, TrainingMode.DOUBLE_OPEN):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                trainer = SelfPlayTrainer(training_settings(mode), run_directory=directory)
                try:
                    actor = FrozenPolicyActor(trainer.policy.eval(), max_batch_size=2)
                    critic = FrozenValueActor(trainer.critic, amp_dtype=None, max_batch_size=2)
                    with patch.object(JunqiGame, "clone", side_effect=AssertionError("cloned PPO game")):
                        samples, outcomes, metrics = collect_ppo_samples(
                            trainer.pool, actor, critic, trainer.layout.eval(), count=17,
                            behavior_version=0, discount=1, gae_lambda=.95,
                        )
                    self.assertEqual(len(samples), 17)
                    self.assertEqual(metrics.environment_plies, 17)
                    self.assertEqual(metrics.base_games_completed, 4)
                    self.assertEqual(len(outcomes), 16)
                    self.assertEqual(metrics.terminal_continuations, 0)
                    self.assertEqual(metrics.root_candidates, 0)
                    self.assertTrue(all(sample.action in sample.state.legal_actions for sample in samples))
                finally:
                    trainer.logger.close()

    def test_checkpoint_preserves_critic_optimizer_and_inference_needs_no_critic(self):
        from junqi.training.inference import InferenceEngine
        with tempfile.TemporaryDirectory() as directory:
            settings = training_settings()
            trainer = SelfPlayTrainer(settings, run_directory=directory)
            batch = states()
            samples = [PPOSample(state, state.legal_actions[0], -1, 0, 1, 1, 0)
                       for state in batch]
            trainer._update_critic(samples)
            trainer.train()
            checkpoint = trainer.checkpoints.latest_path
            payload = torch.load(checkpoint, weights_only=False)
            self.assertEqual(payload["algorithm"], "ppo")
            self.assertIsNone(payload["reference_policy"])
            self.assertTrue(payload["critic_optimizer"]["state"])
            resumed = SelfPlayTrainer(settings, run_directory=directory)
            try:
                for expected, actual in zip(trainer.critic.parameters(), resumed.critic.parameters()):
                    self.assertTrue(torch.equal(expected, actual))
                self.assertEqual(resumed.cumulative["environment_plies"], 8)
                self.assertEqual(resumed.update, 1)
                engine = InferenceEngine.from_checkpoint(checkpoint, device="cpu")
                self.assertFalse(hasattr(engine, "critic"))
                game, history = engine.new_game(seed=12, max_plies=4)
                self.assertFalse(game.is_terminal)
            finally:
                resumed.logger.close()

    def test_legacy_migration_is_explicit_and_failure_retains_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = training_settings()
            trainer = SelfPlayTrainer(settings, run_directory=root / "source")
            checkpoint = trainer.checkpoints.latest_path
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            with patch.object(trainer, "_update_critic", side_effect=RuntimeError("injected")):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    trainer.train()
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)
            payload = torch.load(checkpoint, weights_only=False)
            payload.pop("algorithm")  # legacy v4 implicitly means GRPO
            payload.pop("critic")
            payload.pop("critic_optimizer")
            payload["reference_policy"] = payload["policy"]
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "algorithm"):
                SelfPlayTrainer(settings, run_directory=root / "source")
            migrated = SelfPlayTrainer(settings, run_directory=root / "migrated",
                                       initialize_from=checkpoint)
            try:
                self.assertEqual(migrated.update, 0)
                self.assertEqual(len(migrated.critic_optimizer.state), 0)
                self.assertTrue(torch.equal(migrated.policy.board_encoder.board_token,
                                             migrated.critic.board_encoder.board_token))
            finally:
                migrated.logger.close()

    def test_only_four_player_modes_use_ppo(self):
        for mode in TrainingMode:
            settings = training_settings(mode)
            self.assertEqual(settings.algorithm, "grpo" if mode is TrainingMode.TWO_PLAYER else "ppo")
        settings = TrainingSettings.from_yaml(CONFIG, TrainingMode.FOUR_DARK, model_scale="main")
        self.assertEqual(settings.policy_microbatch, 8)
        self.assertEqual(settings.base_game_pool_size, 20)
        self.assertEqual(settings.model.inference_temporal_cache_entries, 240)

    def test_no_dead_rules_and_real_step_budget(self):
        settings = TrainingSettings.from_yaml(
            CONFIG, TrainingMode.FOUR_DARK, tiny=True, dead_rules_enabled=False,
            overrides={"device": "cpu", "anchor_batch": 8, "policy_microbatch": 2,
                       "total_updates": 3, "target_environment_plies": 8},
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer = SelfPlayTrainer(settings, run_directory=directory)
            self.assertIsNone(trainer.critic.board_encoder.casualty_projection)
            trainer.train()
            self.assertEqual(trainer.update, 1)
            self.assertEqual(trainer.cumulative["environment_plies"], 8)
            self.assertEqual(trainer.cumulative["continuation_plies"], 0)


if __name__ == "__main__":
    unittest.main()
