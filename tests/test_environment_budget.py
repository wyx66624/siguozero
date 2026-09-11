"""Environment interaction accounting, including simulated branches and resume."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("PyTorch training extra is not installed") from error

from junqi import JunqiGame
from junqi.training.cli import build_parser
from junqi.training.encoding import GameHistory
from junqi.training.models import GamePolicyTransformer, ModelConfig
from junqi.training.modes import TrainingMode, new_game
from junqi.training.ppo import collect_ppo_samples
from junqi.training.rollout import AnchorSnapshot, FrozenPolicyActor, collect_policy_groups
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer

CONFIG = Path(__file__).parents[1] / "configs/bootstrap.yaml"


class EnvironmentBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_canonical_budget_alias_and_legacy_grpo_are_unambiguous(self):
        parser = build_parser()
        args = parser.parse_args(["--mode", "four_dark", "--target-environment-plies", "3000000000"])
        self.assertEqual(args.target_environment_plies, 3_000_000_000)
        self.assertIsNone(args.target_continuation_plies)
        for mode in (TrainingMode.FOUR_DARK, TrainingMode.DOUBLE_OPEN):
            canonical = TrainingSettings.from_yaml(CONFIG, mode, overrides={"target_environment_plies": 13})
            old_alias = TrainingSettings.from_yaml(CONFIG, mode, overrides={"target_continuation_plies": 13})
            self.assertEqual(canonical.step_budget_target, old_alias.step_budget_target)
            self.assertEqual(canonical.step_budget_counter, "environment_plies")
            self.assertIsNone(old_alias.target_continuation_plies)
            self.assertEqual(canonical.serializable()["step_budget_unit"], "training_environment_transitions")
        legacy = TrainingSettings.from_yaml(CONFIG, "two_player", overrides={"target_continuation_plies": 256})
        total = TrainingSettings.from_yaml(CONFIG, "two_player", overrides={"target_environment_plies": 256})
        self.assertEqual(legacy.step_budget_counter, "continuation_plies")
        self.assertEqual(total.step_budget_counter, "environment_plies")
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                TrainingSettings.from_yaml(CONFIG, "four_dark", overrides={"target_environment_plies": invalid})
        with self.assertRaises(ValueError):
            TrainingSettings.from_yaml(CONFIG, "four_dark", overrides={
                "target_environment_plies": 13, "target_continuation_plies": 13})

    def test_eight_actual_32_step_branches_count_256_including_root_actions(self):
        torch.manual_seed(12)
        mode = TrainingMode.FOUR_DARK
        game = new_game(mode, seed=41, max_plies=32)
        history = GameHistory.initialize(game, mode, max_transitions=16)
        anchor = AnchorSnapshot(game, history, history.state_for(game), game.current_player)
        actor = FrozenPolicyActor(GamePolicyTransformer(ModelConfig.tiny()).eval(), max_batch_size=8)
        original_step = JunqiGame.step
        calls = 0

        def counted_step(game, *args, **kwargs):
            nonlocal calls
            result = original_step(game, *args, **kwargs)
            calls += 1
            return result

        with patch.object(JunqiGame, "step", counted_step):
            groups, metrics = collect_policy_groups(
                [anchor], actor, behavior_version=0, advantage_epsilon=1e-4,
                anchor_wave_size=1, environment_workers=1)
        self.assertEqual(game.ply_count, 0)  # Only the eight copies advanced.
        self.assertEqual(calls, 8 * 32)
        self.assertEqual(metrics.environment_plies, calls)
        self.assertEqual(metrics.continuation_plies, calls)
        self.assertEqual(metrics.base_plies, 0)
        self.assertEqual(groups[0].continuation_plies, calls)
        # Executing an action on the original game is one additional interaction.
        game.step(anchor.state.legal_actions[0])
        metrics.record_environment_steps()
        self.assertEqual(metrics.environment_plies, 257)
        self.assertEqual(metrics.as_dict()["rollout/plies_per_second"], 257 / metrics.wall_seconds)

    def test_ppo_extra_simulation_counts_towards_stop_and_survives_resume(self):
        # Test-only extra rollouts exercise the budget contract. Production PPO
        # still uses its critic and does not restore Monte Carlo simulations.
        settings = TrainingSettings.from_yaml(CONFIG, "four_dark", tiny=True, overrides={
            "device": "cpu", "anchor_batch": 8, "policy_microbatch": 2,
            "total_updates": 5, "target_environment_plies": 13,
            "policy_epochs": 2, "critic_epochs": 2, "arena_enabled": False,
        })

        def with_simulations(pool, actor, critic, layout, **kwargs):
            samples, layouts, metrics = collect_ppo_samples(pool, actor, critic, layout, **kwargs)
            for _ in range(3):
                branch = pool.slots[0].game.clone()
                history = pool.slots[0].history.clone()
                for _ in range(2):
                    state = history.state_for(branch)
                    actions, _ = actor.sample([state], count=1, return_log_probs=False)
                    branch.step(actions[0][0])
                    history.append_after_step(branch)
                    metrics.record_environment_steps(continuation=True)
            return samples, layouts, metrics

        with tempfile.TemporaryDirectory() as directory:
            trainer = SelfPlayTrainer(settings, run_directory=directory)
            with patch("junqi.training.trainer.collect_ppo_samples", side_effect=with_simulations) as collector:
                trainer.train()
            self.assertEqual(collector.call_count, 1)
            self.assertEqual(trainer.update, 1)
            self.assertEqual(trainer.cumulative["environment_plies"], 14)
            self.assertEqual(trainer.cumulative["base_plies"], 8)
            self.assertEqual(trainer.cumulative["continuation_plies"], 6)
            self.assertEqual(trainer.cumulative["policy_samples"], 8)
            metrics = json.loads(trainer.logger.jsonl_path.read_text().splitlines()[-1])
            self.assertEqual(metrics["training/target_environment_plies"], 13)
            self.assertEqual(metrics["training/step_budget_counter"], "environment_plies")
            checkpoint = trainer.checkpoints.latest_path
            payload = torch.load(checkpoint, weights_only=False)
            self.assertEqual(payload["trainer_state"]["cumulative"]["environment_plies"], 14)

            for old_format in (False, True):
                if old_format:
                    payload["trainer_state"]["cumulative"].pop("environment_plies")
                    payload["trainer_state"].pop("environment_step_definition", None)
                    torch.save(payload, checkpoint)
                resumed = SelfPlayTrainer(settings, run_directory=directory)
                self.assertEqual(resumed.cumulative["environment_plies"], 14)
                with patch("junqi.training.trainer.collect_ppo_samples", side_effect=AssertionError("budget already reached")):
                    resumed.train()
                self.assertEqual(resumed.update, 1)
                self.assertEqual(resumed.cumulative["environment_plies"], 14)


if __name__ == "__main__":
    unittest.main()
