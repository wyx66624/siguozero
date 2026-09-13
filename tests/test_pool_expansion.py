"""Explicit pool growth preserves unfinished games and the saved random stream."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from junqi.training.models import ModelConfig, PieceConditionedLayoutPointerDecoder
from junqi.training.modes import TrainingMode
from junqi.training.rollout import BaseGamePool
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer


class PoolExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def pool(self, size):
        return BaseGamePool(TrainingMode.FOUR_DARK, pool_size=size, max_transitions=16,
                            max_game_plies=32, seed=139, layout_prefetch_games=4)

    def test_growth_preserves_every_game_queue_and_rng(self):
        layout = PieceConditionedLayoutPointerDecoder(ModelConfig.tiny()).eval()
        old = self.pool(2)
        old.fill(layout, 7)
        state = copy.deepcopy(old.state_dict())
        expanded = self.pool(3)
        with self.assertRaisesRegex(ValueError, 'pool size changed'):
            expanded.load_state_dict(state)
        expanded.load_state_dict(state, allow_expansion=True)
        before_fill = expanded.state_dict()
        before_fill['pool_size'] = 2
        self.assertEqual(before_fill, state)
        expanded.fill(layout, 7)
        self.assertEqual(len(expanded.slots), 3)
        self.assertEqual(expanded.state_dict()['slots'][:2], state['slots'])
        expected_rng = copy.deepcopy(old.rng)
        expected_rng.randrange(2**63)
        self.assertEqual(expanded.rng.getstate(), expected_rng.getstate())
        self.assertEqual(len(expanded._layout_queue), len(old._layout_queue)-4)

    def test_shrinking_and_corrupt_saved_capacity_remain_errors(self):
        layout = PieceConditionedLayoutPointerDecoder(ModelConfig.tiny()).eval()
        old = self.pool(2)
        old.fill(layout, 7)
        state = old.state_dict()
        with self.assertRaisesRegex(ValueError, 'pool size changed'):
            self.pool(1).load_state_dict(state, allow_expansion=True)
        state['pool_size'] = 1
        with self.assertRaisesRegex(ValueError, 'too many'):
            self.pool(3).load_state_dict(state, allow_expansion=True)

    def test_trainer_resume_grows_pool_and_keeps_optimizer_and_progress(self):
        config = Path(__file__).resolve().parents[1] / 'configs/local_4090_training.yaml'
        settings = TrainingSettings.from_yaml(config, 'four_dark', tiny=True, overrides={
            'device': 'cpu', 'base_game_pool_size': 2, 'arena_enabled': False,
            'historical_enabled': False, 'arena_after_half_historical_only': False,
            'checkpoint_policy': 'periodic', 'inference_snapshot_every_updates': 0,
            'rollout_environment_workers': 1, 'ppo_pipeline_groups': 1})
        with tempfile.TemporaryDirectory() as directory:
            original = SelfPlayTrainer(settings, run_directory=directory, auto_resume=False)
            original.pool.fill(original.layout.eval(), 0)
            sum(p.sum()*0 for p in original.policy.parameters()).backward()
            original.policy_optimizer.step()
            original.policy_optimizer.zero_grad(set_to_none=True)
            saved_adam = copy.deepcopy(original.policy_optimizer.state_dict())
            original.update = 9
            original.cumulative['environment_plies'] = 12345
            original.save_checkpoint(reason='pool expansion test', archive=False)
            saved = original.pool.state_dict()
            original.logger.close()
            expanded = SelfPlayTrainer(replace(settings, base_game_pool_size=3),
                                       run_directory=directory, expand_game_pool=True)
            self.assertEqual(expanded.update, 9)
            self.assertTrue(expanded.pool_expanded_on_resume)
            self.assertEqual(expanded.cumulative['environment_plies'], 12345)
            restored_adam = expanded.policy_optimizer.state_dict()
            for parameter, values in saved_adam['state'].items():
                for key, value in values.items():
                    torch.testing.assert_close(restored_adam['state'][parameter][key], value, rtol=0, atol=0)
            self.assertEqual(expanded.pool.state_dict()['slots'], saved['slots'])
            expanded.pool.fill(expanded.layout.eval(), expanded.update)
            self.assertEqual(len(expanded.pool.slots), 3)
            self.assertEqual(expanded.pool.state_dict()['slots'][:2], saved['slots'])
            expanded.stop_requested = True
            expanded.train()
            ordinary = SelfPlayTrainer(replace(settings, base_game_pool_size=3), run_directory=directory)
            self.assertEqual(len(ordinary.pool.slots), 3)
            self.assertEqual(ordinary.update, 9)
            self.assertFalse(ordinary.pool_expanded_on_resume)
            ordinary.logger.close()

    def test_saved_oom_limits_change_only_with_explicit_retune(self):
        config = Path(__file__).resolve().parents[1] / 'configs/local_4090_training.yaml'
        settings = TrainingSettings.from_yaml(config, 'four_dark', tiny=True, overrides={
            'device': 'cpu', 'base_game_pool_size': 2, 'arena_enabled': False,
            'historical_enabled': False, 'arena_after_half_historical_only': False,
            'checkpoint_policy': 'periodic', 'inference_snapshot_every_updates': 0,
            'rollout_environment_workers': 1, 'ppo_pipeline_groups': 1,
            'actor_inference_batch': 8, 'policy_microbatch': 4, 'anchor_batch': 8})
        with tempfile.TemporaryDirectory() as directory:
            original = SelfPlayTrainer(settings, run_directory=directory, auto_resume=False)
            original.pool.fill(original.layout.eval(), 0)
            original.effective_actor_inference_batch = 2
            original.effective_policy_microbatch = 1
            original.save_checkpoint(reason='simulated OOM caps', archive=False)
            expected_pool = original.pool.state_dict()
            original.logger.close()
            ordinary = SelfPlayTrainer(settings, run_directory=directory)
            self.assertEqual(ordinary.effective_actor_inference_batch, 2)
            self.assertEqual(ordinary.effective_policy_microbatch, 1)
            ordinary.logger.close()
            retuned = SelfPlayTrainer(settings, run_directory=directory, reset_oom_batch_limits=True)
            self.assertEqual(retuned.effective_actor_inference_batch, 8)
            self.assertEqual(retuned.effective_policy_microbatch, 4)
            self.assertEqual(retuned.pool.state_dict(), expected_pool)
            for actual, expected in zip(retuned.policy.parameters(), original.policy.parameters(), strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            retuned.logger.close()


if __name__ == '__main__':
    unittest.main()
