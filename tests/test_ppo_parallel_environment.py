"""Serial/parallel replay parity, worker failure and checkpointable pool state."""
from __future__ import annotations

import copy
import pickle
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from junqi.training.models import GamePolicyTransformer, GameValueTransformer, ModelConfig, PieceConditionedLayoutPointerDecoder, PreNormEncoderBlock
from junqi.training.modes import TrainingMode
from junqi.training.ppo import FrozenValueActor, PPOSample, collect_ppo_samples, policy_ppo_loss, critic_ppo_loss
from junqi.training.ppo_environment import ParallelPPOEnvironment, _serialize
from junqi.training.rollout import BaseGamePool, FrozenPolicyActor, RolloutMetrics
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer
from junqi.training.metrics import MetricLogger


class ParallelEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_serial_parallel_samples_resets_rng_and_pool_match(self):
        for mode in (TrainingMode.FOUR_DARK, TrainingMode.DOUBLE_OPEN):
            for dead, arrays in ((True, True), (False, False)):
                with self.subTest(mode=mode, dead=dead, arrays=arrays):
                    torch.manual_seed(741)
                    config = replace(ModelConfig.tiny(dead_rules_enabled=dead), max_transitions=7,
                                     ppo_array_history=arrays, ppo_fixed_kv=False)
                    policy = GamePolicyTransformer(config).eval()
                    critic = GameValueTransformer(config).eval()
                    torch.nn.init.normal_(critic.value_head.weight, std=.03)
                    layout = PieceConditionedLayoutPointerDecoder(config).eval()
                    serial = BaseGamePool(mode, pool_size=3, max_transitions=7,
                                          max_game_plies=9, dead_rules_enabled=dead, seed=8)
                    serial.fill(layout, 3)
                    parallel = BaseGamePool(mode, pool_size=3, max_transitions=7,
                                            max_game_plies=9, dead_rules_enabled=dead, seed=8)
                    parallel.load_state_dict(serial.state_dict())
                    with ParallelPPOEnvironment(2) as workers:
                        for count in (17, 25):
                            rng = torch.get_rng_state()
                            left = collect_ppo_samples(serial, FrozenPolicyActor(policy),
                                FrozenValueActor(critic, amp_dtype=None, max_batch_size=3), layout,
                                count=count, behavior_version=3, discount=1., gae_lambda=.95)
                            expected_rng = torch.get_rng_state()
                            torch.set_rng_state(rng)
                            right = collect_ppo_samples(parallel, FrozenPolicyActor(policy),
                                FrozenValueActor(critic, amp_dtype=None, max_batch_size=3), layout,
                                count=count, behavior_version=3, discount=1., gae_lambda=.95, environment=workers)
                            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                            self.assertEqual(len(left[0]), count)
                            self.assertEqual(len(right[0]), count)
                            for a, b in zip(left[0], right[0], strict=True):
                                self.assertEqual(tuple(a.state.records), tuple(b.state.records))
                                self.assertEqual(a.state.legal_actions, b.state.legal_actions)
                                for name in ('action', 'old_log_prob', 'old_value', 'advantage', 'value_target', 'behavior_version'):
                                    self.assertEqual(getattr(a, name), getattr(b, name), name)
                            self.assertEqual(left[1], right[1])
                            for name in ('environment_plies', 'base_plies', 'base_games_completed', 'wins', 'draws', 'losses'):
                                self.assertEqual(getattr(left[2], name), getattr(right[2], name), name)
                            self.assertEqual(serial.state_dict(), parallel.state_dict())
                            self.assertEqual(right[2].environment_workers, 2)
                        processes = list(workers.processes)
                    self.assertTrue(all(p.poll() is not None for p in processes))

    def test_failed_worker_propagates_and_cleanup_reaps_processes(self):
        config = ModelConfig.tiny()
        layout = PieceConditionedLayoutPointerDecoder(config).eval()
        pool = BaseGamePool(TrainingMode.FOUR_DARK, pool_size=2, max_transitions=7,
                            max_game_plies=20, seed=6)
        pool.fill(layout, 1)
        with ParallelPPOEnvironment(2) as workers:
            workers.begin(pool)
            processes = list(workers.processes)
            workers.submit([0], [(-999, -999)])
            with self.assertRaisesRegex(RuntimeError, 'worker 0 failed'):
                workers.receive()
        self.assertTrue(all(p.poll() is not None for p in processes))

    def test_transport_omits_only_shared_immutable_board_graphs(self):
        from junqi.training.modes import new_game
        for mode in TrainingMode:
            with self.subTest(mode=mode):
                game = new_game(mode, seed=883, max_plies=2000)
                for _ in range(12):
                    game.step(game.legal_actions()[0])
                payload = _serialize([game, game.clone()])
                normal = pickle.dumps([game, game.clone()], protocol=pickle.HIGHEST_PROTOCOL)
                self.assertLess(len(payload), len(normal) / 2)
                restored, other = pickle.loads(payload)
                self.assertIs(restored._boards, other._boards)
                for _ in range(11):
                    self.assertEqual(game.legal_actions(), restored.legal_actions())
                    for seat in range(game.config.player_count):
                        self.assertEqual(game.observe(seat), restored.observe(seat))
                    if game.is_terminal:
                        break
                    action = game.legal_actions()[0]
                    game.step(action)
                    restored.step(action)

    def test_direct_causal_attention_matches_masked_outputs_and_gradients(self):
        torch.manual_seed(831)
        original = PreNormEncoderBlock(32, 4, 64, 0.).train()
        direct = copy.deepcopy(original)
        x = torch.randn(3, 19, 32, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        valid = torch.arange(19)[None, :] < torch.tensor([19, 11, 1])[:, None]
        mask = torch.zeros(19, 19).masked_fill(torch.ones(19, 19, dtype=torch.bool).triu(1), float('-inf'))
        a = original(x, valid_mask=valid, attention_mask=mask)
        b = direct(y, valid_mask=valid, is_causal=True)
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
        a.square().sum().backward()
        b.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, atol=1e-5, rtol=1e-5)
        for p, q in zip(original.parameters(), direct.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-5)

    def test_causal_packed_policy_critic_losses_and_gradients_match(self):
        from test_ppo_pipeline import histories
        game, history = histories(window=20)
        states = []
        for _ in range(13):
            states.append(history.state_for(game))
            game.step(game.legal_actions()[0])
            history.append_after_step(game)
        samples = [PPOSample(state, state.legal_actions[0], -3., 0., (-1.) ** i, .2, 0)
                   for i, state in enumerate(states)]
        config = replace(ModelConfig.tiny(), max_transitions=20, temporal_causal_sdpa=False,
                         activation_checkpointing=False)
        for cls, loss in ((GamePolicyTransformer, policy_ppo_loss), (GameValueTransformer, critic_ppo_loss)):
            with self.subTest(model=cls.__name__):
                old = cls(config).train()
                if cls is GameValueTransformer:
                    torch.nn.init.normal_(old.value_head.weight, std=.03)
                new = cls(replace(config, temporal_causal_sdpa=True)).train()
                new.load_state_dict(old.state_dict())
                options = dict(clip_epsilon=.2, sequence_training=True)
                options['entropy_coefficient' if cls is GamePolicyTransformer else 'value_coefficient'] = .01
                a, b = loss(old, samples, **options), loss(new, samples, defer_metrics=True, **options)
                torch.testing.assert_close(a.loss, b.loss, rtol=1e-5, atol=2e-6)
                materialized = SelfPlayTrainer._materialize_metrics(b.metrics)
                for key, value in a.metrics.items():
                    self.assertAlmostEqual(value, materialized[key], places=5)
                a.loss.backward()
                b.loss.backward()
                for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters(), strict=True):
                    if p.grad is not None or q.grad is not None:
                        torch.testing.assert_close(p.grad, q.grad, rtol=3e-5, atol=3e-6, msg=name)

    def test_resume_upgrades_configured_batch_but_preserves_oom_reduction(self):
        config = Path(__file__).parents[1] / 'configs/bootstrap.yaml'
        for effective in (2, 1):
            with self.subTest(effective=effective), tempfile.TemporaryDirectory() as directory:
                settings = TrainingSettings.from_yaml(config, 'four_dark', tiny=True,
                    overrides={'device': 'cpu', 'anchor_batch': 8, 'policy_microbatch': 2,
                               'arena_enabled': False})
                with patch.object(MetricLogger, 'start_resource_monitor'):
                    original = SelfPlayTrainer(settings, run_directory=directory, auto_resume=False)
                    original.effective_policy_microbatch = effective
                    original.save_checkpoint(reason='test_microbatch', archive=False)
                    resumed = SelfPlayTrainer(replace(settings, policy_microbatch=4), run_directory=directory)
                try:
                    self.assertEqual(resumed.effective_policy_microbatch, 4 if effective == 2 else 1)
                    row = resumed._aggregate_rollout_metrics(RolloutMetrics(
                        environment_workers=4, environment_worker_seconds=3.5, environment_sync_seconds=.7))
                    self.assertEqual(row.environment_workers, 4)
                    self.assertEqual(row.environment_worker_seconds, 3.5)
                    self.assertEqual(row.environment_sync_seconds, .7)
                finally:
                    original.logger.close()
                    resumed.logger.close()


if __name__ == '__main__':
    unittest.main()
