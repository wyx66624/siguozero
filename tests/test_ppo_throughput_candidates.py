"""Semantic contracts for PPO throughput changes, including resume RNG."""
import copy
from dataclasses import replace
import unittest
import random
import numpy as np
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import torch

from junqi.training.encoding import GameHistory
from junqi.training.models import GamePolicyTransformer, ModelConfig, PieceConditionedLayoutPointerDecoder
from junqi.training.modes import TrainingMode, new_game
from junqi.training.rollout import BaseGamePool
from junqi.training.rollout import FrozenPolicyActor
from junqi.training.models import GameValueTransformer
from junqi.training.ppo import FrozenValueActor, collect_ppo_samples
from junqi.training.ppo_pipeline import collect_pipelined
from junqi.training.ppo_environment import ParallelPPOEnvironment
from junqi.training.packed_observation import observation_rows
from junqi.training.encoding import ActionFeatures, _record_from_observation
from junqi.training.history_arrays import record_array
from junqi.training.trainer import SelfPlayTrainer
from junqi.training.settings import TrainingSettings


class ThroughputCandidatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_resume_reconfigures_fused_backend_for_cpu_without_changing_moments(self):
        layer = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(layer.parameters())
        layer(torch.ones(1, 2)).sum().backward()
        optimizer.step()

        moments = [optimizer.state[p]['exp_avg'] for p in layer.parameters()]
        for group in optimizer.param_groups:
            group.update(fused=True, foreach=False)
        trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
        trainer.settings = SimpleNamespace(algorithm='ppo', ppo_fused_optimizer=True)
        trainer.device = torch.device('cpu')
        trainer.policy_optimizer, trainer.critic_optimizer, trainer.layout_optimizer = optimizer, None, None
        trainer._configure_optimizer_backend()
        for group in optimizer.param_groups:
            self.assertIsNone(group['fused'])
            self.assertIsNone(group['foreach'])
        for parameter, expected in zip(layer.parameters(), moments, strict=True):
            self.assertIs(optimizer.state[parameter]['exp_avg'], expected)
        optimizer.step()

    def test_throughput_profile_preserves_model_rules_and_environment_based_schedules(self):
        root = Path(__file__).resolve().parents[1]
        base = TrainingSettings.from_yaml(root/'configs/bootstrap.yaml', 'four_dark', model_scale='main')
        fast = TrainingSettings.from_yaml(root/'configs/ppo_4090_throughput.yaml', 'four_dark', model_scale='main')
        self.assertEqual(base.model, fast.model)
        for key in ('rules', 'models', 'history', 'board_piece_codes', 'model_selection'):
            self.assertEqual(base.raw_config[key], fast.raw_config[key])
        for key in ('policy_epochs', 'critic_epochs', 'ppo_minibatch_samples', 'target_environment_plies'):
            self.assertEqual(getattr(base, key), getattr(fast, key))
        for key in ('warmup_updates', 'layout_update_interval', 'reference_refresh_updates'):
            self.assertEqual(getattr(base, key) * base.anchor_batch, getattr(fast, key) * fast.anchor_batch)

    def test_disabling_direct_causal_sdpa_keeps_reference_fallback_available(self):
        config = replace(ModelConfig.tiny(), ppo_tensor_learner=True, temporal_causal_sdpa=False)
        model = GameValueTransformer(config).eval()
        game = new_game('four_dark', seed=81)
        history = GameHistory.initialize(game, 'four_dark', max_transitions=16)
        history.enable_array_storage()
        with torch.no_grad():
            result = model([history.state_for(game)], pack_sequences=True)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(getattr(model, '_tensor_learner_calls', 0), 0)

    def test_serial_and_online_value_overrides_select_compatible_collection(self):
        path = Path(__file__).resolve().parents[1]/'configs/bootstrap.yaml'
        for overrides in ({'rollout_environment_workers': 1}, {'ppo_deferred_values': False}):
            settings = TrainingSettings.from_yaml(path, 'four_dark', overrides=overrides)
            self.assertEqual(settings.ppo_pipeline_groups, 1)
            with self.assertRaises(ValueError):
                TrainingSettings.from_yaml(path, 'four_dark', overrides={**overrides, 'ppo_pipeline_groups': 2})

    def test_direct_observations_equal_public_views_in_every_mode(self):
        for mode in TrainingMode:
            for dead in (False, True):
                for seed in (61, 99):
                    game = new_game(mode, seed=seed, dead_rules_enabled=dead, max_plies=90)
                    rng = random.Random(seed)
                    while True:
                        rows = observation_rows(game, mode)
                        for seat, row in enumerate(rows):
                            obs = game.observe(seat, history_limit=1, include_legal_masks=False,
                                               include_candidate_masks=False)
                            action = ActionFeatures.from_event(obs.history[-1]) if obs.history else None
                            expected = record_array(_record_from_observation(obs, action), mode, dead)
                            np.testing.assert_array_equal(row, expected)
                        if game.is_terminal:
                            break
                        game.step(rng.choice(game.legal_actions()))

    def test_board_chunk_outputs_and_all_gradients_match(self):
        for dead in (True, False):
            for mode in TrainingMode:
                with self.subTest(dead=dead, mode=mode):
                    torch.manual_seed(43)
                    a = GamePolicyTransformer(replace(ModelConfig.tiny(dead_rules_enabled=dead), board_chunk_size=1)).train()
                    b = copy.deepcopy(a)
                    b.config = replace(a.config, board_chunk_size=2048)
                    game = new_game(mode, seed=8, dead_rules_enabled=dead)
                    history = GameHistory.initialize(game, mode, max_transitions=16)
                    states = []
                    for _ in range(9):
                        states.append(history.state_for(game))
                        game.step(game.legal_actions()[0])
                        history.append_after_step(game)
                    observations = []
                    for model in (a, b):
                        output = model(states, [s.legal_actions[:2] for s in states])
                        observations.append(output)
                        sum(x.square().mean() for x in output).backward()
                    for x, y in zip(*observations, strict=True):
                        torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-4)
                    for x, y in zip(a.parameters(), b.parameters(), strict=True):
                        torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=2e-4)

    def test_prefetch_resume_preserves_unused_layouts_and_rng(self):
        torch.manual_seed(91)
        layout = PieceConditionedLayoutPointerDecoder(ModelConfig.tiny()).eval()
        def pool():
            return BaseGamePool('four_dark', pool_size=2, max_transitions=16,
                                max_game_plies=50, seed=5, layout_prefetch_games=4)
        left = pool()
        left.fill(layout, 3)
        saved = copy.deepcopy(left.state_dict())
        rng = torch.get_rng_state()
        expected = [left._slot_state_dict(left._new_slot(layout, 3)) for _ in range(5)]
        end_rng = torch.get_rng_state()
        right = pool()
        right.load_state_dict(saved)
        torch.set_rng_state(rng)
        actual = [right._slot_state_dict(right._new_slot(layout, 3)) for _ in range(5)]
        self.assertEqual(expected, actual)
        self.assertTrue(torch.equal(end_rng, torch.get_rng_state()))
        with patch.object(layout, 'sample_layouts', wraps=layout.sample_layouts) as sampler:
            right._new_slot(layout, 4)
            sampler.assert_called_once_with(16, right.mode, temperature=0.7)
        legacy = dict(saved)
        legacy.pop('layout_queue')
        legacy.pop('layout_queue_version')
        right.load_state_dict(legacy)
        self.assertEqual(right._layout_queue, [])

    def test_tensor_path_matches_full_history_outputs_and_gradients(self):
        for mode in TrainingMode:
            for dead in (False, True):
                with self.subTest(mode=mode, dead=dead):
                    torch.manual_seed(121)
                    reference = GameValueTransformer(ModelConfig.tiny(dead_rules_enabled=dead)).train()
                    torch.nn.init.normal_(reference.value_head.weight, std=.1)
                    fast = copy.deepcopy(reference)
                    fast.config = replace(fast.config, ppo_tensor_learner=True, temporal_causal_sdpa=True)
                    game = new_game(mode, seed=31, dead_rules_enabled=dead)
                    history = GameHistory.initialize(game, mode, max_transitions=7)
                    history.enable_array_storage()
                    states = []
                    for _ in range(15):
                        states.append(history.state_for(game))
                        game.step(game.legal_actions()[0])
                        history.append_after_step(game)
                    a = reference(states, pack_sequences=True)
                    b = fast(states, pack_sequences=True)
                    self.assertEqual(fast._tensor_learner_calls, 1)
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
                    a.square().mean().backward()
                    b.square().mean().backward()
                    for x, y in zip(reference.parameters(), fast.parameters(), strict=True):
                        torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=2e-4)

    def test_deferred_values_preserve_actions_bootstrap_and_gae_across_resets(self):
        for mode in (TrainingMode.FOUR_DARK, TrainingMode.DOUBLE_OPEN):
            for workers in (1, 2):
                torch.manual_seed(101)
                config = replace(ModelConfig.tiny(), ppo_tensor_learner=True, temporal_causal_sdpa=True)
                policy = GamePolicyTransformer(config).eval()
                critic = GameValueTransformer(config).eval()
                torch.nn.init.normal_(critic.value_head.weight, std=.1)
                layout = PieceConditionedLayoutPointerDecoder(config).eval()
                def pool():
                    return BaseGamePool(mode, pool_size=2, max_transitions=7,
                                        max_game_plies=11, seed=18, layout_prefetch_games=4)
                left, right = pool(), pool()
                left.fill(layout, 4)
                right.load_state_dict(copy.deepcopy(left.state_dict()))
                rng = torch.get_rng_state()
                def collect(p, deferred):
                    return collect_ppo_samples(p, FrozenPolicyActor(policy),
                        FrozenValueActor(critic, amp_dtype=None, max_batch_size=2, deferred=deferred),
                        layout, count=53, behavior_version=4, discount=.99, gae_lambda=.95,
                        environment_workers=workers)
                a = collect(left, False)
                end_rng = torch.get_rng_state()
                torch.set_rng_state(rng)
                b = collect(right, True)
                self.assertTrue(torch.equal(end_rng, torch.get_rng_state()))
                self.assertEqual(left.state_dict(), right.state_dict())
                self.assertEqual(a[1], b[1])
                for x, y in zip(a[0], b[0], strict=True):
                    self.assertEqual(x.action, y.action)
                    self.assertEqual(x.old_log_prob, y.old_log_prob)
                    for field in ('old_value', 'advantage', 'value_target'):
                        self.assertAlmostEqual(getattr(x, field), getattr(y, field), places=5)

    def test_pipeline_keeps_exact_steps_histories_and_terminal_returns(self):
        for games in (1, 4):
            for mode in (TrainingMode.FOUR_DARK, TrainingMode.DOUBLE_OPEN):
                torch.manual_seed(511)
                config = replace(ModelConfig.tiny(), ppo_tensor_learner=True, temporal_causal_sdpa=True)
                policy = GamePolicyTransformer(config).eval()
                critic = GameValueTransformer(config).eval()
                torch.nn.init.normal_(critic.value_head.weight, std=.1)
                layout = PieceConditionedLayoutPointerDecoder(config).eval()
                def pool():
                    return BaseGamePool(mode, pool_size=games, max_transitions=7,
                                        max_game_plies=11, seed=18, layout_prefetch_games=4)
                left, right = pool(), pool()
                left.fill(layout, 4)
                right.load_state_dict(copy.deepcopy(left.state_dict()))
                rng = torch.get_rng_state()
                # Deterministic actions isolate scheduling from the changed
                # CUDA categorical batch size, which can change RNG order.
                def actor():
                    a = FrozenPolicyActor(policy)
                    a.sample = lambda states, **kw: (
                        [[s.legal_actions[0]] for s in states], [torch.tensor([0.]) for s in states])
                    return a
                value = lambda: FrozenValueActor(critic, amp_dtype=None, max_batch_size=4, deferred=True)
                kwargs = dict(count=53, behavior_version=4, discount=.99, gae_lambda=.95)
                a = collect_ppo_samples(left, actor(), value(), layout, **kwargs, environment_workers=2)
                torch.set_rng_state(rng)
                with ParallelPPOEnvironment(2) as env:
                    b = collect_pipelined(right, actor(), value(), layout, **kwargs, environment=env)
                self.assertEqual(left.state_dict(), right.state_dict())
                self.assertEqual(a[1], b[1])
                self.assertEqual(b[2].environment_plies, 53)
                for x, y in zip(a[0], b[0], strict=True):
                    self.assertEqual(tuple(x.state.records), tuple(y.state.records))
                    self.assertEqual(x.action, y.action)
                    for field in ('old_value', 'advantage', 'value_target'):
                        self.assertAlmostEqual(getattr(x, field), getattr(y, field), places=5)


if __name__ == '__main__':
    unittest.main()
