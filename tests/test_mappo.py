"""Expanded MAPPO architecture, information boundary, returns and recovery."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from junqi.training.checkpoint import require_current_checkpoint
from junqi.training.checkpoint_format import checkpoint_format_for_model
from junqi.training.encoding import GameHistory
from junqi.training.mappo import MAPPOTransition, collect_mappo_samples, team_generalized_advantages
from junqi.training.mappo_encoding import CentralizedHistory, centralized_observation_rows
from junqi.training.models import (
    GameCentralValueTransformer, GamePolicyTransformer, ModelConfig,
    PieceConditionedLayoutPointerDecoder, SwiGLU,
)
from junqi.training.modes import TrainingMode, new_game
from junqi.training.packed_observation import observation_rows
from junqi.training.ppo import FrozenValueActor, PPOSample, critic_ppo_loss, policy_ppo_loss
from junqi.training.ppo_environment import _advance
from junqi.training.rollout import BaseGamePool, FrozenPolicyActor
from junqi.training.settings import TrainingSettings


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "expand_mappo_training.yaml"
MODE = TrainingMode.FOUR_DARK


def tiny_config(**overrides):
    return replace(ModelConfig.tiny(), temporal_layers=2, token_fusion_hidden_dim=64,
                   ppo_cuda_graphs=False, **overrides)


def histories(mode=MODE, *, window=16, dead=True):
    game = new_game(mode, seed=421, max_plies=80, dead_rules_enabled=dead)
    private = GameHistory.initialize(game, mode, max_transitions=window)
    central = CentralizedHistory.initialize(game, mode, max_transitions=window)
    private.enable_array_storage()
    central.enable_array_storage()
    return game, private, central


class MAPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_expanded_profile_and_legacy_mode_isolation(self):
        for mode in (MODE, TrainingMode.DOUBLE_OPEN):
            settings = TrainingSettings.from_yaml(CONFIG, mode, model_scale="main")
            self.assertEqual(settings.algorithm, "mappo")
            self.assertEqual(settings.model.board_dim, 384)
            self.assertEqual(settings.model.action_dim, 128)
            self.assertTrue(settings.model.joint_action_context)
            self.assertEqual(settings.model.board_encoder_type, "whole_board_inventory_candidates_linear")
            self.assertTrue(settings.model.action_pass_context)
            self.assertEqual(settings.model.temporal_dim, 512)
            self.assertEqual(settings.model.token_fusion_hidden_dim, 512)
            self.assertEqual(settings.model.temporal_heads, 8)
            self.assertEqual(settings.model.temporal_dim // settings.model.temporal_heads, 64)
            self.assertEqual(settings.model.temporal_layers, 16)
            self.assertEqual(settings.critic_layers, 16)
            self.assertEqual(settings.step_budget_counter, "environment_plies")
            self.assertIn("runs_expand_mappo", str(settings.resolve_run_directory()))
            self.assertEqual(checkpoint_format_for_model(settings.model), 17)
        legacy = TrainingSettings.from_yaml(ROOT / "configs/local_4090_training.yaml", MODE, model_scale="main")
        self.assertEqual((legacy.algorithm, legacy.model.board_dim, legacy.model.token_fusion_hidden_dim), ("ppo", 128, 0))
        self.assertEqual(checkpoint_format_for_model(legacy.model), 15)
        self.assertFalse(legacy.model.joint_action_context)
        two = TrainingSettings.from_yaml(CONFIG, TrainingMode.TWO_PLAYER, model_scale="main")
        self.assertEqual(two.algorithm, "mappo")
        self.assertEqual(two.model.token_fusion_hidden_dim, 512)
        self.assertEqual(two.critic_layers, 16)
        self.assertTrue(two.model.joint_action_context)
        self.assertEqual(two.model.board_point_capacity, 60)
        legacy_two = TrainingSettings.from_yaml(ROOT / "configs/bootstrap.yaml", TrainingMode.TWO_PLAYER)
        self.assertEqual(legacy_two.algorithm, "grpo")

    def test_critic_has_one_head_and_independent_matching_backbone(self):
        config = tiny_config()
        actor = GamePolicyTransformer(config)
        critic = GameCentralValueTransformer(config)
        critic.initialize_from_policy(actor)
        self.assertIsInstance(actor.token_fusion, SwiGLU)
        self.assertEqual(actor.token_fusion.up.in_features, config.temporal_dim)
        self.assertEqual(actor.token_fusion.down.out_features, config.temporal_dim)
        self.assertEqual((len(actor.temporal_layers), len(critic.temporal_layers)), (2, 2))
        self.assertEqual((critic.value_head.in_features, critic.value_head.out_features), (64, 1))
        self.assertFalse(hasattr(critic, "source_query"))
        self.assertFalse(hasattr(critic, "destination_query"))
        self.assertFalse(hasattr(critic, "draw_value_head"))
        actor_parameters = dict(actor.named_parameters())
        for name, parameter in critic.named_parameters():
            if name.startswith("value_head."):
                continue
            torch.testing.assert_close(parameter, actor_parameters[name], atol=0, rtol=0)
            self.assertNotEqual(parameter.data_ptr(), actor_parameters[name].data_ptr())
        game, private, central = histories()
        globals_ = [central.value_state_for(team) for team in (0, 1)]
        self.assertEqual(critic(globals_).shape, (2,))
        with self.assertRaisesRegex(ValueError, "global-state"):
            critic([private.state_for(game)])
        for function in (
            lambda: actor(globals_, [[], []]),
            lambda: actor.ppo_statistics(globals_, [(0, 0), (0, 0)]),
            lambda: actor.sample_action_groups(globals_),
            lambda: actor.log_probs_for_action_groups(globals_, [[], []]),
            lambda: FrozenPolicyActor(actor).sample(globals_),
        ):
            with self.assertRaisesRegex(ValueError, "centralized"):
                function()

    def test_hidden_rank_changes_global_state_but_not_actor_view(self):
        game, private, central = histories()
        before_private = observation_rows(game, MODE)[0].copy()
        before_global = centralized_observation_rows(game, MODE)[0].copy()
        unknown = [point for point, piece in game._pieces.items()
                   if piece.owner == 1 and game._known_piece_kind(0, point, piece) is None]
        first = unknown[0]
        second = next(point for point in unknown if game._pieces[point].kind != game._pieces[first].kind)
        changed = game.clone()
        changed._pieces[first], changed._pieces[second] = changed._pieces[second], changed._pieces[first]
        np.testing.assert_array_equal(before_private, observation_rows(changed, MODE)[0])
        self.assertFalse(np.array_equal(before_global, centralized_observation_rows(changed, MODE)[0]))
        self.assertNotEqual(private.players[0].records.identity, central.value_state_for(0).records.identity)
        self.assertFalse(private.state_for(game).centralized)
        self.assertTrue(central.value_state_for(0).centralized)

    def test_worker_and_checkpoint_keep_global_history_separate(self):
        for mode in (MODE, TrainingMode.DOUBLE_OPEN):
            for dead in (True,):
                game, private, central = histories(mode, window=7, dead=dead)
                retained = central.value_state_for(0)
                retained_records = tuple(retained.records)
                for _ in range(12):
                    row = _advance(0, game, game.legal_actions()[0], mode, centralized_critic=True)
                    self.assertFalse(np.shares_memory(row.records, row.centralized_records))
                    np.testing.assert_array_equal(row.records, observation_rows(game, mode))
                    np.testing.assert_array_equal(row.centralized_records, centralized_observation_rows(game, mode))
                    private.append_encoded_rows(row.records)
                    central.append_encoded_rows(row.centralized_records)
                saved = central.state_dict()
                restored = CentralizedHistory.from_state_dict(saved)
                restored.enable_array_storage()
                self.assertEqual(saved, restored.state_dict())
                self.assertEqual(tuple(retained.records), retained_records)
                self.assertEqual(len(restored.value_state_for(0).records), 8)
                with self.assertRaisesRegex(ValueError, "explicitly saved"):
                    CentralizedHistory.from_state_dict(private.state_dict())

    def test_scalar_team_gae_draw_capture_elimination_and_bootstrap(self):
        game, private, central = histories()
        state = private.state_for(game)
        def transition(team, rewards=(0.0, 0.0), terminal=False, values=(0.0, 0.0), learnable=True):
            return MAPPOTransition(state, state.legal_actions[0], -1.0,
                                   (central.value_state_for(0), central.value_state_for(1)),
                                   rewards, terminal, team, learnable, values)
        def targets(trace, bootstrap=(0.0, 0.0), discount=1.0):
            return team_generalized_advantages(trace, bootstrap_values=bootstrap,
                discount=discount, gae_lambda=1.0, behavior_version=9)
        # Draws penalize both teams, regardless of actor order or skipped seats.
        draw = targets([transition(0), transition(1), transition(1, (-.15, -.15), True)], (999., 999.))
        for item in draw:
            self.assertAlmostEqual(item.value_target, -.15)
            self.assertEqual((item.old_draw_value, item.draw_value_target), (0., 0.))
        # A capture on the same step as a draw is additive, not zero sum.
        capture = targets([transition(0), transition(1, (.05, -.35), True)])
        self.assertAlmostEqual(capture[0].value_target, .05)
        self.assertAlmostEqual(capture[1].value_target, -.35)
        # Frozen opponent turns are bridges for the learning team, not samples
        # for that opponent's actor. Consecutive allies need no sign correction.
        bridge = targets([transition(0), transition(0, learnable=False), transition(0, (1., -1.), True)])
        self.assertEqual([s.value_target for s in bridge], [1., 1., 1.])
        self.assertFalse(bridge[1].learnable)
        truncated = targets([transition(1, values=(.2, -.4))], (.7, -.3), discount=.9)[0]
        self.assertAlmostEqual(truncated.value_target, -.27)
        self.assertAlmostEqual(truncated.advantage, .13)
        reset = targets([transition(0, (1., -1.), True), transition(0, (-.15, -.15), True)])
        self.assertEqual([s.value_target for s in reset], [1., -.15])

    def test_fusion_matches_full_tensor_packed_and_incremental_paths(self):
        config = tiny_config(max_transitions=7, ppo_tensor_learner=False)
        ordinary = GamePolicyTransformer(config).eval()
        tensor = GamePolicyTransformer(replace(config, ppo_tensor_learner=True)).eval()
        cached = copy.deepcopy(ordinary)
        tensor.load_state_dict(ordinary.state_dict())
        cached.start_ppo_inference_cache(capacity=4, behavior_version=0)
        game, history, _ = histories(window=7)
        held = []
        with torch.inference_mode():
            for _ in range(12):
                state = history.state_for(game)
                held.append(state)
                expected = ordinary._encode_full([state]).context
                torch.testing.assert_close(tensor._encode_full([state]).context, expected, atol=5e-6, rtol=5e-6)
                torch.testing.assert_close(cached.encode([state]).context, expected, atol=5e-6, rtol=5e-6)
                game.step(game.legal_actions()[0])
                history.append_after_step(game)
            expected = ordinary._encode_full(held).context
            torch.testing.assert_close(tensor._encode_full(held, pack_prefixes=True).context,
                                       expected, atol=5e-6, rtol=5e-6)
        # The added fusion weights receive gradients in the learner path.
        ordinary.train()
        states = held[-2:]
        logs = ordinary(states, [[s.legal_actions[0]] for s in states])
        sum(log.sum() for log in logs).backward()
        self.assertGreater(float(ordinary.token_fusion.up.weight.grad.abs().sum()), 0.)

    def test_collection_counts_once_and_pool_restores_both_histories(self):
        config = tiny_config(ppo_fixed_kv=False)
        policy = GamePolicyTransformer(config).eval()
        critic = GameCentralValueTransformer(replace(config, temporal_layers=1)).eval()
        layout = PieceConditionedLayoutPointerDecoder(config).eval()
        def make_pool(centralized=True):
            return BaseGamePool(MODE, pool_size=2, max_transitions=16, max_game_plies=4,
                                seed=33, centralized_critic=centralized)
        pool = make_pool()
        samples, outcomes, metrics = collect_mappo_samples(pool, FrozenPolicyActor(policy),
            FrozenValueActor(critic, amp_dtype=None, max_batch_size=2, deferred=True), layout,
            count=17, behavior_version=0, discount=1., gae_lambda=.95, flag_capture_reward=.2)
        self.assertEqual((len(samples), metrics.environment_plies, metrics.base_plies), (17, 17, 17))
        self.assertEqual((metrics.base_games_completed, len(outcomes)), (4, 16))
        self.assertEqual(metrics.terminal_continuations, 0)
        for sample in samples:
            self.assertFalse(sample.state.centralized)
            self.assertTrue(sample.value_state.centralized)
            self.assertTrue(sample.other_team_value.state.centralized)
            self.assertIsNot(sample.other_team_value.state, sample.value_state)
            self.assertEqual(sample.value_state.legal_actions, ())
            self.assertIn(sample.action, sample.state.legal_actions)
        restored = make_pool()
        restored.load_state_dict(pool.state_dict())
        self.assertEqual(restored.state_dict(), pool.state_dict())
        with self.assertRaisesRegex(ValueError, "observation contract"):
            make_pool(False).load_state_dict(pool.state_dict())
        broken = copy.deepcopy(pool.state_dict())
        broken["slots"][0].pop("critic_history")
        with self.assertRaisesRegex(ValueError, "global histories"):
            restored.load_state_dict(broken)

    def test_actor_and_scalar_value_losses_use_their_own_observations(self):
        config = tiny_config()
        policy = GamePolicyTransformer(config).train()
        critic = GameCentralValueTransformer(replace(config, temporal_layers=1)).train()
        game, private, central = histories()
        state = private.state_for(game)
        sample = PPOSample(state, state.legal_actions[0], -1., 0., 1., 1., 0,
                           value_state=central.value_state_for(0))
        with patch.object(policy, "forward", wraps=policy.forward) as forward:
            policy_ppo_loss(policy, [sample], clip_epsilon=.2, entropy_coefficient=.01).loss.backward()
            self.assertIs(forward.call_args.args[0][0], state)
        self.assertTrue(all(p.grad is None for p in critic.parameters()))
        policy.zero_grad(set_to_none=True)
        loss = critic_ppo_loss(critic, [sample], clip_epsilon=.2, value_coefficient=.5)
        self.assertAlmostEqual(float(loss.loss.detach()), .25)
        loss.loss.backward()
        self.assertGreater(float(critic.value_head.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in policy.parameters()))

    def test_full_trainer_checkpoint_and_actor_only_inference(self):
        from junqi.training.inference import InferenceEngine
        from junqi.training.trainer import SelfPlayTrainer
        settings = TrainingSettings.from_yaml(CONFIG, MODE, tiny=True, overrides={
            "device": "cpu", "anchor_batch": 8, "policy_microbatch": 2,
            "adaptive_learning_rate": False, "inference_snapshot_every_updates": 0,
        })
        with tempfile.TemporaryDirectory() as directory:
            trainer = SelfPlayTrainer(settings, run_directory=directory)
            trainer.train()
            self.assertEqual(trainer.cumulative["environment_plies"], 8)
            payload = torch.load(trainer.checkpoints.latest_path, map_location="cpu", weights_only=False)
            require_current_checkpoint(payload)
            self.assertEqual(payload["format_version"], 17)
            for key in ('policy', 'critic'):
                self.assertEqual(payload[key]['action_encoder.projection.weight'].shape[1], 14)
                self.assertEqual(payload[key]['board_encoder.projection.weight'].shape[1], 3357)
                self.assertFalse(any(name in payload[key] for name in (
                    'no_interaction_embedding.weight', 'active_embedding.weight', 'revealed_embedding.weight')))
            with self.assertRaisesRegex(ValueError, 'version does not match'):
                require_current_checkpoint({**payload, 'format_version': 15})
            self.assertEqual(payload["algorithm"], "mappo")
            self.assertTrue(payload["critic_optimizer"]["state"])
            self.assertFalse(any("draw_value_head" in key for key in payload["critic"]))
            resumed = SelfPlayTrainer(settings, run_directory=directory)
            try:
                self.assertEqual(resumed.cumulative["environment_plies"], 8)
                self.assertEqual(resumed.pool.state_dict(), trainer.pool.state_dict())
                for p, q in zip(trainer.critic.parameters(), resumed.critic.parameters(), strict=True):
                    torch.testing.assert_close(p, q, atol=0, rtol=0)
                engine = InferenceEngine.from_checkpoint(trainer.checkpoints.latest_path, device="cpu")
                self.assertFalse(hasattr(engine, "critic"))
                game, history = engine.new_game(seed=87, max_plies=4)
                self.assertIn(engine.select_action(game, history), game.legal_actions())
            finally:
                resumed.historical.close()
                resumed.logger.close()
            invalid = {**payload, "algorithm": "ppo"}
            torch.save(invalid, trainer.checkpoints.latest_path)
            with self.assertRaisesRegex(RuntimeError, "algorithm does not match"):
                SelfPlayTrainer(settings, run_directory=directory)
