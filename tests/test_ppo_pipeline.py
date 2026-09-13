"""Numerical and lifecycle checks for indexed histories and fixed-slot decoding."""
from __future__ import annotations

import copy
from dataclasses import replace
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from junqi.training.encoding import GameHistory, PolicyState, StateTokenRecord, ActionFeatures, history_prefix_groups, policy_history_key
from junqi.training.history_arrays import ArrayHistory, BLOCK_ROWS, HistoryArrayView
from junqi.training.models import GamePolicyTransformer, GameValueTransformer, ModelConfig, collate_policy_states
from junqi.training.modes import TrainingMode, mode_spec, new_game
from junqi.training.ppo import PPOSample, policy_ppo_loss


def histories(mode=TrainingMode.FOUR_DARK, *, dead=True, window=32, seed=411):
    game = new_game(mode, seed=seed, max_plies=90, dead_rules_enabled=dead)
    history = GameHistory.initialize(game, mode, max_transitions=window)
    history.enable_array_storage()
    return game, history


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_old_samples_survive_block_retirement_and_new_games_have_distinct_keys(self):
        initial = StateTokenRecord((0,) * 129, (0,) * 75, None, 0, 15, 0, 0)
        store = ArrayHistory(TrainingMode.FOUR_DARK, [initial], 17, (91, 0))
        expected = [initial]
        held = []
        for step in range(BLOCK_ROWS * 2 + 19):
            record = replace(initial, action=ActionFeatures(0, 1, step % 4), no_interaction_plies=step % 71)
            store.append(record)
            expected = [initial, *expected[1:][-16:], record]
            if step in (6, 200, 301, 530):
                held.append((store.view(), tuple(expected)))
        self.assertLessEqual(len(store.blocks), 2)
        for view, records in held:
            self.assertEqual(tuple(view), records)
        a = PolicyState(TrainingMode.FOUR_DARK, store.view(), ((0, 1),))
        other = ArrayHistory(TrainingMode.FOUR_DARK, list(a.records), 17, (92, 0))
        b = PolicyState(TrainingMode.FOUR_DARK, other.view(), a.legal_actions)
        self.assertNotEqual(policy_history_key(a), policy_history_key(b))

    def test_array_history_matches_legacy_observations_and_checkpoint_replay(self):
        for mode in TrainingMode:
            for dead in (False, True):
                with self.subTest(mode=mode, dead=dead):
                    game, array = histories(mode, dead=dead, window=8)
                    legacy = GameHistory.from_state_dict(array.state_dict())
                    rng = random.Random(311)
                    for _ in range(19):
                        if game.is_terminal:
                            break
                        left, right = array.state_for(game), legacy.state_for(game)
                        self.assertEqual(tuple(left.records), right.records)
                        self.assertEqual(left.legal_actions, right.legal_actions)
                        one = collate_policy_states([left], device="cpu", dead_rules_enabled=dead)
                        two = collate_policy_states([right], device="cpu", dead_rules_enabled=dead)
                        for name in ("board_codes", "casualty_bits", "action_fields", "action_present", "no_interaction",
                                     "active_mask", "revealed_mask", "current_player", "token_mask", "valid_indices"):
                            if getattr(one, name) is not None:
                                torch.testing.assert_close(getattr(one, name), getattr(two, name), rtol=0, atol=0)
                        game.step(rng.choice(game.legal_actions()))
                        array.append_after_step(game)
                        legacy.append_after_step(game)
                    self.assertEqual(array.state_dict(), legacy.state_dict())
                    restored = GameHistory.from_state_dict(array.state_dict())
                    restored.enable_array_storage()
                    self.assertEqual(restored.state_dict(), legacy.state_dict())
                    self.assertNotEqual(restored.players[0].records.identity, array.players[0].records.identity)

    def test_fast_grouping_and_collation_do_not_query_history_records(self):
        game, history = histories()
        states = []
        for _ in range(12):
            states.append(history.state_for(game, player=0))
            game.step(game.legal_actions()[0])
            history.append_after_step(game)
        with patch.object(HistoryArrayView, "__getitem__", side_effect=AssertionError("record lookup")), \
                patch.object(StateTokenRecord, "__hash__", side_effect=AssertionError("record hash")):
            self.assertEqual(len(history_prefix_groups(states)), 1)
            collate_policy_states([states[-1]], device="cpu")
            policy_history_key(states[-1])

    def test_flat_policy_loss_preserves_all_action_probabilities_entropy_and_gradients(self):
        game, history = histories()
        states = []
        rng = random.Random(45)
        for _ in range(10):
            states.append(history.state_for(game))
            game.step(rng.choice(game.legal_actions()))
            history.append_after_step(game)
        config = replace(ModelConfig.tiny(), max_transitions=32, activation_checkpointing=False)
        model = GamePolicyTransformer(config).train()
        reference = copy.deepcopy(model)

        class LegacyLossPolicy(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        samples = [PPOSample(state, state.legal_actions[0], -3., 0., (-1.) ** i, .3, 0)
                   for i, state in enumerate(states)]
        args = dict(clip_epsilon=.2, entropy_coefficient=.01, sequence_training=True)
        actual = policy_ppo_loss(model, samples, **args)
        expected = policy_ppo_loss(LegacyLossPolicy(reference), samples, **args)
        torch.testing.assert_close(actual.loss, expected.loss, rtol=2e-6, atol=2e-6)
        self.assertAlmostEqual(actual.metrics["policy/entropy"], expected.metrics["policy/entropy"], places=6)
        actual.loss.backward()
        expected.loss.backward()
        for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
            if p.grad is not None or q.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=3e-5, atol=3e-6, msg=name)

    def check_fixed(self, device="cpu", graphs=False):
        config = replace(ModelConfig.tiny(), board_dim=128, temporal_dim=256, temporal_heads=8,
                         temporal_layers=2, temporal_ffn_dim=128, max_transitions=20,
                         ppo_cuda_graphs=graphs, activation_checkpointing=False)
        policy = GamePolicyTransformer(config).to(device).eval()
        reference = copy.deepcopy(policy)
        policy.start_ppo_inference_cache(capacity=8, behavior_version=5)
        game, history = histories(window=20)
        rng = random.Random(7)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
            for step in range(32):
                if game.is_terminal:
                    break
                # Four seats, suffixes of 2-4 tokens, cached repeat queries, and
                # exact learned-position resets when the history window slides.
                if step % 3 != 1:
                    state = history.state_for(game)
                    actual = policy.encode([state]).context
                    expected = reference.encode([state]).context
                    tolerance = 2e-2 if device == "cuda" else 3e-6
                    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                    torch.testing.assert_close(policy.encode([state]).context, actual, rtol=0, atol=0)
                game.step(rng.choice(game.legal_actions()))
                history.append_after_step(game)
        store = policy._fixed_kv_store
        self.assertGreater(store.decode_tokens, store.decode_states)
        self.assertLessEqual(len(store.entries), 8)
        if graphs:
            self.assertIsNotNone(store.kernels)
            self.assertGreater(store.graph_captures, 0)
            self.assertGreaterEqual(store.graph_replays, store.graph_captures)
        policy.train()
        self.assertIsNone(policy._fixed_kv_store)

    def test_fixed_cache_outputs_match_full_attention_through_window_slides(self):
        self.check_fixed()

    def test_failed_append_keeps_old_prefix_valid_and_can_retry(self):
        config = replace(ModelConfig.tiny(), max_transitions=32, temporal_layers=2)
        policy = GamePolicyTransformer(config).eval()
        reference = copy.deepcopy(policy)
        game, history = histories()
        policy.start_ppo_inference_cache(capacity=4, behavior_version=1)
        with torch.inference_mode():
            policy.encode([history.state_for(game, player=0)])
            for _ in range(4):
                game.step(game.legal_actions()[0])
                history.append_after_step(game)
            state = history.state_for(game, player=0)
            store = policy._fixed_kv_store
            write = store.write

            def fail(layer, *args):
                if layer == 1:
                    raise torch.OutOfMemoryError("injected fixed KV write failure")
                write(layer, *args)

            with patch.object(store, "write", side_effect=fail), self.assertRaises(torch.OutOfMemoryError):
                policy.encode([state])
            self.assertEqual(store.entries[state.records.identity].length, 1)
            torch.testing.assert_close(policy.encode([state]).context, reference.encode([state]).context, rtol=3e-6, atol=3e-6)
        policy.start_ppo_inference_cache(capacity=4, behavior_version=2)
        self.assertEqual(len(policy._fixed_kv_store.entries), 0)
        policy.load_state_dict(reference.state_dict())
        self.assertIsNone(policy._fixed_kv_store)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and Triton")
    def test_cuda_direct_attention_without_graphs(self):
        self.check_fixed("cuda", False)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and Triton")
    def test_cuda_graph_decode_matches_reference(self):
        self.check_fixed("cuda", True)
