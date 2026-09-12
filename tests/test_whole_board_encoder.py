from __future__ import annotations

import copy
from dataclasses import replace
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("PyTorch training extra is not installed") from error

from junqi.training.checkpoint import CHECKPOINT_FORMAT_VERSION, require_current_checkpoint
from junqi.training.encoding import (
    BOARD_CODE_VOCAB_SIZE, BOARD_PAD_CODE, MAX_BOARD_POINTS, GameHistory,
)
from junqi.training.models import (
    GamePolicyTransformer, ModelConfig, WholeBoardEncoder, collate_policy_states,
)
from junqi.training.modes import TrainingMode, new_game


class WholeBoardEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_modes_use_one_whole_vector_projection_without_point_tokens(self):
        for dead_rules in (False, True):
            config = ModelConfig.tiny(dead_rules_enabled=dead_rules)
            encoder = WholeBoardEncoder(config)
            self.assertEqual(list(encoder.children()), [encoder.projection])
            self.assertIsInstance(encoder.projection, torch.nn.Linear)
            for mode in TrainingMode:
                with self.subTest(dead_rules=dead_rules, mode=mode):
                    game = new_game(mode, seed=501, dead_rules_enabled=dead_rules)
                    state = GameHistory.initialize(game, mode).state_for(game)
                    batch = collate_policy_states([state], device="cpu", dead_rules_enabled=dead_rules)
                    seen = []
                    handle = encoder.projection.register_forward_pre_hook(
                        lambda module, args: seen.append(tuple(args[0].shape)))
                    output = encoder(batch.board_codes, batch.point_mask, batch.mode_ids, batch.casualty_bits)
                    handle.remove()
                    width = MAX_BOARD_POINTS * BOARD_CODE_VOCAB_SIZE + 3 + (75 if dead_rules else 0)
                    self.assertEqual(seen, [(1, width)])
                    self.assertEqual(output.shape, (1, config.board_dim))
                    self.assertTrue(torch.isfinite(output).all())
                    output.square().sum().backward()
                    self.assertGreater(float(encoder.projection.weight.grad.abs().sum()), 0)
                    encoder.zero_grad(set_to_none=True)

    def test_vector_preserves_categories_positions_modes_and_padding(self):
        encoder = WholeBoardEncoder(ModelConfig.tiny(dead_rules_enabled=False))
        codes = torch.tensor([[30, 31, BOARD_PAD_CODE], [31, 30, BOARD_PAD_CODE]])
        mask = torch.tensor([[True, True, False], [True, True, False]])
        vector = encoder.encode_vector(codes, mask, torch.tensor([0, 1]), None)
        expected = torch.zeros_like(vector)
        for row, (first, second) in enumerate(((30, 31), (31, 30))):
            expected[row, first] = 1
            expected[row, BOARD_CODE_VOCAB_SIZE + second] = 1
            expected[row, encoder.board_feature_dim + row] = 1
        torch.testing.assert_close(vector, expected)
        padded_codes = torch.nn.functional.pad(codes, (0, 126), value=BOARD_PAD_CODE)
        padded_mask = torch.nn.functional.pad(mask, (0, 126), value=False)
        torch.testing.assert_close(
            encoder(codes, mask, torch.tensor([0, 1]), None),
            encoder(padded_codes, padded_mask, torch.tensor([0, 1]), None),
        )
        # An empty real square is a category; a missing/padded square is absent.
        empty = encoder.encode_vector(torch.tensor([[0]]), torch.tensor([[True]]), torch.tensor([0]), None)
        absent = encoder.encode_vector(torch.tensor([[0]]), torch.tensor([[False]]), torch.tensor([0]), None)
        self.assertEqual(float((empty - absent).sum()), 1)

    def test_casualties_are_part_of_the_same_projection_and_removed_when_disabled(self):
        enabled = WholeBoardEncoder(ModelConfig.tiny())
        disabled = WholeBoardEncoder(ModelConfig.tiny(dead_rules_enabled=False))
        self.assertEqual(enabled.projection.weight.numel() - disabled.projection.weight.numel(), 75 * 32)
        codes, mask, modes = torch.tensor([[30]]), torch.tensor([[True]]), torch.tensor([0])
        dead = torch.zeros(1, 75)
        dead[0, 7] = 1
        vector = enabled.encode_vector(codes, mask, modes, dead)
        torch.testing.assert_close(vector[:, -75:], dead)
        with self.assertRaisesRegex(ValueError, "absent"):
            disabled(codes, mask, modes, dead)
        with self.assertRaisesRegex(ValueError, "casualty_bits"):
            enabled(codes, mask, modes, None)

    def test_invalid_categories_cannot_alias_another_point_or_auxiliary_field(self):
        encoder = WholeBoardEncoder(ModelConfig.tiny(dead_rules_enabled=False))
        for codes, modes in (([[139]], [0]), ([[-1]], [0]), ([[0]], [-1]), ([[0]], [3])):
            with self.subTest(codes=codes, modes=modes), self.assertRaises(RuntimeError):
                encoder.encode_vector(
                    torch.tensor(codes), torch.tensor([[True]]), torch.tensor(modes), None
                )

    def test_repeated_cached_state_needs_no_board_encoding_and_matches_uncached_policy(self):
        for mode in TrainingMode:
            config = ModelConfig.tiny()
            model = GamePolicyTransformer(config).eval()
            reference = copy.deepcopy(model)
            game = new_game(mode, seed=503)
            history = GameHistory.initialize(game, mode, max_transitions=16)
            state = history.state_for(game)
            model.start_inference_board_cache()
            with torch.inference_mode():
                first = model.encode([state])
                count = model._board_encoder_tokens
                cached = model.encode([state])
                self.assertEqual(model._board_encoder_tokens, count)
                torch.testing.assert_close(cached.context, first.context)
                logs = model([state], [state.legal_actions])[0]
                expected = reference([state], [state.legal_actions])[0]
                torch.testing.assert_close(logs, expected)
                self.assertAlmostEqual(float(logs.exp().sum()), 1.0, places=5)
                sampled, sampled_logs = model.sample_action_groups([state], count=8)
                self.assertTrue(all(action in state.legal_actions for action in sampled[0]))
                recomputed = reference([state], sampled)[0]
                torch.testing.assert_close(sampled_logs[0], recomputed)

    def test_old_spatial_checkpoints_are_rejected_before_loading_weights(self):
        require_current_checkpoint({"format_version": CHECKPOINT_FORMAT_VERSION})
        with self.assertRaisesRegex(ValueError, "whole-board linear.*version 6"):
            require_current_checkpoint({"format_version": 4})
        with self.assertRaisesRegex(ValueError, "unsupported board encoder"):
            replace(ModelConfig.tiny(), board_encoder_type="graph_transformer")


if __name__ == "__main__":
    unittest.main()
