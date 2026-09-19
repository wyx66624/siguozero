from __future__ import annotations

import copy
from dataclasses import asdict, replace
import unittest
from pathlib import Path
import tempfile
import numpy as np
import torch
import yaml

from junqi.inventory import INVENTORY_ENCODER_TYPE
from junqi.training.checkpoint import CHECKPOINT_FORMAT_VERSION, require_current_checkpoint
from junqi.training.encoding import BOARD_PAD_CODE, MAX_BOARD_POINTS, GameHistory
from junqi.training.history_arrays import HistoryArrayView, record_array
from junqi.training.models import GamePolicyTransformer, ModelConfig, WholeBoardEncoder, collate_policy_states
from junqi.training.modes import TrainingMode, new_game
from junqi.training.settings import TrainingSettings


class WholeBoardEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_modes_default_to_multihot_without_point_tokens(self):
        config = ModelConfig.tiny()
        self.assertEqual(config.board_encoder_type, INVENTORY_ENCODER_TYPE)
        encoder = WholeBoardEncoder(config)
        self.assertEqual(list(encoder.children()), [encoder.projection])
        self.assertEqual(encoder.input_dim, MAX_BOARD_POINTS * 26 + 3 + 75 + 4)
        for mode in TrainingMode:
            with self.subTest(mode=mode):
                game = new_game(mode, seed=501)
                state = GameHistory.initialize(game, mode).state_for(game)
                batch = collate_policy_states([state], device='cpu')
                self.assertIsNotNone(game._inventory)
                self.assertEqual(batch.inventory_masks.shape, batch.board_codes.shape)
                output = encoder(batch.board_codes, batch.point_mask, batch.mode_ids,
                                 batch.casualty_bits, batch.passes_remaining, batch.inventory_masks)
                self.assertEqual(output.shape, (1, config.board_dim))
                self.assertTrue(torch.isfinite(output).all())
                output.square().sum().backward()
                self.assertGreater(float(encoder.projection.weight.grad.abs().sum()), 0)
                encoder.zero_grad(set_to_none=True)

    def test_vector_preserves_candidate_slots_ownership_positions_and_padding(self):
        encoder = WholeBoardEncoder(ModelConfig.tiny())
        codes = torch.tensor([[30, 1, BOARD_PAD_CODE], [1, 30, BOARD_PAD_CODE]])
        mask = torch.tensor([[True, True, False], [True, True, False]])
        slots = torch.tensor([[1, 3, 0], [3, 1, 0]])
        modes, casualties = torch.tensor([0, 1]), torch.zeros(2, 75)
        vector = encoder.encode_vector(codes, mask, modes, casualties, inventory_masks=slots)
        expected = torch.zeros_like(vector)
        expected[:, 0] = expected[:, 26] = 1
        expected[0, 27] = expected[1, 1] = 1
        expected[:, 25] = torch.tensor([1, 2])
        expected[:, 51] = torch.tensor([2, 1])
        expected[0, encoder.board_feature_dim] = expected[1, encoder.board_feature_dim + 1] = 1
        expected[:, -4:] = 4
        torch.testing.assert_close(vector, expected)
        padded_codes = torch.nn.functional.pad(codes, (0, 126), value=BOARD_PAD_CODE)
        padded_mask = torch.nn.functional.pad(mask, (0, 126), value=False)
        padded_slots = torch.nn.functional.pad(slots, (0, 126), value=0)
        torch.testing.assert_close(
            encoder(codes, mask, modes, casualties, inventory_masks=slots),
            encoder(padded_codes, padded_mask, modes, casualties, inventory_masks=padded_slots))
        values = (torch.tensor([[0]]), torch.tensor([0]), casualties[:1])
        empty = encoder.encode_vector(values[0], torch.tensor([[True]]), values[1], values[2], inventory_masks=torch.tensor([[0]]))
        absent = encoder.encode_vector(values[0], torch.tensor([[False]]), values[1], values[2], inventory_masks=torch.tensor([[0]]))
        torch.testing.assert_close(empty, absent)

    def test_candidate_masks_and_casualties_are_required(self):
        encoder = WholeBoardEncoder(ModelConfig.tiny())
        codes, mask, modes = torch.tensor([[30]]), torch.tensor([[True]]), torch.tensor([0])
        dead = torch.zeros(1, 75)
        dead[0, 7] = 1
        vector = encoder.encode_vector(codes, mask, modes, dead, inventory_masks=torch.ones_like(codes))
        torch.testing.assert_close(vector[:, -79:-4], dead)
        with self.assertRaisesRegex(ValueError, 'candidate masks'):
            encoder(codes, mask, modes, dead)
        with self.assertRaisesRegex(ValueError, 'casualty_bits'):
            encoder(codes, mask, modes, None, inventory_masks=torch.ones_like(codes))

    def test_legacy_history_is_rejected_instead_of_dropping_candidates(self):
        mode = TrainingMode.FOUR_DARK
        game = new_game(mode, seed=504)
        history = GameHistory.initialize(game, mode)
        serialized = history.state_dict()
        for player in serialized['players']:
            for record in player['records']:
                record.pop('inventory_masks')
        with self.assertRaisesRegex(ValueError, 'inventory candidate masks'):
            GameHistory.from_state_dict(serialized)
        with self.assertRaisesRegex(ValueError, 'inventory candidate masks'):
            record_array(replace(history.players[0].records[0], inventory_masks=None), mode, True)
        with self.assertRaisesRegex(ValueError, 'observation width'):
            HistoryArrayView(mode, True, (0, 0), 0, (np.zeros((1, 129 + 75 + 14), dtype=np.int16),))

    def test_repeated_cached_state_matches_uncached_policy(self):
        for mode in TrainingMode:
            model = GamePolicyTransformer(ModelConfig.tiny()).eval()
            reference = copy.deepcopy(model)
            game = new_game(mode, seed=503)
            state = GameHistory.initialize(game, mode, max_transitions=16).state_for(game)
            model.start_inference_board_cache()
            with torch.inference_mode():
                first = model.encode([state])
                count = model._board_encoder_tokens
                cached = model.encode([state])
                self.assertEqual(model._board_encoder_tokens, count)
                torch.testing.assert_close(cached.context, first.context)
                logs = model([state], [state.legal_actions])[0]
                torch.testing.assert_close(logs, reference([state], [state.legal_actions])[0])
                self.assertAlmostEqual(float(logs.exp().sum()), 1.0, places=5)
                sampled, sampled_logs = model.sample_action_groups([state], count=8)
                self.assertTrue(all(action in state.legal_actions for action in sampled[0]))
                torch.testing.assert_close(sampled_logs[0], reference([state], sampled)[0])

    def test_old_board_architectures_are_rejected_before_loading_weights(self):
        payload = {'format_version': CHECKPOINT_FORMAT_VERSION, 'config': {'model': asdict(ModelConfig.tiny())}}
        require_current_checkpoint(payload)
        for version in range(4, 15):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'multihot.*version 15'):
                require_current_checkpoint({**payload, 'format_version': version})
        for architecture in ('whole_board_one_hot_linear', 'graph_transformer'):
            with self.subTest(architecture=architecture):
                with self.assertRaisesRegex(ValueError, 'unsupported board encoder'):
                    replace(ModelConfig.tiny(), board_encoder_type=architecture)
                with self.assertRaisesRegex(ValueError, 'multihot'):
                    require_current_checkpoint({**payload, 'config': {'model': {'board_encoder_type': architecture}}})
        with self.assertRaisesRegex(ValueError, '3436 board inputs'):
            require_current_checkpoint({**payload, 'policy': {'board_encoder.projection.weight': torch.empty(32, 129 * 139 + 82)}})

    def test_yaml_cannot_select_the_removed_encoder_even_with_tiny_or_scale_overrides(self):
        base = yaml.safe_load((Path(__file__).parents[1] / 'configs/bootstrap.yaml').read_text(encoding='utf-8'))
        for scope in ('policy', 'four_player'):
            data = copy.deepcopy(base)
            if scope == 'policy':
                data['models']['policy']['board_encoder']['architecture'] = 'whole_board_one_hot_linear'
            else:
                data['models']['four_player']['board_encoder_type'] = 'whole_board_one_hot_linear'
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'old_encoder.yaml'
                path.write_text(yaml.safe_dump(data), encoding='utf-8')
                for tiny in (False, True):
                    for scale in ('bootstrap', 'main', 'extended'):
                        with self.subTest(scope=scope, tiny=tiny, scale=scale):
                            with self.assertRaisesRegex(ValueError, 'unsupported board encoder'):
                                TrainingSettings.from_yaml(path, TrainingMode.FOUR_DARK, tiny=tiny, model_scale=scale)


if __name__ == '__main__':
    unittest.main()
