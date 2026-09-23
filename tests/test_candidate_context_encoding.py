"""Candidate-only boards and pass-aware actions across Actor/Critic paths."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from junqi import ArmPoint, GameConfig, InformationMode, JunqiGame, Piece, PieceType
from junqi.inventory import INVENTORY_ENCODER_TYPE, INVENTORY_ONLY_ENCODER_TYPE
from junqi.pieces import PIECE_COUNTS
from junqi.training.checkpoint import require_current_checkpoint, restore_training_state
from junqi.training.checkpoint_format import checkpoint_format_for_model
from junqi.training.encoding import (
    CONTEXT_ACTION_ENCODER_TYPE, PASS_CONTEXT_ACTION_ENCODER_TYPE, GameHistory,
)
from junqi.training.history_arrays import record_array
from junqi.training.mappo_encoding import CentralizedHistory, centralized_observation_rows
from junqi.training.models import (
    GameCentralValueTransformer, GamePolicyTransformer, ModelConfig, PublicActionEncoder,
)
from junqi.training.modes import TrainingMode, new_game


MODE = TrainingMode.FOUR_DARK


def candidate_config(**overrides):
    return replace(ModelConfig.tiny(), board_dim=48, action_dim=16,
                   board_encoder_type=INVENTORY_ONLY_ENCODER_TYPE,
                   action_encoder_type=PASS_CONTEXT_ACTION_ENCODER_TYPE,
                   token_fusion_hidden_dim=64, temporal_causal_sdpa=True,
                   ppo_cuda_graphs=False, **overrides)


def test_initial_action_preserves_all_pass_counts_and_relative_seat_order():
    encoder = PublicActionEncoder(14, joint_context=True, passes_context=True)
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.eye(14))
        encoder.projection.bias.zero_()
    fields = torch.tensor([[float('nan')] * 6, [-2., -3., -2., -2., 3., 69.]])
    present = torch.tensor([False, True])
    context = dict(no_interaction=torch.tensor([69, 1]), active_mask=torch.tensor([5, 15]))
    encoded = encoder(fields, present, passes_remaining=torch.tensor([[4, 3, 2, 1], [0, 1, 2, 3]]),
                      **context)
    torch.testing.assert_close(encoded, torch.tensor([
        [0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 4., 3., 2., 1.],
        [-2., -3., -2., -2., 3., 69., 1., 1., 1., 1., 0., 1., 2., 3.]]))
    with pytest.raises(ValueError, match='four remaining-pass counts'):
        encoder(fields, present, **context)
    with pytest.raises(ValueError, match='four remaining-pass counts'):
        encoder(fields, present, passes_remaining=torch.ones(2, 3), **context)


@pytest.mark.parametrize('model_type', [GamePolicyTransformer, GameCentralValueTransformer])
@pytest.mark.parametrize('mode', [MODE, TrainingMode.DOUBLE_OPEN])
def test_casualty_metadata_is_not_a_network_input_and_passes_only_change_actions(model_type, mode):
    torch.set_num_threads(1)
    torch.manual_seed(201)
    model = model_type(candidate_config()).eval()
    game = new_game(mode, seed=21)
    if model_type is GameCentralValueTransformer:
        state = CentralizedHistory.initialize(game, mode).value_state_for(0)
    else:
        state = GameHistory.initialize(game, mode).state_for(game)
    record = state.records[0]
    casualties = replace(record, known_casualty_bits=tuple(1 - bit for bit in record.known_casualty_bits))
    passes = replace(record, passes_remaining=(0, 1, 2, 3))
    records = [record, casualties, passes]
    states = [replace(state, records=(r,)) for r in records]
    assert model.board_encoder.input_dim == 3357
    assert model.board_encoder.casualty_feature_dim == model.board_encoder.pass_feature_dim == 0
    assert model.action_encoder.projection.in_features == 14
    with torch.inference_mode():
        board, _ = model._record_board_batch(records, mode)
        torch.testing.assert_close(board[0], board[1], rtol=0, atol=0)
        torch.testing.assert_close(board[0], board[2], rtol=0, atol=0)
        tokens = model._record_temporal_tokens(records, mode, board, 0)
        torch.testing.assert_close(tokens[0], tokens[1], rtol=0, atol=0)
        assert not torch.allclose(tokens[0], tokens[2])
        rows = torch.from_numpy(np.stack([record_array(r, mode, True) for r in records]))[:, None, :]
        raw = model._embed_observation_rows(rows, mode, torch.zeros((3, 1), dtype=torch.long))
        torch.testing.assert_close(tokens, raw, rtol=2e-6, atol=2e-6)
        features = model._encode_full(states).context
        torch.testing.assert_close(features[0], features[1], rtol=0, atol=0)
        assert not torch.allclose(features[0], features[2])
        masks = list(record.inventory_masks)
        occupied = next(i for i, mask in enumerate(masks) if mask)
        masks[occupied] ^= 1 << 8
        changed_board, _ = model._record_board_batch([replace(record, inventory_masks=tuple(masks))], mode)
        assert not torch.allclose(changed_board[0], board[0])


@pytest.mark.parametrize('model_type', [GamePolicyTransformer, GameCentralValueTransformer])
def test_new_encoding_matches_list_array_packed_and_sliding_kv_with_gradients(model_type):
    torch.set_num_threads(1)
    torch.manual_seed(317)
    ordinary = model_type(candidate_config(max_transitions=5)).eval()
    tensor = model_type(candidate_config(max_transitions=5, ppo_tensor_learner=True)).eval()
    if model_type is GameCentralValueTransformer:
        with torch.no_grad():
            ordinary.value_head.weight.normal_(std=.1)
    tensor.load_state_dict(ordinary.state_dict())
    cached = copy.deepcopy(ordinary)
    cached.start_ppo_inference_cache(capacity=8, behavior_version=0)
    game = new_game(MODE, seed=91, max_plies=100)
    private = GameHistory.initialize(game, MODE, max_transitions=5)
    central = CentralizedHistory.initialize(game, MODE, max_transitions=5)
    private.enable_array_storage()
    central.enable_array_storage()
    held = []
    with torch.inference_mode():
        for turn in range(12):
            state = (central.value_state_for(game.current_player) if model_type is GameCentralValueTransformer
                     else private.state_for(game))
            held.append(state)
            expected = ordinary._encode_full([state]).context
            torch.testing.assert_close(cached.encode([state]).context, expected, atol=6e-6, rtol=6e-6)
            tuple_state = replace(state, records=tuple(state.records))
            torch.testing.assert_close(ordinary._encode_full([tuple_state]).context, expected, atol=6e-6, rtol=6e-6)
            action = (0, 0) if turn < 4 else next(a for a in game.legal_actions() if a != (0, 0))
            game.step(action)
            private.append_after_step(game)
            central.append_encoded_rows(centralized_observation_rows(game, MODE))
        torch.testing.assert_close(tensor._encode_full(held, pack_prefixes=True).context,
                                   ordinary._encode_full(held).context, atol=6e-6, rtol=6e-6)
    states = held[-3:]
    for model in (ordinary, tensor):
        model.train()
        if model_type is GameCentralValueTransformer:
            model(states, pack_sequences=True).sum().backward()
        else:
            model.ppo_statistics(states, [s.legal_actions[-1] for s in states], pack_sequences=True)[0].sum().backward()
        assert model.board_encoder.projection.weight.grad.abs().sum() > 0
        assert model.action_encoder.projection.weight.grad[:, -4:].abs().sum() > 0
    for p, q in zip(ordinary.parameters(), tensor.parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4)


def test_checkpoint_contract_keeps_legacy_encoders_and_rejects_mixed_shapes():
    new = candidate_config()
    for config, version, board_inputs, action_inputs in (
        (new, 17, 3357, 14),
        (replace(new, board_encoder_type=INVENTORY_ENCODER_TYPE,
                 action_encoder_type=CONTEXT_ACTION_ENCODER_TYPE), 16, 3436, 10),
        (ModelConfig.tiny(), 15, 3436, 6),
    ):
        model = GamePolicyTransformer(config)
        payload = dict(format_version=version, config={'model': asdict(config)}, policy=model.state_dict())
        require_current_checkpoint(payload)
        assert checkpoint_format_for_model(config) == version
        assert model.board_encoder.input_dim == board_inputs
        assert model.action_encoder.projection.in_features == action_inputs
        if version == 17:
            for key, columns, message in (
                ('board_encoder.projection.weight', 3436, '3357 board inputs'),
                ('action_encoder.projection.weight', 10, '14 action inputs'),
            ):
                corrupted = {**payload, 'policy': {**payload['policy'], key: torch.zeros(16, columns)}}
                with pytest.raises(ValueError, match=message):
                    require_current_checkpoint(corrupted)
            with pytest.raises(ValueError, match='version does not match'):
                require_current_checkpoint({**payload, 'format_version': 16})
    with pytest.raises(ValueError, match='pass counts'):
        replace(new, action_encoder_type=CONTEXT_ACTION_ENCODER_TYPE)
    with pytest.raises(ValueError, match='pass counts'):
        replace(new, board_encoder_type=INVENTORY_ENCODER_TYPE)


def test_resume_rejects_v16_into_v17_before_loading_any_weights():
    config = candidate_config()
    old = replace(config, board_encoder_type=INVENTORY_ENCODER_TYPE,
                  action_encoder_type=CONTEXT_ACTION_ENCODER_TYPE)
    payload = dict(format_version=16, config={'model': asdict(old), 'critic_layers': old.temporal_layers},
                   mode=MODE.value, dead_rules_enabled=True, algorithm='mappo')
    policy = GamePolicyTransformer(config)
    critic = GameCentralValueTransformer(config)
    with pytest.raises(RuntimeError, match='architecture/critic depth changed'):
        restore_training_state(payload, expected_mode=MODE.value, expected_dead_rules_enabled=True,
            expected_algorithm='mappo', policy=policy, critic=critic, layout=None,
            reference_policy=None, reference_layout=None, policy_optimizer=None,
            layout_optimizer=None)


def test_whole_board_union_reconstructs_casualties_before_and_after_known_engineer_death():
    pieces = {ArmPoint(owner, 6, 2): Piece(owner, PieceType.FLAG) for owner in range(4)}
    pieces.update({ArmPoint(owner, 3, 3): Piece(owner, PieceType.COMMANDER) for owner in range(4)})
    start, end = ArmPoint(0, 5, 1), ArmPoint(0, 1, 3)
    attacker = ArmPoint(0, 1, 2)
    pieces.update({start: Piece(0, PieceType.ENGINEER), attacker: Piece(3, PieceType.ARMY_COMMANDER),
                   ArmPoint(0, 2, 3): Piece(0, PieceType.ENGINEER),
                   ArmPoint(0, 3, 2): Piece(0, PieceType.ENGINEER)})
    game = JunqiGame.from_position(GameConfig(information_mode=InformationMode.FOUR_DARK), pieces)
    game.enable_inventory_tracking()
    from junqi.inventory import INVENTORY_SLOTS
    type_slots = {kind: sum(1 << i for i, k in enumerate(INVENTORY_SLOTS) if k == kind)
                  for kind in PIECE_COUNTS}
    for step in range(3):
        history = GameHistory.initialize(game, MODE)
        for player in history.players:
            record = player.records[0]
            for owner in (1, 2, 3):
                union = 0
                for code, mask in zip(record.board_codes, record.inventory_masks, strict=True):
                    relative = (code - 30) // 32 if code >= 30 else code
                    if code and relative == owner:
                        union |= mask
                deaths = {kind: n - (union & type_slots[kind]).bit_count() for kind, n in PIECE_COUNTS.items()}
                assert JunqiGame._casualty_bits(deaths) == record.known_casualty_bits[(owner - 1) * 25:owner * 25]
        if step == 0:
            game.step((20, 2))  # The railway turn publicly proves the first engineer.
        elif step == 1:
            board = game.board_for(3)
            game.step((board.encode(attacker), board.encode(end)))  # The identified engineer dies.
