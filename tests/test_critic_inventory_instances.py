"""Physical copy identities in the inventory Critic, including continuation."""
from dataclasses import replace
import copy
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from junqi import ArmPoint, GameConfig, InformationMode, JunqiGame, Piece, PieceType as K
from junqi.inventory import INVENTORY_ENCODER_TYPE, INVENTORY_SLOTS
from junqi.training.encoding import GameHistory
from junqi.training.history_arrays import inventory_words
from junqi.training.mappo_encoding import CentralizedHistory, centralized_observation_rows
from junqi.training.models import GameCentralValueTransformer, ModelConfig, PieceConditionedLayoutPointerDecoder, collate_policy_states
from junqi.training.modes import TrainingMode, new_game
from junqi.training.packed_observation import observation_rows
from junqi.training.ppo_environment import ParallelPPOEnvironment, _advance, _serialize
from junqi.training.rollout import BaseGamePool


MODE = TrainingMode.FOUR_DARK


def config():
    return replace(ModelConfig.tiny(), board_dim=48, action_dim=16,
                   board_encoder_type=INVENTORY_ENCODER_TYPE, token_fusion_hidden_dim=64,
                   ppo_cuda_graphs=False)


def sparse(extra):
    pieces = {ArmPoint(owner, 6, 2): Piece(owner, K.FLAG) for owner in range(4)}
    pieces.update({ArmPoint(owner, 3, 3): Piece(owner, K.COMMANDER) for owner in range(4)})
    pieces.update(extra)
    game = JunqiGame.from_position(GameConfig(information_mode=InformationMode.FOUR_DARK), pieces)
    game.enable_inventory_tracking()
    centralized_observation_rows(game, MODE)
    return game


@pytest.mark.parametrize('mode', [MODE, TrainingMode.DOUBLE_OPEN, TrainingMode.TWO_PLAYER])
def test_every_piece_has_one_distinct_critic_slot_and_actor_keeps_candidates(mode):
    torch.set_num_threads(1)
    game = new_game(mode, seed=41)
    GameHistory.initialize(game, mode)
    private = observation_rows(game, mode).copy()
    history = CentralizedHistory.initialize(game, mode)
    np.testing.assert_array_equal(observation_rows(game, mode), private)
    states = [player.as_value_state() for player in history.history.players]
    batch = collate_policy_states(states, device='cpu')
    encoder = GameCentralValueTransformer(config()).board_encoder
    vector = encoder.encode_vector(batch.board_codes, batch.point_mask, batch.mode_ids,
                                   batch.casualty_bits, batch.passes_remaining, batch.inventory_masks)
    features = vector[:, :129 * 26].reshape(len(states), 129, 26)
    for viewer in range(game.config.player_count):
        used = [set() for _ in range(game.config.player_count)]
        board = game.board_for(viewer)
        for point, piece in game.pieces.items():
            code = board.encode(point)
            assert features[viewer, code, :25].sum() == 1
            slot = int(features[viewer, code, :25].argmax())
            assert INVENTORY_SLOTS[slot] is piece.kind
            assert slot not in used[piece.owner]
            used[piece.owner].add(slot)
            assert slot == game._critic_inventory_slots[point]
        assert all(row == set(range(25)) for row in used)
    for point, piece in game.pieces.items():
        assert game.inventory_mask(piece.owner, point) == 1 << game._critic_inventory_slots[point]
        if mode is TrainingMode.DOUBLE_OPEN:
            assert game.inventory_mask((piece.owner + 2) % 4, point) == game.inventory_mask(piece.owner, point)


@pytest.mark.parametrize('defender', [None, K.MINE, K.ARMY_COMMANDER, K.ENGINEER])
def test_move_and_combat_preserve_survivor_slots_without_renumbering(defender):
    start, end = ArmPoint(0, 2, 3), ArmPoint(0, 1, 3)
    if defender is K.MINE:
        start, end = ArmPoint(1, 4, 3), ArmPoint(1, 5, 3)
    peers = (ArmPoint(0, 3, 2), ArmPoint(0, 4, 3))
    pieces = {p: Piece(0, K.ENGINEER) for p in (start, *peers)}
    if defender is not None:
        pieces[end] = Piece(1, defender)
    game = sparse(pieces)
    before = dict(game._critic_inventory_slots)
    game.step((game.board_for(0).encode(start), game.board_for(0).encode(end)))
    expected = {p: slot for p, slot in before.items() if p in game.pieces and p != end}
    if defender in (None, K.MINE):
        expected[end] = before[start]
    elif defender is K.ARMY_COMMANDER:
        expected[end] = before[end]
    assert game._critic_inventory_slots == expected
    for point, piece in game.pieces.items():
        assert game.inventory_mask(piece.owner, point) == 1 << expected[point]
    assert [expected[p] for p in peers] == [before[p] for p in peers]
    if defender in (K.ARMY_COMMANDER, K.ENGINEER):
        assert [expected[p] for p in peers] == [7, 8]
    masks = inventory_words(torch.from_numpy(centralized_observation_rows(game, MODE)), 129, True)
    for viewer in range(4):
        for point, slot in expected.items():
            assert masks[viewer, game.board_for(viewer).encode(point)] == 1 << slot


def test_slots_follow_reordering_moves_and_survive_clone_process_and_elimination():
    start, end = ArmPoint(0, 5, 1), ArmPoint(0, 1, 3)
    enemy = ArmPoint(0, 1, 2)
    game = sparse({start: Piece(0, K.ENGINEER), enemy: Piece(3, K.ARMY_COMMANDER),
                   ArmPoint(0, 2, 3): Piece(0, K.ENGINEER), ArmPoint(0, 3, 2): Piece(0, K.ENGINEER)})
    original_slot = game._critic_inventory_slots[start]
    clone = game.clone()
    pool = SimpleNamespace(mode=MODE, slots=[SimpleNamespace(game=clone)])
    with ParallelPPOEnvironment(1) as worker:
        worker.begin(pool, centralized_critic=True)
        actions = ((game.board_for(0).encode(start), game.board_for(0).encode(end)),
                   (game.board_for(3).encode(enemy), game.board_for(3).encode(end)))
        for action in actions:
            worker.submit([0], [action])
            expected = _advance(0, game, action, MODE, centralized_critic=True)
            actual = worker.receive()[0]
            np.testing.assert_array_equal(actual.records, expected.records)
            np.testing.assert_array_equal(actual.centralized_records, expected.centralized_records)
            if game.current_player == 3:
                assert game._critic_inventory_slots[end] == original_slot == 8
                assert clone._critic_inventory_slots[start] == original_slot
        worker.synchronize(pool)
    assert pool.slots[0].game.state_key() == game.state_key()
    assert pickle.loads(_serialize(game)).state_key() == game.state_key()
    game._eliminate_player(3)
    assert set(game._critic_inventory_slots) == set(game.pieces)
    assert all(game.pieces[p].owner != 3 for p in game._critic_inventory_slots)


def test_pool_resume_keeps_copy_ids_and_rejects_missing_or_mismatched_ids():
    torch.set_num_threads(1)
    def make_pool():
        return BaseGamePool(MODE, pool_size=1, max_transitions=3, max_game_plies=100,
                            seed=33, centralized_critic=True)
    pool = make_pool()
    pool.fill(PieceConditionedLayoutPointerDecoder(config()).eval(), 0)
    slot = pool.slots[0]
    for _ in range(7):
        row = _advance(0, slot.game, slot.game.legal_actions()[0], MODE, centralized_critic=True)
        slot.history.append_encoded_rows(row.records)
        slot.critic_history.append_encoded_rows(row.centralized_records)
    saved = pool.state_dict()
    resumed = make_pool()
    resumed.load_state_dict(saved)
    assert resumed.state_dict() == saved
    action = slot.game.legal_actions()[0]
    expected = _advance(0, slot.game, action, MODE, centralized_critic=True)
    actual = _advance(0, resumed.slots[0].game, action, MODE, centralized_critic=True)
    np.testing.assert_array_equal(actual.centralized_records, expected.centralized_records)
    for corruption in ('missing', 'duplicate', 'swapped', 'old_contract'):
        broken = copy.deepcopy(saved)
        raw = broken['slots'][0]
        if corruption == 'missing':
            raw['game'].pop('critic_inventory_slots')
        elif corruption == 'old_contract':
            raw['critic_history']['observation_contract'] = 'referee_global_v1'
        else:
            pieces = raw['game']['pieces']
            peers = [p for p, piece in pieces.items() if piece.owner == 0 and piece.kind is K.MINE]
            first, second = peers[:2]
            indices = raw['game']['critic_inventory_slots']
            if corruption == 'duplicate':
                indices[first] = indices[second]
            else:
                indices[first], indices[second] = indices[second], indices[first]
        with pytest.raises(ValueError, match='critic'):
            make_pool().load_state_dict(broken)
