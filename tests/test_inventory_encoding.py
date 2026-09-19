"""Public inference, capacity constraints and every 384+128 input path."""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import replace
from itertools import permutations
import pickle
import random

import numpy as np
import pytest
import torch

from junqi import ArmPoint, GameConfig, InformationMode, JunqiGame, Piece, PieceType as K
from junqi.inventory import INVENTORY_ENCODER_TYPE, INVENTORY_SLOTS, TYPE_BITS, supported_domains, guaranteed_casualties
from junqi.pieces import PIECE_COUNTS, PIECE_TYPE_ORDER
from junqi.training.encoding import CONTEXT_ACTION_ENCODER_TYPE, GameHistory, PolicyState
from junqi.training.history_arrays import HistoryArrayView, record_array
from junqi.training.mappo_encoding import CentralizedHistory, centralized_observation_rows
from junqi.training.models import (GameCentralValueTransformer, GamePolicyTransformer, ModelConfig,
                                   PublicActionEncoder, collate_policy_states)
from junqi.training.modes import TrainingMode, new_game
from junqi.training.packed_observation import observation_rows
from junqi.training.ppo_environment import _advance, _serialize


MODE = TrainingMode.FOUR_DARK


def config(**kwargs):
    return replace(ModelConfig.tiny(), board_dim=48, action_dim=16,
                   board_encoder_type=INVENTORY_ENCODER_TYPE, token_fusion_hidden_dim=64,
                   action_encoder_type=CONTEXT_ACTION_ENCODER_TYPE,
                   temporal_causal_sdpa=True, ppo_cuda_graphs=False, **kwargs)


def sparse(extra, **kwargs):
    pieces = {ArmPoint(owner, 6, 2): Piece(owner, K.FLAG) for owner in range(4)}
    pieces.update({ArmPoint(owner, 3, 3): Piece(owner, K.COMMANDER) for owner in range(4)})
    pieces.update(extra)
    game = JunqiGame.from_position(GameConfig(information_mode=InformationMode.FOUR_DARK), pieces, **kwargs)
    game.enable_inventory_tracking()
    return game


def slots(mask, kind):
    return tuple((mask >> i) & 1 for i, k in enumerate(INVENTORY_SLOTS) if k is kind)


def restore(game):
    return JunqiGame.from_position(game.config, game.pieces,
        current_player=game.current_player, active_players=game.active_players,
        revealed_flags=game.revealed_flags, ply_count=game.ply_count,
        no_interaction_plies=game.no_interaction_plies, public_candidates=game.public_candidates,
        known_identities=game.known_identities, known_casualties=game.known_casualties,
        public_history=game.public_history, passes_remaining=game.passes_remaining,
        inventory_knowledge=game._inventory.state_dict(),
        critic_inventory_slots=game._critic_inventory_slots)


def test_matching_support_agrees_with_exhaustive_small_assignments():
    unknown = (K.ARMY_COMMANDER, K.DIVISION_COMMANDER, K.DIVISION_COMMANDER,
               K.BRIGADE_COMMANDER, K.BRIGADE_COMMANDER)
    fixed = list(INVENTORY_SLOTS)
    for kind in unknown:
        fixed.remove(kind)
    rng = random.Random(57)
    for _ in range(35):
        domains = [{kind for kind in set(unknown) if rng.random() < .6} | {actual} for actual in unknown]
        feasible = [assignment for assignment in set(permutations(unknown))
                    if all(kind in d for kind, d in zip(assignment, domains))]
        masks = tuple(TYPE_BITS[k] for k in fixed) + tuple(sum(TYPE_BITS[k] for k in d) for d in domains)
        result = supported_domains(masks)[-5:]
        assert result == tuple(frozenset(a[i] for a in feasible) for i in range(5))
        live = tuple(sorted([TYPE_BITS[k] for k in fixed] + [sum(TYPE_BITS[k] for k in d) for d in result[:2]]))
        dead = tuple(sorted(sum(TYPE_BITS[k] for k in d) for d in result[2:]))
        minimum = guaranteed_casualties(live, dead)
        assert minimum == tuple(min(Counter(a[2:])[kind] for a in feasible) for kind in PIECE_TYPE_ORDER)


def test_initial_26_features_empty_ownership_and_deployment_limits():
    torch.set_num_threads(1)
    game = new_game(MODE, seed=57)
    history = GameHistory.initialize(game, MODE)
    states = [history.state_for(game, seat) for seat in range(4)]
    batch = collate_policy_states(states, device='cpu')
    encoder = GamePolicyTransformer(config()).board_encoder
    vectors = encoder.encode_vector(batch.board_codes, batch.point_mask, batch.mode_ids,
                                    batch.casualty_bits, batch.passes_remaining, batch.inventory_masks)
    board = vectors[:, :129 * 26].reshape(4, 129, 26)
    assert encoder.input_dim == 129 * 26 + 3 + 75 + 4
    assert Counter(INVENTORY_SLOTS) == Counter(PIECE_COUNTS)
    assert (INVENTORY_SLOTS[0], INVENTORY_SLOTS[-1]) == (K.FLAG, K.COMMANDER)
    for viewer in range(4):
        for code, point in enumerate(game.observe(viewer).points):
            if point is None:
                assert not board[viewer, code].any()
            else:
                assert board[viewer, code, 25] == point.owner + 1
                assert torch.all((board[viewer, code, :25] == 0) | (board[viewer, code, :25] == 1))
        enemy = (viewer + 1) % 4
        for row in range(1, 7):
            for column in range(1, 6):
                point = ArmPoint(enemy, row, column)
                if point not in game._pieces:
                    continue
                mask = game.inventory_mask(viewer, point)
                if row == 1:
                    assert slots(mask, K.BOMB) == (0, 0)
                if row < 5:
                    assert slots(mask, K.MINE) == (0, 0, 0)
                if (row, column) not in ((6, 2), (6, 4)):
                    assert slots(mask, K.FLAG) == (0,)
    assert all(not counts for owners in game._known_casualties for counts in owners)


def test_front_row_restriction_follows_piece_into_second_row():
    start, end = ArmPoint(0, 1, 3), ArmPoint(0, 2, 3)
    game = sparse({start: Piece(0, K.DIVISION_COMMANDER)})
    game.step((game.board_for(0).encode(start), game.board_for(0).encode(end)))
    for viewer in (1, 2, 3):
        mask = game.inventory_mask(viewer, end)
        assert slots(mask, K.BOMB) == (0, 0)
        assert slots(mask, K.MINE) == (0, 0, 0)


def test_ambiguous_survivor_domain_follows_moves_clone_and_restore():
    # Beating a brigade leaves {division, army, commander}, not just "unknown".
    start, victim = ArmPoint(0, 2, 3), ArmPoint(0, 1, 3)
    game = sparse({start: Piece(0, K.ARMY_COMMANDER), victim: Piece(1, K.BRIGADE_COMMANDER)})
    game.step((7, 2))
    expected = frozenset((K.COMMANDER, K.ARMY_COMMANDER, K.DIVISION_COMMANDER))
    assert game._inventory.live[1][victim] == expected
    for _ in range(3):
        game.step((0, 0))
    game.step((2, 7))
    assert game._inventory.live[1][start] == expected
    assert game.clone().state_key() == game.state_key()
    restored = restore(game)
    assert restored.state_key() == game.state_key()
    assert restored._inventory.live[1][start] == expected
    restored._inventory.live[1][start] = frozenset((K.ARMY_COMMANDER,))
    assert game._inventory.live[1][start] == expected


def test_flag_reveal_excludes_other_hq_flag_and_all_live_commanders():
    commander, bomb = ArmPoint(0, 3, 3), ArmPoint(0, 2, 3)
    other_hq = ArmPoint(0, 6, 4)
    game = sparse({bomb: Piece(1, K.BOMB), other_hq: Piece(0, K.MINE)})
    game.step((game.board_for(0).encode(commander), game.board_for(0).encode(bomb)))
    assert game.revealed_flags[0]
    for viewer in range(4):
        assert slots(game.inventory_mask(viewer, other_hq), K.FLAG) == (0,)
        for point, piece in game._pieces.items():
            if piece.owner == 0:
                assert slots(game.inventory_mask(viewer, point), K.COMMANDER) == (0,)
        assert game._known_casualties[viewer][0][K.COMMANDER] == 1


def test_proven_engineer_death_shrinks_unknown_candidates_preserves_confirmed_slots():
    a, b = ArmPoint(0, 5, 1), ArmPoint(0, 1, 3)
    game = sparse({a: Piece(0, K.ENGINEER), ArmPoint(0, 1, 2): Piece(3, K.ARMY_COMMANDER),
                   ArmPoint(0, 2, 3): Piece(0, K.ENGINEER), ArmPoint(0, 3, 2): Piece(0, K.ENGINEER)})
    own = {p: game.inventory_mask(0, p) for p, piece in game.pieces.items() if piece.owner == 0}
    game.step((20, 2))  # An engineer-only railway turn, publicly proved.
    assert game._inventory.live[1][b] == frozenset((K.ENGINEER,))
    board = game.board_for(3)
    game.step((board.encode(ArmPoint(0, 1, 2)), board.encode(b)))
    for viewer in range(4):
        assert game._known_casualties[viewer][0][K.ENGINEER] == 1
        for point in (ArmPoint(0, 2, 3), ArmPoint(0, 3, 2)):
            if viewer == 0:
                assert game.inventory_mask(viewer, point) == own[point]
                assert own[point].bit_count() == 1
            else:
                assert slots(game.inventory_mask(viewer, point), K.ENGINEER) == (0, 1, 1)


def test_unknown_equal_rank_or_bomb_exchange_stays_ambiguous():
    target = ArmPoint(0, 2, 3)
    common = {ArmPoint(0, 1, 3): Piece(0, K.ARMY_COMMANDER),
              ArmPoint(1, 2, 3): Piece(1, K.ENGINEER), ArmPoint(1, 4, 3): Piece(1, K.REGIMENT_COMMANDER)}
    twins = [sparse({**common, target: Piece(1, kind)}) for kind in (K.ARMY_COMMANDER, K.BOMB)]
    before = [observation_rows(g, MODE)[0].copy() for g in twins]
    np.testing.assert_array_equal(*before)
    # The training-only histories know which kind entered the exchange even
    # when the two positions become identical after mutual removal.
    central = [CentralizedHistory.initialize(g, MODE) for g in twins]
    assert central[0].value_state_for(0).records != central[1].value_state_for(0).records
    for game in twins:
        game.step((2, 7))
        assert game._inventory.dead[0][1][-1] == frozenset((K.ARMY_COMMANDER, K.BOMB))
        assert game._known_casualties[0][1][K.BOMB] == 0
        assert game._known_casualties[0][1][K.ARMY_COMMANDER] == 0
    np.testing.assert_array_equal(*(observation_rows(g, MODE)[0] for g in twins))
    for history, game in zip(central, twins):
        history.append_encoded_rows(centralized_observation_rows(game, MODE))
    assert central[0].value_state_for(0).records != central[1].value_state_for(0).records


@pytest.mark.parametrize('mode', [MODE, TrainingMode.DOUBLE_OPEN])
def test_critic_inventory_has_only_the_true_kind_before_and_after_exchange(mode):
    target, survivor = ArmPoint(0, 2, 3), ArmPoint(1, 2, 3)
    pieces = {ArmPoint(owner, 6, 2): Piece(owner, K.FLAG) for owner in range(4)}
    pieces.update({ArmPoint(owner, 3, 3): Piece(owner, K.COMMANDER) for owner in range(4)})
    pieces.update({ArmPoint(0, 1, 3): Piece(0, K.ARMY_COMMANDER),
                   target: Piece(1, K.BOMB), survivor: Piece(1, K.ENGINEER),
                   ArmPoint(1, 4, 3): Piece(1, K.ENGINEER)})
    game = JunqiGame.from_position(GameConfig(information_mode=InformationMode(mode.value)), pieces)
    game.enable_inventory_tracking()
    model = GameCentralValueTransformer(config()).eval()
    for turn in range(2):
        central = CentralizedHistory.initialize(game, mode)
        for team in (0, 1):
            state = central.value_state_for(team)
            batch = collate_policy_states([state], device='cpu')
            vector = model.board_encoder.encode_vector(
                batch.board_codes, batch.point_mask, batch.mode_ids, batch.casualty_bits,
                batch.passes_remaining, batch.inventory_masks)
            features = vector[0, :129 * 26].reshape(129, 26)
            board = game.board_for(team)
            order = game._relative_player_order(team)
            for code in range(129):
                piece = game._pieces.get(board.decode(code))
                enabled = [i for i, value in enumerate(features[code, :25]) if value]
                if piece is None:
                    assert not features[code].any()
                else:
                    assert {INVENTORY_SLOTS[i] for i in enabled} == {piece.kind}
                    assert len(enabled) == 1
                    assert enabled[0] == game._critic_inventory_slots[board.decode(code)]
                    assert features[code, 25] == order.index(piece.owner) + 1
            engineer = features[board.encode(survivor), :25]
            assert sum(int(engineer[i]) for i, k in enumerate(INVENTORY_SLOTS) if k is K.ENGINEER) == 1
        if turn == 0:
            game.step((2, 7))  # Known to the referee: army commander versus bomb.


def test_joint_action_preserves_countdown_and_alive_bits_without_preceding_move():
    encoder = PublicActionEncoder(10, joint_context=True)
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.eye(10))
        encoder.projection.bias.zero_()
    fields = torch.tensor([[float('nan')] * 6, [-2., -3., -2., -2., 3., 69.]])
    present = torch.tensor([False, True])
    encoded = encoder(fields, present, no_interaction=torch.tensor([69, 1]),
                      active_mask=torch.tensor([0b0101, 0b1111]))
    torch.testing.assert_close(encoded, torch.tensor([
        [0., 0., 0., 0., 0., 1., 1., 0., 1., 0.],
        [-2., -3., -2., -2., 3., 69., 1., 1., 1., 1.]]))
    with pytest.raises(ValueError, match='no-capture and active-player'):
        encoder(fields, present)


@pytest.mark.parametrize('model_type', [GamePolicyTransformer, GameCentralValueTransformer])
def test_joint_context_tokens_match_record_and_array_paths_and_ignore_reveal_metadata(model_type):
    torch.set_num_threads(1)
    model = model_type(config()).eval()
    assert all(not hasattr(model, name) for name in (
        'no_interaction_embedding', 'active_embedding', 'revealed_embedding'))
    game = new_game(MODE, seed=172)
    private = GameHistory.initialize(game, MODE)
    central = CentralizedHistory.initialize(game, MODE)
    for turn in range(2):
        state = (central.value_state_for(0) if model_type is GameCentralValueTransformer
                 else private.players[0].as_value_state())
        record = replace(state.records[-1], no_interaction_plies=69, active_mask=0b0101)
        states = [replace(state, records=(record,)),
                  replace(state, records=(replace(record, revealed_mask=record.revealed_mask ^ 15),))]
        with torch.inference_mode():
            features = model._encode_full(states).context
            torch.testing.assert_close(features[0], features[1], rtol=0, atol=0)
            rows = torch.from_numpy(np.stack([record_array(s.records[0], MODE, True) for s in states]))[:, None, :]
            raw_tokens = model._embed_observation_rows(rows, MODE, torch.zeros((2, 1), dtype=torch.long))
            torch.testing.assert_close(raw_tokens[0], raw_tokens[1], rtol=0, atol=0)
            board, _ = model._record_board_batch([record], MODE)
            record_tokens = model._record_temporal_tokens([record], MODE, board, 0)
            torch.testing.assert_close(record_tokens[0], raw_tokens[0], atol=2e-6, rtol=2e-6)
        if turn == 0:
            game.step(game.legal_actions()[0])
            private.append_after_step(game)
            central.append_encoded_rows(centralized_observation_rows(game, MODE))


def test_restored_elimination_counts_inventory_without_hidden_death_labels():
    game = new_game(MODE, seed=77)
    game._eliminate_player(3)
    restored = JunqiGame.from_position(game.config, game.pieces, active_players=game.active_players,
                                      revealed_flags=game.revealed_flags, current_player=0)
    restored.enable_inventory_tracking()
    for viewer in range(4):
        assert restored._known_casualties[viewer][3] == Counter(PIECE_COUNTS)


def test_history_rejects_mixing_old_and_inventory_records():
    game = new_game(MODE, seed=59)
    history = GameHistory.initialize(game, MODE)
    game.step((0, 0))
    history.append_after_step(game)
    payload = history.state_dict()
    payload['players'][0]['records'][1]['inventory_masks'] = None
    with pytest.raises(ValueError, match='inventory candidate masks'):
        GameHistory.from_state_dict(payload)


@pytest.mark.parametrize('mode', [MODE, TrainingMode.DOUBLE_OPEN, TrainingMode.TWO_PLAYER])
def test_actor_privacy_compact_worker_and_sliding_history_roundtrip(mode):
    game = new_game(mode, seed=98, max_plies=80)
    history = GameHistory.initialize(game, mode, max_transitions=3)
    original = history.state_dict()
    initial = history.players[0].records[0]
    history.enable_array_storage()
    assert history.state_dict() == original
    before = observation_rows(game, mode)[0].copy()
    old = centralized_observation_rows(game, mode)[0].copy()
    unknown = [p for p, piece in game._pieces.items()
               if piece.owner == 1 and game._known_piece_kind(0, p, piece) is None]
    first = next(p for p in unknown if game._pieces[p].kind is K.ENGINEER)
    second = next(p for p in unknown if game._pieces[p].kind is K.ARMY_COMMANDER)
    changed = game.clone()
    changed._pieces[first], changed._pieces[second] = changed._pieces[second], changed._pieces[first]
    changed._critic_inventory_slots[first], changed._critic_inventory_slots[second] = (
        changed._critic_inventory_slots[second], changed._critic_inventory_slots[first])
    np.testing.assert_array_equal(before, observation_rows(changed, mode)[0])
    assert not np.array_equal(old, centralized_observation_rows(changed, mode)[0])
    for _ in range(8):
        result = _advance(0, game, game.legal_actions()[0], mode, centralized_critic=True)
        history.append_encoded_rows(result.records)
        for viewer in range(game.config.player_count):
            record = history.players[viewer].records[-1]
            np.testing.assert_array_equal(record_array(record, mode, True), result.records[viewer])
            direct = GameHistory.initialize(game, mode).players[viewer].records[0]
            assert record.inventory_masks == direct.inventory_masks
    assert len(history.players[0].records) == 4
    assert history.players[0].records[0] == initial
    assert GameHistory.from_state_dict(history.state_dict()).state_dict() == history.state_dict()
    assert pickle.loads(_serialize(game)).state_key() == game.state_key()
    assert restore(game).state_key() == game.state_key()


def test_full_array_incremental_forward_and_gradients_match():
    torch.set_num_threads(1)
    ordinary = GamePolicyTransformer(config(max_transitions=5)).eval()
    tensor = GamePolicyTransformer(config(max_transitions=5, ppo_tensor_learner=True)).eval()
    tensor.load_state_dict(ordinary.state_dict())
    cached = copy.deepcopy(ordinary)
    cached.start_ppo_inference_cache(capacity=4, behavior_version=0)
    game = new_game(MODE, seed=91, max_plies=100)
    history = GameHistory.initialize(game, MODE, max_transitions=5)
    history.enable_array_storage()
    held = []
    with torch.inference_mode():
        for _ in range(10):
            state = history.state_for(game)
            held.append(state)
            expected = ordinary._encode_full([state]).context
            torch.testing.assert_close(cached.encode([state]).context, expected, atol=6e-6, rtol=6e-6)
            tuple_state = replace(state, records=tuple(state.records))
            torch.testing.assert_close(ordinary._encode_full([tuple_state]).context, expected, atol=6e-6, rtol=6e-6)
            game.step(game.legal_actions()[0])
            history.append_after_step(game)
        torch.testing.assert_close(tensor._encode_full(held, pack_prefixes=True).context,
                                   ordinary._encode_full(held).context, atol=6e-6, rtol=6e-6)
    ordinary.train()
    tensor.train()
    states = held[-3:]
    for model in (ordinary, tensor):
        model.ppo_statistics(states, [s.legal_actions[-1] for s in states], pack_sequences=True)[0].sum().backward()
        assert model.board_encoder.projection.weight.grad.abs().sum() > 0
        assert model.action_encoder.projection.weight.grad.abs().sum() > 0
    for p, q in zip(ordinary.parameters(), tensor.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4)
    assert ordinary.action_encoder.projection.out_features == 16
    assert ordinary.board_encoder.projection.out_features == 48


def test_actual_384_128_split_keeps_512_backbone_and_256_action_heads():
    actual = replace(config(), board_dim=384, action_dim=128, temporal_dim=512,
                     temporal_heads=8, token_fusion_hidden_dim=512)
    model = GamePolicyTransformer(actual).train()
    game = new_game(MODE, seed=39)
    state = GameHistory.initialize(game, MODE).state_for(game)
    assert model.encode([state]).context.shape == (1, 512)
    assert model.board_encoder.projection.weight.shape == (384, 3436)
    assert model.action_encoder.projection.weight.shape == (128, 10)
    assert model.source_query[0].out_features == 256
    model([state], [[state.legal_actions[-1]]])[0].sum().backward()
    assert model.board_encoder.projection.weight.grad.abs().sum() > 0


def test_board_cache_distinguishes_same_codes_with_different_candidates():
    model = GamePolicyTransformer(config(incremental_inference=False)).eval()
    model.start_inference_board_cache(8)
    game = new_game(MODE, seed=39)
    state = GameHistory.initialize(game, MODE).state_for(game)
    record = state.records[0]
    masks = list(record.inventory_masks)
    index = next(i for i, code in enumerate(record.board_codes) if code == 1 and masks[i].bit_count() > 5)
    masks[index] &= ~(sum(1 << i for i, k in enumerate(INVENTORY_SLOTS) if k is K.ENGINEER))
    changed = replace(state, records=(replace(record, inventory_masks=tuple(masks)),))
    with torch.inference_mode():
        first = model.encode([state]).context
        second = model.encode([changed]).context
    assert len(model._inference_board_cache) == 2
    assert not torch.allclose(first, second)


def test_worker_process_preserves_private_inventory_and_global_critic_rows():
    from types import SimpleNamespace
    from junqi.training.ppo_environment import ParallelPPOEnvironment
    game = sparse({ArmPoint(0, 2, 3): Piece(0, K.ARMY_COMMANDER),
                   ArmPoint(0, 1, 3): Piece(1, K.BRIGADE_COMMANDER)})
    pool = SimpleNamespace(mode=MODE, slots=[SimpleNamespace(game=game.clone())])
    with ParallelPPOEnvironment(1) as worker:
        worker.begin(pool, centralized_critic=True)
        for action in ((7, 2), (0, 0), (0, 0), (0, 0), (2, 7)):
            worker.submit([0], [action])
            expected = _advance(0, game, action, MODE, centralized_critic=True)
            actual = worker.receive()[0]
            np.testing.assert_array_equal(actual.records, expected.records)
            np.testing.assert_array_equal(actual.centralized_records, expected.centralized_records)
        worker.synchronize(pool)
    assert pool.slots[0].game.state_key() == game.state_key()


@pytest.mark.parametrize('mode', [MODE, TrainingMode.DOUBLE_OPEN])
def test_random_public_inference_never_excludes_truth_or_invents_deaths(mode):
    rng = random.Random(413)
    for seed in range(3):
        game = new_game(mode, seed=seed, max_plies=140)
        game.enable_inventory_tracking()
        while not game.is_terminal:
            game.step(rng.choice(game.legal_actions()))
            remaining = [Counter() for _ in range(4)]
            for point, piece in game._pieces.items():
                remaining[piece.owner][piece.kind] += 1
                for viewer in range(4):
                    assert piece.kind in game._inventory.live[viewer][point]
                    assert any(slots(game.inventory_mask(viewer, point), piece.kind))
                    if len(game._inventory.live[viewer][point]) == 1:
                        assert game.inventory_mask(viewer, point).bit_count() == 1
            for viewer in range(4):
                assigned = [(game._pieces[p].owner, slot)
                            for p, slot in game._inventory.known_slots[viewer].items()]
                assert len(assigned) == len(set(assigned))
                for owner in range(4):
                    expected = sum(1 << slot for own, slot in assigned if own == owner)
                    assert game._inventory.live_slots[viewer][owner] == expected
                for point, domain in game._inventory.live[viewer].items():
                    owner = game._pieces[point].owner
                    if len(domain) > 1:
                        assert not game.inventory_mask(viewer, point) & game._inventory.used_slots[viewer][owner]
                for owner in range(4):
                    for kind, capacity in PIECE_COUNTS.items():
                        assert game._known_casualties[viewer][owner][kind] <= capacity - remaining[owner][kind]
