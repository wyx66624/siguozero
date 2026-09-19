"""Confirmed Actor identities are distinct, public, and persistent."""
from collections import Counter
import copy
import pickle

import numpy as np
import pytest

from junqi import ArmPoint as A, GameConfig, InformationMode, JunqiGame, Piece, PieceType as K
from junqi.inventory import INVENTORY_SLOTS, INVENTORY_MASK_ENCODING, slot_mask
from junqi.pieces import PIECE_COUNTS
from junqi.training.encoding import GameHistory
from junqi.training.mappo_encoding import centralized_observation_rows
from junqi.training.modes import TrainingMode, new_game
from junqi.training.packed_observation import observation_rows
from junqi.training.ppo_environment import _serialize


MODE = TrainingMode.FOUR_DARK


def kind_bits(game, viewer, point, kind=K.ENGINEER):
    mask = game.inventory_mask(viewer, point)
    return tuple((mask >> i) & 1 for i, item in enumerate(INVENTORY_SLOTS) if item is kind)


def restored(game, state=None):
    return JunqiGame.from_position(
        game.config, game.pieces, current_player=game.current_player,
        active_players=game.active_players, revealed_flags=game.revealed_flags,
        ply_count=game.ply_count, no_interaction_plies=game.no_interaction_plies,
        public_candidates=game.public_candidates, known_identities=game.known_identities,
        known_casualties=game.known_casualties, public_history=game.public_history,
        passes_remaining=game.passes_remaining,
        inventory_knowledge=game._inventory.state_dict() if state is None else state,
        critic_inventory_slots=game._critic_inventory_slots)


def confirm(game, viewer, point, kind=K.ENGINEER):
    # Supply the same exact identity evidence consumed by the rule engine,
    # keeping it private to this observer rather than revealing referee IDs.
    game._known_identities[viewer][point] = kind
    game._inventory.propagate(game)


def test_enemy_confirmations_follow_discovery_order_not_hidden_copy_order():
    game = new_game(MODE, seed=57)
    centralized_observation_rows(game, MODE)
    points = sorted((p for p, piece in game.pieces.items() if piece.owner == 1 and piece.kind is K.ENGINEER),
                    key=game.board_for(1).encode)
    old = GameHistory.initialize(game, MODE).players[0].records[0]
    other_view = observation_rows(game, MODE)[2].copy()
    assert all(kind_bits(game, 0, p) == (1, 1, 1) for p in points)
    for index, point in enumerate(reversed(points)):
        confirm(game, 0, point)
        expected = tuple(int(i == index) for i in range(3))
        assert kind_bits(game, 0, point) == expected
        assert game.inventory_mask(0, point).bit_count() == 1
        for other, piece in game.pieces.items():
            if piece.owner == 1 and other != point:
                assert game.inventory_mask(0, other) & game.inventory_mask(0, point) == 0
    assert game._inventory.known_slots[0][points[-1]] != game._critic_inventory_slots[points[-1]]
    assert Counter(game._inventory.known_slots[0][p] for p in points) == Counter((6, 7, 8))
    np.testing.assert_array_equal(observation_rows(game, MODE)[2], other_view)
    for point in points:
        mask = old.inventory_masks[game.board_for(0).encode(point)]
        assert tuple((mask >> i) & 1 for i in (6, 7, 8)) == (1, 1, 1)
    for clone in (game.clone(), restored(game), pickle.loads(_serialize(game))):
        assert clone.state_key() == game.state_key()
        np.testing.assert_array_equal(observation_rows(clone, MODE), observation_rows(game, MODE))


def test_engineer_reveal_death_and_next_confirmation_keep_distinct_ids():
    start, end, enemy = A(0, 5, 1), A(0, 1, 3), A(0, 1, 2)
    peers = (A(0, 2, 3), A(0, 3, 2))
    pieces = {A(owner, 6, 2): Piece(owner, K.FLAG) for owner in range(4)}
    pieces.update({A(owner, 3, 3): Piece(owner, K.COMMANDER) for owner in range(4)})
    pieces.update({p: Piece(0, K.ENGINEER) for p in (start, *peers)})
    pieces[enemy] = Piece(3, K.ARMY_COMMANDER)
    game = JunqiGame.from_position(GameConfig(information_mode=InformationMode.FOUR_DARK), pieces)
    game.enable_inventory_tracking()
    history = GameHistory.initialize(game, MODE, max_transitions=2)
    history.enable_array_storage()
    initial = history.players[1].records[0]
    own_masks = {p: game.inventory_mask(0, p) for p in (start, *peers)}
    assert {mask for mask in own_masks.values()} == {1 << 6, 1 << 7, 1 << 8}

    # Only an engineer can make this turning railway move.
    game.step((20, 2))
    history.append_after_step(game)
    assert game.inventory_mask(0, end) == own_masks[start] == 1 << 8
    assert kind_bits(game, 1, end) == (1, 0, 0)
    # Request the Critic for the first time after a move: own IDs still agree.
    centralized_observation_rows(game, MODE)
    assert game._critic_inventory_slots[end] == 8
    assert game.inventory_mask(0, end) == 1 << game._critic_inventory_slots[end]
    board = game.board_for(3)
    game.step((board.encode(enemy), board.encode(end)))
    history.append_after_step(game)
    assert end not in game._inventory.known_slots[1]
    for point in peers:
        assert game.inventory_mask(0, point) == own_masks[point]
        assert kind_bits(game, 1, point) == (0, 1, 1)
    before = observation_rows(game, MODE)
    game = restored(game)
    np.testing.assert_array_equal(observation_rows(game, MODE), before)
    confirm(game, 1, peers[0])
    assert kind_bits(game, 1, peers[0]) == (0, 1, 0)
    assert kind_bits(game, 1, peers[1]) == (0, 0, 1)
    confirm(game, 1, peers[1])
    assert kind_bits(game, 1, peers[1]) == (0, 0, 1)
    assert initial == history.players[1].records[0]
    assert kind_bits(game, 0, peers[1]) == (0, 1, 0)


@pytest.mark.parametrize('corrupt', ['missing', 'duplicate', 'wrong_type', 'unreserved', 'unknown'])
def test_restore_rejects_corrupt_confirmed_slot_state(corrupt):
    game = new_game(MODE, seed=57)
    points = [p for p, piece in game.pieces.items() if piece.owner == 1 and piece.kind is K.ENGINEER]
    for point in points:
        confirm(game, 0, point)
    state = game._inventory.state_dict()
    known = state['known_slots'][0]
    if corrupt == 'missing':
        state.pop('known_slots')
    elif corrupt == 'duplicate':
        known[points[1]] = known[points[0]]
    elif corrupt == 'wrong_type':
        known[points[0]] = 24
    elif corrupt == 'unreserved':
        state['used_slots'][0][1] &= ~(1 << known[points[0]])
    else:
        unknown = next(p for p, domain in game._inventory.live[0].items() if len(domain) > 1)
        known[unknown] = 0
    with pytest.raises(ValueError, match='confirmed piece slots'):
        restored(game, state)


@pytest.mark.parametrize('old_encoding', [None, 'confirmed_onehot_unknown_multihot_v1'])
def test_policy_history_marks_new_semantics_and_rejects_old_multihot_records(old_encoding):
    history = GameHistory.initialize(new_game(MODE, seed=58), MODE)
    state = history.state_dict()
    assert all(p['inventory_mask_encoding'] == INVENTORY_MASK_ENCODING for p in state['players'])
    assert GameHistory.from_state_dict(state).state_dict() == state
    old = copy.deepcopy(state)
    if old_encoding is None:
        old['players'][0].pop('inventory_mask_encoding')
    else:
        old['players'][0]['inventory_mask_encoding'] = old_encoding
    with pytest.raises(ValueError, match='confirmed onehot'):
        GameHistory.from_state_dict(old)


def test_unknown_engineer_slots_are_removed_on_confirmation_not_twice_on_death():
    starts = (A(0, 2, 1), A(0, 2, 3), A(0, 2, 5))
    ends = (A(0, 1, 1), A(0, 1, 3), A(0, 1, 5))
    probe = A(0, 4, 3)
    pieces = {A(owner, 6, 2): Piece(owner, K.FLAG) for owner in range(4)}
    pieces.update({A(owner, 3, 3): Piece(owner, K.COMMANDER) for owner in range(4)})
    pieces.update({p: Piece(0, K.ENGINEER) for p in starts})
    pieces.update({p: Piece(3, K.COMPANY_COMMANDER) for p in ends})
    pieces[probe] = Piece(0, K.PLATOON_COMMANDER)
    game = JunqiGame.from_position(GameConfig(information_mode=InformationMode.FOUR_DARK), pieces)
    game.enable_inventory_tracking()
    assert kind_bits(game, 1, probe) == (1, 1, 1)
    for number, (start, end) in enumerate(zip(starts, ends), start=1):
        confirm(game, 1, start)
        assert kind_bits(game, 1, start) == tuple(int(i == number - 1) for i in range(3))
        remaining = tuple(int(i >= number) for i in range(3))
        assert kind_bits(game, 1, probe) == remaining
        board = game.board_for(0)
        game.step((board.encode(start), board.encode(end)))
        assert game._known_casualties[1][0][K.ENGINEER] == number
        assert kind_bits(game, 1, probe) == remaining
        if number < 3:
            for _ in range(3):
                game.step((0, 0))


@pytest.mark.parametrize('kind', list(K))
def test_confirmed_slot_is_exclusive_for_all_types_and_only_its_owner_and_viewer(kind):
    game = new_game(MODE, seed=57)
    points = [p for p, piece in game.pieces.items() if piece.owner == 1 and piece.kind is kind]
    point = points[-1]
    other_view = observation_rows(game, MODE)[2].copy()
    other_owners = {p: game.inventory_mask(0, p) for p, piece in game.pieces.items() if piece.owner != 1}
    history = GameHistory.initialize(game, MODE)
    initial = history.players[0].records[0]
    confirm(game, 0, point, kind)
    fixed = game.inventory_mask(0, point)
    assert fixed.bit_count() == 1
    assert INVENTORY_SLOTS[fixed.bit_length() - 1] is kind
    game.step((0, 0))
    history.append_after_step(game)
    latest = history.players[0].records[-1]
    for other, piece in game.pieces.items():
        encoded = latest.inventory_masks[game.board_for(0).encode(other)]
        assert encoded == game.inventory_mask(0, other)
        if piece.owner == 1 and other != point:
            assert encoded & fixed == 0
        elif piece.owner != 1:
            assert encoded == other_owners[other]
    # Only the trailing action/counters changed for the uninformed observer.
    np.testing.assert_array_equal(observation_rows(game, MODE)[2, :129 * 3 + 75],
                                  other_view[:129 * 3 + 75])
    assert history.players[0].records[0] == initial


def test_free_slot_counts_cover_live_confirmations_identified_and_anonymous_deaths():
    # Exhaust all disjoint live/dead ID assignments for each inventory size.
    for kind, capacity in PIECE_COUNTS.items():
        slots = [i for i, k in enumerate(INVENTORY_SLOTS) if k is kind]
        for live_pattern in range(1 << capacity):
            for dead_pattern in range(1 << capacity):
                if live_pattern & dead_pattern:
                    continue
                live = sum(1 << slot for i, slot in enumerate(slots) if live_pattern & (1 << i))
                dead = sum(1 << slot for i, slot in enumerate(slots) if dead_pattern & (1 << i))
                free_count = capacity - live_pattern.bit_count() - dead_pattern.bit_count()
                for anonymous in range(free_count + 1):
                    result = slot_mask({kind}, {kind: dead_pattern.bit_count() + anonymous}, live | dead, live)
                    assert not result & (live | dead)
                    assert result.bit_count() == free_count - anonymous
                    assert all(INVENTORY_SLOTS[i] is kind for i in range(25) if result & (1 << i))
    engineer = 1 << 6
    # First anonymous loss retains 110; a later live confirmation reserves 100.
    assert slot_mask({K.ENGINEER}, {K.ENGINEER: 1}) == (1 << 6) | (1 << 7)
    assert slot_mask({K.ENGINEER}, {K.ENGINEER: 1}, engineer, engineer) == 1 << 7
    # Its subsequent death increases casualties but does not consume 010 too.
    assert slot_mask({K.ENGINEER}, {K.ENGINEER: 2}, engineer, 0) == 1 << 7
