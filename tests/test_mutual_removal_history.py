"""Retrospective exchange deductions from viewer-visible evidence only."""
from __future__ import annotations

import copy
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest

from junqi import ArmPoint as A, GameConfig, InformationMode, JunqiGame, Piece, PieceType as K
from junqi.inventory import INVENTORY_SLOTS
from junqi.pieces import PIECE_COUNTS
from junqi.training.encoding import GameHistory
from junqi.training.modes import TrainingMode, new_game
from junqi.training.packed_observation import observation_rows
from junqi.training.ppo_environment import ParallelPPOEnvironment, _advance, _serialize


START, TARGET, SURVIVOR = A(0, 1, 3), A(0, 2, 3), A(1, 2, 3)
PLATOONS = (A(0, 1, 1), A(0, 1, 5), A(0, 2, 5))


def position(kind=K.COMPANY_COMMANDER, *, mode=InformationMode.FOUR_DARK, mine_owner=0,
             unknown_attacker=False):
    pieces = {A(owner, 6, 2): Piece(owner, K.FLAG) for owner in range(4)}
    pieces.update({A(owner, 3, 3): Piece(owner, K.COMMANDER) for owner in range(4)})
    pieces.update({START: Piece(0, kind), TARGET: Piece(1, kind),
                   SURVIVOR: Piece(1, kind)})
    for column in (1, 5):
        pieces[A(mine_owner, 4, column)] = Piece(1, K.BOMB)
        pieces[A(mine_owner, 5, column)] = Piece(mine_owner, K.MINE)
    if unknown_attacker:
        pieces.update({point: Piece(0, K.PLATOON_COMMANDER) for point in PLATOONS})
    game = JunqiGame.from_position(GameConfig(information_mode=mode), pieces)
    if unknown_attacker:
        candidates = dict(game.public_candidates)
        candidates[START] = frozenset((K.COMPANY_COMMANDER, K.PLATOON_COMMANDER))
        candidates[TARGET] = frozenset((K.COMPANY_COMMANDER, K.PLATOON_COMMANDER, K.BOMB))
        game = JunqiGame.from_position(game.config, pieces, public_candidates=candidates)
    game.enable_inventory_tracking()
    return game


def sequence(mine_owner=0):
    return ((START, TARGET), None, None,
            (A(mine_owner, 4, 1), A(mine_owner, 5, 1)), None, None, None,
            (A(mine_owner, 4, 5), A(mine_owner, 5, 5)))


def action(game, move):
    if move is None:
        return (0, 0)
    board = game.board_for(game.current_player)
    return tuple(board.encode(point) for point in move)


def restored(game, *, inventory=None):
    return JunqiGame.from_position(
        game.config, game.pieces, current_player=game.current_player,
        active_players=game.active_players, revealed_flags=game.revealed_flags,
        ply_count=game.ply_count, no_interaction_plies=game.no_interaction_plies,
        public_candidates=game.public_candidates, known_identities=game.known_identities,
        known_casualties=game.known_casualties, public_history=game.public_history,
        passes_remaining=game.passes_remaining,
        inventory_knowledge=game._inventory.state_dict() if inventory is None else inventory,
        critic_inventory_slots=game._critic_inventory_slots)


def kind_slots(mask, kind):
    return tuple((mask >> index) & 1 for index, value in enumerate(INVENTORY_SLOTS) if value is kind)


@pytest.mark.parametrize('kind', [K.COMPANY_COMMANDER, K.PLATOON_COMMANDER,
                                 K.BATTALION_COMMANDER, K.REGIMENT_COMMANDER])
def test_two_identified_bomb_losses_resolve_earlier_middle_rank_exchange(kind):
    game = position(kind)
    for index, move in enumerate(sequence()):
        game.step(action(game, move))
        event = game.mutual_removal_history(0)[0]
        if index < 7:
            assert event.classification == 'ambiguous'
            assert (kind, K.BOMB) in event.pairs
        else:
            assert event.classification == 'equal_rank'
            assert event.pairs == frozenset(((kind, kind),))
    assert [event.ply for event in game.mutual_removal_history(0)] == [1, 4, 8]
    assert [event.classification for event in game.mutual_removal_history(0)] == ['equal_rank', 'bomb', 'bomb']
    assert game._known_casualties[0][1][K.BOMB] == 2
    assert game._known_casualties[0][1][kind] >= 1
    assert kind_slots(game.inventory_mask(0, SURVIVOR), kind)[-1] == 0
    # Player 2 did not know player 0's mines, so cannot inherit that evidence.
    assert game.mutual_removal_history(2)[0].classification == 'ambiguous'


def test_equal_rank_relation_survives_unknown_ranks_and_propagates_later_inventory_evidence():
    game = position(mine_owner=2, unknown_attacker=True)
    for move in sequence(2):
        game.step(action(game, move))
    event = game.mutual_removal_history(2)[0]
    assert event.classification == 'equal_rank'
    assert event.pairs == frozenset(((K.COMPANY_COMMANDER, K.COMPANY_COMMANDER),
                                     (K.PLATOON_COMMANDER, K.PLATOON_COMMANDER)))
    assert game._inventory.dead[2][1][event.defender_index] == frozenset(
        (K.COMPANY_COMMANDER, K.PLATOON_COMMANDER))
    # Simulate subsequently proven identities of all three surviving platoons.
    # Their domains are legitimate viewer knowledge, not read from hidden ranks.
    game._known_identities[2].update({point: K.PLATOON_COMMANDER for point in PLATOONS})
    game._inventory.propagate(game)
    resolved = game.mutual_removal_history(2)[0]
    assert resolved.pairs == frozenset(((K.COMPANY_COMMANDER, K.COMPANY_COMMANDER),))
    assert game._inventory.dead[2][0][resolved.attacker_index] == frozenset((K.COMPANY_COMMANDER,))
    assert game._inventory.dead[2][1][resolved.defender_index] == frozenset((K.COMPANY_COMMANDER,))
    assert game._known_casualties[2][1][K.COMPANY_COMMANDER] >= 1
    # Held historical knowledge is immutable, just like previously emitted tokens.
    assert len(event.pairs) == 2


def army_bomb_position(*, commander_alive=False, proof='attacks'):
    pieces = {A(owner, 6, 2): Piece(owner, K.FLAG) for owner in range(4)}
    pieces.update({A(owner, 3, 3): Piece(owner, K.COMMANDER) for owner in (2, 3)})
    if commander_alive:
        pieces[A(1, 3, 3)] = Piece(1, K.COMMANDER)
    enemy_army, division = {
        'attacks': (A(1, 3, 1), A(1, 2, 1)),
        'moved_defender': (A(1, 3, 1), A(1, 1, 1)),
        'unmoved_defender': (A(1, 5, 1), A(1, 4, 1)),
    }[proof]
    pieces.update({
        START: Piece(0, K.ARMY_COMMANDER), TARGET: Piece(1, K.BOMB),
        enemy_army: Piece(1, K.ARMY_COMMANDER), division: Piece(0, K.DIVISION_COMMANDER),
        A(0, 1, 5): Piece(0, K.COMPANY_COMMANDER),
        A(0, 2, 5): Piece(1, K.COMPANY_COMMANDER),
        SURVIVOR: Piece(1, K.COMPANY_COMMANDER),
        A(0, 4, 1): Piece(1, K.BOMB), A(0, 5, 1): Piece(0, K.MINE),
        A(0, 3, 5): Piece(0, K.ENGINEER),
    })
    game = JunqiGame.from_position(GameConfig(information_mode=InformationMode.FOUR_DARK), pieces)
    game.enable_inventory_tracking()
    return game, enemy_army, division


@pytest.mark.parametrize('proof', ['attacks', 'moved_defender'])
def test_later_proven_live_army_identifies_earlier_bomb_loss(proof):
    game, enemy_army, division = army_bomb_position(proof=proof)
    history = GameHistory.initialize(game, TrainingMode.FOUR_DARK)
    game.step(action(game, (START, TARGET)))
    history.append_after_step(game)
    old_event = game.mutual_removal_history(0)[0]
    old_record = history.players[0].records[-1]
    spare_bomb_code = game.board_for(0).encode(A(0, 4, 1))
    assert old_event.pairs == frozenset(((K.ARMY_COMMANDER, K.ARMY_COMMANDER),
                                        (K.ARMY_COMMANDER, K.BOMB)))
    assert game._known_casualties[0][1][K.BOMB] == 0
    assert kind_slots(old_record.inventory_masks[spare_bomb_code], K.BOMB) == (1, 1)
    # Recover the unresolved exchange before the later identifying battle.
    game = restored(game)
    if proof == 'attacks':
        moves = (None, None, (enemy_army, division))
        survivor = division
    else:
        survivor = A(1, 2, 1)
        moves = (None, None, (enemy_army, survivor), (division, survivor))
    for move in moves:
        game.step(action(game, move))
        history.append_after_step(game)
    event = game.mutual_removal_history(0)[0]
    assert game._inventory.live[0][survivor] == frozenset((K.ARMY_COMMANDER,))
    assert event.pairs == frozenset(((K.ARMY_COMMANDER, K.BOMB),))
    assert event.classification == 'bomb'
    assert game._inventory.dead[0][1][event.defender_index] == frozenset((K.BOMB,))
    assert game._known_casualties[0][1][K.BOMB] == 1
    assert game._known_casualties[0][1][K.ARMY_COMMANDER] == 0
    assert kind_slots(game.inventory_mask(0, A(0, 4, 1)), K.BOMB) == (1, 0)
    assert kind_slots(history.players[0].records[-1].inventory_masks[spare_bomb_code], K.BOMB) == (1, 0)
    assert kind_slots(old_record.inventory_masks[spare_bomb_code], K.BOMB) == (1, 1)
    assert old_event.classification == 'ambiguous'
    # Player 2 does not know player 0's army or division identities.
    assert game.mutual_removal_history(2)[0].classification == 'ambiguous'


@pytest.mark.parametrize('uncertainty', ['commander_alive', 'unmoved_mine'])
def test_beating_division_does_not_prove_army_with_other_possible_winners(uncertainty):
    proof = 'unmoved_defender' if uncertainty == 'unmoved_mine' else 'attacks'
    game, enemy_army, division = army_bomb_position(
        proof=proof, commander_alive=uncertainty == 'commander_alive')
    if proof == 'attacks':
        moves = ((START, TARGET), None, None, (enemy_army, division))
        survivor, alternative = division, K.COMMANDER
    else:
        moves = ((START, TARGET), None, None, None, (division, enemy_army))
        survivor, alternative = enemy_army, K.MINE
    for move in moves:
        game.step(action(game, move))
    assert game._inventory.live[0][survivor] == frozenset((K.ARMY_COMMANDER, alternative))
    assert game.mutual_removal_history(0)[0].classification == 'ambiguous'
    assert game._known_casualties[0][1][K.BOMB] == 0


@pytest.mark.parametrize('kind', [K.DIVISION_COMMANDER, K.BRIGADE_COMMANDER,
                                 K.REGIMENT_COMMANDER, K.BATTALION_COMMANDER,
                                 K.COMPANY_COMMANDER, K.PLATOON_COMMANDER, K.ENGINEER])
def test_multiple_copy_ranks_require_all_copies_accounted_for_before_inferring_bomb(kind):
    survivors = (A(1, 2, 3), A(1, 3, 1), A(1, 4, 3))[:PIECE_COUNTS[kind]]
    pieces = {A(owner, 6, 2): Piece(owner, K.FLAG) for owner in range(4)}
    pieces.update({A(owner, 3, 3): Piece(owner, K.COMMANDER) for owner in (2, 3)})
    pieces.update({START: Piece(0, kind), TARGET: Piece(1, K.BOMB),
                   A(0, 3, 5): Piece(0, K.ENGINEER), A(1, 4, 1): Piece(1, K.BOMB)})
    pieces.update({point: Piece(1, kind) for point in survivors})
    game = JunqiGame.from_position(GameConfig(information_mode=InformationMode.FOUR_DARK), pieces)
    game.enable_inventory_tracking()
    game.step(action(game, (START, TARGET)))
    for index, point in enumerate(survivors):
        assert game.mutual_removal_history(0)[0].classification == 'ambiguous'
        assert game._known_casualties[0][1][K.BOMB] == 0
        # Supply later legitimate exact-identity evidence, one live copy at a time.
        game._known_identities[0][point] = kind
        game._inventory.propagate(game)
        if index < len(survivors) - 1:
            assert game.mutual_removal_history(0)[0].classification == 'ambiguous'
    assert game.mutual_removal_history(0)[0].pairs == frozenset(((kind, K.BOMB),))
    assert game._known_casualties[0][1][K.BOMB] == 1
    assert game._known_casualties[0][1][kind] == 0


def test_inferred_army_bomb_loss_cascades_to_old_company_exchange_without_recounting():
    game, enemy_army, division = army_bomb_position()
    moves = ((A(0, 1, 5), A(0, 2, 5)), None, None,
             (A(0, 4, 1), A(0, 5, 1)),  # First bomb independently proved.
             (START, TARGET), None, None, (enemy_army, division))
    for index, move in enumerate(moves):
        game.step(action(game, move))
        if index < 7:
            assert game.mutual_removal_history(0)[0].classification == 'ambiguous'
        if index == 4:
            assert game._known_casualties[0][1][K.BOMB] == 1
            assert game.mutual_removal_history(0)[-1].classification == 'ambiguous'
            game = restored(game)
    assert [event.classification for event in game.mutual_removal_history(0)] == ['equal_rank', 'bomb', 'bomb']
    assert game.mutual_removal_history(0)[0].pairs == frozenset(((K.COMPANY_COMMANDER, K.COMPANY_COMMANDER),))
    assert game.mutual_removal_history(0)[-1].pairs == frozenset(((K.ARMY_COMMANDER, K.BOMB),))
    assert game._known_casualties[0][1][K.BOMB] == 2
    assert game._known_casualties[0][1][K.COMPANY_COMMANDER] >= 1
    assert kind_slots(game.inventory_mask(0, SURVIVOR), K.BOMB) == (0, 0)
    assert kind_slots(game.inventory_mask(0, SURVIVOR), K.COMPANY_COMMANDER)[-1] == 0
    before = game.state_key()
    game._inventory.propagate(game)
    assert restored(game).state_key() == game.state_key() == before


@pytest.mark.parametrize('roundtrip', ['clone', 'state_dict', 'worker_pickle'])
def test_unresolved_pairs_survive_roundtrip_then_resolve(roundtrip):
    game = position(mine_owner=2, unknown_attacker=True)
    game.step(action(game, sequence(2)[0]))
    before = game.state_key()
    original = game
    game = {'clone': lambda: game.clone(), 'state_dict': lambda: restored(game),
            'worker_pickle': lambda: pickle.loads(_serialize(game))}[roundtrip]()
    assert game.state_key() == before
    for move in sequence(2)[1:]:
        game.step(action(game, move))
    assert game.mutual_removal_history(2)[0].classification == 'equal_rank'
    assert original.mutual_removal_history(2)[0].classification == 'ambiguous'
    assert original.state_key() == before


def test_history_window_and_old_policy_tokens_do_not_gain_future_information():
    game = position()
    history = GameHistory.initialize(game, TrainingMode.FOUR_DARK, max_transitions=2)
    history.enable_array_storage()
    moves = sequence()
    game.step(action(game, moves[0]))
    history.append_after_step(game)
    old_state = history.players[0].as_value_state()
    old_records = tuple(old_state.records)
    frozen = copy.deepcopy(old_records)
    code = game.board_for(0).encode(SURVIVOR)
    assert kind_slots(old_records[-1].inventory_masks[code], K.COMPANY_COMMANDER) == (1, 1, 1)
    for move in moves[1:]:
        game.step(action(game, move))
        history.append_after_step(game)
    assert tuple(old_state.records) == frozen
    assert len(history.players[0].records) == 3
    latest = history.players[0].records[-1]
    assert kind_slots(latest.inventory_masks[code], K.COMPANY_COMMANDER)[-1] == 0
    assert game.mutual_removal_history(0)[0].classification == 'equal_rank'


def test_hidden_bomb_and_rank_swap_does_not_change_observer_exchange_knowledge():
    game = position()
    pieces = dict(game.pieces)
    bomb = A(0, 4, 1)
    pieces[TARGET], pieces[bomb] = pieces[bomb], pieces[TARGET]
    twin = JunqiGame.from_position(game.config, pieces, public_candidates=game.public_candidates)
    twin.enable_inventory_tracking()
    for current in (game, twin):
        current.step(action(current, sequence()[0]))
    assert game.mutual_removal_history(0) == twin.mutual_removal_history(0)
    np.testing.assert_array_equal(observation_rows(game, TrainingMode.FOUR_DARK)[0],
                                  observation_rows(twin, TrainingMode.FOUR_DARK)[0])


def test_all_bombs_dead_without_identifying_their_exchanges_does_not_resolve_history():
    # The third party can prove all of player 1's bombs are gone, yet the old
    # exchange may itself be one of those bomb losses.
    game = position(mine_owner=2, unknown_attacker=True)
    game.step(action(game, sequence(2)[0]))
    game._eliminate_player(1)
    game._inventory.propagate(game)
    assert game._known_casualties[2][1][K.BOMB] == 2
    event = game.mutual_removal_history(2)[0]
    assert event.classification == 'ambiguous'
    assert any(b is K.BOMB for _, b in event.pairs)
    assert restored(game).state_key() == game.state_key()


def test_bombs_of_only_one_side_accounted_for_do_not_rule_out_the_other_side_bomb():
    game = position(mine_owner=2)
    # Both initial attackers and defenders may be bombs in this fixture.
    pieces = dict(game.pieces)
    start = A(0, 2, 1)
    pieces[start] = pieces.pop(START)
    target = A(0, 3, 1)
    pieces[target] = pieces.pop(TARGET)
    game = JunqiGame.from_position(game.config, pieces)
    game.enable_inventory_tracking()
    moves = ((start, target),) + sequence(2)[1:]
    for move in moves:
        game.step(action(game, move))
    event = game.mutual_removal_history(2)[0]
    assert game._known_casualties[2][1][K.BOMB] == 2
    assert all(b is not K.BOMB for _, b in event.pairs)
    assert any(a is K.BOMB for a, _ in event.pairs)
    assert event.classification == 'ambiguous'


def test_double_open_shares_deductions_with_the_legal_ally():
    game = position(mode=InformationMode.DOUBLE_OPEN)
    for move in sequence():
        game.step(action(game, move))
    assert game.mutual_removal_history(0) == game.mutual_removal_history(2)
    assert game.mutual_removal_history(2)[0].classification == 'equal_rank'
    assert game._known_casualties[0] == game._known_casualties[2]


def test_legacy_inventory_restores_without_inventing_lost_pair_correlations():
    game = position(mine_owner=2, unknown_attacker=True)
    game.step(action(game, sequence(2)[0]))
    state = game._inventory.state_dict()
    state['version'] = 1
    state.pop('mutual_removals')
    legacy = restored(game, inventory=state)
    assert legacy.mutual_removal_history(2) == ()
    np.testing.assert_array_equal(observation_rows(game, TrainingMode.FOUR_DARK),
                                  observation_rows(legacy, TrainingMode.FOUR_DARK))
    assert game.state_key() != legacy.state_key()


@pytest.mark.parametrize('corrupt', ['endpoint', 'ply', 'pairs'])
def test_corrupt_history_is_rejected(corrupt):
    game = position()
    game.step(action(game, sequence()[0]))
    state = game._inventory.state_dict()
    event = state['mutual_removals'][0][0]
    if corrupt == 'endpoint':
        event['attacker_index'] = 25
    elif corrupt == 'ply':
        event['ply'] = game.ply_count + 1
    else:
        event['pairs'] = ((K.COMPANY_COMMANDER, K.PLATOON_COMMANDER),)
    with pytest.raises(ValueError, match='mutual-removal'):
        restored(game, inventory=state)


def test_environment_process_preserves_retrospective_relations():
    game = position(mine_owner=2, unknown_attacker=True)
    pool = SimpleNamespace(mode=TrainingMode.FOUR_DARK, slots=[SimpleNamespace(game=game.clone())])
    with ParallelPPOEnvironment(1) as worker:
        worker.begin(pool)
        for move in sequence(2):
            chosen = action(game, move)
            worker.submit([0], [chosen])
            expected = _advance(0, game, chosen, TrainingMode.FOUR_DARK)
            actual = worker.receive()[0]
            np.testing.assert_array_equal(actual.records, expected.records)
        worker.synchronize(pool)
    assert pool.slots[0].game.state_key() == game.state_key()
    assert pool.slots[0].game.mutual_removal_history(2)[0].classification == 'equal_rank'


@pytest.mark.parametrize('mode', list(TrainingMode))
def test_random_games_never_prune_actual_historical_pairs(mode):
    rng = random.Random(762)
    for seed in range(2):
        game = new_game(mode, seed=seed, max_plies=140)
        game.enable_inventory_tracking()
        actual = {}
        while not game.is_terminal:
            result = game.step(rng.choice(game.legal_actions()))
            if result.combat.value == 'both_removed':
                actual[game.ply_count] = (result.attacker.kind, result.defender.kind)
            for viewer in range(game.config.player_count):
                for event in game.mutual_removal_history(viewer):
                    assert actual[event.ply] in event.pairs
