"""Replay the reported game through UI, public observations and PPO rows."""
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from junqi import ArmPoint, GameConfig, InformationMode, JunqiGame, Piece, PieceType as K
from junqi.inventory import INVENTORY_SLOTS
from junqi.training.encoding import GameHistory
from junqi.training.history_arrays import record_array
from junqi.training.modes import TrainingMode
from junqi.training.packed_observation import observation_rows
from junqi.training.ppo_environment import _advance, _serialize
from junqi.web.cpu_game import CpuGame


MODE = TrainingMode.FOUR_DARK
COMMANDER_MASK = sum(1 << i for i, kind in enumerate(INVENTORY_SLOTS) if kind is K.COMMANDER)


@pytest.fixture(scope='module')
def reported_game():
    # Completed-game ranks reconstruct only the referee. Actor observations
    # and training rows must derive every identity from legitimate evidence.
    data = json.loads((Path(__file__).parent / 'fixtures/flag_location_commander_alive.json').read_text())
    board = JunqiGame.new_four_player(seed=0).board_for(0)
    pieces = {board.decode(code): Piece(owner, kind) for code, owner, kind in data['pieces']}
    game = JunqiGame.from_position(GameConfig(information_mode=InformationMode.FOUR_DARK), pieces)
    history = GameHistory.initialize(game, MODE)
    snapshots = {}
    for start, end in data['actions']:
        actor_board = game.board_for(game.current_player)
        action = (0, 0) if start == end == 0 else tuple(
            actor_board.encode(board.decode(code)) for code in (start, end))
        game.step(action)
        history.append_after_step(game)
        rows = observation_rows(game, MODE)
        for viewer, player in enumerate(history.players):
            np.testing.assert_array_equal(rows[viewer], record_array(player.records[-1], MODE, True))
            observation = game.observe(viewer, history_limit=0, include_legal_masks=False,
                                       include_inventory_masks=True)
            for point, actual in game.pieces.items():
                observed = observation.points[game.board_for(viewer).encode(point)]
                assert observed.kind is None or observed.kind is actual.kind
                if game.revealed_flags[actual.owner]:
                    assert not observed.inventory_mask & COMMANDER_MASK
        if game.ply_count in (42, 87, 100, 122):
            # The actual environment-worker transport preserves both meanings.
            restored = pickle.loads(_serialize(game))
            np.testing.assert_array_equal(observation_rows(restored, MODE), rows)
            snapshots[game.ply_count] = restored
    assert game.ply_count == 126 and game.is_terminal
    return snapshots


def test_deduced_flag_does_not_imply_commander_death(reported_game):
    game = reported_game[87]
    observation = game.observe(0, include_legal_masks=False)
    assert observation.points[56].kind is K.FLAG
    assert not observation.revealed_flags[1]
    assert game.known_casualties[0][1].get(K.COMMANDER, 0) == 0
    assert any(game.inventory_mask(0, point) & COMMANDER_MASK
               for point, piece in game.pieces.items() if piece.owner == 1)

    game = reported_game[122]
    observation = game.observe(0, include_legal_masks=False)
    assert observation.points[53].kind is K.COMMANDER
    assert game.piece_at(game.board_for(0).decode(53)).kind is K.COMMANDER
    assert not observation.revealed_flags[1]


@pytest.mark.parametrize('ply,owner', [(42, 3), (100, 2)])
def test_real_commander_death_excludes_commander_from_every_live_candidate(reported_game, ply, owner):
    game = reported_game[ply]
    assert game.revealed_flags[owner]
    for viewer in range(4):
        assert game.known_casualties[viewer][owner][K.COMMANDER] == 1
        for point, piece in game.pieces.items():
            if piece.owner == owner:
                assert not game.inventory_mask(viewer, point) & COMMANDER_MASK


@pytest.mark.parametrize('mode', [TrainingMode.FOUR_DARK, TrainingMode.DOUBLE_OPEN])
@pytest.mark.parametrize('army_attacks', [True, False])
def test_actual_reveal_and_surviving_capture_of_division_force_army_in_training(mode, army_attacks):
    pieces = {ArmPoint(owner, 6, 2): Piece(owner, K.FLAG) for owner in range(4)}
    pieces.update({ArmPoint(owner, 3, 3): Piece(owner, K.COMMANDER) for owner in (0, 2, 3)})
    start, end = ArmPoint(0, 2, 3), ArmPoint(0, 1, 3)
    army, division = Piece(1, K.ARMY_COMMANDER, has_moved=True), Piece(0, K.DIVISION_COMMANDER)
    pieces[start], pieces[end] = (army, division) if army_attacks else (division, army)
    game = JunqiGame.from_position(GameConfig(information_mode=InformationMode(mode.value)),
                                   pieces, current_player=1 if army_attacks else 0)
    history = GameHistory.initialize(game, mode)
    assert game.revealed_flags[1]
    board = game.board_for(game.current_player)
    result = _advance(0, game, (board.encode(start), board.encode(end)), mode)
    history.append_after_step(game)
    np.testing.assert_array_equal(result.records[0], record_array(history.players[0].records[-1], mode, True))
    observed = game.observe(0, include_legal_masks=False, include_inventory_masks=True).points[
        game.board_for(0).encode(end)]
    assert observed.kind is K.ARMY_COMMANDER
    assert not observed.inventory_mask & COMMANDER_MASK


@pytest.mark.parametrize('viewer', range(4))
@pytest.mark.parametrize('ply', [42, 87, 122])
def test_cpu_page_preserves_relative_reveal_status_and_events(reported_game, viewer, ply):
    game = reported_game[ply]
    cpu = CpuGame.__new__(CpuGame)
    cpu.game, cpu.human = game, viewer
    cpu.metadata = {'mode': 'four_dark'}
    cpu.ai_moves, cpu.inference_seconds = 0, 0.0
    frame = cpu.state()
    observation = game.observe(viewer, history_limit=40, include_legal_masks=False)
    assert frame['revealed_flags'] == list(observation.revealed_flags)
    assert [event['newly_revealed_flags'] for event in frame['history']] == [
        list(event.newly_revealed_flags) for event in observation.history]
    if viewer == 0 and ply == 87:
        assert frame['pieces'][56]['kind'] == 'flag'
        assert frame['revealed_flags'][1] is False
