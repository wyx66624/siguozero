"""Immobility, legacy replay, and lossless checkpoint-rule adoption."""
import copy
from dataclasses import fields
import pickle

import numpy as np
import pytest
import torch

from junqi import ArmPoint, GameConfig, InformationMode, JunqiGame, Piece, PieceType
from junqi.training.mappo_encoding import centralized_observation_rows
from junqi.training.history_arrays import record_array
from junqi.training.no_move_migration import reconcile_no_move_rule
from junqi.training.packed_observation import observation_rows
import test_mappo_tactical_rewards as helpers


def position(*, legacy=False, current=0, terminal=False):
    active = (True, True, True, not terminal)
    immobile = 1 if terminal else 0
    pieces = {ArmPoint(p,6,2):Piece(p,PieceType.FLAG) for p in range(4) if active[p]}
    for p in range(4):
        if not active[p]:
            continue
        # A nominally movable piece trapped permanently in HQ also has no move.
        pieces[ArmPoint(p,6,4) if p==immobile else ArmPoint(p,2,3)] = Piece(p,PieceType.COMMANDER)
    pieces[ArmPoint(immobile,6,1)] = Piece(immobile,PieceType.MINE)
    return JunqiGame.from_position(GameConfig(information_mode=InformationMode.FOUR_DARK,
        eliminate_no_moves=not legacy),pieces,current_player=current,active_players=active)


def test_unused_passes_do_not_save_hq_trapped_player_or_add_a_ply():
    game = position()
    assert not game.active_players[0]
    assert game.current_player == 3
    assert game.passes_remaining == (4,4,4,4)
    assert game.ply_count == 0 and not game.public_history
    assert game.legal_actions(0) == ()
    assert not any(p.owner==0 for p in game.pieces.values())
    assert game.legal_actions()[0] == (0,0)  # A real choice for the movable seat.


def test_no_move_last_enemy_finishes_team_without_spending_passes():
    game = position(current=1,terminal=True)
    assert game.is_terminal and game.result.winner_team == 0
    assert game.rewards() == (1.,-1.,1.,-1.)
    assert game.passes_remaining == (4,4,4,4)
    assert game.ply_count == 0 and not game.public_history


def test_legacy_config_pickle_keeps_original_rule_and_new_roundtrip_keeps_new_rule():
    current = GameConfig()
    values = [getattr(current,f.name) for f in fields(current)]
    old = GameConfig.__new__(GameConfig)
    old.__setstate__(values[:-1])
    assert old.eliminate_no_moves is False
    assert pickle.loads(pickle.dumps(current)).eliminate_no_moves is True
    assert pickle.loads(pickle.dumps(old)).eliminate_no_moves is False
    legacy_game = position(legacy=True)
    assert legacy_game.active_players[0] and legacy_game.legal_actions() == ((0,0),)
    legacy_game.step((0,0))
    assert legacy_game.passes_remaining[0] == 3


def test_adoption_preserves_state_and_histories_resolves_current_seat_and_is_idempotent():
    torch.set_num_threads(1)
    helper = helpers.MAPPOTacticalRewardTests()
    positions = [(position(legacy=True,current=p), (0,0)) for p in (0,1,2)]
    positions.append((position(legacy=True,current=1,terminal=True),(0,0)))
    _,_,_,pool = helper.models_and_pool(positions)
    # The second game reaches the immobile human via a real opponent pass.
    slot = pool.slots[1]
    slot.game.step((0,0))
    slot.history.append_after_step(slot.game)
    slot.critic_history.append_encoded_rows(centralized_observation_rows(slot.game,pool.mode))
    payload = dict(update=42, config=dict(eliminate_no_moves=False),
        policy={'weight':torch.tensor([1.,2.])}, policy_optimizer={'step':torch.tensor(9.)},
        rng_state={'torch_cpu':torch.get_rng_state().clone()},
        trainer_state=dict(base_game_pool=pool.state_dict(),cumulative={'environment_plies':900},
                           adaptive_learning_rate={'scale':.05},layout_buffer=[]))
    original = copy.deepcopy(payload)
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError,match='adopt-no-move-elimination'):
        reconcile_no_move_rule(payload)
    helpers.assert_tree_equal(helper,payload,original)
    reconcile_no_move_rule(payload,adopt=True)
    for key in ('policy','policy_optimizer','rng_state'):
        helpers.assert_tree_equal(helper,payload[key],original[key])
    for key in ('cumulative','adaptive_learning_rate','layout_buffer'):
        helpers.assert_tree_equal(helper,payload['trainer_state'][key],original['trainer_state'][key])
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    raw = payload['trainer_state']['base_game_pool']
    assert raw['rng_state'] == original['trainer_state']['base_game_pool']['rng_state']
    assert len(raw['slots']) == 3
    assert len(payload['trainer_state']['no_move_rule_migration']['terminal_games']) == 1
    pool.load_state_dict(raw)
    for i,slot in enumerate(pool.slots):
        assert slot.game.config.eliminate_no_moves
        if i < 2:
            assert not slot.game.active_players[0]
        assert slot.game.ply_count == original['trainer_state']['base_game_pool']['slots'][i]['game']['ply_count']
        public = observation_rows(slot.game,pool.mode)
        central = centralized_observation_rows(slot.game,pool.mode)
        for view in range(4):
            before = original['trainer_state']['base_game_pool']['slots'][i]
            for history, name, rows in ((slot.history,'history',public),
                    (slot.critic_history.history,'critic_history',central)):
                saved = before[name] if name=='history' else before[name]['history']
                assert len(history.players[view].records) == len(saved['players'][view]['records'])
                if len(saved['players'][view]['records'])>1:
                    assert history.players[view].state_dict()['records'][:-1] == saved['players'][view]['records'][:-1]
                np.testing.assert_array_equal(record_array(history.players[view].records[-1],pool.mode,True), rows[view])
    assert pool.slots[1].game.public_history[-1].eliminated_players == (0,)
    result = copy.deepcopy(payload)
    reconcile_no_move_rule(payload)
    helpers.assert_tree_equal(helper,payload,result)


def test_rule_marker_cannot_mask_legacy_saved_games():
    payload = dict(config={'eliminate_no_moves':True},trainer_state={'base_game_pool':{
        'slots':[{'game':{'config':GameConfig(eliminate_no_moves=False)}}]}})
    with pytest.raises(ValueError,match='contradicts'):
        reconcile_no_move_rule(payload)
