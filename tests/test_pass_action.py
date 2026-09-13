"""Voluntary-pass rules, public inputs, legal policy masks and resume migration."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from junqi.board import PASS_ACTION
from junqi.game import CombatOutcome, IllegalActionError, JunqiGame, TerminationReason
from junqi.training.encoding import ActionFeatures, GameHistory, _record_from_observation
from junqi.training.history_arrays import record_array
from junqi.training.models import GamePolicyTransformer, ModelConfig, collate_policy_states
from junqi.training.modes import TrainingMode, mode_spec, new_game
from junqi.training.packed_observation import observation_rows
from junqi.training.ppo_environment import ParallelPPOEnvironment
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


@pytest.mark.parametrize("mode", list(TrainingMode))
def test_four_opportunities_are_independent_noops_and_then_hard_masked(mode):
    game = new_game(mode, seed=29)
    before = dict(game.pieces)
    history = GameHistory.initialize(game, mode)
    board = game.board_for(0)
    assert board.decode_action(0) == PASS_ACTION
    assert board.action_index(0, 0) == 0
    assert board.action(0, 1) == (0, 1)
    policy = GamePolicyTransformer(ModelConfig.tiny()).eval()
    for _ in range(4):
        for _ in range(mode_spec(mode).player_count):
            actor = game.current_player
            previous = game.passes_remaining
            assert game.legal_action_mask()[0]
            result = game.step(0)
            history.append_after_step(game)
            assert result.action == PASS_ACTION and result.combat is CombatOutcome.PASS
            assert result.attacker is None and result.defender is None
            assert game.passes_remaining == tuple(n - int(i == actor) for i, n in enumerate(previous))
            assert dict(game.pieces) == before
            for viewer in range(mode_spec(mode).player_count):
                observation = game.observe(viewer)
                assert observation.history[-1].action == PASS_ACTION
                order = game._relative_player_order(viewer)
                assert observation.passes_remaining == tuple(game.passes_remaining[i] for i in order)
                assert history.players[viewer].records[-1].passes_remaining == observation.passes_remaining + (0,) * (4-len(order))
    assert game.no_interaction_plies == game.ply_count == 4 * mode_spec(mode).player_count
    assert game.current_player == 0
    state = history.state_for(game)
    assert PASS_ACTION not in state.legal_actions
    assert not game.legal_action_mask()[0]
    key = game.state_key()
    for action in (0, PASS_ACTION):
        with pytest.raises(IllegalActionError):
            game.step(action)
        assert game.state_key() == key
    with torch.inference_mode():
        _, masks = policy._sampling_legal_masks([state], board.point_count)
        assert not masks[0, 0, 0]
        actions, _ = policy.sample_action_groups([state], count=64)
        assert all(a != PASS_ACTION and a in state.legal_actions for a in actions[0])
    cloned = game.clone()
    assert cloned.state_key() == key and cloned._passes_remaining is not game._passes_remaining


@pytest.mark.parametrize("mode", list(TrainingMode))
@pytest.mark.parametrize("dead", [False, True])
def test_pass_inputs_match_workers_array_history_and_cached_policy(mode, dead):
    game = new_game(mode, seed=32, dead_rules_enabled=dead)
    history = GameHistory.initialize(game, mode, max_transitions=16)
    policy = GamePolicyTransformer(ModelConfig.tiny(dead_rules_enabled=dead)).eval()
    reference = copy.deepcopy(policy)
    policy.start_inference_board_cache()
    for _ in range(3):
        game.step(PASS_ACTION)
        history.append_after_step(game)
        rows = observation_rows(game, mode)
        for viewer in range(mode_spec(mode).player_count):
            obs = game.observe(viewer)
            record = history.players[viewer].records[-1]
            np.testing.assert_array_equal(rows[viewer], record_array(record, mode, dead))
            assert record.action.as_vector(mode, no_capture_plies=game.no_interaction_plies)[:4] == (0, 0, 0, 0)
            assert record.action.as_vector(mode, no_capture_plies=game.no_interaction_plies)[-1] == 70-game.ply_count
        state = history.state_for(game)
        with torch.inference_mode():
            logs = policy([state], [state.legal_actions])[0]
            expected = reference([state], [state.legal_actions])[0]
            torch.testing.assert_close(logs, expected)
    restored = GameHistory.from_state_dict(history.state_dict())
    restored.enable_array_storage()
    assert restored.state_dict() == history.state_dict()
    array_state = restored.state_for(game)
    batch = collate_policy_states([array_state], device='cpu', dead_rules_enabled=dead)
    assert tuple(batch.passes_remaining[-1].int().tolist()) == state.records[-1].passes_remaining
    with torch.inference_mode():
        torch.testing.assert_close(reference([array_state], [array_state.legal_actions])[0], expected)
    # Same board with different remaining budgets must not share a cached board token.
    altered = replace(state.records[-1], passes_remaining=(0, 0, 0, 0))
    changed = replace(state, records=(*state.records[:-1], altered))
    with torch.inference_mode():
        a = policy.encode([state]).context
        b = policy.encode([changed]).context
        c = reference.encode([changed]).context
    torch.testing.assert_close(b, c)
    assert not torch.allclose(a, b)


def test_pass_is_the_seventieth_no_capture_ply_and_cannot_be_replayed_twice():
    game = new_game('four_dark', seed=11)
    game.no_interaction_plies = 69
    game.step(0)
    assert game.result.reason is TerminationReason.NO_CAPTURE_DRAW
    assert game.passes_remaining == (3, 4, 4, 4)
    assert game.no_interaction_plies == 70
    assert all(not x for x in game.legal_action_mask())


def test_process_workers_preserve_pass_budgets_and_publish_updated_inputs():
    mode = TrainingMode.FOUR_DARK
    pool = SimpleNamespace(mode=mode, slots=[SimpleNamespace(game=new_game(mode, seed=i)) for i in range(2)])
    env = ParallelPPOEnvironment(2)
    try:
        env.begin(pool)
        for _ in range(16):
            env.submit([0, 1], [PASS_ACTION, PASS_ACTION])
            rows = env.receive()
            assert all(np.count_nonzero(row.records[:, -4:] < 4) for row in rows)
        assert all(PASS_ACTION not in state.legal_actions for state in env.states)
        env.synchronize(pool)
        assert all(slot.game.passes_remaining == (0, 0, 0, 0) for slot in pool.slots)
    finally:
        env.close()


def test_v8_resume_preserves_weights_adam_rng_progress_and_adds_zero_columns(tmp_path):
    config = Path(__file__).parents[1] / 'configs/bootstrap.yaml'
    settings = TrainingSettings.from_yaml(config, 'four_dark', tiny=True, overrides={
        'device': 'cpu', 'anchor_batch': 8, 'policy_microbatch': 2, 'base_game_pool_size': 2,
        'actor_inference_batch': 2, 'total_updates': 2})
    original = SelfPlayTrainer(settings, run_directory=tmp_path, auto_resume=False)
    try:
        original.pool.fill(original.layout, 0)
        for model, optimizer in ((original.policy, original.policy_optimizer), (original.critic, original.critic_optimizer)):
            sum(p.square().sum() for p in model.parameters()).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        original.update = 1
        original.cumulative['environment_plies'] = 8
        original.save_checkpoint(reason='test_fixture', archive=False)
        path = original.checkpoints.latest_path
        payload = torch.load(path, weights_only=False)
        expected_pool = copy.deepcopy(payload['trainer_state']['base_game_pool'])
        payload['format_version'] = 8
        payload['config'].pop('max_passes_per_player')
        for slot in payload['trainer_state']['base_game_pool']['slots']:
            slot['game'].pop('passes_remaining')
            for player in slot['history']['players']:
                for record in player['records']:
                    record.pop('passes_remaining')
        for kind, model in (('policy', original.policy), ('critic', original.critic)):
            name = 'board_encoder.projection.weight'
            payload[kind][name] = payload[kind][name][:, :-4].clone()
            ids = payload[kind + '_optimizer']['param_groups'][0]['params']
            index = [n for n, _ in model.named_parameters()].index(name)
            for moment in ('exp_avg', 'exp_avg_sq'):
                value = payload[kind + '_optimizer']['state'][ids[index]][moment]
                payload[kind + '_optimizer']['state'][ids[index]][moment] = value[:, :-4].clone()
        torch.save(payload, path)
    finally:
        original.logger.close()
    with pytest.raises(RuntimeError, match='adopt-pass-rule'):
        SelfPlayTrainer(settings, run_directory=tmp_path)
    resumed = SelfPlayTrainer(settings, run_directory=tmp_path, adopt_pass_rule=True)
    try:
        assert resumed.update == 1 and resumed.cumulative == payload['trainer_state']['cumulative']
        assert resumed.pool.state_dict() == expected_pool
        torch.testing.assert_close(torch.get_rng_state(), payload['rng_state']['torch_cpu'], rtol=0, atol=0)
        for kind, model, optimizer in (('policy', resumed.policy, resumed.policy_optimizer), ('critic', resumed.critic, resumed.critic_optimizer)):
            for name, value in payload[kind].items():
                actual = model.state_dict()[name]
                if name == 'board_encoder.projection.weight':
                    assert actual[:, -4:].count_nonzero() == 0
                    actual = actual[:, :-4]
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
            parameter = model.board_encoder.projection.weight
            ids = payload[kind + '_optimizer']['param_groups'][0]['params']
            index = [n for n, _ in model.named_parameters()].index('board_encoder.projection.weight')
            for moment in ('exp_avg', 'exp_avg_sq'):
                saved = payload[kind + '_optimizer']['state'][ids[index]][moment]
                torch.testing.assert_close(optimizer.state[parameter][moment][:, :-4], saved, rtol=0, atol=0)
                assert optimizer.state[parameter][moment][:, -4:].count_nonzero() == 0
        resumed.train()
        saved = torch.load(path, weights_only=False)
        assert saved['format_version'] == 9 and saved['update'] == 2
        assert saved['trainer_state']['cumulative']['environment_plies'] == 16
        assert saved['trainer_state']['pass_rule_migration']['update'] == 1
    finally:
        resumed.logger.close()
