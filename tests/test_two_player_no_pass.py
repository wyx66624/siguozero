"""Two-player pass masking, legacy compatibility and complete-state continuation."""
import copy
from dataclasses import fields, replace
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from junqi.board import PASS_ACTION
from junqi.game import GameConfig, GameVariant, IllegalActionError
from junqi.training.arena import MatchSettings
from junqi.training.encoding import GameHistory
from junqi.training.models import GamePolicyTransformer, ModelConfig
from junqi.training.modes import TrainingMode, new_game
from junqi.training.ppo_environment import ParallelPPOEnvironment
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer
from junqi.training.two_player_no_pass_migration import reconcile_two_player_no_pass
from junqi.training.no_pass_features import WEIGHT, remove_two_player_pass_inputs
from junqi.training.encoding import PASS_CONTEXT_ACTION_ENCODER_TYPE
from junqi.web.cpu_game import CpuGame

CONFIG = Path(__file__).parents[1] / "configs/expand_mappo_training.yaml"


def settings(**overrides):
    return TrainingSettings.from_yaml(CONFIG, "two_player", tiny=True, overrides=dict(
        device="cpu", anchor_batch=16, base_game_pool_size=2, policy_microbatch=2,
        max_game_plies=64, policy_epochs=1, critic_epochs=1, inference_snapshot_every_updates=0,
        **overrides))


def test_two_player_rejects_pass_without_changing_state_or_model_shape():
    torch.set_num_threads(1)
    game = new_game("two_player", seed=29)
    history = GameHistory.initialize(game, "two_player")
    assert game.passes_remaining == (0, 0)
    assert not game.legal_action_mask()[0]
    assert PASS_ACTION not in game.legal_actions()
    assert all(record.records[-1].passes_remaining == (0, 0, 0, 0) for record in history.players)
    before = game.state_key()
    for action in (0, PASS_ACTION):
        with pytest.raises(IllegalActionError):
            game.step(action)
        assert game.state_key() == before
    policy = GamePolicyTransformer(ModelConfig.tiny()).eval()
    with torch.inference_mode():
        state = history.state_for(game)
        _, mask = policy._sampling_legal_masks([state], game.board_for(0).point_count)
        assert not mask[0, 0, 0]
        actions, _ = policy.sample_action_groups([state], count=64)
        assert all(a != PASS_ACTION and a in game.legal_actions() for a in actions[0])
    # Exercise the actual CPU game's request handler as well as the rule engine.
    cpu = CpuGame.__new__(CpuGame)
    cpu.game, cpu.history, cpu.human = game, history, 0
    cpu.ended, cpu.review = False, SimpleNamespace()
    with pytest.raises((IllegalActionError, ValueError)):
        cpu.command({"op": "move", "action": [0, 0], "expected_ply": 0})
    assert game.state_key() == before


def test_mode_contracts_and_legacy_pickle():
    assert settings().max_passes_per_player == MatchSettings().max_passes_per_player == 0
    assert MatchSettings(mode="four_dark").max_passes_per_player == 4
    assert new_game("four_dark", seed=1).passes_remaining == (4, 4, 4, 4)
    current = GameConfig(variant=GameVariant.TWO_PLAYER)
    values = [getattr(current, f.name) for f in fields(current)]
    for length in (6, 7):
        legacy = GameConfig.__new__(GameConfig)
        legacy.__setstate__(values[:length])
        assert legacy.allow_passes is True
        assert legacy.eliminate_no_moves is (length == 7)
        assert pickle.loads(pickle.dumps(legacy)) == legacy
    assert pickle.loads(pickle.dumps(current)).allow_passes is False
    with pytest.raises(ValueError):
        settings(max_passes_per_player=4)


def legacy_payload():
    game = new_game("two_player", seed=21)
    game.config = replace(game.config, allow_passes=True)
    game._passes_remaining = [4, 4]
    game._invalidate_legal_actions()
    history = GameHistory.initialize(game, "two_player")
    game.step(PASS_ACTION)
    history.append_after_step(game)
    slot = {"game": {"config": game.config, "passes_remaining": game.passes_remaining,
                     "public_history": game.public_history}, "history": history.state_dict()}
    slot["critic_history"] = {"history": copy.deepcopy(slot["history"])}
    return {"config": {"max_passes_per_player": 4}, "update": 42,
            "trainer_state": {"distributed": {"rank_states": [
                {"base_game_pool": {"slots": [copy.deepcopy(slot)]}} for _ in range(2)]},
                "cumulative": {"environment_plies": 900}}}


def test_all_ranks_migrate_atomically_and_past_passes_are_retained():
    payload = legacy_payload()
    before = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="adopt-two-player-no-pass"):
        reconcile_two_player_no_pass(payload, settings())
    assert payload == before
    broken = copy.deepcopy(payload)
    broken["trainer_state"]["distributed"]["rank_states"][1]["base_game_pool"]["slots"][0]["game"]["passes_remaining"] = (5, 4)
    saved = copy.deepcopy(broken)
    with pytest.raises(ValueError, match="contradicts"):
        reconcile_two_player_no_pass(broken, settings(), adopt=True)
    assert broken == saved
    assert reconcile_two_player_no_pass(payload, settings(), adopt=True)
    for rank, old in zip(payload["trainer_state"]["distributed"]["rank_states"], before["trainer_state"]["distributed"]["rank_states"]):
        slot, previous = rank["base_game_pool"]["slots"][0], old["base_game_pool"]["slots"][0]
        assert slot["game"]["passes_remaining"] == (0, 0)
        assert slot["game"]["public_history"] == previous["game"]["public_history"]
        for key in ("history", "critic_history"):
            history = slot[key] if key == "history" else slot[key]["history"]
            old_history = previous[key] if key == "history" else previous[key]["history"]
            for p, q in zip(history["players"], old_history["players"]):
                assert p["records"][:-1] == q["records"][:-1]
                assert p["records"][-1] == dict(q["records"][-1], passes_remaining=(0, 0, 0, 0))
    after = copy.deepcopy(payload)
    assert not reconcile_two_player_no_pass(payload, settings())
    assert payload == after


def test_process_workers_keep_pass_masked():
    mode = TrainingMode.TWO_PLAYER
    pool = SimpleNamespace(mode=mode, slots=[SimpleNamespace(game=new_game(mode, seed=i)) for i in range(2)])
    env = ParallelPPOEnvironment(2)
    try:
        env.begin(pool)
        for _ in range(4):
            assert all(PASS_ACTION not in state.legal_actions for state in env.states)
            env.submit([0, 1], [state.legal_actions[0] for state in env.states])
            rows = env.receive()
            assert all(np.count_nonzero(row.records[:, -4:]) == 0 for row in rows)
        env.synchronize(pool)
        assert all(slot.game.passes_remaining == (0, 0) for slot in pool.slots)
    finally:
        env.close()


def test_full_resume_preserves_weights_adam_rng_games_and_progress(tmp_path):
    torch.set_num_threads(1)
    old_settings = settings()
    original = SelfPlayTrainer(old_settings, run_directory=tmp_path)
    original.train()
    path = original.checkpoints.latest_path
    before = torch.load(path, map_location="cpu", weights_only=False)
    _legacy_feature_columns(before, {"policy": original.policy, "critic": original.critic})
    # Construct the previous on-disk rule contract with valid legacy histories.
    before["config"]["max_passes_per_player"] = 4
    for slot in before["trainer_state"]["base_game_pool"]["slots"]:
        slot["game"]["config"] = replace(slot["game"]["config"], allow_passes=True)
        slot["game"]["passes_remaining"] = (4, 4)
        for history in (slot["history"], slot["critic_history"]["history"]):
            for player in history["players"]:
                for record in player["records"]:
                    record["passes_remaining"] = (4, 4, 0, 0)
    torch.save(before, path)
    modern = replace(old_settings, total_updates=2)
    with pytest.raises(ValueError, match="adopt-two-player-no-pass"):
        SelfPlayTrainer(modern, run_directory=tmp_path)
    resumed = SelfPlayTrainer(modern, run_directory=tmp_path, adopt_two_player_no_pass=True)
    try:
        after = torch.load(path, map_location="cpu", weights_only=False)
        assert after["reason"] == "two_player_no_pass_migration"
        assert after["update"] == before["update"] == 1
        assert after["trainer_state"]["cumulative"] == before["trainer_state"]["cumulative"]
        expected = copy.deepcopy(before)
        reconcile_two_player_no_pass(expected, modern, adopt=True)
        remove_two_player_pass_inputs(expected, optimizer_modules={"policy": resumed.policy, "critic": resumed.critic})
        for key in ("policy", "critic", "layout", "reference_layout", "policy_optimizer",
                    "critic_optimizer", "layout_optimizer", "rng_state"):
            torch.testing.assert_close(after[key], expected[key], rtol=0, atol=0)
        assert after["format_version"] == 18
        for name in ("policy", "critic"):
            assert after[name][WEIGHT].shape[1] == 10
            torch.testing.assert_close(after[name][WEIGHT], before[name][WEIGHT][:, :10], rtol=0, atol=0)
        assert resumed.policy.action_encoder.pass_feature_dim == resumed.critic.action_encoder.pass_feature_dim == 0
        assert after["trainer_state"]["base_game_pool"] == expected["trainer_state"]["base_game_pool"]
        resumed.train()
        assert resumed.update == 2 and resumed.cumulative["environment_plies"] == 32
        ordinary = SelfPlayTrainer(modern, run_directory=tmp_path)
        assert ordinary.update == 2
        assert all(PASS_ACTION not in s.game.legal_actions() for s in ordinary.pool.slots)
        ordinary.historical.close()
        ordinary.logger.close()
    finally:
        resumed.historical.close()
        resumed.logger.close()


def _legacy_feature_columns(payload, modules):
    payload["format_version"] = 17
    payload["config"]["model"]["action_encoder_type"] = PASS_CONTEXT_ACTION_ENCODER_TYPE
    for name, module in modules.items():
        weight = payload[name][WEIGHT]
        payload[name][WEIGHT] = torch.cat((weight, weight.new_full((weight.shape[0], 4), 0.125)), dim=1)
        optimizer = payload.get(name + "_optimizer")
        if optimizer:
            names = [n for n, _ in module.named_parameters()]
            ids = [i for g in optimizer["param_groups"] for i in g["params"]]
            moments = optimizer["state"].get(ids[names.index(WEIGHT)], {})
            for key, value in list(moments.items()):
                if value.ndim:
                    moments[key] = torch.cat((value, value.new_full((value.shape[0], 4), 0.25)), dim=1)


def test_old_inference_and_packed_history_load_without_pass_dimensions(tmp_path):
    from junqi.training.inference import InferenceEngine
    from junqi.training.inference_snapshot import export_inference_snapshot
    from junqi.training.models import PieceConditionedLayoutPointerDecoder, PublicActionEncoder
    from junqi.training.tactical_opponent import pack_state_dict
    from junqi.training.historical_cache import HistoricalWeightCache
    torch.set_num_threads(1)
    configured = settings()
    policy = GamePolicyTransformer(configured.model)
    layout = PieceConditionedLayoutPointerDecoder(configured.model)
    path = export_inference_snapshot(tmp_path, policy, layout, configured, 7)
    payload = torch.load(path, weights_only=False)
    _legacy_feature_columns(payload, {"policy": policy})
    torch.save(payload, path)
    old_weight = payload["policy"][WEIGHT].clone()
    engine = InferenceEngine.from_checkpoint(path, device="cpu")
    assert engine.policy.action_encoder.projection.in_features == 10
    assert engine.policy.action_encoder.pass_feature_dim == 0
    assert torch.load(path, weights_only=False)["format_version"] == 17
    game, history = engine.new_game(seed=23)
    assert PASS_ACTION not in game.legal_actions()
    assert engine.select_action(game, history, temperature=0) in game.legal_actions()
    # Mathematical equivalence to the old layer with zero pass inputs.
    old = PublicActionEncoder(configured.model.action_embedding_dim, joint_context=True, passes_context=True)
    old.projection.load_state_dict({"weight": old_weight, "bias": policy.action_encoder.projection.bias})
    field = torch.zeros((2, 6)); present = torch.tensor([False, True])
    kwargs = dict(no_interaction=torch.tensor([0, 20]), active_mask=torch.tensor([3, 3]))
    torch.testing.assert_close(old(field, present, **kwargs, passes_remaining=torch.zeros((2, 4))),
                               engine.policy.action_encoder(field, present, **kwargs), rtol=1e-6, atol=1e-6)
    packed = copy.deepcopy(payload)
    packed["packed_weights"] = {name: pack_state_dict(packed.pop(name)) for name in ("policy", "layout")}
    cache = HistoricalWeightCache(tmp_path, configured, matrix_names=set())
    result = cache._prepare(packed)
    assert next(p["views"][WEIGHT][2][1] for p in result["policy"].values() if WEIGHT in p["views"]) == 10
    cache.close()


def test_evaluation_migration_keeps_champion_reports_and_uses_new_seeds(tmp_path):
    from junqi.training.distributed import DistributedContext
    from junqi.training.model_selection import ModelSelection
    from junqi.training.models import PieceConditionedLayoutPointerDecoder
    torch.set_num_threads(1)
    configured = settings(arena_enabled=True, arena_games=2)
    context = DistributedContext(0, 1, 0, torch.device("cpu"))
    selection = ModelSelection(configured, tmp_path, context)
    policy = GamePolicyTransformer(configured.model)
    layout = PieceConditionedLayoutPointerDecoder(configured.model)
    selection.initialize(policy, layout, update=0, cumulative={"environment_plies": 0})
    previous = selection.state
    previous["contract"]["max_passes_per_player"] = 4
    # A completed old-rule report is immutable; do not mix it into the new family.
    report_path = selection.directory / "completed_old_rule.json"
    report = {"schedule_contract": {"max_passes_per_player": 4}, "score": 0.5}
    report_path.write_text(json.dumps(report))
    previous["rounds"] = [report_path.name]
    selection.state_path.write_text(json.dumps(previous))
    migrated = ModelSelection(configured, tmp_path, context)
    migrated.initialize(policy, layout, update=0, cumulative={"environment_plies": 0},
                        adopt_two_player_no_pass=True)
    assert migrated.state["contract"]["max_passes_per_player"] == 0
    assert migrated.state["best_sha256"] == previous["best_sha256"]
    assert migrated.state["rounds"] == previous["rounds"]
    assert json.loads(report_path.read_text()) == report
    assert migrated.state["schedule_seed_base"] > previous["contract"]["seed"]
    assert migrated.state["statistical_family"] == "two_player_no_pass"
    again = ModelSelection(configured, tmp_path, context)
    again.initialize(policy, layout, update=0, cumulative={"environment_plies": 0})
    assert again.state == migrated.state


def test_existing_historical_catalog_survives_feature_removal(tmp_path):
    from junqi.training.distributed import DistributedContext
    from junqi.training.historical_opponents import HistoricalOpponents
    configured = TrainingSettings.from_yaml(CONFIG, "two_player", overrides={"device": "cpu"})
    context = DistributedContext(0, 1, 0, torch.device("cpu"))
    league = HistoricalOpponents(configured, tmp_path, context)
    old_contract = dict(league.contract(), checkpoint_format=17)
    entries = [dict(id="old", environment_plies=0, sha256="unchanged", file="old.pt")]
    league.directory.mkdir(parents=True)
    league.catalog_path.write_text(json.dumps(dict(contract=old_contract, opponents=entries)))
    loaded = HistoricalOpponents(configured, tmp_path, context)
    assert loaded.entries == entries and loaded._catalog_dirty
    assert loaded._compatible_contract(old_contract, 0)
    assert not loaded._compatible_contract(dict(old_contract, target=123), 0)
    assert not loaded._compatible_contract(dict(old_contract, mode="four_dark"), 0)
