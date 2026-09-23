"""Public-information labels, observational equivalence and exact replay/GAE."""
from contextlib import nullcontext
import copy
from dataclasses import replace
import gzip
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from junqi import ArmPoint, GameConfig, GameVariant, InformationMode, JunqiGame, Piece, PieceType
from junqi.training.mappo import collect_mappo_samples
from junqi.training.models import GameCentralValueTransformer, GamePolicyTransformer, PieceConditionedLayoutPointerDecoder
from junqi.training.ppo import FrozenValueActor, normalize_advantages
from junqi.training.ppo_environment import _advance
from junqi.training.rollout import BaseGamePool, FrozenPolicyActor
from junqi.training.tactical_events import public_tactical_event
from junqi.training.tactical_logging import MAPPOTrainingAudit, TacticalLogSettings
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer
from test_mappo import tiny_config
from test_mappo_tactical_rewards import assert_tree_equal


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("report_mappo_tactics", ROOT/"tools/report_mappo_tactics.py")
report_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_tool)


def position(extra, *, known=()):
    pieces = {ArmPoint(p, 6, 4): Piece(p, PieceType.FLAG) for p in range(4)}
    pieces.update({ArmPoint(p, 2, 3): Piece(p, PieceType.ENGINEER) for p in range(4)})
    pieces.update(extra)
    game = JunqiGame.from_position(GameConfig(variant=GameVariant.FOUR_PLAYER,
        information_mode=InformationMode.FOUR_DARK), pieces, current_player=0,
        known_identities=[{p: pieces[p].kind for p in known}, {}, {}, {}])
    game.enable_inventory_tracking()
    return game


def action(game, start, end):
    pair = (game.board_for(0).encode(start), game.board_for(0).encode(end))
    assert pair in game.legal_actions()
    return pair


def test_bomb_labels_do_not_use_hidden_referee_kind():
    source, target = ArmPoint(1, 4, 1), ArmPoint(1, 5, 1)
    outputs = []
    for kind in (PieceType.COMMANDER, PieceType.BRIGADE_COMMANDER):
        other_kind = PieceType.BRIGADE_COMMANDER if kind is PieceType.COMMANDER else PieceType.COMMANDER
        game = position({source: Piece(0, PieceType.BOMB), target: Piece(1, kind, has_moved=True),
                         ArmPoint(1, 4, 3): Piece(1, other_kind, has_moved=True)})
        pair = action(game, source, target)
        before = game.state_key()
        public = public_tactical_event(game, pair)
        assert game.state_key() == before
        assert not any(n.startswith("bomb_") for n in public["opportunities"])
        outputs.append(public)
    assert outputs[0] == outputs[1]
    game = position({source: Piece(0, PieceType.BOMB), target: Piece(1, PieceType.COMMANDER)}, known=[target])
    assert "bomb_known_high" in public_tactical_event(game, action(game, source, target))["taken"]


def test_known_flag_defense_is_a_legal_public_capture_and_unknown_is_not_peeked():
    enemy, defender = ArmPoint(0, 5, 4), ArmPoint(0, 5, 3)
    game = position({enemy: Piece(1, PieceType.ENGINEER, has_moved=True),
                     defender: Piece(0, PieceType.BATTALION_COMMANDER)}, known=[enemy])
    pair = action(game, defender, enemy)
    event = public_tactical_event(game, pair)
    assert event["own_flag_threats"] == [pair[1]]
    assert pair in event["opportunities"]["flag_threat_removal"]
    result = _advance(0, game, pair, "four_dark", centralized_critic=True, tactical_logging=True)
    assert result.tactical["referee_outcome"]["combat"] == "attacker_wins"
    assert "flag_threat_removal" in result.tactical["taken"]


@pytest.mark.parametrize("kind,category,enemy_kind", [
    (PieceType.ENGINEER, "engineer_escape", PieceType.COMMANDER),
    (PieceType.COMMANDER, "high_piece_escape", PieceType.BOMB)])
def test_escape_to_camp_and_engineer_exposure(kind, category, enemy_kind):
    source, enemy = ArmPoint(0, 3, 1), ArmPoint(0, 1, 1)
    game = position({source: Piece(0, kind), enemy: Piece(1, enemy_kind, has_moved=True)}, known=[enemy])
    safe = action(game, source, ArmPoint(0, 2, 2))
    event = public_tactical_event(game, safe)
    assert safe in event["opportunities"][category]
    assert safe not in event["opportunities"].get("engineer_exposure", [])
    if kind is PieceType.ENGINEER:
        risky = action(game, source, ArmPoint(0, 2, 1))
        assert risky in event["opportunities"]["engineer_exposure"]


def test_quiet_reversal_is_a_pattern_not_a_reward_change():
    start, end = ArmPoint(0, 3, 1), ArmPoint(0, 2, 1)
    game = position({start: Piece(0, PieceType.ENGINEER)})
    game.step(action(game, start, end))
    while game.current_player != 0:
        game.step((0, 0))
    back = action(game, end, start)
    assert "quiet_reversal" in public_tactical_event(game, back)["taken"]


def collect_pair():
    torch.set_num_threads(1)
    torch.manual_seed(151)
    config = tiny_config(ppo_fixed_kv=False)
    policy = GamePolicyTransformer(config).eval()
    critic = GameCentralValueTransformer(config).eval()
    layout = PieceConditionedLayoutPointerDecoder(config).eval()
    pool = BaseGamePool("four_dark", pool_size=2, max_transitions=16, max_game_plies=12,
                        seed=421, centralized_critic=True)
    pool.fill(layout, 0)
    initial = copy.deepcopy(pool.state_dict())
    rng = torch.get_rng_state()
    results = []
    for enabled in (False, True):
        pool.load_state_dict(copy.deepcopy(initial))
        torch.set_rng_state(rng)
        actor = FrozenPolicyActor(policy, max_batch_size=2)
        value_actor = FrozenValueActor(critic, amp_dtype=None, max_batch_size=2, deferred=True)
        audit = MAPPOTrainingAudit(TacticalLogSettings(enabled=True, environment_lanes=1, probe_samples=8), update=1) if enabled else None
        samples, outcomes, metrics = collect_mappo_samples(pool, actor, value_actor, layout,
            count=32, behavior_version=0, discount=1., gae_lambda=.95, environment_workers=2,
            pipeline_groups=2, advantage_mode="player_decisions", reward_scope="player", training_audit=audit)
        results.append((samples, outcomes, metrics, copy.deepcopy(pool.state_dict()), torch.get_rng_state(), audit))
    return policy, results


def test_logging_is_observational_and_replay_reproduces_actual_advantages(tmp_path):
    policy, (plain, logged) = collect_pair()
    for a, b in zip(plain[0], logged[0], strict=True):
        for field in ("action", "old_log_prob", "old_value", "advantage", "value_target", "value_player"):
            assert getattr(a, field) == getattr(b, field)
        assert a.state.records[-1] == b.state.records[-1]
    # Deep exact comparison includes the environment/pool RNG, inventories, histories and queued layouts.
    from unittest import TestCase
    assert_tree_equal(TestCase(), plain[3], logged[3])
    torch.testing.assert_close(plain[4], logged[4], rtol=0, atol=0)
    samples, audit = logged[0], logged[-1]
    raw = [s.advantage for s in samples]
    mean, std = float(np.mean(raw)), float(np.std(raw))
    groups = normalize_advantages(samples, mean=mean, std=std, epsilon=1e-4)
    objective = dict(mode="four_dark", advantage_mode="player_decisions", reward_scope="player",
        discount=1., gae_lambda=.95, draw_reward=-.15, flag_capture_reward=0., futile_attack_penalty=0.,
        bomb_commander_reward=0., bomb_army_commander_reward=0., bomb_division_commander_reward=0.)
    audit.bind(groups, normalization=dict(mean=mean, std=std, epsilon=1e-4), objective=objective)
    torch_rng, py_rng = torch.get_rng_state(), random.getstate()
    trainer = SimpleNamespace(policy=policy, _autocast=nullcontext,
                              _learner_chunks=lambda items: [items[i:i+2] for i in range(0, len(items), 2)])
    audit.measure(trainer, "before")
    audit.measure(trainer, "after")
    torch.testing.assert_close(torch_rng, torch.get_rng_state(), rtol=0, atol=0)
    assert py_rng == random.getstate()
    for s in audit.probes:
        assert s.training_diagnostic["before"] == s.training_diagnostic["after"]
    audit.commit_step(groups)
    metrics = audit.finish()
    assert metrics["tactical/sampled_decisions"] == 16
    assert metrics["tactical/complete_games"] == 1
    audit.write(tmp_path, environment_plies=32)
    archive = next((tmp_path/"tactical_diagnostics/details").glob("*.pt.gz"))
    with gzip.open(archive, "rb") as stream:
        payload = torch.load(stream, weights_only=False)
    result = report_tool.verify_replay(payload)
    assert result["verified_decisions"] == 16
    assert result["verified_games_or_segments"] == 2
    rows = [step["row"] for game in payload["games"] for step in game["steps"]]
    assert all(len(row["critic_values"]) == 4 and row["optimizer_uses"] == 1 for row in rows)
    assert json.loads((tmp_path/"tactical_diagnostics/latest.rank000.json").read_text())["sampled_decisions"] == 16
    # Summary retention is independent; detail pairs expire together.
    audit.settings = replace(audit.settings, keep_updates=1)
    audit.update = 2
    audit.write(tmp_path, environment_plies=64)
    assert not archive.exists()
    assert len(list((tmp_path/"tactical_diagnostics/updates").glob("*.json"))) == 2
    assert len(list((tmp_path/"tactical_diagnostics/details").glob("*"))) == 2


@pytest.mark.parametrize("value", [dict(enabled="true"), dict(environment_lanes=0),
    dict(probe_samples=-1), dict(probe_samples=513), dict(keep_updates=0), dict(max_detail_mib=False)])
def test_invalid_logging_config_rejected(value):
    with pytest.raises(ValueError):
        TacticalLogSettings.from_config({"runtime": {"tactical_logging": value}})


@pytest.mark.parametrize("skip_optimizer", [False, True])
def test_real_trainer_optimizer_and_replay_logs_do_not_change_training(tmp_path, skip_optimizer):
    torch.set_num_threads(1)
    settings = TrainingSettings.from_yaml(ROOT/"configs/expand_mappo_training.yaml", "four_dark", tiny=True,
        overrides=dict(device="cpu", amp="float32", total_updates=1, anchor_batch=32,
            base_game_pool_size=2, actor_inference_batch=2, rollout_environment_workers=2,
            policy_microbatch=4, ppo_minibatch_samples=8, policy_epochs=1, critic_epochs=1,
            critic_layers=1, max_game_plies=12, arena_enabled=False, historical_enabled=False,
            layout_update_interval=10, inference_snapshot_every_updates=0,
            lr_probe_samples=8, ppo_learner_cuda_graphs=False))
    snapshots = []
    for enabled in (False, True):
        raw = copy.deepcopy(settings.raw_config)
        raw["runtime"]["tactical_logging"] = dict(enabled=enabled, environment_lanes=1, probe_samples=8)
        trainer = SelfPlayTrainer(replace(settings, raw_config=raw), run_directory=tmp_path/str(enabled))
        if skip_optimizer:
            # Simulates GradScaler refusing a policy optimizer step. No post-hook may fire.
            trainer.grad_scaler.step = lambda optimizer: None
        trainer.train()
        snapshots.append(copy.deepcopy(trainer.policy.state_dict()))
        if enabled:
            report = json.loads((trainer.run_directory/"tactical_diagnostics/latest.rank000.json").read_text())
            assert report["optimizer_steps_applied"] == (0 if skip_optimizer else 4)
            with gzip.open(trainer.run_directory/"tactical_diagnostics"/report["replay_file"], "rb") as stream:
                payload = torch.load(stream, weights_only=False)
            assert report_tool.verify_replay(payload)["verified_decisions"] == 16
            assert all(s["row"]["optimizer_uses"] == (0 if skip_optimizer else 1)
                       for g in payload["games"] for s in g["steps"])
    for name in snapshots[0]:
        torch.testing.assert_close(snapshots[0][name], snapshots[1][name], atol=0, rtol=0)
