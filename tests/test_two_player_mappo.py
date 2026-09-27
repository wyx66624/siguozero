"""Two-player expanded encoding, terminal-only MAPPO and frozen opponents."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from junqi.training.checkpoint import require_current_checkpoint
from junqi.training.clipping import scheduled_policy_clip
from junqi.training.cli import build_parser
from junqi.training.distributed import DistributedContext
from junqi.training.encoding import GameHistory
from junqi.training.historical_opponents import HistoricalOpponents
from junqi.training.inference import InferenceEngine
from junqi.training.mappo import collect_mappo_samples, player_step_rewards
from junqi.training.mappo_encoding import CentralizedHistory
from junqi.training.model_selection import ModelSelection
from junqi.training.models import GameCentralValueTransformer, GamePolicyTransformer, PieceConditionedLayoutPointerDecoder
from junqi.training.modes import TrainingMode, new_game
from junqi.training.ppo import FrozenValueActor
from junqi.training.rollout import BaseGamePool, FrozenPolicyActor
from junqi.training.settings import TrainingSettings
from junqi.training.trainer import SelfPlayTrainer


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/expand_mappo_training.yaml"
MODE = TrainingMode.TWO_PLAYER


def config(**kwargs):
    return TrainingSettings.from_yaml(CONFIG, MODE, **kwargs)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_training_contract_matches_four_except_explicit_two_player_rules():
    two = config(model_scale="main")
    four = TrainingSettings.from_yaml(CONFIG, "four_dark", model_scale="main")
    allowed = {"mode", "model", "flag_capture_reward", "known_nonflag_hq_penalty",
               "bomb_commander_reward", "bomb_army_commander_reward", "bomb_division_commander_reward",
               "futile_attack_penalty", "historical_tactical_current", "historical_start_fraction",
               "historical_start_environment_plies", "historical_checkpoint_start_fraction",
               "arena_historical_teammate_fraction"}
    for field, value in asdict(two).items():
        if field not in allowed:
            assert value == asdict(four)[field], field
    assert replace(two.model, board_point_capacity=129) == four.model
    assert two.algorithm == "mappo" and two.ppo_clip_mode == "acpo"
    assert two.mappo_advantage_mode == "player_decisions" and two.mappo_reward_scope == "player"
    assert two.historical_teammate_fraction == two.arena_historical_teammate_fraction == 0
    assert not two.historical_tactical_current and not two.historical_tactical_champion
    assert two.checkpoint_interval_environment_plies == 25_000_000
    assert two.historical_start_environment_plies == 500_000_000
    assert two.draw_reward == -.15
    for field in ("flag_capture_reward", "known_nonflag_hq_penalty", "bomb_commander_reward",
                  "bomb_army_commander_reward", "bomb_division_commander_reward", "futile_attack_penalty"):
        assert getattr(two, field) == 0
        assert getattr(four, field) > 0
    assert [four.bomb_commander_reward, four.bomb_army_commander_reward,
            four.bomb_division_commander_reward, four.futile_attack_penalty] == [.03, .01, .003, .01]
    for step in (0, 500_000_000, 3_000_000_000):
        assert scheduled_policy_clip(two, step) == scheduled_policy_clip(four, step)
    assert build_parser(MODE).parse_args([]).mode == "two_player"


def test_60_point_encoding_matches_tensor_packed_and_fixed_kv():
    model = replace(config(tiny=True).model, ppo_tensor_learner=False,
                    ppo_cuda_graphs=False, max_transitions=7)
    ordinary = GamePolicyTransformer(model).eval()
    tensor = GamePolicyTransformer(replace(model, ppo_tensor_learner=True)).eval()
    cached = copy.deepcopy(ordinary)
    tensor.load_state_dict(ordinary.state_dict())
    assert ordinary.board_encoder.input_dim == 60 * 26 + 3
    assert ordinary.source_query[-1].out_features == 60
    assert ordinary.destination_query[0].in_features == model.temporal_dim + 60
    assert ordinary.destination_query[-1].out_features == 60
    cached.start_ppo_inference_cache(capacity=2, behavior_version=0)
    game = new_game(MODE, seed=29)
    history = GameHistory.initialize(game, MODE, max_transitions=7)
    history.enable_array_storage()
    central = CentralizedHistory.initialize(game, MODE, max_transitions=7)
    private_masks = history.state_for(game).records[0].inventory_masks
    assert any(mask.bit_count() > 1 for mask in private_masks)
    assert all(mask.bit_count() <= 1 for mask in central.value_state_for(0).records[0].inventory_masks)
    held = []
    with torch.inference_mode():
        for _ in range(12):
            state = history.state_for(game)
            held.append(state)
            expected = ordinary._encode_full([state]).context
            torch.testing.assert_close(tensor._encode_full([state]).context, expected, atol=5e-6, rtol=5e-6)
            torch.testing.assert_close(cached.encode([state]).context, expected, atol=5e-6, rtol=5e-6)
            action = ordinary.sample_action_groups([state])[0][0][0]
            assert action in state.legal_actions
            game.step(game.legal_actions()[0])
            history.append_after_step(game)
        torch.testing.assert_close(tensor._encode_full(held, pack_prefixes=True).context,
                                   ordinary._encode_full(held).context, atol=5e-6, rtol=5e-6)


@pytest.mark.parametrize("raw,expected", [((1., -1.), (1., -1.)), ((-1., 1.), (-1., 1.)),
                                        ((0., 0.), (-.15, -.15))])
def test_terminal_rewards_are_exact_with_capture_on_terminal_step(raw, expected):
    s = config()
    row = SimpleNamespace(terminal=True, rewards=raw, flag_captured_owner=1,
                          bombed_kind=None, bomb_owner=None)
    assert player_step_rewards(row, player=0, player_count=2, draw_reward=s.draw_reward,
        flag_capture_reward=s.flag_capture_reward, bomb_rewards={}, futile_reward=0.,
        known_nonflag_hq_reward=0.) == expected
    row.terminal = False
    assert player_step_rewards(row, player=0, player_count=2, draw_reward=s.draw_reward,
        flag_capture_reward=s.flag_capture_reward, bomb_rewards={}, futile_reward=0.) == (0., 0.)


def test_checkpoint_and_history_thresholds_are_environment_steps(tmp_path):
    s = config(model_scale="main")
    league = HistoricalOpponents(s, tmp_path, DistributedContext(0, 1, 0, torch.device("cpu")))
    assert league.threshold == 500_000_000 and league.checkpoint_threshold == 25_000_000
    league.settings = replace(s, target_environment_plies=2_000_000_000)
    assert league.threshold == 500_000_000
    league.progress = 499_999_999
    assert not league.active and league.assignment() == (None, None, 0)
    league.progress = 500_000_000
    assert league.active
    trainer = SelfPlayTrainer.__new__(SelfPlayTrainer)
    trainer.settings = s
    trainer.last_checkpoint_environment_plies = 0
    for step, due in ((24_999_999, False), (25_000_000, True), (25_067_520, True)):
        trainer.cumulative = {"environment_plies": step}
        assert trainer._periodic_checkpoint_due() is due
    trainer.last_checkpoint_environment_plies = 25_067_520
    trainer.cumulative = {"environment_plies": 49_999_999}
    assert not trainer._periodic_checkpoint_due()


@pytest.mark.parametrize("workers", [1, 2])
def test_frozen_opponent_rotation_collection_and_restore(tmp_path, workers):
    s = config(tiny=True, overrides=dict(
        device="cpu", target_environment_plies=1000, historical_enabled=True,
        historical_start_fraction=.5, historical_start_environment_plies=500,
        historical_snapshot_fractions=(.1, .2), historical_checkpoint_start_fraction=.1,
        historical_cache_gib=0., historical_cohort_games=1000,
        ppo_pipeline_groups=2, rollout_environment_workers=2,
        arena_enabled=True, arena_interval_environment_plies=100, arena_games=10,
        historical_eval_games=2, historical_eval_total_games=10))
    model = replace(s.model, max_transitions=8, ppo_cuda_graphs=False)
    s = replace(s, model=model)
    context = DistributedContext(0, 1, 0, torch.device("cpu"))
    league = HistoricalOpponents(s, tmp_path, context)
    policy = GamePolicyTransformer(model).eval()
    layout = PieceConditionedLayoutPointerDecoder(model).eval()
    critic = GameCentralValueTransformer(model).eval()
    pool = BaseGamePool(MODE, pool_size=10, max_transitions=8, max_game_plies=8,
                        seed=77, layout_prefetch_games=4, centralized_critic=True)
    pool.historical = league
    try:
        for step in (0, 100, 200, 500):
            league.update_progress(policy, layout, environment_plies=step, update=step)
        samples, outcomes, metrics = collect_mappo_samples(pool, FrozenPolicyActor(policy),
            FrozenValueActor(critic, amp_dtype=None, max_batch_size=4, deferred=True), layout,
            count=20, behavior_version=500, discount=1., gae_lambda=.95,
            advantage_mode="player_decisions", reward_scope="player", draw_reward=-.15,
            environment_workers=workers, pipeline_groups=workers)
        assert metrics.environment_plies == len(samples) == 20
        assert sum(not s.learnable for s in samples) == 2
        assert all(len(s.additional_values) == 1 for s in samples)
        mixed = [slot for slot in pool.slots if slot.opponent_id]
        assert len(mixed) == 2 and {slot.primary_seat for slot in mixed} == {0, 1}
        assert all(slot.teammate_id is None for slot in pool.slots)
        assert all(sum(slot.learner_owns(p) for p in range(2)) == 1 for slot in mixed)
        restored = BaseGamePool(MODE, pool_size=10, max_transitions=8, max_game_plies=8,
                               seed=77, layout_prefetch_games=4, centralized_critic=True)
        restored.historical = HistoricalOpponents(s, tmp_path, context)
        restored.load_state_dict(pool.state_dict())
        assert restored.state_dict() == pool.state_dict()
        assert [league.assignment() for _ in range(20)] == [restored.historical.assignment() for _ in range(20)]
        selection = ModelSelection(s, tmp_path, context)
        assert selection._historical_game_counts([{}, {}, {}]) == [4, 4, 2]
        with pytest.raises(ValueError, match="teammates"):
            replace(s, historical_teammate_fraction=.2).validate()
    finally:
        league.close()


def test_training_resume_and_actor_only_inference(tmp_path):
    s = config(tiny=True, overrides=dict(device="cpu", anchor_batch=8, policy_microbatch=2,
        policy_epochs=3, critic_epochs=3, ppo_adaptive_clip=True,
        lr_epoch_completion=True, inference_snapshot_every_updates=0))
    trainer = SelfPlayTrainer(s, run_directory=tmp_path)
    trainer.train()
    path = trainer.checkpoints.latest_path
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require_current_checkpoint(payload)
    assert payload["algorithm"] == "mappo" and payload["format_version"] == 17
    assert trainer.cumulative["environment_plies"] == 8
    assert payload["critic"]["board_encoder.projection.weight"].shape[1] == 1563
    assert payload["policy_optimizer"]["state"] and payload["critic_optimizer"]["state"]
    resumed = SelfPlayTrainer(s, run_directory=tmp_path)
    try:
        assert resumed.pool.state_dict() == trainer.pool.state_dict()
        for name in ("policy", "critic", "layout", "policy_optimizer", "critic_optimizer", "layout_optimizer"):
            torch.testing.assert_close(getattr(resumed, name).state_dict(), payload[name], rtol=0, atol=0)
        engine = InferenceEngine.from_checkpoint(path, device="cpu")
        assert not hasattr(engine, "critic")
        game, history = engine.new_game(seed=87, max_plies=4)
        assert engine.select_action(game, history) in game.legal_actions()
        broken = copy.deepcopy(payload)
        broken["config"]["model"]["board_point_capacity"] = 129
        with pytest.raises(ValueError, match="board inputs"):
            require_current_checkpoint(broken)
    finally:
        resumed.historical.close()
        resumed.logger.close()


@pytest.mark.skipif(os.name != "posix", reason="requires bash")
def test_main_launcher_uses_expanded_profile(tmp_path):
    fake = tmp_path / "python"
    fake.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o700)
    result = subprocess.run(["bash", str(ROOT / "scripts/start_two_player_main.sh")],
        env={**os.environ, "PYTHON_BIN": str(fake), "RUN_DIR": "runs_expand_mappo"},
        text=True, capture_output=True, check=True)
    args = json.loads(result.stdout)
    assert args[args.index("--config") + 1] == "configs/expand_mappo_training.yaml"
    assert "--target-continuation-plies" not in args
    assert args[args.index("--model-scale") + 1] == "main"
