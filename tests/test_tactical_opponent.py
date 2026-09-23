"""Hard rules, public-information limits, champion refresh and frozen PPO seats."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from junqi import ArmPoint, GameConfig, GameVariant, InformationMode, JunqiGame, Piece, PieceType
from junqi.training.arena import atomic_json, sha256_file
from junqi.training.checkpoint_format import checkpoint_format_for_model
from junqi.training.distributed import DistributedContext
from junqi.training.encoding import GameHistory
from junqi.training.historical_opponents import HistoricalOpponents
from junqi.training.mappo import collect_mappo_samples, critic_training_samples
from junqi.training.mappo_encoding import CentralizedHistory
from junqi.training.models import GameCentralValueTransformer, GamePolicyTransformer, PieceConditionedLayoutPointerDecoder
from junqi.training.ppo import FrozenValueActor
from junqi.training.rollout import BaseGamePool, FrozenPolicyActor
from junqi.training.settings import TrainingSettings
from junqi.training.tactical_opponent import (LIVE_CHAMPION_ID, TacticalOpponentActor,
                                            forced_tactical_actions)
from test_historical_opponents import settings


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def position(extra, *, known=(), current_player=0):
    pieces = {ArmPoint(p, 6, 4): Piece(p, PieceType.FLAG) for p in range(4)}
    pieces.update({ArmPoint(p, 2, 3): Piece(p, PieceType.ENGINEER) for p in range(4)})
    pieces.update(extra)
    for owner in range(4):
        if not any(p.owner == owner and p.kind is PieceType.COMMANDER for p in pieces.values()):
            pieces[ArmPoint(owner, 3, 4)] = Piece(owner, PieceType.COMMANDER)
    knowledge = [{} for _ in range(4)]
    knowledge[current_player] = {p: pieces[p].kind for p in known}
    game = JunqiGame.from_position(GameConfig(variant=GameVariant.FOUR_PLAYER,
        information_mode=InformationMode.FOUR_DARK), pieces, current_player=current_player,
        known_identities=knowledge)
    game.enable_inventory_tracking()
    return game, GameHistory.initialize(game, "four_dark", max_transitions=8).state_for(game)


def pair(game, source, target):
    b = game.board_for(game.current_player)
    result = b.encode(source), b.encode(target)
    assert result in game.legal_actions()
    return result


@pytest.mark.parametrize("kind,forced", [
    (PieceType.DIVISION_COMMANDER, True), (PieceType.ARMY_COMMANDER, True),
    (PieceType.COMMANDER, True), (PieceType.BRIGADE_COMMANDER, False),
    (PieceType.REGIMENT_COMMANDER, False)])
def test_bomb_requires_public_division_rank_or_higher(kind, forced):
    a, b = ArmPoint(1, 4, 1), ArmPoint(1, 5, 1)
    game, state = position({a: Piece(0, PieceType.BOMB), b: Piece(1, kind)}, known=[b])
    rule, actions = forced_tactical_actions(state)
    assert (pair(game, a, b) in actions) == forced
    assert rule == ("bomb_known_high" if forced else None)


@pytest.mark.parametrize("kind", [PieceType.COMMANDER, PieceType.ENGINEER, PieceType.FLAG])
def test_referee_identity_is_not_a_trigger(kind):
    a, b = ArmPoint(1, 4, 1), ArmPoint(1, 5, 1)
    if kind is PieceType.FLAG:
        a, b = ArmPoint(1, 5, 4), ArmPoint(1, 6, 4)
    _, state = position({a: Piece(0, PieceType.BOMB), b: Piece(1, kind)})
    assert forced_tactical_actions(state) == (None, ())
    with pytest.raises(ValueError, match="centralized"):
        forced_tactical_actions(replace(state, centralized=True))


def test_flag_then_bomb_then_engineer_priority():
    flag = ArmPoint(1, 6, 4)
    flag_attacker = ArmPoint(1, 5, 4)
    bomb, high = ArmPoint(3, 4, 1), ArmPoint(3, 5, 1)
    defender, engineer = ArmPoint(0, 5, 3), ArmPoint(0, 5, 4)
    extra = {flag_attacker: Piece(0, PieceType.COMPANY_COMMANDER),
             bomb: Piece(0, PieceType.BOMB), high: Piece(3, PieceType.DIVISION_COMMANDER),
             defender: Piece(0, PieceType.BATTALION_COMMANDER), engineer: Piece(1, PieceType.ENGINEER)}
    game, state = position(extra, known=[flag, high, engineer])
    assert forced_tactical_actions(state) == ("capture_flag", (pair(game, flag_attacker, flag),))
    game, state = position(extra, known=[high, engineer])
    assert forced_tactical_actions(state) == ("bomb_known_high", (pair(game, bomb, high),))
    game, state = position(extra, known=[engineer])
    assert forced_tactical_actions(state) == ("capture_known_engineer", (pair(game, defender, engineer),))


@pytest.mark.parametrize("seat", range(4))
def test_relative_seats_and_engineer_trades(seat):
    src, dst = ArmPoint(seat, 5, 3), ArmPoint(seat, 5, 4)
    enemy = (seat + 1) % 4
    game, state = position({src: Piece(seat, PieceType.ENGINEER), dst: Piece(enemy, PieceType.ENGINEER)},
                           known=[dst], current_player=seat)
    rule, actions = forced_tactical_actions(state)
    assert rule == "capture_known_engineer"
    assert pair(game, src, dst) in actions
    # The legal action list and observation both prevent own/allied captures.
    for owner in (seat, (seat + 2) % 4):
        _, friendly = position({src: Piece(seat, PieceType.BOMB), dst: Piece(owner, PieceType.COMMANDER)},
                               known=[dst], current_player=seat)
        assert forced_tactical_actions(friendly) == (None, ())


def test_camp_is_not_capturable_and_legal_actions_are_never_invented():
    a, b = ArmPoint(1, 3, 1), ArmPoint(1, 2, 2)
    _, state = position({a: Piece(0, PieceType.BOMB), b: Piece(1, PieceType.COMMANDER)}, known=[b])
    assert forced_tactical_actions(state) == (None, ())


def test_actor_guarantees_trigger_and_fallback_is_identical():
    cfg = settings()
    policy = GamePolicyTransformer(cfg.model).eval()
    plain = FrozenPolicyActor(policy)
    tactical = TacticalOpponentActor(policy)
    a, b = ArmPoint(1, 4, 1), ArmPoint(1, 5, 1)
    game, triggered = position({a: Piece(0, PieceType.BOMB), b: Piece(1, PieceType.COMMANDER)}, known=[b])
    selected, logs = tactical.sample([triggered], count=64)
    assert selected == [[pair(game, a, b)] * 64]
    torch.testing.assert_close(logs[0], torch.zeros_like(logs[0]), atol=0, rtol=0)
    _, fallback = position({})
    rng = torch.get_rng_state()
    expected = plain.sample([fallback], count=16)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    actual = tactical.sample([fallback], count=16)
    assert actual[0] == expected[0]
    torch.testing.assert_close(actual[1][0], expected[1][0], atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
    assert tactical.rule_counts == {"bomb_known_high": 64, "model_fallback": 16}


def make_league(tmp_path, *, tactical=True):
    cfg = settings(historical_tactical_champion=tactical)
    policy = GamePolicyTransformer(cfg.model).eval()
    layout = PieceConditionedLayoutPointerDecoder(cfg.model).eval()
    context = DistributedContext(0, 1, 0, torch.device("cpu"))
    league = HistoricalOpponents(cfg, tmp_path, context)
    for step in (0, 100, 200, 500):
        league.update_progress(policy, layout, environment_plies=step, update=step // 10)
    pool = BaseGamePool("four_dark", pool_size=4, max_transitions=8, max_game_plies=8,
                        seed=43, layout_prefetch_games=2, centralized_critic=True)
    pool.historical = league
    return cfg, policy, layout, league, pool


def publish_champion(tmp_path, cfg, policy, layout, update):
    directory = tmp_path / "model_selection"
    path = directory / "snapshots" / f"champion_{update}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format_version=checkpoint_format_for_model(cfg.model), update=update,
                    mode=cfg.mode.value, dead_rules_enabled=cfg.dead_rules_enabled,
                    config=cfg.serializable(), policy=policy.state_dict(), layout=layout.state_dict()), path)
    atomic_json(directory / "state.json", dict(best_snapshot=f"snapshots/{path.name}",
                best_sha256=sha256_file(path), best_update=update))
    return path


def test_disabled_mode_does_not_read_champion_and_config_is_off(tmp_path):
    _, policy, _, league, pool = make_league(tmp_path, tactical=False)
    with patch("junqi.training.historical_opponents.committed_champion", side_effect=AssertionError("disabled")):
        league.begin_rollout(pool, FrozenPolicyActor(policy), 1)
    assert type(league.actor) is FrozenPolicyActor
    assert "champion_entry" not in league.state_dict()
    cfg = TrainingSettings.from_yaml(Path(__file__).parents[1] / "configs/expand_mappo_training.yaml",
                                     "four_dark", model_scale="main")
    assert cfg.historical_tactical_champion is False
    for value in (1, "false"):
        with pytest.raises(ValueError, match="boolean"):
            replace(cfg, historical_tactical_champion=value).validate()
    with pytest.raises(ValueError, match="no frozen teammate"):
        replace(cfg, historical_tactical_champion=True, historical_teammate_fraction=.2).validate()


def test_champion_refresh_keeps_learner_rng_and_snapshot_fixed_within_rollout(tmp_path):
    cfg, policy, layout, league, pool = make_league(tmp_path)
    publish_champion(tmp_path, cfg, policy, layout, 10)
    actor = FrozenPolicyActor(policy)
    rng = torch.get_rng_state()
    league.begin_rollout(pool, actor, 1)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert isinstance(league.actor, TacticalOpponentActor)
    assert league.champion_entry['update'] == 10
    before = {k: v.clone() for k, v in league.policy.state_dict().items()}
    with torch.no_grad():
        policy.action_encoder.projection.weight.add_(.05)
    publish_champion(tmp_path, cfg, policy, layout, 11)
    # Publication alone never changes the active rollout's parameters.
    for key, value in before.items():
        torch.testing.assert_close(league.policy.state_dict()[key], value, atol=0, rtol=0)
    league.begin_rollout(pool, actor, 2)
    assert league.champion_entry['update'] == 11 and league.load_count == 2
    torch.testing.assert_close(league.policy.action_encoder.projection.weight,
                               policy.action_encoder.projection.weight, atol=0, rtol=0)
    league.begin_rollout(pool, actor, 3)
    assert league.load_count == 2
    assigned = [league.assignment() for _ in range(100)]
    assert sum(o is not None for o, _, _ in assigned) == 20
    assert {s for o, _, s in assigned if o} == set(range(4))
    assert all(t is None for _, t, _ in assigned)
    assert {o for o, _, _ in assigned if o} == {LIVE_CHAMPION_ID}


def test_enabled_without_committed_champion_fails_instead_of_using_latest(tmp_path):
    _, policy, _, league, pool = make_league(tmp_path)
    with pytest.raises(RuntimeError, match="live champion"):
        league.begin_rollout(pool, FrozenPolicyActor(policy), 1)


def test_pristine_history_checkpoint_can_select_the_new_mode(tmp_path):
    cfg, policy, layout, ordinary, pool = make_league(tmp_path, tactical=False)
    saved = copy.deepcopy(ordinary.state_dict())
    enabled = replace(cfg, historical_tactical_champion=True)
    special = HistoricalOpponents(enabled, tmp_path, ordinary.context)
    special.load_state_dict(saved)
    publish_champion(tmp_path, cfg, policy, layout, 10)
    pool.historical = special
    special.begin_rollout(pool, FrozenPolicyActor(policy), 1)
    assert special.active_id == LIVE_CHAMPION_ID
    assert special.sampling_contract()['priority'] == ['capture_flag', 'bomb_known_high', 'capture_known_engineer']


def test_forced_move_is_a_frozen_opponent_transition_not_a_learner_sample(tmp_path):
    cfg, policy, layout, league, pool = make_league(tmp_path)
    publish_champion(tmp_path, cfg, policy, layout, 10)
    actor = FrozenPolicyActor(policy, max_batch_size=4)
    league.begin_rollout(pool, actor, 1)
    pool.fill(layout, 1)
    source, target = ArmPoint(3, 5, 3), ArmPoint(3, 5, 4)
    game, state = position({source: Piece(3, PieceType.COMPANY_COMMANDER),
                            target: Piece(0, PieceType.ENGINEER)}, known=[target], current_player=3)
    expected = pair(game, source, target)
    pool.slots[0] = replace(pool.slots[0], game=game,
        history=GameHistory.initialize(game, 'four_dark', max_transitions=8),
        critic_history=CentralizedHistory.initialize(game, 'four_dark', max_transitions=8),
        opponent_id=LIVE_CHAMPION_ID, learner_team=0, learner_seat=0,
        teammate_id=None, historical_scenario='historical_opponents')
    critic = GameCentralValueTransformer(cfg.model).eval()
    samples, _, _ = collect_mappo_samples(pool, actor,
        FrozenValueActor(critic, amp_dtype=None, max_batch_size=4, deferred=True), layout,
        count=1, behavior_version=1, discount=1., gae_lambda=.95, environment_workers=2,
        pipeline_groups=2, advantage_mode='player_decisions', reward_scope='player')
    assert len(samples) == 1 and samples[0].action == expected
    assert samples[0].learnable is False
    assert len(critic_training_samples(samples)) == 4
    assert league.tactical_rule_counts['capture_known_engineer'] == 1


def test_collect_resume_and_keep_rule_actions_out_of_actor_training(tmp_path):
    cfg, policy, layout, league, pool = make_league(tmp_path)
    publish_champion(tmp_path, cfg, policy, layout, 10)
    critic = GameCentralValueTransformer(cfg.model).eval()
    samples, _, _ = collect_mappo_samples(pool, FrozenPolicyActor(policy, max_batch_size=4),
        FrozenValueActor(critic, amp_dtype=None, max_batch_size=4, deferred=True), layout,
        count=80, behavior_version=1, discount=1., gae_lambda=.95, environment_workers=2,
        pipeline_groups=2, advantage_mode="player_decisions", reward_scope="player")
    frozen = [s for s in samples if not s.learnable]
    assert frozen and any(s.learnable for s in samples)
    for s in frozen:
        rule, candidates = forced_tactical_actions(s.state)
        assert not rule or s.action in candidates
    assert len(critic_training_samples(samples)) == 4 * len(samples)
    league.old_layouts(2)
    saved = copy.deepcopy(pool.state_dict())
    assert saved['historical_opponents']['champion_entry']['update'] == 10
    other = HistoricalOpponents(cfg, tmp_path, league.context)
    pool.historical = other
    pool.load_state_dict(saved)
    other.begin_rollout(pool, FrozenPolicyActor(policy), 2)
    assert other.champion_entry == league.champion_entry
    assert [s.position_indices for s in other.layout_queue] == [s.position_indices for s in league.layout_queue]
    assert other.tactical_rule_counts == league.tactical_rule_counts
    assert [other.assignment() for _ in range(20)] == [league.assignment() for _ in range(20)]
    assert other.quality_update_games == 0
    # A regular history scheduler must not silently reinterpret special seats.
    ordinary = HistoricalOpponents(replace(cfg, historical_tactical_champion=False), tmp_path, league.context)
    with pytest.raises(ValueError, match="sampling changed"):
        ordinary.load_state_dict(saved['historical_opponents'])
