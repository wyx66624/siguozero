"""Public tactical responses and one adaptively rated current-model pool entry."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from junqi import ArmPoint, Piece, PieceType
from junqi.training.current_tactical_opponent import LIVE_CURRENT_ID, current_tactical_actions
from junqi.training.distributed import DistributedContext
from junqi.training.encoding import GameHistory
from junqi.training.historical_opponents import HistoricalOpponents
from junqi.training.mappo import collect_mappo_samples, critic_training_samples
from junqi.training.mappo_encoding import CentralizedHistory
from junqi.training.models import GameCentralValueTransformer, GamePolicyTransformer, PieceConditionedLayoutPointerDecoder
from junqi.training.ppo import FrozenValueActor
from junqi.training.rollout import BaseGamePool, FrozenPolicyActor
from junqi.training.settings import TrainingSettings
from junqi.training.tactical_opponent import TacticalOpponentActor
from test_historical_opponents import settings
from test_tactical_opponent import position, pair


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.mark.parametrize("seat", range(4))
def test_immediate_flag_defense_uses_largest_piece_before_bomb_or_engineer(seat):
    enemy = (seat + 1) % 4
    threat = ArmPoint(seat, 5, 4)
    largest, small = ArmPoint(seat, 5, 5), ArmPoint(seat, 4, 4)
    bomb, high = ArmPoint(enemy, 4, 1), ArmPoint(enemy, 5, 1)
    game, state = position({threat: Piece(enemy, PieceType.ENGINEER),
        largest: Piece(seat, PieceType.ARMY_COMMANDER), small: Piece(seat, PieceType.COMPANY_COMMANDER),
        bomb: Piece(seat, PieceType.BOMB), high: Piece(enemy, PieceType.COMMANDER)},
        known=[threat, high], current_player=seat)
    rule, actions = current_tactical_actions(state)
    assert rule == "defend_flag"
    assert actions == (pair(game, largest, threat),)


def test_defense_bomb_fallback_when_ordinary_pieces_cannot_remove_commander():
    threat, bomb, small = ArmPoint(0, 5, 4), ArmPoint(0, 5, 5), ArmPoint(0, 4, 4)
    game, state = position({threat: Piece(1, PieceType.COMMANDER),
                           bomb: Piece(0, PieceType.BOMB), small: Piece(0, PieceType.COMPANY_COMMANDER)}, known=[threat])
    assert current_tactical_actions(state) == ("defend_flag", (pair(game, bomb, threat),))


@pytest.mark.parametrize("kind,forced", [(PieceType.DIVISION_COMMANDER, True),
    (PieceType.ARMY_COMMANDER, True), (PieceType.COMMANDER, True), (PieceType.BRIGADE_COMMANDER, False)])
def test_bomb_known_big_piece(kind, forced):
    a, b = ArmPoint(1, 4, 1), ArmPoint(1, 5, 1)
    game, state = position({a: Piece(0, PieceType.BOMB), b: Piece(1, kind)}, known=[b])
    rule, actions = current_tactical_actions(state)
    assert (pair(game, a, b) in actions) == forced
    assert rule == ("bomb_known_high" if forced else None)


def test_bomb_before_engineer_and_fallback_keeps_rng_and_probabilities():
    a, b, c, d = ArmPoint(1, 4, 1), ArmPoint(1, 5, 1), ArmPoint(3, 4, 1), ArmPoint(3, 5, 1)
    game, state = position({a: Piece(0, PieceType.BOMB), b: Piece(1, PieceType.COMMANDER),
                            c: Piece(0, PieceType.COMPANY_COMMANDER), d: Piece(3, PieceType.ENGINEER)}, known=[b, d])
    policy = GamePolicyTransformer(settings().model).eval()
    tactical = TacticalOpponentActor(policy, rules=current_tactical_actions)
    selected, logs = tactical.sample([state], count=8)
    assert selected == [[pair(game, a, b)] * 8]
    torch.testing.assert_close(logs[0], torch.zeros_like(logs[0]))
    _, ordinary = position({})
    rng = torch.get_rng_state()
    expected = FrozenPolicyActor(policy).sample([ordinary], count=8)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    actual = tactical.sample([ordinary], count=8)
    assert actual[0] == expected[0]
    torch.testing.assert_close(actual[1][0], expected[1][0], atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)


def test_known_engineer_is_removed_even_by_equal_trade():
    a, b = ArmPoint(1, 4, 1), ArmPoint(1, 5, 1)
    game, state = position({a: Piece(0, PieceType.ENGINEER), b: Piece(1, PieceType.ENGINEER)}, known=[b])
    assert current_tactical_actions(state) == ("capture_known_engineer", (pair(game, a, b),))


@pytest.mark.parametrize("hidden", [PieceType.COMMANDER, PieceType.ENGINEER])
def test_referee_kind_not_used_and_centralized_input_rejected(hidden):
    a, b = ArmPoint(1, 4, 1), ArmPoint(1, 5, 1)
    _, state = position({a: Piece(0, PieceType.BOMB), b: Piece(1, hidden)})
    assert current_tactical_actions(state) == (None, ())
    with pytest.raises(ValueError, match="centralized"):
        current_tactical_actions(replace(state, centralized=True))


def test_camp_and_allied_piece_do_not_trigger():
    a, camp = ArmPoint(1, 3, 1), ArmPoint(1, 2, 2)
    _, state = position({a: Piece(0, PieceType.BOMB), camp: Piece(1, PieceType.COMMANDER)}, known=[camp])
    assert current_tactical_actions(state) == (None, ())
    for owner in (0, 2):
        _, state = position({a: Piece(0, PieceType.BOMB), ArmPoint(1, 4, 1): Piece(owner, PieceType.COMMANDER)},
                            known=[ArmPoint(1, 4, 1)])
        assert current_tactical_actions(state) == (None, ())


def test_free_capture_rejects_public_recapture_and_self_defeating_trade():
    a, b, recapture = ArmPoint(1, 4, 1), ArmPoint(1, 5, 1), ArmPoint(1, 5, 2)
    pieces = {a: Piece(0, PieceType.ARMY_COMMANDER), b: Piece(1, PieceType.PLATOON_COMMANDER)}
    game, state = position(pieces, known=[b])
    assert current_tactical_actions(state) == ("free_capture", (pair(game, a, b),))
    _, state = position({**pieces, recapture: Piece(1, PieceType.COMMANDER)}, known=[b, recapture])
    assert current_tactical_actions(state) == (None, ())
    _, state = position({a: Piece(0, PieceType.PLATOON_COMMANDER), b: Piece(1, PieceType.PLATOON_COMMANDER)}, known=[b])
    assert current_tactical_actions(state) == (None, ())


def make_current_league(tmp_path, **overrides):
    cfg = settings(historical_tactical_current=True, historical_sampling="strength",
                   historical_initial_loss_rate=.5, historical_stage_mix_fraction=0.,
                   historical_uniform_fraction=0., **overrides)
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


def select_current(league):
    p = league.probabilities()
    league.pending_cohort = dict(id=LIVE_CURRENT_ID, probability=p[LIVE_CURRENT_ID], panel_size=len(p))


def test_one_identity_equal_prior_adaptive_losses_draws_and_snapshot_filter(tmp_path):
    _, policy, layout, league, pool = make_current_league(tmp_path, historical_checkpoint_start_fraction=.1)
    p = league.probabilities()
    assert len(p) == 3 and len(set(p.values())) == 1  # 100, 200, one current; initial snapshot excluded
    assert len(league.evaluation_panel()) == 3 and all(e['id'] != LIVE_CURRENT_ID for e in league.entries)
    select_current(league)
    league.begin_rollout(pool, FrozenPolicyActor(policy), 1)
    slot = SimpleNamespace(opponent_id=LIVE_CURRENT_ID, teammate_id=None)
    league.record_result(slot, -1.)
    rate = league.loss_rates[LIVE_CURRENT_ID]
    assert rate == pytest.approx(.505)
    assert league.probabilities()[LIVE_CURRENT_ID] > p[LIVE_CURRENT_ID]
    league.record_result(slot, 0.)
    assert league.loss_rates[LIVE_CURRENT_ID] == rate
    league.record_result(slot, 1.)
    assert league.loss_rates[LIVE_CURRENT_ID] < rate
    league.update_progress(policy, layout, environment_plies=600, update=60)
    assert set(league.probabilities()) == set(p)
    assert league.results[LIVE_CURRENT_ID] == dict(wins=1, losses=1, draws=1)
    league.write_status(pool)
    status = json.loads((league.directory / 'status.json').read_text())
    assert sum(e['id'] == LIVE_CURRENT_ID for e in status['opponents']) == 1
    assert status['sampling']['strategy'] == 'strength'
    assert status['historical/tactical_current_enabled'] == 1


def test_current_copy_is_frozen_refreshes_at_boundary_and_preserves_rng(tmp_path):
    _, policy, layout, league, pool = make_current_league(tmp_path)
    select_current(league)
    rng = torch.get_rng_state()
    league.begin_rollout(pool, FrozenPolicyActor(policy), 1)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert isinstance(league.actor, TacticalOpponentActor)
    before = league.policy.action_encoder.projection.weight.clone()
    assert before.data_ptr() != policy.action_encoder.projection.weight.data_ptr()
    with torch.no_grad():
        policy.action_encoder.projection.weight.add_(.1)
        next(layout.parameters()).add_(.1)
    torch.testing.assert_close(league.policy.action_encoder.projection.weight, before, atol=0, rtol=0)
    league.update_progress(policy, layout, environment_plies=600, update=60)
    assert league.current_entry['update'] == 50
    league.begin_rollout(pool, FrozenPolicyActor(policy), 2)
    assert league.current_entry['update'] == 60 and league.load_count == 2
    for source, replica in ((policy, league.policy), (layout, league.layout)):
        for a, b in zip(source.parameters(), replica.parameters(), strict=True):
            assert a.data_ptr() != b.data_ptr() and not b.requires_grad
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    league.begin_rollout(pool, FrozenPolicyActor(policy), 2)
    assert league.load_count == 2


def test_prefetch_current_skips_disk_and_switches_to_real_frozen_weights(tmp_path):
    _, policy, layout, league, pool = make_current_league(tmp_path, historical_checkpoint_start_fraction=.1)
    select_current(league)
    league.weight_cache = SimpleNamespace(trim=lambda: None, prefetch=lambda _: pytest.fail('virtual read'))
    league.update_progress(policy, layout, environment_plies=500, update=50)
    league.weight_cache = None
    league.begin_rollout(pool, FrozenPolicyActor(policy), 1)
    old_id = league.panel()[1]['id']
    league.cohort_started = league.cohort_limit
    league.pending_cohort = dict(id=old_id, probability=.25, panel_size=4)
    league.begin_rollout(pool, FrozenPolicyActor(policy), 2)
    assert type(league.actor) is FrozenPolicyActor and league.loaded_id == old_id


def test_current_rule_moves_remain_frozen_in_real_mappo_collection_and_resume(tmp_path):
    cfg, policy, layout, league, pool = make_current_league(tmp_path)
    select_current(league)
    actor = FrozenPolicyActor(policy, max_batch_size=4)
    league.begin_rollout(pool, actor, 1)
    pool.fill(layout, 1)
    a, b = ArmPoint(3, 4, 1), ArmPoint(3, 5, 1)
    game, _ = position({a: Piece(3, PieceType.BOMB), b: Piece(0, PieceType.COMMANDER)}, known=[b], current_player=3)
    pool.slots[0] = replace(pool.slots[0], game=game,
        history=GameHistory.initialize(game, 'four_dark', max_transitions=8),
        critic_history=CentralizedHistory.initialize(game, 'four_dark', max_transitions=8),
        opponent_id=LIVE_CURRENT_ID, learner_team=0, learner_seat=0,
        teammate_id=None, historical_scenario='historical_opponents')
    critic = GameCentralValueTransformer(cfg.model).eval()
    samples, _, _ = collect_mappo_samples(pool, actor,
        FrozenValueActor(critic, amp_dtype=None, max_batch_size=4, deferred=True), layout,
        count=1, behavior_version=1, discount=1., gae_lambda=.95, environment_workers=2,
        pipeline_groups=2, advantage_mode='player_decisions', reward_scope='player')
    assert len(samples) == 1 and samples[0].action == pair(game, a, b)
    assert samples[0].learnable is False and len(critic_training_samples(samples)) == 4
    assert league.tactical_rule_counts['bomb_known_high'] == 1
    league.record_result(SimpleNamespace(opponent_id=LIVE_CURRENT_ID, teammate_id=None), -1.)
    saved = copy.deepcopy(pool.state_dict())
    other = HistoricalOpponents(cfg, tmp_path, league.context)
    pool.historical = other
    pool.load_state_dict(saved)
    other.update_progress(policy, layout, environment_plies=500, update=50)
    other.begin_rollout(pool, actor, 2)
    assert other.loss_rates == league.loss_rates and other.probabilities() == league.probabilities()
    assert other.rng.getstate() == league.rng.getstate() and other.current_entry == league.current_entry
    assert other.tactical_rule_counts == league.tactical_rule_counts
    assert [other.assignment() for _ in range(20)] == [league.assignment() for _ in range(20)]
    ordinary = HistoricalOpponents(replace(cfg, historical_tactical_current=False), tmp_path, league.context)
    with pytest.raises(ValueError, match='sampling changed'):
        ordinary.load_state_dict(saved['historical_opponents'])


def test_config_and_pristine_adoption_preserve_schedule(tmp_path):
    cfg, _, _, league, pool = make_current_league(tmp_path)
    saved = league.state_dict()
    saved['sampling_contract'].pop('current_tactical')
    saved.pop('current_entry')
    league.load_state_dict(saved)
    league.progress = league.threshold - 1
    assert league.assignment() == (None, None, 0)
    cfg = TrainingSettings.from_yaml(Path(__file__).parents[1] / 'configs/expand_mappo_training.yaml',
                                    'four_dark', model_scale='main')
    assert cfg.historical_tactical_current and not cfg.historical_tactical_champion
    assert cfg.historical_training_fraction == .2 and cfg.historical_sampling == 'strength'
    assert cfg.historical_start_fraction * cfg.target_environment_plies == 1_000_000_000
    for changes in (dict(historical_tactical_current='true'), dict(historical_tactical_champion=True),
                    dict(historical_teammate_fraction=.1), dict(historical_stage_mix_fraction=.1)):
        with pytest.raises(ValueError):
            replace(cfg, **changes).validate()
