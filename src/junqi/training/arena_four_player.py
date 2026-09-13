"""Four-country team protocol, separate from two-player win/loss accounting.

Opposite seats 0/2 and 1/3 are allies. The default compares complete teams.
Balanced teammate evaluation alternates current and frozen allies within each
four-rotation sampling unit, while scoring the focal player's whole team.
"""

from __future__ import annotations

from typing import Any

from .arena import MatchSettings, paired_statistics, play_game, seed_inference, summarize_groups, validate_engines
from .inference import InferenceEngine
from .modes import TrainingMode


EVALUATION_TYPE = "historical_team_rotations_four_player"
LAYOUT_PROTOCOL = "four_fixed_own_layouts_rotate_through_all_seats"
FOUR_PLAYER_MODES = (TrainingMode.FOUR_DARK.value, TrainingMode.DOUBLE_OPEN.value)


def historical_teammate(group_index, rotation):
    # Each condition occupies both teams in a group; across groups rotate which
    # two focal seats get the frozen ally to avoid fixing it to the starting seat.
    return (group_index + rotation) % 4 >= 2


def play_group(candidate: InferenceEngine, opponent: InferenceEngine,
               group_index: int, settings: MatchSettings) -> list[dict[str, Any]]:
    records = []
    for spec in prepare_group(candidate, opponent, group_index, settings):
        record = play_game(candidate, opponent, settings, setups=spec["setups"],
                           candidate_seats=spec["candidate_seats"], action_seed=spec["action_seed"])
        record.update(spec["metadata"])
        records.append(record)
    return records


def prepare_group(candidate: InferenceEngine, opponent: InferenceEngine,
                  group_index: int, settings: MatchSettings) -> list[dict[str, Any]]:
    """Fix the four layout rotations independently of game scheduling."""
    if settings.mode not in FOUR_PLAYER_MODES:
        raise ValueError("four-player protocol requires four_dark or double_open settings")
    validate_engines(candidate, opponent, mode=settings.mode)
    if not 0 <= group_index < settings.pairs:
        raise ValueError("rotation group index is out of range")
    group_seed = settings.seed + settings.seed_stride * group_index
    layouts = []
    for index, engine in enumerate((candidate, opponent)):
        seed_inference(group_seed + index, engine)
        layouts.append([sample.setup for sample in engine.sample_layouts(
            3 if index == 1 and settings.historical_teammate_fraction else 2,
            temperature=settings.layout_temperature
        )])
    # Each checkpoint still has exactly one Policy + one Layout instance.
    original = [layouts[0][0], layouts[1][0], layouts[0][1], layouts[1][1]]
    specs = []
    for rotation in range(4):
        team = rotation % 2
        frozen_ally = bool(settings.historical_teammate_fraction and historical_teammate(group_index, rotation))
        group_layouts = list(original)
        if frozen_ally:
            group_layouts[2] = layouts[1][2]
        mapping = [(seat - rotation) % 4 for seat in range(4)]
        metadata = {"group_index": group_index, "group_seed": group_seed,
                    "rotation": rotation, "layout_origin_seats": mapping}
        if settings.historical_teammate_fraction:
            metadata.update(focal_seat=rotation, teammate_seat=(rotation + 2) % 4,
                            teammate_version="historical" if frozen_ally else "current")
        specs.append({"candidate_seats": (rotation,) if frozen_ally else (team, team + 2),
                      "setups": [group_layouts[index] for index in mapping],
                      "action_seed": group_seed + 2 + rotation,
                      "metadata": metadata})
    return specs


def summarize_games(records: list[dict[str, Any]], settings: MatchSettings,
                    *, alpha: float) -> dict[str, Any]:
    if settings.mode not in FOUR_PLAYER_MODES:
        raise ValueError("four-player statistics require four_dark or double_open settings")
    for row in records:
        team = row["rotation"] % 2
        if row["mode"] != settings.mode or row["candidate_team"] != team:
            raise ValueError("mode/team does not match the four-seat rotation")
        frozen_ally = bool(settings.historical_teammate_fraction and historical_teammate(row["group_index"], row["rotation"]))
        expected = [row["rotation"]] if frozen_ally else [team, team + 2]
        if row["candidate_seats"] != expected:
            raise ValueError("candidate seats must be opposite allies, not adjacent enemies")
        if settings.historical_teammate_fraction and (
                row.get("focal_seat") != row["rotation"]
                or row.get("teammate_seat") != (row["rotation"] + 2) % 4
                or row.get("teammate_version") != ("historical" if frozen_ally else "current")):
            raise ValueError("teammate assignment does not match the balanced rotation protocol")
        winner = row["winner_team"]
        if winner not in (None, 0, 1):
            raise ValueError("invalid winning team")
        reward = 0 if winner is None else 1 if winner == team else -1
        if row["candidate_reward"] != reward or row["player_rewards"] != [
            reward if seat % 2 == team else -reward for seat in range(4)
        ]:
            raise ValueError("reward must describe the whole team, including eliminated allies")
    score_alpha = alpha / 3 if settings.historical_teammate_fraction else alpha
    stats = summarize_groups(records, settings, alpha=score_alpha,
                             group_key="group_index", leg_key="rotation")
    stats["rotation_groups"] = stats.pop("pairs")
    stats.update({
        "statistical_unit": "four_game_rotation_group", "result_unit": "team_game",
        "ci_method": "rotation_group_hoeffding",
        "team_scores": {
            str(team): sum(r["candidate_score"] for r in records if r["candidate_team"] == team)
            / (2 * settings.pairs) for team in (0, 1)
        },
        "rotation_scores": {
            str(rotation): sum(r["candidate_score"] for r in records if r["rotation"] == rotation)
            / settings.pairs for rotation in range(4)
        },
    })
    if settings.historical_teammate_fraction:
        stats["statistical_unit"] = "balanced_teammate_four_game_group"
        stats["teammate_protocol"] = "paired_current_historical_v1"
        stats["teammate_results"] = {}
        for version in ("current", "historical"):
            rows = [row for row in records if row["teammate_version"] == version]
            scores = [sum(row["candidate_score"] for row in rows if row["group_index"] == group) / 2
                      for group in range(settings.pairs)]
            result = paired_statistics(scores, alpha=score_alpha, bootstrap_seed=settings.seed)
            counts = {name: sum(row["candidate_reward"] == reward for row in rows)
                      for name, reward in (("wins", 1), ("draws", 0), ("losses", -1))}
            result.update(counts, games=len(rows), score=sum(scores) / len(scores),
                          statistical_unit="two_games_within_balanced_four_game_group")
            if settings.smoke_test:
                result["verdict"] = "smoke_test_not_strength_evidence"
            stats["teammate_results"][version] = result
        stats["score_alpha"] = score_alpha
    return stats
