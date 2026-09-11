"""Two-player strength protocol: two games, fixed own layouts, exchanged seats."""

from __future__ import annotations

from typing import Any

from .arena import MatchSettings, play_game, seed_inference, summarize_groups, validate_engines
from .inference import InferenceEngine
from .modes import TrainingMode


EVALUATION_TYPE = "historical_paired_two_player"
LAYOUT_PROTOCOL = "own_layout_follows_model_when_swapping_seats"


def play_group(candidate: InferenceEngine, opponent: InferenceEngine,
               pair_index: int, settings: MatchSettings) -> list[dict[str, Any]]:
    records = []
    for spec in prepare_group(candidate, opponent, pair_index, settings):
        record = play_game(candidate, opponent, settings, setups=spec["setups"],
                           candidate_seats=spec["candidate_seats"], action_seed=spec["action_seed"])
        record.update(spec["metadata"])
        records.append(record)
    return records


def prepare_group(candidate: InferenceEngine, opponent: InferenceEngine,
                  pair_index: int, settings: MatchSettings) -> list[dict[str, Any]]:
    """Fix own layouts once; games can subsequently run in any batch order."""
    if settings.mode != TrainingMode.TWO_PLAYER.value:
        raise ValueError("two-player protocol requires two_player settings")
    validate_engines(candidate, opponent, mode=TrainingMode.TWO_PLAYER)
    if not 0 <= pair_index < settings.pairs:
        raise ValueError("pair index is out of range")
    pair_seed = settings.seed + settings.seed_stride * pair_index
    setups = []
    for index, engine in enumerate((candidate, opponent)):
        seed_inference(pair_seed + index, engine)
        setups.append(engine.sample_layouts(1, temperature=settings.layout_temperature)[0].setup)
    specs = []
    for candidate_seat in (0, 1):
        specs.append({"candidate_seats": (candidate_seat,),
                      "setups": setups if candidate_seat == 0 else setups[::-1],
                      "action_seed": pair_seed + 2 + candidate_seat,
                      "metadata": {"pair_index": pair_index, "pair_seed": pair_seed,
                                   "candidate_seat": candidate_seat}})
    return specs


def summarize_games(records: list[dict[str, Any]], settings: MatchSettings,
                    *, alpha: float) -> dict[str, Any]:
    if settings.mode != TrainingMode.TWO_PLAYER.value:
        raise ValueError("two-player statistics require two_player settings")
    stats = summarize_groups(records, settings, alpha=alpha,
                             group_key="pair_index", leg_key="candidate_seat")
    stats.update({
        "statistical_unit": "two_game_seat_pair", "result_unit": "individual_game",
        "seat_scores": {
            str(seat): sum(r["candidate_score"] for r in records if r["candidate_seat"] == seat)
            / settings.pairs for seat in (0, 1)
        },
    })
    return stats
