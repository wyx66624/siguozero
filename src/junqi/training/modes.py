"""The three supported training and inference profiles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from ..game import GameVariant, InformationMode, JunqiGame
from ..pieces import PlayerSetup


class TrainingMode(str, Enum):
    FOUR_DARK = "four_dark"
    DOUBLE_OPEN = "double_open"
    TWO_PLAYER = "two_player"


@dataclass(frozen=True, slots=True)
class ModeSpec:
    mode: TrainingMode
    mode_index: int
    variant: GameVariant
    information_mode: InformationMode
    player_count: int
    point_count: int


MODE_SPECS: dict[TrainingMode, ModeSpec] = {
    TrainingMode.FOUR_DARK: ModeSpec(
        mode=TrainingMode.FOUR_DARK,
        mode_index=0,
        variant=GameVariant.FOUR_PLAYER,
        information_mode=InformationMode.FOUR_DARK,
        player_count=4,
        point_count=129,
    ),
    TrainingMode.DOUBLE_OPEN: ModeSpec(
        mode=TrainingMode.DOUBLE_OPEN,
        mode_index=1,
        variant=GameVariant.FOUR_PLAYER,
        information_mode=InformationMode.DOUBLE_OPEN,
        player_count=4,
        point_count=129,
    ),
    TrainingMode.TWO_PLAYER: ModeSpec(
        mode=TrainingMode.TWO_PLAYER,
        mode_index=2,
        variant=GameVariant.TWO_PLAYER,
        information_mode=InformationMode.DARK,
        player_count=2,
        point_count=60,
    ),
}


def normalize_mode(mode: TrainingMode | str) -> TrainingMode:
    try:
        return TrainingMode(mode)
    except ValueError as error:
        choices = ", ".join(item.value for item in TrainingMode)
        raise ValueError(f"unsupported training mode {mode!r}; choose {choices}") from error


def mode_spec(mode: TrainingMode | str) -> ModeSpec:
    return MODE_SPECS[normalize_mode(mode)]


def new_game(
    mode: TrainingMode | str,
    *,
    setups: Sequence[PlayerSetup] | None = None,
    seed: int | None = None,
    max_plies: int | None = None,
    dead_rules_enabled: bool = True,
) -> JunqiGame:
    """Create a rules-engine game whose information mode exactly matches profile."""

    spec = mode_spec(mode)
    if spec.variant is GameVariant.FOUR_PLAYER:
        return JunqiGame.new_four_player(
            setups=setups,
            seed=seed,
            information_mode=spec.information_mode,
            dead_rules_enabled=dead_rules_enabled,
            no_interaction_draw_plies=60,
            max_plies=max_plies,
        )
    return JunqiGame.new_two_player(
        setups=setups,
        seed=seed,
        information_mode=spec.information_mode,
        dead_rules_enabled=dead_rules_enabled,
        no_interaction_draw_plies=60,
        max_plies=max_plies,
    )
