"""Leak-free player-view encodings and replayable temporal histories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..game import CombatOutcome, JunqiGame, Observation, ObservedEvent
from ..pieces import PieceType
from .modes import TrainingMode, mode_spec, normalize_mode


MAX_BOARD_POINTS = 129
EXACT_PIECE_CODE_STRIDE = 32
BOARD_PAD_CODE = 138
BOARD_CODE_VOCAB_SIZE = 139
ACTION_POINT_PAD = 129
ACTION_PLAYER_PAD = 4
ACTION_COMBAT_PAD = 4
CASUALTY_SLOTS_PER_PLAYER = 25
MAX_CASUALTY_BITS = 75

OWN_PIECE_CODES: dict[PieceType, int] = {
    PieceType.FLAG: 30,
    PieceType.BOMB: 31,
    PieceType.ENGINEER: 32,
    PieceType.PLATOON_COMMANDER: 33,
    PieceType.COMPANY_COMMANDER: 34,
    PieceType.BATTALION_COMMANDER: 35,
    PieceType.REGIMENT_COMMANDER: 36,
    PieceType.BRIGADE_COMMANDER: 37,
    PieceType.DIVISION_COMMANDER: 38,
    PieceType.ARMY_COMMANDER: 39,
    PieceType.COMMANDER: 40,
    PieceType.MINE: 41,
}


def exact_piece_code(
    kind: PieceType,
    relative_owner: int,
    mode: TrainingMode | str,
) -> int:
    """Encode a legitimately known identity in a collision-free seat block.

    Four-player relative owners use blocks 0/1/2/3.  The sole two-player
    opponent is physically the opposite player and therefore uses block 2,
    even though ``ObservedPiece.owner`` is 1 in the compact two-player order.
    """

    normalized = normalize_mode(mode)
    try:
        base = OWN_PIECE_CODES[PieceType(kind)]
    except (KeyError, ValueError) as error:
        raise ValueError(f"unknown exact piece type: {kind!r}") from error
    if isinstance(relative_owner, bool) or not isinstance(relative_owner, int):
        raise ValueError("relative piece owner must be an integer")
    if normalized is TrainingMode.TWO_PLAYER:
        if relative_owner not in (0, 1):
            raise ValueError("two-player relative owner must be 0 or 1")
        block = 0 if relative_owner == 0 else 2
    else:
        if relative_owner not in (0, 1, 2, 3):
            raise ValueError("four-player relative owner must be in 0..3")
        block = relative_owner
    return base + EXACT_PIECE_CODE_STRIDE * block

COMBAT_INDICES: dict[CombatOutcome, int] = {
    CombatOutcome.MOVE: 0,
    CombatOutcome.ATTACKER_WINS: 1,
    CombatOutcome.DEFENDER_WINS: 2,
    CombatOutcome.BOTH_REMOVED: 3,
}


def encode_visible_board(observation: Observation) -> tuple[int, ...]:
    """Encode exactly what ``observation.viewer`` may see, never referee state."""

    try:
        mode = (
            TrainingMode.TWO_PLAYER
            if observation.variant.value == "two_player"
            else TrainingMode(observation.information_mode.value)
        )
    except ValueError as error:
        raise ValueError(
            f"information mode {observation.information_mode.value!r} has no code map"
        ) from error

    expected = mode_spec(mode).point_count
    if len(observation.points) != expected:
        raise ValueError(
            f"{mode.value} observation requires {expected} points, "
            f"got {len(observation.points)}"
        )

    encoded: list[int] = []
    for piece in observation.points:
        if piece is None:
            encoded.append(0)
            continue

        if piece.kind is not None:
            encoded.append(exact_piece_code(piece.kind, piece.owner, mode))
            continue

        if piece.owner == 0:
            raise ValueError("the viewer's own piece identity cannot be hidden")

        if mode is TrainingMode.TWO_PLAYER:
            encoded.append(2)
            continue

        if piece.owner == 2 and mode is TrainingMode.DOUBLE_OPEN:
            raise ValueError("double-open ally identity unexpectedly hidden")

        if piece.owner not in (1, 2, 3):
            raise ValueError(f"invalid relative piece owner: {piece.owner}")
        encoded.append(piece.owner)

    return tuple(encoded)


def encode_known_casualties(observation: Observation) -> tuple[int, ...]:
    """Flatten exact casualty rows in deterministic relative-seat order."""

    if not observation.dead_rules_enabled:
        if observation.casualty_players or observation.known_casualties:
            raise ValueError("disabled dead rules must not expose casualty rows")
        return ()

    mode = (
        TrainingMode.TWO_PLAYER
        if observation.variant.value == "two_player"
        else TrainingMode(observation.information_mode.value)
    )
    expected_players = {
        TrainingMode.FOUR_DARK: (1, 2, 3),
        TrainingMode.DOUBLE_OPEN: (1, 3),
        TrainingMode.TWO_PLAYER: (1,),
    }[mode]
    if observation.casualty_players != expected_players:
        raise ValueError(
            f"{mode.value} casualty rows require relative players "
            f"{expected_players}, got {observation.casualty_players}"
        )
    if len(observation.known_casualties) != len(expected_players):
        raise ValueError("casualty row count does not match represented players")
    flattened: list[int] = []
    for row in observation.known_casualties:
        if len(row) != CASUALTY_SLOTS_PER_PLAYER:
            raise ValueError("each casualty row must contain exactly 25 bits")
        flattened.extend(int(value) for value in row)
    return tuple(flattened)


@dataclass(frozen=True, slots=True)
class ActionFeatures:
    source: int
    destination: int
    actor: int
    combat: int
    was_attack: int
    flag_captured_owner: int
    newly_revealed_count: int
    eliminated_count: int

    @classmethod
    def from_event(cls, event: ObservedEvent) -> ActionFeatures:
        source, destination = event.action
        return cls(
            source=source,
            destination=destination,
            actor=event.actor,
            combat=COMBAT_INDICES[event.combat],
            was_attack=int(event.was_attack),
            flag_captured_owner=(
                ACTION_PLAYER_PAD
                if event.flag_captured_owner is None
                else event.flag_captured_owner
            ),
            newly_revealed_count=min(len(event.newly_revealed_flags), 4),
            eliminated_count=min(len(event.eliminated_players), 4),
        )

    def as_tuple(self) -> tuple[int, ...]:
        return (
            self.source,
            self.destination,
            self.actor,
            self.combat,
            self.was_attack,
            self.flag_captured_owner,
            self.newly_revealed_count,
            self.eliminated_count,
        )


@dataclass(frozen=True, slots=True)
class StateTokenRecord:
    board_codes: tuple[int, ...]
    # None is a structural ablation: no casualty tensor is built or passed to
    # the policy when deterministic dead-rule features are disabled.
    known_casualty_bits: tuple[int, ...] | None
    action: ActionFeatures | None
    no_interaction_plies: int
    active_mask: int
    revealed_mask: int
    current_player: int


@dataclass(frozen=True, slots=True)
class _HistoryNode:
    """Persistent transition node shared by cloned rollout branches."""

    record: StateTokenRecord
    previous: _HistoryNode | None
    length: int


def _bit_mask(values: Iterable[bool]) -> int:
    result = 0
    for index, value in enumerate(values):
        if value:
            result |= 1 << index
    return result


def _record_from_observation(
    observation: Observation,
    action: ActionFeatures | None,
) -> StateTokenRecord:
    return StateTokenRecord(
        board_codes=encode_visible_board(observation),
        known_casualty_bits=(
            encode_known_casualties(observation)
            if observation.dead_rules_enabled
            else None
        ),
        action=action,
        no_interaction_plies=min(observation.no_interaction_plies, 60),
        active_mask=_bit_mask(observation.active_players),
        revealed_mask=_bit_mask(observation.revealed_flags),
        current_player=(
            ACTION_PLAYER_PAD
            if observation.current_player is None
            else observation.current_player
        ),
    )


@dataclass(frozen=True, slots=True)
class PolicyState:
    mode: TrainingMode
    records: tuple[StateTokenRecord, ...]
    legal_actions: tuple[tuple[int, int], ...]

    @property
    def current_board(self) -> tuple[int, ...]:
        return self.records[-1].board_codes


class PlayerHistory:
    """One player's pinned initial state plus a sliding transition history."""

    __slots__ = (
        "mode",
        "max_transitions",
        "dead_rules_enabled",
        "_initial",
        "_transition_tail",
        "_records_cache",
    )

    def __init__(
        self,
        mode: TrainingMode,
        initial: StateTokenRecord,
        *,
        max_transitions: int = 1000,
    ) -> None:
        if max_transitions <= 0:
            raise ValueError("max_transitions must be positive")
        if initial.action is not None:
            raise ValueError("initial state token cannot contain an action")
        self.mode = normalize_mode(mode)
        self.max_transitions = max_transitions
        self.dead_rules_enabled = initial.known_casualty_bits is not None
        self._initial = initial
        self._transition_tail: _HistoryNode | None = None
        self._records_cache: tuple[StateTokenRecord, ...] | None = (initial,)

    @classmethod
    def from_initial_observation(
        cls,
        mode: TrainingMode | str,
        observation: Observation,
        *,
        max_transitions: int = 1000,
    ) -> PlayerHistory:
        return cls(
            normalize_mode(mode),
            _record_from_observation(observation, None),
            max_transitions=max_transitions,
        )

    @property
    def records(self) -> tuple[StateTokenRecord, ...]:
        if self._records_cache is not None:
            return self._records_cache
        recent: list[StateTokenRecord] = []
        node = self._transition_tail
        while node is not None and len(recent) < self.max_transitions:
            recent.append(node.record)
            node = node.previous
        recent.reverse()
        self._records_cache = (self._initial, *recent)
        return self._records_cache

    def append_post_action(self, observation: Observation) -> None:
        if not observation.history:
            raise ValueError("post-action observation has no public event")
        if observation.dead_rules_enabled != self.dead_rules_enabled:
            raise ValueError("dead-rule feature mode changed inside one history")
        action = ActionFeatures.from_event(observation.history[-1])
        record = _record_from_observation(observation, action)
        previous = self._transition_tail
        self._transition_tail = _HistoryNode(
            record=record,
            previous=previous,
            length=1 if previous is None else previous.length + 1,
        )
        self._records_cache = None

    def as_policy_state(
        self, legal_actions: Iterable[tuple[int, int]]
    ) -> PolicyState:
        actions = tuple(legal_actions)
        if not actions:
            raise ValueError("a non-terminal policy state must have legal actions")
        return PolicyState(self.mode, self.records, actions)

    def clone(self) -> PlayerHistory:
        copied = self.__class__(
            self.mode,
            self._initial,
            max_transitions=self.max_transitions,
        )
        # Immutable linked nodes provide O(1) structural sharing.  The first
        # append on a child creates only one new node (true copy-on-write),
        # rather than copying up to 1000 transition records eight times.
        copied._transition_tail = self._transition_tail
        copied._records_cache = self._records_cache
        return copied

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "max_transitions": self.max_transitions,
            "dead_rules_enabled": self.dead_rules_enabled,
            "records": [_record_state_dict(record) for record in self.records],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> PlayerHistory:
        records = tuple(
            _record_from_state_dict(record) for record in state["records"]
        )
        if not records:
            raise ValueError("serialized player history has no initial record")
        result = cls(
            normalize_mode(state["mode"]),
            records[0],
            max_transitions=int(state["max_transitions"]),
        )
        serialized_dead_rules = state["dead_rules_enabled"]
        if not isinstance(serialized_dead_rules, bool):
            raise ValueError("serialized dead-rule history flag is not boolean")
        if serialized_dead_rules != result.dead_rules_enabled:
            raise ValueError("serialized dead-rule history flag contradicts records")
        if len(records) - 1 > result.max_transitions:
            raise ValueError("serialized player history exceeds its context window")
        for record in records[1:]:
            previous = result._transition_tail
            result._transition_tail = _HistoryNode(
                record=record,
                previous=previous,
                length=1 if previous is None else previous.length + 1,
            )
        result._records_cache = records
        return result


class GameHistory:
    """All player-view histories associated with one mutable referee game."""

    __slots__ = ("mode", "players")

    def __init__(self, mode: TrainingMode, players: tuple[PlayerHistory, ...]) -> None:
        self.mode = normalize_mode(mode)
        if not players:
            raise ValueError("game history requires at least one player")
        if any(player.mode is not self.mode for player in players):
            raise ValueError("game history contains a different training mode")
        dead_rules_enabled = players[0].dead_rules_enabled
        if any(
            player.dead_rules_enabled != dead_rules_enabled for player in players
        ):
            raise ValueError("game history cannot mix dead-rule variants")
        self.players = players

    @classmethod
    def initialize(
        cls,
        game: JunqiGame,
        mode: TrainingMode | str,
        *,
        max_transitions: int = 1000,
    ) -> GameHistory:
        normalized = normalize_mode(mode)
        players = tuple(
            PlayerHistory.from_initial_observation(
                normalized,
                game.observe(
                    player,
                    history_limit=0,
                    include_legal_masks=False,
                    include_candidate_masks=False,
                ),
                max_transitions=max_transitions,
            )
            for player in range(game.config.player_count)
        )
        return cls(normalized, players)

    def append_after_step(self, game: JunqiGame) -> None:
        for player, history in enumerate(self.players):
            history.append_post_action(
                game.observe(
                    player,
                    history_limit=1,
                    include_legal_masks=False,
                    include_candidate_masks=False,
                )
            )

    def state_for(self, game: JunqiGame, player: int | None = None) -> PolicyState:
        actor = game.current_player if player is None else player
        if actor is None:
            raise ValueError("terminal game has no current policy actor")
        return self.players[actor].as_policy_state(game.legal_actions(actor))

    def clone(self) -> GameHistory:
        return self.__class__(self.mode, tuple(item.clone() for item in self.players))

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "dead_rules_enabled": self.players[0].dead_rules_enabled,
            "players": [history.state_dict() for history in self.players],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> GameHistory:
        mode = normalize_mode(state["mode"])
        players = tuple(
            PlayerHistory.from_state_dict(item) for item in state["players"]
        )
        expected_players = mode_spec(mode).player_count
        if len(players) != expected_players:
            raise ValueError(
                f"serialized {mode.value} history requires {expected_players} players"
            )
        if any(player.mode is not mode for player in players):
            raise ValueError("serialized player-history mode mismatch")
        dead_rules_enabled = state["dead_rules_enabled"]
        if not isinstance(dead_rules_enabled, bool):
            raise ValueError("serialized game-history dead-rule flag is not boolean")
        if any(
            player.dead_rules_enabled != dead_rules_enabled for player in players
        ):
            raise ValueError("serialized player histories mix dead-rule modes")
        return cls(mode, players)


def _record_state_dict(record: StateTokenRecord) -> dict[str, Any]:
    return {
        "board_codes": record.board_codes,
        "known_casualty_bits": record.known_casualty_bits,
        "action": None if record.action is None else record.action.as_tuple(),
        "no_interaction_plies": record.no_interaction_plies,
        "active_mask": record.active_mask,
        "revealed_mask": record.revealed_mask,
        "current_player": record.current_player,
    }


def _record_from_state_dict(state: Mapping[str, Any]) -> StateTokenRecord:
    raw_action = state["action"]
    action = None
    if raw_action is not None:
        if len(raw_action) != 8:
            raise ValueError("serialized action requires exactly eight fields")
        action = ActionFeatures(*(int(value) for value in raw_action))
    raw_casualties = state["known_casualty_bits"]
    return StateTokenRecord(
        board_codes=tuple(int(value) for value in state["board_codes"]),
        known_casualty_bits=(
            None
            if raw_casualties is None
            else tuple(int(value) for value in raw_casualties)
        ),
        action=action,
        no_interaction_plies=int(state["no_interaction_plies"]),
        active_mask=int(state["active_mask"]),
        revealed_mask=int(state["revealed_mask"]),
        current_player=int(state["current_player"]),
    )
