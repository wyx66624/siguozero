"""Deterministic Junqi rules engine suitable for self-play training."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import random
from types import MappingProxyType
from typing import TypeAlias

from .board import (
    Action,
    ArmPoint,
    BoardEncodingError,
    CenterPoint,
    FourPlayerBoard,
    FourPlayerSeat,
    HEADQUARTERS_COORDINATES,
    PathKind,
    PhysicalPoint,
    PointKind,
    RailwayDirection,
    TwoPlayerBoard,
    TwoPlayerSeat,
)
from .pieces import (
    DEPLOYMENT_PIECE_SEQUENCE,
    MOVABLE_PIECE_TYPES,
    PIECE_COUNTS,
    PIECE_RANKS,
    PIECE_TYPE_ORDER,
    SETUP_COORDINATES,
    Piece,
    PieceType,
    PlayerSetup,
    allowed_piece_types,
)


Board: TypeAlias = FourPlayerBoard | TwoPlayerBoard
SetupInput: TypeAlias = PlayerSetup | Mapping[tuple[int, int], PieceType | str]


class GameRuleError(ValueError):
    """Raised when a game configuration or restored position is invalid."""


class IllegalActionError(GameRuleError):
    """Raised when an action is not legal in the current position."""


class TerminalGameError(GameRuleError):
    """Raised when attempting to move after the game has ended."""


class GameVariant(str, Enum):
    FOUR_PLAYER = "four_player"
    TWO_PLAYER = "two_player"


class InformationMode(str, Enum):
    FOUR_DARK = "four_dark"
    DOUBLE_OPEN = "double_open"
    FULL_OPEN = "full_open"
    DARK = "dark"
    OPEN = "open"


class CombatOutcome(str, Enum):
    MOVE = "move"
    ATTACKER_WINS = "attacker_wins"
    DEFENDER_WINS = "defender_wins"
    BOTH_REMOVED = "both_removed"


class TerminationReason(str, Enum):
    TEAM_ELIMINATED = "team_eliminated"
    NO_INTERACTION_DRAW = "no_interaction_draw"
    MAX_PLIES_DRAW = "max_plies_draw"


@dataclass(frozen=True, slots=True)
class GameConfig:
    variant: GameVariant = GameVariant.FOUR_PLAYER
    information_mode: InformationMode | None = None
    dead_rules_enabled: bool = True
    first_player: int = 0
    no_interaction_draw_plies: int = 60
    max_plies: int | None = None

    def __post_init__(self) -> None:
        try:
            variant = GameVariant(self.variant)
        except ValueError as error:
            raise GameRuleError(f"unknown game variant: {self.variant!r}") from error
        object.__setattr__(self, "variant", variant)

        default_mode = (
            InformationMode.FOUR_DARK
            if variant is GameVariant.FOUR_PLAYER
            else InformationMode.DARK
        )
        try:
            mode = (
                default_mode
                if self.information_mode is None
                else InformationMode(self.information_mode)
            )
        except ValueError as error:
            raise GameRuleError(
                f"unknown information mode: {self.information_mode!r}"
            ) from error
        allowed_modes = (
            {
                InformationMode.FOUR_DARK,
                InformationMode.DOUBLE_OPEN,
                InformationMode.FULL_OPEN,
            }
            if variant is GameVariant.FOUR_PLAYER
            else {InformationMode.DARK, InformationMode.OPEN}
        )
        if mode not in allowed_modes:
            raise GameRuleError(
                f"information mode {mode.value!r} is invalid for {variant.value}"
            )
        object.__setattr__(self, "information_mode", mode)

        if not isinstance(self.dead_rules_enabled, bool):
            raise GameRuleError("dead_rules_enabled must be a boolean")

        if (
            isinstance(self.first_player, bool)
            or not isinstance(self.first_player, int)
            or not 0 <= self.first_player < self.player_count
        ):
            raise GameRuleError(
                f"first player must be in 0..{self.player_count - 1}: "
                f"{self.first_player!r}"
            )
        if (
            isinstance(self.no_interaction_draw_plies, bool)
            or not isinstance(self.no_interaction_draw_plies, int)
            or self.no_interaction_draw_plies <= 0
        ):
            raise GameRuleError("no-interaction draw limit must be a positive integer")
        if self.max_plies is not None and (
            isinstance(self.max_plies, bool)
            or not isinstance(self.max_plies, int)
            or self.max_plies <= 0
        ):
            raise GameRuleError("maximum plies must be a positive integer or None")

    @property
    def player_count(self) -> int:
        return 4 if self.variant is GameVariant.FOUR_PLAYER else 2

    @property
    def turn_step(self) -> int:
        # FourPlayerSeat is clockwise, while the rules require anticlockwise.
        return -1 if self.variant is GameVariant.FOUR_PLAYER else 1


@dataclass(frozen=True, slots=True)
class GameResult:
    reason: TerminationReason
    winner_team: int | None

    @property
    def is_draw(self) -> bool:
        return self.winner_team is None


@dataclass(frozen=True, slots=True)
class ObservedPiece:
    """A piece as visible from one player's rotated perspective."""

    owner: int
    kind: PieceType | None
    identity_visible: bool
    has_moved: bool
    candidate_mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class PublicEvent:
    """One public transition stored without any hidden piece identities."""

    ply: int
    actor: int
    start: PhysicalPoint
    end: PhysicalPoint
    was_attack: bool
    combat: CombatOutcome
    flag_captured_owner: int | None
    newly_revealed_flags: tuple[int, ...]
    eliminated_players: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ObservedEvent:
    """A public event rotated and relabelled for one viewer."""

    ply: int
    actor: int
    action: Action
    was_attack: bool
    combat: CombatOutcome
    flag_captured_owner: int | None
    newly_revealed_flags: tuple[int, ...]
    eliminated_players: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Observation:
    variant: GameVariant
    information_mode: InformationMode
    dead_rules_enabled: bool
    viewer: int
    current_player: int | None
    points: tuple[ObservedPiece | None, ...]
    casualty_players: tuple[int, ...]
    known_casualties: tuple[tuple[bool, ...], ...]
    active_players: tuple[bool, ...]
    revealed_flags: tuple[bool, ...]
    legal_action_mask: tuple[bool, ...]
    legal_origin_mask: tuple[bool, ...]
    history: tuple[ObservedEvent, ...]
    ply_count: int
    no_interaction_plies: int
    result: GameResult | None


@dataclass(frozen=True, slots=True)
class StepResult:
    player: int
    action: Action
    attacker: Piece
    defender: Piece | None
    combat: CombatOutcome
    flag_captured_owner: int | None
    newly_revealed_flags: tuple[int, ...]
    eliminated_players: tuple[int, ...]
    next_player: int | None
    game_result: GameResult | None
    rewards: tuple[float, ...]


class JunqiGame:
    """Mutable referee state with player-relative actions and observations.

    The object is intentionally dependency-free.  Tree-search callers can use
    :meth:`clone`, :meth:`legal_action_mask`, :meth:`observe`, and :meth:`step`
    directly without adapting to a particular reinforcement-learning library.
    """

    def __init__(
        self,
        config: GameConfig | None = None,
        setups: Mapping[int, SetupInput] | Sequence[SetupInput] | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self._initialize_empty(config or GameConfig())
        generator = rng if rng is not None else random.Random()
        normalized_setups = self._normalize_setups(setups, generator)
        for owner, setup in enumerate(normalized_setups):
            for (row, column), kind in setup.items():
                point = ArmPoint(owner, row, column)
                self._pieces[point] = Piece(owner, kind)
                self._piece_candidates[point] = frozenset(
                    allowed_piece_types((row, column))
                )
        self._validate_position()
        self._resolve_current_turn()

    @classmethod
    def new_four_player(
        cls,
        setups: Mapping[int, SetupInput] | Sequence[SetupInput] | None = None,
        *,
        seed: int | None = None,
        information_mode: InformationMode = InformationMode.FOUR_DARK,
        dead_rules_enabled: bool = True,
        first_player: FourPlayerSeat = FourPlayerSeat.SOUTH,
        no_interaction_draw_plies: int = 60,
        max_plies: int | None = None,
    ) -> JunqiGame:
        config = GameConfig(
            variant=GameVariant.FOUR_PLAYER,
            information_mode=information_mode,
            dead_rules_enabled=dead_rules_enabled,
            first_player=int(first_player),
            no_interaction_draw_plies=no_interaction_draw_plies,
            max_plies=max_plies,
        )
        return cls(config, setups, rng=random.Random(seed))

    @classmethod
    def new_two_player(
        cls,
        setups: Mapping[int, SetupInput] | Sequence[SetupInput] | None = None,
        *,
        seed: int | None = None,
        information_mode: InformationMode = InformationMode.DARK,
        dead_rules_enabled: bool = True,
        first_player: TwoPlayerSeat = TwoPlayerSeat.SOUTH,
        no_interaction_draw_plies: int = 60,
        max_plies: int | None = None,
    ) -> JunqiGame:
        config = GameConfig(
            variant=GameVariant.TWO_PLAYER,
            information_mode=information_mode,
            dead_rules_enabled=dead_rules_enabled,
            first_player=int(first_player),
            no_interaction_draw_plies=no_interaction_draw_plies,
            max_plies=max_plies,
        )
        return cls(config, setups, rng=random.Random(seed))

    @classmethod
    def from_position(
        cls,
        config: GameConfig,
        pieces: Mapping[PhysicalPoint, Piece],
        *,
        current_player: int | None = None,
        active_players: Sequence[bool] | None = None,
        revealed_flags: Sequence[bool] | None = None,
        ply_count: int = 0,
        no_interaction_plies: int = 0,
        public_candidates: Mapping[
            PhysicalPoint, Sequence[PieceType | str]
        ]
        | None = None,
        known_identities: Sequence[
            Mapping[PhysicalPoint, PieceType | str]
        ]
        | None = None,
        known_casualties: Sequence[
            Sequence[Mapping[PieceType | str, int]]
        ]
        | None = None,
        public_history: Sequence[PublicEvent] = (),
    ) -> JunqiGame:
        """Restore a validated mid-game referee position.

        Captured inventories may be smaller than a full setup.  Every active
        player must still own exactly one flag in one of their headquarters.
        """

        game = cls.__new__(cls)
        game._initialize_empty(config)
        game._pieces = dict(pieces)
        game._piece_candidates = game._restore_candidate_sets(public_candidates)
        game._known_identities = game._restore_known_identities(known_identities)
        game._known_casualties = game._restore_known_casualties(
            known_casualties
        )
        if not all(isinstance(event, PublicEvent) for event in public_history):
            raise GameRuleError("public history must contain PublicEvent values")
        game._public_history = list(public_history)
        game._current_player = (
            config.first_player if current_player is None else current_player
        )
        if active_players is not None:
            if len(active_players) != config.player_count:
                raise GameRuleError(
                    f"active-player vector requires {config.player_count} values"
                )
            game._active = [bool(value) for value in active_players]
        if revealed_flags is not None:
            if len(revealed_flags) != config.player_count:
                raise GameRuleError(
                    f"revealed-flag vector requires {config.player_count} values"
                )
            game._flag_revealed = [bool(value) for value in revealed_flags]
        game._validate_counter("ply count", ply_count)
        game._validate_counter("no-interaction count", no_interaction_plies)
        game.ply_count = ply_count
        game.no_interaction_plies = no_interaction_plies
        game._validate_player(game._current_player, "current player")
        game._validate_public_history()
        game._validate_position()

        for owner in range(config.player_count):
            commander_present = any(
                piece.owner == owner and piece.kind is PieceType.COMMANDER
                for piece in game._pieces.values()
            )
            if revealed_flags is None:
                game._flag_revealed[owner] = not commander_present
            elif (
                game._active[owner]
                and not commander_present
                and not game._flag_revealed[owner]
            ):
                raise GameRuleError(
                    f"player {owner} has no commander, so its flag must be revealed"
                )

            # A revealed flag is public proof that this owner's sole commander
            # has died.  Repairing this fact on restore prevents an old/truncated
            # history from making Commander a future live candidate.
            if (
                config.dead_rules_enabled
                and game._flag_revealed[owner]
                and not commander_present
            ):
                for viewer in range(config.player_count):
                    game._known_casualties[viewer][owner][
                        PieceType.COMMANDER
                    ] = 1

        game._synchronize_revealed_flag_candidates()
        game._share_double_open_knowledge()

        if not game._finish_if_team_eliminated():
            game._apply_restored_draw_limits()
        if game.result is None:
            game._resolve_current_turn()
        return game

    def _initialize_empty(self, config: GameConfig) -> None:
        if not isinstance(config, GameConfig):
            raise GameRuleError(f"config must be GameConfig, got {type(config)!r}")
        self.config = config
        self._boards: tuple[Board, ...] = (
            tuple(
                FourPlayerBoard(FourPlayerSeat(player))
                for player in range(config.player_count)
            )
            if config.variant is GameVariant.FOUR_PLAYER
            else tuple(
                TwoPlayerBoard(TwoPlayerSeat(player))
                for player in range(config.player_count)
            )
        )
        self._pieces: dict[PhysicalPoint, Piece] = {}
        self._piece_candidates: dict[PhysicalPoint, frozenset[PieceType]] = {}
        # Exact identities inferred from private-but-legitimate information are
        # viewer-specific.  Keeping them on the game state makes the knowledge
        # survive a truncated Transformer history without revealing it to seats
        # that could not make the same deduction.
        self._known_identities: list[dict[PhysicalPoint, PieceType]] = [
            {} for _ in range(config.player_count)
        ]
        self._known_casualties: list[list[Counter[PieceType]]] = [
            [Counter() for _ in range(config.player_count)]
            for _ in range(config.player_count)
        ]
        self._public_history: list[PublicEvent] = []
        self._active = [True] * config.player_count
        self._flag_revealed = [False] * config.player_count
        self._current_player: int | None = config.first_player
        self.ply_count = 0
        self.no_interaction_plies = 0
        self.result: GameResult | None = None
        self._legal_actions_cache: dict[int, tuple[Action, ...]] = {}
        self._legal_action_sets: dict[int, frozenset[Action]] = {}

    def _normalize_setups(
        self,
        setups: Mapping[int, SetupInput] | Sequence[SetupInput] | None,
        rng: random.Random,
    ) -> tuple[PlayerSetup, ...]:
        if setups is None:
            return tuple(
                PlayerSetup.random(rng) for _ in range(self.config.player_count)
            )

        if isinstance(setups, Mapping):
            expected = set(range(self.config.player_count))
            actual: set[int] = set()
            for raw_owner in setups:
                self._validate_player(raw_owner, "setup owner")
                actual.add(raw_owner)
            if actual != expected:
                raise GameRuleError(
                    f"setups must contain players {sorted(expected)}, got "
                    f"{sorted(actual)}"
                )
            ordered = [setups[player] for player in range(self.config.player_count)]
        else:
            if len(setups) != self.config.player_count:
                raise GameRuleError(
                    f"expected {self.config.player_count} setups, got {len(setups)}"
                )
            ordered = list(setups)

        return tuple(
            setup if isinstance(setup, PlayerSetup) else PlayerSetup(setup)
            for setup in ordered
        )

    def _restore_candidate_sets(
        self,
        supplied: Mapping[PhysicalPoint, Sequence[PieceType | str]] | None,
    ) -> dict[PhysicalPoint, frozenset[PieceType]]:
        if supplied is not None and set(supplied) != set(self._pieces):
            raise GameRuleError(
                "public candidate map must contain exactly the occupied points"
            )

        result: dict[PhysicalPoint, frozenset[PieceType]] = {}
        for point, piece in self._pieces.items():
            if supplied is not None:
                try:
                    candidates = frozenset(PieceType(kind) for kind in supplied[point])
                except ValueError as error:
                    raise GameRuleError(
                        f"unknown public candidate type at {point}: {supplied[point]!r}"
                    ) from error
            elif (
                not piece.has_moved
                and isinstance(point, ArmPoint)
                and point.seat == piece.owner
                and (point.row, point.column) in SETUP_COORDINATES
            ):
                candidates = frozenset(
                    allowed_piece_types((point.row, point.column))
                )
            else:
                candidates = frozenset(MOVABLE_PIECE_TYPES)

            if not candidates:
                raise GameRuleError(f"public candidate set cannot be empty: {point}")
            if piece.kind not in candidates:
                raise GameRuleError(
                    f"public candidate set at {point} excludes its referee identity"
                )
            result[point] = candidates
        return result

    def _restore_known_identities(
        self,
        supplied: Sequence[Mapping[PhysicalPoint, PieceType | str]] | None,
    ) -> list[dict[PhysicalPoint, PieceType]]:
        if supplied is None:
            return [{} for _ in range(self.config.player_count)]
        if not self.config.dead_rules_enabled and any(supplied):
            raise GameRuleError(
                "known identities are forbidden when dead rules are disabled"
            )
        if len(supplied) != self.config.player_count:
            raise GameRuleError(
                "known-identity state requires one mapping per viewer"
            )

        restored: list[dict[PhysicalPoint, PieceType]] = []
        for viewer, mapping in enumerate(supplied):
            known: dict[PhysicalPoint, PieceType] = {}
            for point, raw_kind in mapping.items():
                if point not in self._pieces:
                    raise GameRuleError(
                        f"viewer {viewer} knows an identity at an empty point: {point}"
                    )
                try:
                    kind = PieceType(raw_kind)
                except ValueError as error:
                    raise GameRuleError(
                        f"unknown remembered piece type at {point}: {raw_kind!r}"
                    ) from error
                if self._pieces[point].kind is not kind:
                    raise GameRuleError(
                        f"viewer {viewer} remembered the wrong identity at {point}"
                    )
                known[point] = kind
            restored.append(known)
        return restored

    def _restore_known_casualties(
        self,
        supplied: Sequence[Sequence[Mapping[PieceType | str, int]]] | None,
    ) -> list[list[Counter[PieceType]]]:
        if supplied is None:
            return [
                [Counter() for _ in range(self.config.player_count)]
                for _ in range(self.config.player_count)
            ]
        if not self.config.dead_rules_enabled and any(
            mapping
            for owner_tables in supplied
            for mapping in owner_tables
        ):
            raise GameRuleError(
                "known casualties are forbidden when dead rules are disabled"
            )
        if len(supplied) != self.config.player_count:
            raise GameRuleError(
                "known-casualty state requires one owner table per viewer"
            )

        live_counts = [Counter() for _ in range(self.config.player_count)]
        for piece in self._pieces.values():
            live_counts[piece.owner][piece.kind] += 1

        restored: list[list[Counter[PieceType]]] = []
        for viewer, owner_tables in enumerate(supplied):
            if len(owner_tables) != self.config.player_count:
                raise GameRuleError(
                    f"viewer {viewer} casualty state requires "
                    f"{self.config.player_count} owner tables"
                )
            restored_owners: list[Counter[PieceType]] = []
            for owner, mapping in enumerate(owner_tables):
                counter: Counter[PieceType] = Counter()
                for raw_kind, raw_count in mapping.items():
                    try:
                        kind = PieceType(raw_kind)
                    except ValueError as error:
                        raise GameRuleError(
                            f"unknown casualty type for owner {owner}: {raw_kind!r}"
                        ) from error
                    self._validate_counter("known casualty count", raw_count)
                    maximum_dead = PIECE_COUNTS[kind] - live_counts[owner][kind]
                    if raw_count > maximum_dead:
                        raise GameRuleError(
                            f"viewer {viewer} knows {raw_count} dead {kind.value} "
                            f"for owner {owner}, but at most {maximum_dead} are dead"
                        )
                    if raw_count:
                        counter[kind] = raw_count
                restored_owners.append(counter)
            restored.append(restored_owners)
        return restored

    def _synchronize_revealed_flag_candidates(self) -> None:
        for point, piece in self._pieces.items():
            if piece.kind is PieceType.FLAG and self._flag_revealed[piece.owner]:
                self._piece_candidates[point] = frozenset({PieceType.FLAG})

    @staticmethod
    def _validate_counter(label: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GameRuleError(f"{label} must be a non-negative integer: {value!r}")

    def _validate_player(self, player: int | None, label: str = "player") -> None:
        if (
            isinstance(player, bool)
            or not isinstance(player, int)
            or not 0 <= player < self.config.player_count
        ):
            raise GameRuleError(
                f"{label} must be in 0..{self.config.player_count - 1}: {player!r}"
            )

    def _validate_position(self) -> None:
        if set(self._piece_candidates) != set(self._pieces):
            raise GameRuleError(
                "public candidate state must contain exactly the occupied points"
            )
        counts_by_owner = {
            owner: Counter() for owner in range(self.config.player_count)
        }
        flags_by_owner = Counter()
        reference_board = self._boards[0]
        for point, piece in self._pieces.items():
            if not isinstance(point, (ArmPoint, CenterPoint)):
                raise GameRuleError(f"invalid physical point: {point!r}")
            try:
                reference_board.encode(point)
            except BoardEncodingError as error:
                raise GameRuleError(
                    f"point does not belong to {self.config.variant.value}: {point!r}"
                ) from error
            if not isinstance(piece, Piece):
                raise GameRuleError(f"position values must be Piece: {piece!r}")
            self._validate_player(piece.owner, "piece owner")
            if not self._active[piece.owner]:
                raise GameRuleError(
                    f"inactive player {piece.owner} cannot retain pieces"
                )
            counts_by_owner[piece.owner][piece.kind] += 1
            candidates = self._piece_candidates[point]
            if not candidates or piece.kind not in candidates:
                raise GameRuleError(
                    f"invalid public candidate set for piece at {point}"
                )

            if piece.kind in (PieceType.MINE, PieceType.FLAG) and piece.has_moved:
                raise GameRuleError(f"immovable piece cannot be marked moved: {piece}")
            if piece.kind is PieceType.MINE:
                if (
                    not isinstance(point, ArmPoint)
                    or point.seat != piece.owner
                    or point.row not in (5, 6)
                ):
                    raise GameRuleError(
                        f"mine must remain in its owner's rows 5 or 6: {point}"
                    )
            if piece.kind is PieceType.FLAG:
                flags_by_owner[piece.owner] += 1
                if (
                    not isinstance(point, ArmPoint)
                    or point.seat != piece.owner
                    or (point.row, point.column) not in HEADQUARTERS_COORDINATES
                ):
                    raise GameRuleError(
                        f"flag must remain in its owner's headquarters: {point}"
                    )

        for owner, counts in counts_by_owner.items():
            for kind, count in counts.items():
                if count > PIECE_COUNTS[kind]:
                    raise GameRuleError(
                        f"player {owner} has too many {kind.value} pieces: {count}"
                    )
            expected_flags = 1 if self._active[owner] else 0
            if flags_by_owner[owner] != expected_flags:
                raise GameRuleError(
                    f"active player {owner} must have exactly one flag"
                )

    def _validate_public_history(self) -> None:
        previous_ply = -1
        reference_board = self._boards[0]
        for event in self._public_history:
            if event.ply <= previous_ply or event.ply > self.ply_count:
                raise GameRuleError(
                    f"public event ply is out of order or beyond state: {event.ply}"
                )
            previous_ply = event.ply
            self._validate_player(event.actor, "event actor")
            for point in (event.start, event.end):
                try:
                    reference_board.encode(point)
                except BoardEncodingError as error:
                    raise GameRuleError(
                        f"public event point is outside the board: {point!r}"
                    ) from error
            if event.flag_captured_owner is not None:
                self._validate_player(
                    event.flag_captured_owner, "captured flag owner"
                )
            for owner in (
                event.newly_revealed_flags + event.eliminated_players
            ):
                self._validate_player(owner, "event player")

    def _apply_restored_draw_limits(self) -> None:
        if self.no_interaction_plies >= self.config.no_interaction_draw_plies:
            self._finish_draw(TerminationReason.NO_INTERACTION_DRAW)
        elif (
            self.config.max_plies is not None
            and self.ply_count >= self.config.max_plies
        ):
            self._finish_draw(TerminationReason.MAX_PLIES_DRAW)

    @property
    def current_player(self) -> int | None:
        return self._current_player

    @property
    def is_terminal(self) -> bool:
        return self.result is not None

    @property
    def active_players(self) -> tuple[bool, ...]:
        return tuple(self._active)

    @property
    def revealed_flags(self) -> tuple[bool, ...]:
        return tuple(self._flag_revealed)

    @property
    def pieces(self) -> Mapping[PhysicalPoint, Piece]:
        return MappingProxyType(dict(self._pieces))

    @property
    def public_history(self) -> tuple[PublicEvent, ...]:
        return tuple(self._public_history)

    @property
    def public_candidates(
        self,
    ) -> Mapping[PhysicalPoint, frozenset[PieceType]]:
        return MappingProxyType(dict(self._piece_candidates))

    @property
    def known_identities(
        self,
    ) -> tuple[Mapping[PhysicalPoint, PieceType], ...]:
        """Return each viewer's persistent, legitimately inferred identities."""

        return tuple(
            MappingProxyType(dict(known)) for known in self._known_identities
        )

    @property
    def known_casualties(
        self,
    ) -> tuple[tuple[Mapping[PieceType, int], ...], ...]:
        """Return exact dead-piece counts indexed by viewer then physical owner."""

        return tuple(
            tuple(MappingProxyType(dict(counts)) for counts in owner_tables)
            for owner_tables in self._known_casualties
        )

    def board_for(self, player: int) -> Board:
        self._validate_player(player)
        return self._boards[player]

    def piece_at(self, point: PhysicalPoint) -> Piece | None:
        return self._pieces.get(point)

    def team_of(self, player: int) -> int:
        self._validate_player(player)
        return player % 2 if self.config.variant is GameVariant.FOUR_PLAYER else player

    def are_allies(self, first: int, second: int) -> bool:
        self._validate_player(first, "first player")
        self._validate_player(second, "second player")
        return first != second and self.team_of(first) == self.team_of(second)

    def clone(self) -> JunqiGame:
        clone = self.__class__.__new__(self.__class__)
        clone.config = self.config
        clone._boards = self._boards
        clone._pieces = dict(self._pieces)
        clone._piece_candidates = dict(self._piece_candidates)
        clone._known_identities = [
            dict(known) for known in self._known_identities
        ]
        clone._known_casualties = [
            [Counter(counts) for counts in owner_tables]
            for owner_tables in self._known_casualties
        ]
        clone._public_history = list(self._public_history)
        clone._active = list(self._active)
        clone._flag_revealed = list(self._flag_revealed)
        clone._current_player = self._current_player
        clone.ply_count = self.ply_count
        clone.no_interaction_plies = self.no_interaction_plies
        clone.result = self.result
        clone._legal_actions_cache = dict(self._legal_actions_cache)
        clone._legal_action_sets = dict(self._legal_action_sets)
        return clone

    def _invalidate_legal_actions(self) -> None:
        self._legal_actions_cache.clear()
        self._legal_action_sets.clear()

    def _relative_player_order(self, viewer: int) -> tuple[int, ...]:
        if self.config.variant is GameVariant.FOUR_PLAYER:
            return tuple(
                (viewer + relative) % 4 for relative in range(4)
            )
        return (viewer, 1 - viewer)

    def _casualty_owner_order(self, viewer: int) -> tuple[int, ...]:
        """Physical owners represented in this viewer's casualty matrix."""

        if self.config.variant is GameVariant.TWO_PLAYER:
            return (1 - viewer,)
        if self.config.information_mode is InformationMode.DOUBLE_OPEN:
            return ((viewer + 1) % 4, (viewer + 3) % 4)
        return tuple((viewer + relative) % 4 for relative in (1, 2, 3))

    @staticmethod
    def _casualty_bits(counts: Mapping[PieceType, int]) -> tuple[bool, ...]:
        used: Counter[PieceType] = Counter()
        bits: list[bool] = []
        for kind in DEPLOYMENT_PIECE_SEQUENCE:
            used[kind] += 1
            bits.append(used[kind] <= counts.get(kind, 0))
        return tuple(bits)

    def _share_double_open_knowledge(self) -> None:
        """Synchronize exact knowledge inside each double-open partnership."""

        if (
            not self.config.dead_rules_enabled
            or self.config.information_mode is not InformationMode.DOUBLE_OPEN
        ):
            return
        for first, second in ((0, 2), (1, 3)):
            merged_identities = dict(self._known_identities[first])
            for point, kind in self._known_identities[second].items():
                previous = merged_identities.get(point)
                if previous is not None and previous is not kind:
                    raise RuntimeError("allied exact identities contradict")
                merged_identities[point] = kind
            self._known_identities[first] = dict(merged_identities)
            self._known_identities[second] = dict(merged_identities)

            for owner in range(self.config.player_count):
                merged_counts = Counter(
                    {
                        kind: max(
                            self._known_casualties[first][owner][kind],
                            self._known_casualties[second][owner][kind],
                        )
                        for kind in PIECE_TYPE_ORDER
                    }
                )
                merged_counts += Counter()  # discard zero entries
                self._known_casualties[first][owner] = Counter(merged_counts)
                self._known_casualties[second][owner] = Counter(merged_counts)

    def _identity_visible_by_mode(self, viewer: int, piece: Piece) -> bool:
        if piece.owner == viewer:
            return True
        mode = self.config.information_mode
        if mode in (InformationMode.FULL_OPEN, InformationMode.OPEN):
            return True
        if mode is InformationMode.DOUBLE_OPEN and self.are_allies(viewer, piece.owner):
            return True
        return piece.kind is PieceType.FLAG and self._flag_revealed[piece.owner]

    def _known_piece_kind(
        self,
        viewer: int,
        point: PhysicalPoint,
        piece: Piece,
    ) -> PieceType | None:
        """Return an exact type only when this viewer can legitimately know it."""

        if self._identity_visible_by_mode(viewer, piece):
            return piece.kind
        if not self.config.dead_rules_enabled:
            return None
        candidates = self._piece_candidates[point]
        if len(candidates) == 1:
            return next(iter(candidates))
        return self._known_identities[viewer].get(point)

    def _remember_survivor(
        self,
        viewer: int,
        point: PhysicalPoint,
        piece: Piece,
        known_kind: PieceType | None,
    ) -> None:
        remembered = self._known_identities[viewer]
        if known_kind is None:
            remembered.pop(point, None)
            return
        if known_kind is not piece.kind:
            raise RuntimeError("identity inference contradicted the referee state")
        candidates = self._piece_candidates[point]
        if self._identity_visible_by_mode(viewer, piece) or len(candidates) == 1:
            remembered.pop(point, None)
        else:
            remembered[point] = known_kind

    def _known_live_counts(
        self, viewer: int
    ) -> list[Counter[PieceType]]:
        counts = [Counter() for _ in range(self.config.player_count)]
        for point, piece in self._pieces.items():
            known = self._known_piece_kind(viewer, point, piece)
            if known is not None:
                counts[piece.owner][known] += 1
        return counts

    def _viewer_piece_candidates(
        self,
        *,
        viewer: int,
        point: PhysicalPoint,
        piece: Piece,
        public_candidates: frozenset[PieceType],
        known_kind: PieceType | None,
        known_live_counts: Sequence[Counter[PieceType]],
        is_attacker: bool,
        proves_engineer: bool = False,
    ) -> frozenset[PieceType]:
        """Return only identities this viewer still considers possible."""

        if proves_engineer:
            if known_kind is not None and known_kind is not PieceType.ENGINEER:
                raise RuntimeError("engineer path contradicts an exact identity")
            candidates = {PieceType.ENGINEER}
        elif known_kind is not None:
            candidates = {known_kind}
        else:
            candidates = set(public_candidates)
            if is_attacker:
                candidates.difference_update((PieceType.MINE, PieceType.FLAG))
            for kind in tuple(candidates):
                accounted_for = (
                    self._known_casualties[viewer][piece.owner][kind]
                    + known_live_counts[piece.owner][kind]
                )
                if accounted_for >= PIECE_COUNTS[kind]:
                    candidates.remove(kind)

        if not candidates or piece.kind not in candidates:
            raise RuntimeError(
                f"viewer {viewer} candidate inference contradicts referee piece "
                f"at {point}"
            )
        return frozenset(candidates)

    def _record_known_casualty(
        self, viewer: int, owner: int, kind: PieceType
    ) -> None:
        counts = self._known_casualties[viewer][owner]
        counts[kind] += 1
        if counts[kind] > PIECE_COUNTS[kind]:
            raise RuntimeError("known casualty count exceeds initial inventory")

    @staticmethod
    def _combat_outcome_for_kinds(
        attacker: PieceType, defender: PieceType
    ) -> CombatOutcome:
        if attacker is PieceType.BOMB or defender is PieceType.BOMB:
            return CombatOutcome.BOTH_REMOVED
        if defender is PieceType.FLAG:
            return CombatOutcome.ATTACKER_WINS
        if defender is PieceType.MINE:
            return (
                CombatOutcome.ATTACKER_WINS
                if attacker is PieceType.ENGINEER
                else CombatOutcome.DEFENDER_WINS
            )
        attacker_rank = PIECE_RANKS[attacker]
        defender_rank = PIECE_RANKS[defender]
        if attacker_rank > defender_rank:
            return CombatOutcome.ATTACKER_WINS
        if attacker_rank < defender_rank:
            return CombatOutcome.DEFENDER_WINS
        return CombatOutcome.BOTH_REMOVED

    def _combat_pairs_for_viewer(
        self,
        *,
        attacker: Piece,
        defender: Piece,
        attacker_candidates: frozenset[PieceType],
        defender_candidates: frozenset[PieceType],
        combat: CombatOutcome,
        flag_captured_owner: int | None,
        previously_revealed_flags: Sequence[bool],
        newly_revealed_flags: Sequence[int],
    ) -> tuple[tuple[PieceType, PieceType], ...]:
        newly_revealed = frozenset(newly_revealed_flags)
        attacker_destroyed = combat in (
            CombatOutcome.DEFENDER_WINS,
            CombatOutcome.BOTH_REMOVED,
        )
        defender_destroyed = combat in (
            CombatOutcome.ATTACKER_WINS,
            CombatOutcome.BOTH_REMOVED,
        )
        pairs: list[tuple[PieceType, PieceType]] = []
        for attacker_kind in attacker_candidates:
            for defender_kind in defender_candidates:
                if (
                    self._combat_outcome_for_kinds(attacker_kind, defender_kind)
                    is not combat
                ):
                    continue
                if (defender_kind is PieceType.FLAG) != (
                    flag_captured_owner == defender.owner
                ):
                    continue

                reveal_consistent = True
                for owner, kind, destroyed in (
                    (attacker.owner, attacker_kind, attacker_destroyed),
                    (defender.owner, defender_kind, defender_destroyed),
                ):
                    if not destroyed:
                        continue
                    should_reveal = (
                        kind is PieceType.COMMANDER
                        and not previously_revealed_flags[owner]
                    )
                    if (owner in newly_revealed) != should_reveal:
                        reveal_consistent = False
                        break
                if reveal_consistent:
                    pairs.append((attacker_kind, defender_kind))
        return tuple(pairs)

    def _update_exact_knowledge_after_step(
        self,
        *,
        start: PhysicalPoint,
        end: PhysicalPoint,
        attacker: Piece,
        defender: Piece | None,
        combat: CombatOutcome,
        attacker_candidates: Sequence[frozenset[PieceType]],
        defender_candidates: Sequence[frozenset[PieceType]],
        previously_revealed_flags: Sequence[bool],
        newly_revealed_flags: Sequence[int],
        flag_captured_owner: int | None,
    ) -> None:
        """Persist only identities forced by every legal hidden combat pair."""

        for viewer in range(self.config.player_count):
            remembered = self._known_identities[viewer]
            remembered.pop(start, None)

            if defender is None:
                possible = attacker_candidates[viewer]
                self._remember_survivor(
                    viewer,
                    end,
                    self._pieces[end],
                    next(iter(possible)) if len(possible) == 1 else None,
                )
                continue

            pairs = self._combat_pairs_for_viewer(
                attacker=attacker,
                defender=defender,
                attacker_candidates=attacker_candidates[viewer],
                defender_candidates=defender_candidates[viewer],
                combat=combat,
                flag_captured_owner=flag_captured_owner,
                previously_revealed_flags=previously_revealed_flags,
                newly_revealed_flags=newly_revealed_flags,
            )
            if not pairs or (attacker.kind, defender.kind) not in pairs:
                raise RuntimeError("combat knowledge has no referee-consistent pair")

            possible_attackers = frozenset(pair[0] for pair in pairs)
            possible_defenders = frozenset(pair[1] for pair in pairs)
            attacker_exact = (
                next(iter(possible_attackers))
                if len(possible_attackers) == 1
                else None
            )
            defender_exact = (
                next(iter(possible_defenders))
                if len(possible_defenders) == 1
                else None
            )

            if combat in (
                CombatOutcome.DEFENDER_WINS,
                CombatOutcome.BOTH_REMOVED,
            ) and attacker_exact is not None:
                self._record_known_casualty(
                    viewer, attacker.owner, attacker_exact
                )
            if combat in (
                CombatOutcome.ATTACKER_WINS,
                CombatOutcome.BOTH_REMOVED,
            ) and defender_exact is not None:
                self._record_known_casualty(
                    viewer, defender.owner, defender_exact
                )

            if combat is CombatOutcome.ATTACKER_WINS:
                self._remember_survivor(
                    viewer, end, self._pieces[end], attacker_exact
                )
                continue
            if combat is CombatOutcome.DEFENDER_WINS:
                self._remember_survivor(
                    viewer, end, self._pieces[end], defender_exact
                )
                continue

            remembered.pop(end, None)

        self._share_double_open_knowledge()

    def observe(
        self,
        viewer: int | None = None,
        *,
        history_limit: int | None = 1000,
        include_legal_masks: bool = True,
        include_candidate_masks: bool = True,
    ) -> Observation:
        if viewer is None:
            if self._current_player is None:
                raise GameRuleError("terminal observations require an explicit viewer")
            viewer = self._current_player
        self._validate_player(viewer, "viewer")
        if history_limit is not None and (
            isinstance(history_limit, bool)
            or not isinstance(history_limit, int)
            or history_limit < 0
        ):
            raise GameRuleError(
                f"history limit must be a non-negative integer or None: "
                f"{history_limit!r}"
            )
        board = self._boards[viewer]
        visible_points: list[ObservedPiece | None] = [None] * board.point_count
        order = self._relative_player_order(viewer)
        relative_owner = {owner: index for index, owner in enumerate(order)}
        for point, piece in self._pieces.items():
            known_kind = self._known_piece_kind(viewer, point, piece)
            visible = known_kind is not None
            candidates = (
                frozenset({known_kind})
                if visible
                else self._piece_candidates[point]
            )
            visible_points[board.encode(point)] = ObservedPiece(
                owner=relative_owner[piece.owner],
                kind=known_kind,
                identity_visible=visible,
                has_moved=piece.has_moved,
                candidate_mask=(
                    tuple(kind in candidates for kind in PIECE_TYPE_ORDER)
                    if include_candidate_masks
                    else ()
                ),
            )

        current_relative = (
            None
            if self._current_player is None
            else relative_owner[self._current_player]
        )
        mask = (
            self.legal_action_mask(viewer)
            if include_legal_masks and viewer == self._current_player
            else tuple(False for _ in board.actions)
            if include_legal_masks
            else ()
        )
        origin_mask = (
            self.legal_origin_mask(viewer)
            if include_legal_masks and viewer == self._current_player
            else tuple(False for _ in board.action_space.origin_codes)
            if include_legal_masks
            else ()
        )
        selected_history = (
            self._public_history
            if history_limit is None
            else self._public_history[-history_limit:]
            if history_limit
            else []
        )
        observed_history = tuple(
            ObservedEvent(
                ply=event.ply,
                actor=relative_owner[event.actor],
                action=(board.encode(event.start), board.encode(event.end)),
                was_attack=event.was_attack,
                combat=event.combat,
                flag_captured_owner=(
                    None
                    if event.flag_captured_owner is None
                    else relative_owner[event.flag_captured_owner]
                ),
                newly_revealed_flags=tuple(
                    relative_owner[owner]
                    for owner in event.newly_revealed_flags
                ),
                eliminated_players=tuple(
                    relative_owner[owner]
                    for owner in event.eliminated_players
                ),
            )
            for event in selected_history
        )
        casualty_owners = (
            self._casualty_owner_order(viewer)
            if self.config.dead_rules_enabled
            else ()
        )
        return Observation(
            variant=self.config.variant,
            information_mode=self.config.information_mode,
            dead_rules_enabled=self.config.dead_rules_enabled,
            viewer=viewer,
            current_player=current_relative,
            points=tuple(visible_points),
            casualty_players=tuple(
                relative_owner[owner] for owner in casualty_owners
            ),
            known_casualties=tuple(
                self._casualty_bits(self._known_casualties[viewer][owner])
                for owner in casualty_owners
            ),
            active_players=tuple(self._active[owner] for owner in order),
            revealed_flags=tuple(self._flag_revealed[owner] for owner in order),
            legal_action_mask=mask,
            legal_origin_mask=origin_mask,
            history=observed_history,
            ply_count=self.ply_count,
            no_interaction_plies=self.no_interaction_plies,
            result=self.result,
        )

    def _target_is_enterable(self, player: int, target: int, board: Board) -> bool:
        target_point = board.decode(target)
        occupant = self._pieces.get(target_point)
        if occupant is None:
            return True
        if occupant.owner == player or self.are_allies(player, occupant.owner):
            return False
        return board.point(target).kind is not PointKind.CAMP

    def _railway_reachable_targets(
        self,
        board: Board,
        start: int,
        occupied: frozenset[int],
        *,
        can_turn: bool,
    ) -> set[int]:
        reachable: set[int] = set()
        if not board.neighbors(start, PathKind.RAILWAY):
            return reachable

        if can_turn:
            queue = deque([start])
            visited = {start}
            while queue:
                current = queue.popleft()
                for neighbor in board.neighbors(current, PathKind.RAILWAY):
                    if neighbor == start:
                        continue
                    reachable.add(neighbor)
                    if neighbor in occupied or neighbor in visited:
                        continue
                    visited.add(neighbor)
                    queue.append(neighbor)
            return reachable

        queue: deque[tuple[int, RailwayDirection | None]] = deque([(start, None)])
        visited: set[tuple[int, RailwayDirection | None]] = {(start, None)}
        while queue:
            current, arrival_direction = queue.popleft()
            for neighbor in board.neighbors(current, PathKind.RAILWAY):
                if neighbor == start:
                    continue
                directions = board.railway_directions(current, neighbor)
                if directions is None:
                    continue
                departure_direction, next_arrival = directions
                if (
                    arrival_direction is not None
                    and departure_direction is not arrival_direction
                ):
                    continue
                reachable.add(neighbor)
                state = (neighbor, next_arrival)
                if neighbor in occupied or state in visited:
                    continue
                visited.add(state)
                queue.append(state)
        return reachable

    def _geometric_targets(
        self,
        board: Board,
        start: int,
        piece: Piece,
        occupied: frozenset[int],
    ) -> set[int]:
        targets = set(board.neighbors(start))
        targets.update(
            self._railway_reachable_targets(
                board,
                start,
                occupied,
                can_turn=piece.kind is PieceType.ENGINEER,
            )
        )
        return targets

    def _move_proves_engineer(
        self,
        board: Board,
        start: int,
        end: int,
        occupied: frozenset[int],
    ) -> bool:
        if end in board.neighbors(start):
            return False
        return end not in self._railway_reachable_targets(
            board, start, occupied, can_turn=False
        )

    def _legal_actions_for(self, player: int) -> tuple[Action, ...]:
        cached = self._legal_actions_cache.get(player)
        if cached is not None:
            return cached
        if self.result is not None or not self._active[player]:
            return ()

        board = self._boards[player]
        encoded_pieces = tuple(
            (board.encode(point), point, piece)
            for point, piece in self._pieces.items()
        )
        occupants = {code: piece for code, _point, piece in encoded_pieces}
        occupied = frozenset(occupants)
        actions: list[Action] = []
        owned_starts = sorted(
            (
                code,
                point,
                piece,
            )
            for code, point, piece in encoded_pieces
            if piece.owner == player and piece.movable
        )
        for start, point, piece in owned_starts:
            if board.point(start).kind is PointKind.HEADQUARTERS:
                continue
            for target in sorted(
                self._geometric_targets(board, start, piece, occupied)
            ):
                occupant = occupants.get(target)
                if (
                    occupant is None
                    or (
                        occupant.owner != player
                        and not (
                            self.config.variant is GameVariant.FOUR_PLAYER
                            and player % 2 == occupant.owner % 2
                        )
                        and board.point(target).kind is not PointKind.CAMP
                    )
                ):
                    actions.append((start, target))

        result = tuple(actions)
        self._legal_actions_cache[player] = result
        self._legal_action_sets[player] = frozenset(result)
        return result

    def legal_actions(self, player: int | None = None) -> tuple[Action, ...]:
        if player is None:
            if self._current_player is None:
                return ()
            player = self._current_player
        self._validate_player(player)
        return self._legal_actions_for(player)

    def legal_action_indices(self, player: int | None = None) -> tuple[int, ...]:
        if player is None:
            if self._current_player is None:
                return ()
            player = self._current_player
        self._validate_player(player)
        board = self._boards[player]
        return tuple(
            board.action_index(start, end)
            for start, end in self._legal_actions_for(player)
        )

    def legal_action_mask(self, player: int | None = None) -> tuple[bool, ...]:
        if player is None:
            if self._current_player is None:
                return tuple(False for _ in self._boards[0].actions)
            player = self._current_player
        self._validate_player(player)
        board = self._boards[player]
        mask = [False] * len(board.actions)
        for action_index in self.legal_action_indices(player):
            mask[action_index] = True
        return tuple(mask)

    def legal_origin_indices(self, player: int | None = None) -> tuple[int, ...]:
        if player is None:
            if self._current_player is None:
                return ()
            player = self._current_player
        self._validate_player(player)
        board = self._boards[player]
        starts = sorted({start for start, _end in self._legal_actions_for(player)})
        return tuple(board.action_space.origin_index(start) for start in starts)

    def legal_origin_mask(self, player: int | None = None) -> tuple[bool, ...]:
        if player is None:
            if self._current_player is None:
                return tuple(
                    False for _ in self._boards[0].action_space.origin_codes
                )
            player = self._current_player
        self._validate_player(player)
        board = self._boards[player]
        mask = [False] * board.action_space.origin_count
        for origin_index in self.legal_origin_indices(player):
            mask[origin_index] = True
        return tuple(mask)

    def legal_destination_slots(
        self, start: int, player: int | None = None
    ) -> tuple[int, ...]:
        if player is None:
            if self._current_player is None:
                return ()
            player = self._current_player
        self._validate_player(player)
        board = self._boards[player]
        try:
            board.action_space.targets_from(start)
        except BoardEncodingError as error:
            raise GameRuleError(str(error)) from error
        legal_targets = {
            end
            for action_start, end in self._legal_actions_for(player)
            if action_start == start
        }
        try:
            return tuple(
                board.action_space.destination_slot(start, end)
                for end in sorted(legal_targets)
            )
        except BoardEncodingError as error:
            raise GameRuleError(str(error)) from error

    def legal_destination_mask(
        self, start: int, player: int | None = None
    ) -> tuple[bool, ...]:
        if player is None:
            if self._current_player is None:
                return tuple(
                    False
                    for _ in range(
                        self._boards[0].action_space.max_destination_count
                    )
                )
            player = self._current_player
        self._validate_player(player)
        board = self._boards[player]
        mask = [False] * board.action_space.max_destination_count
        for slot in self.legal_destination_slots(start, player):
            mask[slot] = True
        return tuple(mask)

    def is_legal_action(
        self,
        action: Action | int,
        player: int | None = None,
    ) -> bool:
        if player is None:
            if self._current_player is None:
                return False
            player = self._current_player
        self._validate_player(player)
        try:
            normalized = self._normalize_action(action, self._boards[player])
        except (GameRuleError, BoardEncodingError, TypeError, ValueError):
            return False
        self._legal_actions_for(player)
        return normalized in self._legal_action_sets.get(player, frozenset())

    @staticmethod
    def _normalize_action(action: Action | int, board: Board) -> Action:
        if isinstance(action, bool):
            raise IllegalActionError(f"invalid action: {action!r}")
        if isinstance(action, int):
            try:
                return board.decode_action(action)
            except BoardEncodingError as error:
                raise IllegalActionError(str(error)) from error
        try:
            start, end = action
        except (TypeError, ValueError) as error:
            raise IllegalActionError(
                f"action must be an index or (from_code, to_code): {action!r}"
            ) from error
        try:
            board.action_index(start, end)
        except BoardEncodingError as error:
            raise IllegalActionError(str(error)) from error
        return (start, end)

    @staticmethod
    def _combat_outcome(attacker: Piece, defender: Piece | None) -> CombatOutcome:
        if defender is None:
            return CombatOutcome.MOVE
        return JunqiGame._combat_outcome_for_kinds(attacker.kind, defender.kind)

    def _reveal_destroyed_commanders(
        self, pieces: Sequence[Piece]
    ) -> tuple[int, ...]:
        newly_revealed: list[int] = []
        for piece in pieces:
            if (
                piece.kind is PieceType.COMMANDER
                and not self._flag_revealed[piece.owner]
            ):
                self._flag_revealed[piece.owner] = True
                newly_revealed.append(piece.owner)
        self._synchronize_revealed_flag_candidates()
        return tuple(sorted(set(newly_revealed)))

    def _eliminate_player(self, player: int) -> None:
        if not self._active[player]:
            return
        # Elimination removes the owner's complete original inventory.  This is
        # exact public information (most commonly caused by carrying the flag),
        # so every viewer receives a full 25-bit casualty row.
        if self.config.dead_rules_enabled:
            for viewer in range(self.config.player_count):
                self._known_casualties[viewer][player] = Counter(PIECE_COUNTS)
        self._active[player] = False
        self._pieces = {
            point: piece
            for point, piece in self._pieces.items()
            if piece.owner != player
        }
        self._piece_candidates = {
            point: candidates
            for point, candidates in self._piece_candidates.items()
            if point in self._pieces
        }
        self._known_identities = [
            {
                point: kind
                for point, kind in known.items()
                if point in self._pieces
            }
            for known in self._known_identities
        ]
        self._share_double_open_knowledge()
        self._invalidate_legal_actions()

    def _finish_draw(self, reason: TerminationReason) -> None:
        self.result = GameResult(reason=reason, winner_team=None)
        self._current_player = None
        self._invalidate_legal_actions()

    def _finish_if_team_eliminated(self) -> bool:
        if self.result is not None:
            return True
        active_teams = {
            self.team_of(player)
            for player, active in enumerate(self._active)
            if active
        }
        if len(active_teams) > 1:
            return False
        winner = next(iter(active_teams)) if active_teams else None
        self.result = GameResult(
            reason=TerminationReason.TEAM_ELIMINATED,
            winner_team=winner,
        )
        self._current_player = None
        self._invalidate_legal_actions()
        return True

    def _resolve_current_turn(self) -> tuple[int, ...]:
        """Eliminate active players that have no move when their turn arrives."""

        eliminated: list[int] = []
        if self._finish_if_team_eliminated():
            return ()
        while self.result is None and self._current_player is not None:
            player = self._current_player
            if not self._active[player]:
                self._current_player = (
                    player + self.config.turn_step
                ) % self.config.player_count
                continue
            if self._legal_actions_for(player):
                break
            self._eliminate_player(player)
            eliminated.append(player)
            if self._finish_if_team_eliminated():
                break
            self._current_player = (
                player + self.config.turn_step
            ) % self.config.player_count
        return tuple(eliminated)

    def rewards(self) -> tuple[float, ...]:
        if self.result is None or self.result.winner_team is None:
            return tuple(0.0 for _ in range(self.config.player_count))
        return tuple(
            1.0 if self.team_of(player) == self.result.winner_team else -1.0
            for player in range(self.config.player_count)
        )

    def step(self, action: Action | int) -> StepResult:
        if self.result is not None or self._current_player is None:
            raise TerminalGameError("cannot move after the game has ended")
        player = self._current_player
        board = self._boards[player]
        normalized = self._normalize_action(action, board)
        self._legal_actions_for(player)
        if normalized not in self._legal_action_sets[player]:
            raise IllegalActionError(
                f"action is not legal for player {player}: {normalized}"
            )

        start, end = normalized
        start_point = board.decode(start)
        end_point = board.decode(end)
        attacker = self._pieces[start_point]
        defender = self._pieces.get(end_point)
        combat = self._combat_outcome(attacker, defender)
        previously_revealed_flags = tuple(self._flag_revealed)
        occupied = frozenset(board.encode(point) for point in self._pieces)
        proves_engineer = self._move_proves_engineer(
            board, start, end, occupied
        )
        viewer_attacker_candidates: tuple[frozenset[PieceType], ...] = ()
        viewer_defender_candidates: tuple[frozenset[PieceType], ...] = ()
        if self.config.dead_rules_enabled:
            known_attackers = tuple(
                self._known_piece_kind(viewer, start_point, attacker)
                for viewer in range(self.config.player_count)
            )
            known_defenders = tuple(
                None
                if defender is None
                else self._known_piece_kind(viewer, end_point, defender)
                for viewer in range(self.config.player_count)
            )
            known_live_counts = tuple(
                self._known_live_counts(viewer)
                for viewer in range(self.config.player_count)
            )
            public_attacker_candidates = self._piece_candidates[start_point]
            public_defender_candidates = (
                frozenset()
                if defender is None
                else self._piece_candidates[end_point]
            )
            viewer_attacker_candidates = tuple(
                self._viewer_piece_candidates(
                    viewer=viewer,
                    point=start_point,
                    piece=attacker,
                    public_candidates=public_attacker_candidates,
                    known_kind=known_attackers[viewer],
                    known_live_counts=known_live_counts[viewer],
                    is_attacker=True,
                    proves_engineer=proves_engineer,
                )
                for viewer in range(self.config.player_count)
            )
            viewer_defender_candidates = tuple(
                frozenset()
                if defender is None
                else self._viewer_piece_candidates(
                    viewer=viewer,
                    point=end_point,
                    piece=defender,
                    public_candidates=public_defender_candidates,
                    known_kind=known_defenders[viewer],
                    known_live_counts=known_live_counts[viewer],
                    is_attacker=False,
                )
                for viewer in range(self.config.player_count)
            )

        attacker_candidates = self._piece_candidates.pop(start_point)
        if proves_engineer:
            moved_candidates = frozenset({PieceType.ENGINEER})
        else:
            moved_candidates = frozenset(
                kind
                for kind in attacker_candidates
                if kind not in (PieceType.MINE, PieceType.FLAG)
            )

        del self._pieces[start_point]
        destroyed: list[Piece] = []
        if combat in (CombatOutcome.MOVE, CombatOutcome.ATTACKER_WINS):
            if defender is not None:
                destroyed.append(defender)
                del self._piece_candidates[end_point]
            self._pieces[end_point] = attacker.moved()
            self._piece_candidates[end_point] = moved_candidates
        elif combat is CombatOutcome.DEFENDER_WINS:
            destroyed.append(attacker)
        else:
            destroyed.append(attacker)
            if defender is not None:
                destroyed.append(defender)
                del self._pieces[end_point]
                del self._piece_candidates[end_point]

        newly_revealed_flags = self._reveal_destroyed_commanders(destroyed)
        flag_captured_owner = (
            defender.owner
            if defender is not None and defender.kind is PieceType.FLAG
            else None
        )
        if self.config.dead_rules_enabled:
            self._update_exact_knowledge_after_step(
                start=start_point,
                end=end_point,
                attacker=attacker,
                defender=defender,
                combat=combat,
                attacker_candidates=viewer_attacker_candidates,
                defender_candidates=viewer_defender_candidates,
                previously_revealed_flags=previously_revealed_flags,
                newly_revealed_flags=newly_revealed_flags,
                flag_captured_owner=flag_captured_owner,
            )
        eliminated: list[int] = []
        if flag_captured_owner is not None:
            self._eliminate_player(flag_captured_owner)
            eliminated.append(flag_captured_owner)

        self.ply_count += 1
        self.no_interaction_plies = (
            0 if defender is not None else self.no_interaction_plies + 1
        )
        self._invalidate_legal_actions()

        if not self._finish_if_team_eliminated():
            if self.no_interaction_plies >= self.config.no_interaction_draw_plies:
                self._finish_draw(TerminationReason.NO_INTERACTION_DRAW)
            elif (
                self.config.max_plies is not None
                and self.ply_count >= self.config.max_plies
            ):
                self._finish_draw(TerminationReason.MAX_PLIES_DRAW)
            else:
                self._current_player = (
                    player + self.config.turn_step
                ) % self.config.player_count
                eliminated.extend(self._resolve_current_turn())

        self._public_history.append(
            PublicEvent(
                ply=self.ply_count,
                actor=player,
                start=start_point,
                end=end_point,
                was_attack=defender is not None,
                combat=combat,
                flag_captured_owner=flag_captured_owner,
                newly_revealed_flags=newly_revealed_flags,
                eliminated_players=tuple(eliminated),
            )
        )

        return StepResult(
            player=player,
            action=normalized,
            attacker=attacker,
            defender=defender,
            combat=combat,
            flag_captured_owner=flag_captured_owner,
            newly_revealed_flags=newly_revealed_flags,
            eliminated_players=tuple(eliminated),
            next_player=self._current_player,
            game_result=self.result,
            rewards=self.rewards(),
        )

    def state_key(self) -> tuple[object, ...]:
        """Return a deterministic, hashable referee-state key for tree search."""

        reference_board = self._boards[0]
        pieces = tuple(
            sorted(
                (
                    reference_board.encode(point),
                    piece.owner,
                    piece.kind.value,
                    piece.has_moved,
                )
                for point, piece in self._pieces.items()
            )
        )
        public_candidates = tuple(
            sorted(
                (
                    reference_board.encode(point),
                    tuple(
                        kind.value
                        for kind in PIECE_TYPE_ORDER
                        if kind in candidates
                    ),
                )
                for point, candidates in self._piece_candidates.items()
            )
        )
        known_identities = tuple(
            tuple(
                sorted(
                    (
                        reference_board.encode(point),
                        kind.value,
                    )
                    for point, kind in known.items()
                )
            )
            for known in self._known_identities
        )
        known_casualties = tuple(
            tuple(
                tuple(counts[kind] for kind in PIECE_TYPE_ORDER)
                for counts in owner_tables
            )
            for owner_tables in self._known_casualties
        )
        return (
            self.config.variant.value,
            self.config.information_mode.value,
            self.config.dead_rules_enabled,
            self._current_player,
            tuple(self._active),
            tuple(self._flag_revealed),
            self.ply_count,
            self.no_interaction_plies,
            pieces,
            public_candidates,
            known_identities,
            known_casualties,
            tuple(self._public_history),
            self.result,
        )
