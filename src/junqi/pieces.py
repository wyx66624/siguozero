"""Junqi pieces and legal initial-deployment validation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import random
from types import MappingProxyType
from typing import TypeAlias

from .board import (
    ARM_COLUMNS,
    ARM_ROWS,
    CAMP_COORDINATES,
    HEADQUARTERS_COORDINATES,
)


Coordinate: TypeAlias = tuple[int, int]


class SetupError(ValueError):
    """Raised when a player's initial deployment violates the rules."""


class PieceType(str, Enum):
    COMMANDER = "commander"  # 司令
    ARMY_COMMANDER = "army_commander"  # 军长
    DIVISION_COMMANDER = "division_commander"  # 师长
    BRIGADE_COMMANDER = "brigade_commander"  # 旅长
    REGIMENT_COMMANDER = "regiment_commander"  # 团长
    BATTALION_COMMANDER = "battalion_commander"  # 营长
    COMPANY_COMMANDER = "company_commander"  # 连长
    PLATOON_COMMANDER = "platoon_commander"  # 排长
    ENGINEER = "engineer"  # 工兵
    BOMB = "bomb"  # 炸弹
    MINE = "mine"  # 地雷
    FLAG = "flag"  # 军旗

    @property
    def chinese_name(self) -> str:
        return PIECE_CHINESE_NAMES[self]


PIECE_TYPE_ORDER: tuple[PieceType, ...] = tuple(PieceType)
PIECE_TYPE_INDICES: Mapping[PieceType, int] = MappingProxyType(
    {kind: index for index, kind in enumerate(PIECE_TYPE_ORDER)}
)


PIECE_COUNTS: Mapping[PieceType, int] = MappingProxyType(
    {
        PieceType.COMMANDER: 1,
        PieceType.ARMY_COMMANDER: 1,
        PieceType.DIVISION_COMMANDER: 2,
        PieceType.BRIGADE_COMMANDER: 2,
        PieceType.REGIMENT_COMMANDER: 2,
        PieceType.BATTALION_COMMANDER: 2,
        PieceType.COMPANY_COMMANDER: 3,
        PieceType.PLATOON_COMMANDER: 3,
        PieceType.ENGINEER: 3,
        PieceType.BOMB: 2,
        PieceType.MINE: 3,
        PieceType.FLAG: 1,
    }
)

# Stable autoregressive deployment order.  Repeated entries intentionally
# identify the 25 indistinguishable inventory slots used by both the layout
# decoder and the per-opponent exact-casualty prior.
DEPLOYMENT_PIECE_SEQUENCE: tuple[PieceType, ...] = (
    PieceType.FLAG,
    PieceType.MINE,
    PieceType.MINE,
    PieceType.MINE,
    PieceType.BOMB,
    PieceType.BOMB,
    PieceType.COMMANDER,
    PieceType.ARMY_COMMANDER,
    PieceType.DIVISION_COMMANDER,
    PieceType.DIVISION_COMMANDER,
    PieceType.BRIGADE_COMMANDER,
    PieceType.BRIGADE_COMMANDER,
    PieceType.REGIMENT_COMMANDER,
    PieceType.REGIMENT_COMMANDER,
    PieceType.BATTALION_COMMANDER,
    PieceType.BATTALION_COMMANDER,
    PieceType.COMPANY_COMMANDER,
    PieceType.COMPANY_COMMANDER,
    PieceType.COMPANY_COMMANDER,
    PieceType.PLATOON_COMMANDER,
    PieceType.PLATOON_COMMANDER,
    PieceType.PLATOON_COMMANDER,
    PieceType.ENGINEER,
    PieceType.ENGINEER,
    PieceType.ENGINEER,
)

if Counter(DEPLOYMENT_PIECE_SEQUENCE) != Counter(PIECE_COUNTS):
    raise RuntimeError("deployment piece sequence does not match Junqi inventory")

PIECE_RANKS: Mapping[PieceType, int] = MappingProxyType(
    {
        PieceType.COMMANDER: 9,
        PieceType.ARMY_COMMANDER: 8,
        PieceType.DIVISION_COMMANDER: 7,
        PieceType.BRIGADE_COMMANDER: 6,
        PieceType.REGIMENT_COMMANDER: 5,
        PieceType.BATTALION_COMMANDER: 4,
        PieceType.COMPANY_COMMANDER: 3,
        PieceType.PLATOON_COMMANDER: 2,
        PieceType.ENGINEER: 1,
    }
)

PIECE_CHINESE_NAMES: Mapping[PieceType, str] = MappingProxyType(
    {
        PieceType.COMMANDER: "司令",
        PieceType.ARMY_COMMANDER: "军长",
        PieceType.DIVISION_COMMANDER: "师长",
        PieceType.BRIGADE_COMMANDER: "旅长",
        PieceType.REGIMENT_COMMANDER: "团长",
        PieceType.BATTALION_COMMANDER: "营长",
        PieceType.COMPANY_COMMANDER: "连长",
        PieceType.PLATOON_COMMANDER: "排长",
        PieceType.ENGINEER: "工兵",
        PieceType.BOMB: "炸弹",
        PieceType.MINE: "地雷",
        PieceType.FLAG: "军旗",
    }
)

MOVABLE_PIECE_TYPES: tuple[PieceType, ...] = tuple(
    kind
    for kind in PIECE_TYPE_ORDER
    if kind not in (PieceType.MINE, PieceType.FLAG)
)

SETUP_COORDINATES = frozenset(
    (row, column)
    for row in range(1, ARM_ROWS + 1)
    for column in range(1, ARM_COLUMNS + 1)
    if (row, column) not in CAMP_COORDINATES
)

LAYOUT_ORDER: tuple[Coordinate, ...] = (
    tuple(sorted(HEADQUARTERS_COORDINATES))
    + tuple((1, column) for column in range(1, ARM_COLUMNS + 1))
    + tuple(
        (row, column)
        for row in range(2, 5)
        for column in range(1, ARM_COLUMNS + 1)
        if (row, column) not in CAMP_COORDINATES
    )
    + tuple((5, column) for column in range(1, ARM_COLUMNS + 1))
    + tuple(
        (6, column)
        for column in range(1, ARM_COLUMNS + 1)
        if (6, column) not in HEADQUARTERS_COORDINATES
    )
)


@dataclass(frozen=True, slots=True)
class Piece:
    """One referee-visible piece in a running game."""

    owner: int
    kind: PieceType
    has_moved: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.owner, bool) or not isinstance(self.owner, int):
            raise SetupError(f"piece owner must be an integer: {self.owner!r}")
        if self.owner < 0:
            raise SetupError(f"piece owner cannot be negative: {self.owner}")
        try:
            object.__setattr__(self, "kind", PieceType(self.kind))
        except ValueError as error:
            raise SetupError(f"unknown piece type: {self.kind!r}") from error

    @property
    def movable(self) -> bool:
        return self.kind not in (PieceType.MINE, PieceType.FLAG)

    @property
    def rank(self) -> int | None:
        return PIECE_RANKS.get(self.kind)

    def moved(self) -> Piece:
        return self if self.has_moved else replace(self, has_moved=True)


class PlayerSetup(Mapping[Coordinate, PieceType]):
    """Validated immutable deployment for one 6 x 5 player arm."""

    __slots__ = ("_placements", "_items")

    def __init__(
        self,
        placements: Mapping[Coordinate, PieceType | str]
        | Iterable[tuple[Coordinate, PieceType | str]],
    ) -> None:
        source = placements.items() if isinstance(placements, Mapping) else placements
        normalized: dict[Coordinate, PieceType] = {}
        for raw_coordinate, raw_kind in source:
            try:
                row, column = raw_coordinate
            except (TypeError, ValueError) as error:
                raise SetupError(
                    f"setup coordinate must be a (row, column) pair: "
                    f"{raw_coordinate!r}"
                ) from error
            if (
                isinstance(row, bool)
                or not isinstance(row, int)
                or isinstance(column, bool)
                or not isinstance(column, int)
            ):
                raise SetupError(
                    f"setup coordinate values must be integers: {raw_coordinate!r}"
                )
            coordinate = (row, column)
            if coordinate in normalized:
                raise SetupError(f"duplicate setup coordinate: {coordinate}")
            try:
                normalized[coordinate] = PieceType(raw_kind)
            except ValueError as error:
                raise SetupError(f"unknown piece type: {raw_kind!r}") from error

        validate_setup(normalized)
        self._placements = MappingProxyType(normalized)
        self._items = tuple(sorted(normalized.items()))

    def __getitem__(self, coordinate: Coordinate) -> PieceType:
        return self._placements[coordinate]

    def __iter__(self) -> Iterator[Coordinate]:
        return (coordinate for coordinate, _kind in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"PlayerSetup({dict(self._items)!r})"

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Sequence[PieceType | str | None]],
    ) -> PlayerSetup:
        """Create a setup from six rows of five values; camps must be ``None``."""

        if len(rows) != ARM_ROWS:
            raise SetupError(f"setup requires {ARM_ROWS} rows, got {len(rows)}")
        placements: dict[Coordinate, PieceType | str] = {}
        for row_index, row_values in enumerate(rows, start=1):
            if len(row_values) != ARM_COLUMNS:
                raise SetupError(
                    f"setup row {row_index} requires {ARM_COLUMNS} values, "
                    f"got {len(row_values)}"
                )
            for column, value in enumerate(row_values, start=1):
                coordinate = (row_index, column)
                if coordinate in CAMP_COORDINATES:
                    if value is not None:
                        raise SetupError(
                            f"camp must be empty in initial setup: {coordinate}"
                        )
                    continue
                if value is None:
                    raise SetupError(
                        f"non-camp setup point requires a piece: {coordinate}"
                    )
                placements[coordinate] = value
        return cls(placements)

    @classmethod
    def random(cls, rng: random.Random | None = None) -> PlayerSetup:
        """Generate a random legal deployment, using ``rng`` when supplied."""

        generator = rng if rng is not None else random.Random()
        placements: dict[Coordinate, PieceType] = {}

        flag_coordinate = generator.choice(sorted(HEADQUARTERS_COORDINATES))
        placements[flag_coordinate] = PieceType.FLAG

        mine_candidates = sorted(
            coordinate
            for coordinate in SETUP_COORDINATES
            if coordinate[0] in (5, 6) and coordinate not in placements
        )
        for coordinate in generator.sample(
            mine_candidates, PIECE_COUNTS[PieceType.MINE]
        ):
            placements[coordinate] = PieceType.MINE

        bomb_candidates = sorted(
            coordinate
            for coordinate in SETUP_COORDINATES
            if coordinate[0] != 1 and coordinate not in placements
        )
        for coordinate in generator.sample(
            bomb_candidates, PIECE_COUNTS[PieceType.BOMB]
        ):
            placements[coordinate] = PieceType.BOMB

        remaining_coordinates = sorted(SETUP_COORDINATES - placements.keys())
        remaining_pieces = [
            kind
            for kind, count in PIECE_COUNTS.items()
            if kind not in (PieceType.FLAG, PieceType.MINE, PieceType.BOMB)
            for _ in range(count)
        ]
        generator.shuffle(remaining_coordinates)
        generator.shuffle(remaining_pieces)
        placements.update(zip(remaining_coordinates, remaining_pieces, strict=True))
        return cls(placements)

    def to_rows(self) -> tuple[tuple[PieceType | None, ...], ...]:
        return tuple(
            tuple(
                None
                if (row, column) in CAMP_COORDINATES
                else self[(row, column)]
                for column in range(1, ARM_COLUMNS + 1)
            )
            for row in range(1, ARM_ROWS + 1)
        )


def validate_setup(placements: Mapping[Coordinate, PieceType]) -> None:
    """Validate exact inventory and all fixed deployment restrictions."""

    actual_coordinates = set(placements)
    missing = SETUP_COORDINATES - actual_coordinates
    extra = actual_coordinates - SETUP_COORDINATES
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"invalid={sorted(extra)}")
        raise SetupError(
            "initial setup must fill all 25 non-camp points (" + ", ".join(details) + ")"
        )

    actual_counts = Counter(placements.values())
    expected_counts = Counter(PIECE_COUNTS)
    if actual_counts != expected_counts:
        differences = {
            kind.value: actual_counts[kind] - expected_counts[kind]
            for kind in PieceType
            if actual_counts[kind] != expected_counts[kind]
        }
        raise SetupError(f"incorrect piece inventory: {differences}")

    flag_coordinate = next(
        coordinate
        for coordinate, kind in placements.items()
        if kind is PieceType.FLAG
    )
    if flag_coordinate not in HEADQUARTERS_COORDINATES:
        raise SetupError(
            f"flag must be placed in headquarters: {flag_coordinate}"
        )

    invalid_mines = sorted(
        coordinate
        for coordinate, kind in placements.items()
        if kind is PieceType.MINE and coordinate[0] not in (5, 6)
    )
    if invalid_mines:
        raise SetupError(f"mines must be placed in rows 5 or 6: {invalid_mines}")

    invalid_bombs = sorted(
        coordinate
        for coordinate, kind in placements.items()
        if kind is PieceType.BOMB and coordinate[0] == 1
    )
    if invalid_bombs:
        raise SetupError(f"bombs cannot be placed in row 1: {invalid_bombs}")


def random_setup(rng: random.Random | None = None) -> PlayerSetup:
    """Convenience wrapper for :meth:`PlayerSetup.random`."""

    return PlayerSetup.random(rng)


def allowed_piece_types(coordinate: Coordinate) -> tuple[PieceType, ...]:
    """Return locally legal initial piece types for one setup coordinate."""

    if coordinate not in SETUP_COORDINATES:
        raise SetupError(f"not an initial setup coordinate: {coordinate}")
    row, _column = coordinate
    return tuple(
        kind
        for kind in PieceType
        if not (kind is PieceType.FLAG and coordinate not in HEADQUARTERS_COORDINATES)
        and not (kind is PieceType.MINE and row not in (5, 6))
        and not (kind is PieceType.BOMB and row == 1)
    )


def _completion_is_feasible(
    remaining_counts: Mapping[PieceType, int],
    remaining_positions: Sequence[Coordinate],
) -> bool:
    """Solve the small type-capacity/position bipartite matching problem."""

    if sum(remaining_counts.values()) != len(remaining_positions):
        return False
    if not remaining_positions:
        return True

    kinds = tuple(PieceType)
    kind_count = len(kinds)
    position_count = len(remaining_positions)
    source = 0
    first_kind = 1
    first_position = first_kind + kind_count
    sink = first_position + position_count
    node_count = sink + 1
    capacity = [[0] * node_count for _ in range(node_count)]

    for kind_index, kind in enumerate(kinds):
        capacity[source][first_kind + kind_index] = remaining_counts[kind]
    for position_index, coordinate in enumerate(remaining_positions):
        position_node = first_position + position_index
        capacity[position_node][sink] = 1
        allowed = set(allowed_piece_types(coordinate))
        for kind_index, kind in enumerate(kinds):
            if kind in allowed and remaining_counts[kind] > 0:
                capacity[first_kind + kind_index][position_node] = 1

    flow = 0
    while True:
        parent = [-1] * node_count
        parent[source] = source
        queue = [source]
        for current in queue:
            for neighbor, residual in enumerate(capacity[current]):
                if residual > 0 and parent[neighbor] == -1:
                    parent[neighbor] = current
                    queue.append(neighbor)
                    if neighbor == sink:
                        break
            if parent[sink] != -1:
                break
        if parent[sink] == -1:
            break

        current = sink
        while current != source:
            previous = parent[current]
            capacity[previous][current] -= 1
            capacity[current][previous] += 1
            current = previous
        flow += 1
    return flow == position_count


class LayoutBuilder:
    """Autoregressive setup state with an exact future-feasibility mask."""

    __slots__ = ("_placements", "_remaining", "_index")

    def __init__(self) -> None:
        self._placements: dict[Coordinate, PieceType] = {}
        self._remaining = Counter(PIECE_COUNTS)
        self._index = 0

    @property
    def is_complete(self) -> bool:
        return self._index == len(LAYOUT_ORDER)

    @property
    def step_index(self) -> int:
        return self._index

    @property
    def next_coordinate(self) -> Coordinate | None:
        return None if self.is_complete else LAYOUT_ORDER[self._index]

    @property
    def placements(self) -> Mapping[Coordinate, PieceType]:
        return MappingProxyType(dict(self._placements))

    @property
    def remaining_counts(self) -> tuple[int, ...]:
        return tuple(self._remaining[kind] for kind in PIECE_TYPE_ORDER)

    def legal_piece_types(self) -> tuple[PieceType, ...]:
        if self.is_complete:
            return ()
        coordinate = LAYOUT_ORDER[self._index]
        remaining_positions = LAYOUT_ORDER[self._index + 1 :]
        legal: list[PieceType] = []
        for kind in allowed_piece_types(coordinate):
            if self._remaining[kind] <= 0:
                continue
            self._remaining[kind] -= 1
            feasible = _completion_is_feasible(
                self._remaining, remaining_positions
            )
            self._remaining[kind] += 1
            if feasible:
                legal.append(kind)
        return tuple(legal)

    def legal_piece_mask(self) -> tuple[bool, ...]:
        legal = frozenset(self.legal_piece_types())
        return tuple(kind in legal for kind in PIECE_TYPE_ORDER)

    def place(self, kind: PieceType | str) -> Coordinate:
        if self.is_complete:
            raise SetupError("layout is already complete")
        try:
            normalized = PieceType(kind)
        except ValueError as error:
            raise SetupError(f"unknown piece type: {kind!r}") from error
        legal = self.legal_piece_types()
        if normalized not in legal:
            raise SetupError(
                f"piece {normalized.value!r} is not legal at "
                f"{self.next_coordinate}; legal={[kind.value for kind in legal]}"
            )
        coordinate = LAYOUT_ORDER[self._index]
        self._placements[coordinate] = normalized
        self._remaining[normalized] -= 1
        self._index += 1
        return coordinate

    def place_index(self, piece_index: int) -> Coordinate:
        if (
            isinstance(piece_index, bool)
            or not isinstance(piece_index, int)
            or not 0 <= piece_index < len(PIECE_TYPE_ORDER)
        ):
            raise SetupError(
                f"piece index must be in 0..{len(PIECE_TYPE_ORDER) - 1}: "
                f"{piece_index!r}"
            )
        return self.place(PIECE_TYPE_ORDER[piece_index])

    def build(self) -> PlayerSetup:
        if not self.is_complete:
            raise SetupError(
                f"layout is incomplete: {self._index}/{len(LAYOUT_ORDER)} points"
            )
        return PlayerSetup(self._placements)

    def clone(self) -> LayoutBuilder:
        clone = self.__class__.__new__(self.__class__)
        clone._placements = dict(self._placements)
        clone._remaining = Counter(self._remaining)
        clone._index = self._index
        return clone

    @classmethod
    def sample(cls, rng: random.Random | None = None) -> PlayerSetup:
        generator = rng if rng is not None else random.Random()
        builder = cls()
        while not builder.is_complete:
            builder.place(generator.choice(builder.legal_piece_types()))
        return builder.build()
