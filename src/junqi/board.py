"""Player-relative board encodings for four-player and two-player Junqi.

Rows in an arm run from the front line (1) back to headquarters (6).
Columns always run left-to-right from that arm owner's point of view.
Physical board segments are undirected.  Move actions should use ordered
``(from_code, to_code)`` pairs and be masked against the current game state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Iterable, TypeAlias


ARM_ROWS = 6
ARM_COLUMNS = 5
POINTS_PER_ARM = ARM_ROWS * ARM_COLUMNS
CENTER_SIZE = 3

CAMP_COORDINATES = frozenset(
    {(2, 2), (2, 4), (3, 3), (4, 2), (4, 4)}
)
HEADQUARTERS_COORDINATES = frozenset({(6, 2), (6, 4)})


class BoardEncodingError(ValueError):
    """Raised when a point or code does not belong to the selected board."""


class FourPlayerSeat(IntEnum):
    """Absolute spatial seats used by the physical four-player board.

    The numeric order is clockwise on the rendered board starting at SOUTH:
    SOUTH -> WEST -> NORTH -> EAST.  It is an encoding convention and does not
    define the game's turn order.
    """

    SOUTH = 0
    WEST = 1
    NORTH = 2
    EAST = 3


class TwoPlayerSeat(IntEnum):
    SOUTH = 0
    NORTH = 1


class FourPlayerRelativeArm(IntEnum):
    SELF = 0
    LEFT_OPPONENT = 1
    PARTNER = 2
    RIGHT_OPPONENT = 3


class TwoPlayerRelativeArm(IntEnum):
    SELF = 0
    OPPONENT = 1


class PointKind(str, Enum):
    STATION = "station"
    CAMP = "camp"
    HEADQUARTERS = "headquarters"
    CENTRAL_STATION = "central_station"


class PathKind(str, Enum):
    ROAD = "road"
    RAILWAY = "railway"


class RailwayDirection(IntEnum):
    """Absolute travel direction at one end of a railway segment."""

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    @property
    def opposite(self) -> RailwayDirection:
        return RailwayDirection((int(self) + 2) % 4)


@dataclass(frozen=True, slots=True)
class ArmPoint:
    """A physical point in one player's 6 x 5 arm."""

    seat: int
    row: int
    column: int

    def __post_init__(self) -> None:
        if not 1 <= self.row <= ARM_ROWS:
            raise BoardEncodingError(f"arm row must be in 1..{ARM_ROWS}: {self.row}")
        if not 1 <= self.column <= ARM_COLUMNS:
            raise BoardEncodingError(
                f"arm column must be in 1..{ARM_COLUMNS}: {self.column}"
            )


@dataclass(frozen=True, slots=True)
class CenterPoint:
    """A physical point in the four-player board's central 3 x 3 grid.

    Absolute center coordinates are expressed from SOUTH's view: row 1 is
    nearest SOUTH and column 1 is on SOUTH's left (the west side).
    """

    row: int
    column: int

    def __post_init__(self) -> None:
        if not 1 <= self.row <= CENTER_SIZE:
            raise BoardEncodingError(
                f"center row must be in 1..{CENTER_SIZE}: {self.row}"
            )
        if not 1 <= self.column <= CENTER_SIZE:
            raise BoardEncodingError(
                f"center column must be in 1..{CENTER_SIZE}: {self.column}"
            )


PhysicalPoint: TypeAlias = ArmPoint | CenterPoint
RelativeArm: TypeAlias = FourPlayerRelativeArm | TwoPlayerRelativeArm
Action: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class PointRecord:
    code: int
    physical: PhysicalPoint
    kind: PointKind
    relative_arm: RelativeArm | None


@dataclass(frozen=True, order=True, slots=True)
class PathRecord:
    """One undirected physical segment in canonical endpoint order."""

    start: int
    end: int
    kind: PathKind

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise BoardEncodingError(
                "undirected path records require start < end, "
                f"got ({self.start}, {self.end})"
            )

    @property
    def pair(self) -> tuple[int, int]:
        return (self.start, self.end)


class StaticActionSpace:
    """State-independent move-action vocabulary for one encoded board.

    An action is retained when at least one movable piece could traverse it in
    some board state.  In practice that is an adjacent printed segment, or a
    route within one connected railway component (an engineer may turn).  An
    action beginning in headquarters is never retained.

    Occupancy, piece identity, straight-line restrictions for non-engineers,
    camp protection, and side ownership are deliberately left to the dynamic
    legal-action mask.
    """

    __slots__ = (
        "actions",
        "origin_codes",
        "targets_by_origin",
        "max_destination_count",
        "_action_indices",
        "_origin_indices",
    )

    def __init__(self, targets_by_origin: Iterable[Iterable[int]]) -> None:
        rows = tuple(
            tuple(sorted(set(destinations)))
            for destinations in targets_by_origin
        )
        point_count = len(rows)
        for start, destinations in enumerate(rows):
            for end in destinations:
                if isinstance(end, bool) or not isinstance(end, int):
                    raise BoardEncodingError(
                        f"action target must be an integer: {end!r}"
                    )
                if not 0 <= end < point_count:
                    raise BoardEncodingError(
                        f"action target must be in 0..{point_count - 1}: {end}"
                    )
                if start == end:
                    raise BoardEncodingError(
                        f"static action space cannot contain ({start}, {end})"
                    )

        self.targets_by_origin = rows
        self.origin_codes = tuple(
            start for start, destinations in enumerate(rows) if destinations
        )
        self.actions = tuple(
            (start, end)
            for start in self.origin_codes
            for end in rows[start]
        )
        self.max_destination_count = max(
            (len(destinations) for destinations in rows), default=0
        )
        self._action_indices = {
            action: index for index, action in enumerate(self.actions)
        }
        self._origin_indices = {
            code: index for index, code in enumerate(self.origin_codes)
        }

    @property
    def point_count(self) -> int:
        return len(self.targets_by_origin)

    @property
    def action_count(self) -> int:
        return len(self.actions)

    @property
    def origin_count(self) -> int:
        return len(self.origin_codes)

    def _validate_point(self, code: int) -> None:
        if isinstance(code, bool) or not isinstance(code, int):
            raise BoardEncodingError(f"point code must be an integer: {code!r}")
        if not 0 <= code < self.point_count:
            raise BoardEncodingError(
                f"point code must be in 0..{self.point_count - 1}: {code}"
            )

    @staticmethod
    def _validate_index(index: int, size: int, label: str) -> None:
        if isinstance(index, bool) or not isinstance(index, int):
            raise BoardEncodingError(f"{label} must be an integer: {index!r}")
        if not 0 <= index < size:
            raise BoardEncodingError(
                f"{label} must be in 0..{size - 1}: {index}"
            )

    def targets_from(self, start: int) -> tuple[int, ...]:
        """Return statically possible destination codes in ascending order."""

        self._validate_point(start)
        return self.targets_by_origin[start]

    def contains(self, start: int, end: int) -> bool:
        """Return whether the ordered pair belongs to this vocabulary."""

        self._validate_point(start)
        self._validate_point(end)
        return (start, end) in self._action_indices

    def action_index(self, start: int, end: int) -> int:
        """Encode an endpoint pair as its stable flat-vocabulary index."""

        self._validate_point(start)
        self._validate_point(end)
        try:
            return self._action_indices[(start, end)]
        except KeyError as error:
            raise BoardEncodingError(
                f"action is outside the static action space: ({start}, {end})"
            ) from error

    def action_at(self, index: int) -> Action:
        """Decode a flat-vocabulary index to ``(from_code, to_code)``."""

        self._validate_index(index, self.action_count, "action index")
        return self.actions[index]

    def origin_index(self, start: int) -> int:
        """Encode a board point for a compact piece-selection head."""

        self._validate_point(start)
        try:
            return self._origin_indices[start]
        except KeyError as error:
            raise BoardEncodingError(
                f"point can never be a move origin: {start}"
            ) from error

    def origin_at(self, index: int) -> int:
        """Decode a compact piece-selection index to a board point code."""

        self._validate_index(index, self.origin_count, "origin index")
        return self.origin_codes[index]

    def destination_slot(self, start: int, end: int) -> int:
        """Encode a destination for the compact head conditioned on ``start``."""

        destinations = self.targets_from(start)
        self._validate_point(end)
        try:
            return destinations.index(end)
        except ValueError as error:
            raise BoardEncodingError(
                f"action is outside the static action space: ({start}, {end})"
            ) from error

    def destination_at(self, start: int, slot: int) -> int:
        """Decode a source-conditioned destination-head slot."""

        destinations = self.targets_from(start)
        self._validate_index(slot, len(destinations), "destination slot")
        return destinations[slot]


PhysicalEdgeMap: TypeAlias = dict[frozenset[PhysicalPoint], PathKind]


def _arm_offset(row: int, column: int) -> int:
    return (row - 1) * ARM_COLUMNS + (column - 1)


def _row_column(offset: int) -> tuple[int, int]:
    zero_based_row, zero_based_column = divmod(offset, ARM_COLUMNS)
    return zero_based_row + 1, zero_based_column + 1


def _point_kind(point: PhysicalPoint) -> PointKind:
    if isinstance(point, CenterPoint):
        return PointKind.CENTRAL_STATION
    coordinate = (point.row, point.column)
    if coordinate in CAMP_COORDINATES:
        return PointKind.CAMP
    if coordinate in HEADQUARTERS_COORDINATES:
        return PointKind.HEADQUARTERS
    return PointKind.STATION


def _add_edge(
    edges: PhysicalEdgeMap,
    start: PhysicalPoint,
    end: PhysicalPoint,
    kind: PathKind,
) -> None:
    if start == end:
        raise BoardEncodingError(f"self-loop is not a board path: {start}")
    key = frozenset((start, end))
    existing = edges.get(key)
    if existing is not None and existing != kind:
        raise BoardEncodingError(
            f"path has conflicting kinds {existing!s} and {kind!s}: {start}, {end}"
        )
    edges[key] = kind


def _add_arm_edges(edges: PhysicalEdgeMap, seat: int) -> None:
    """Add the standard printed graph for one 6 x 5 player arm."""

    # Every row has adjacent horizontal links.  Rows 1 and 5 are railway.
    for row in range(1, ARM_ROWS + 1):
        kind = PathKind.RAILWAY if row in (1, 5) else PathKind.ROAD
        for column in range(1, ARM_COLUMNS):
            _add_edge(
                edges,
                ArmPoint(seat, row, column),
                ArmPoint(seat, row, column + 1),
                kind,
            )

    # Every column has adjacent vertical links.  The outside columns are
    # railway through rows 1..5; their final link to row 6 is road.
    for row in range(1, ARM_ROWS):
        for column in range(1, ARM_COLUMNS + 1):
            kind = (
                PathKind.RAILWAY
                if column in (1, 5) and row <= 4
                else PathKind.ROAD
            )
            _add_edge(
                edges,
                ArmPoint(seat, row, column),
                ArmPoint(seat, row + 1, column),
                kind,
            )

    # Each camp has the four printed diagonal road links.  Shared links such
    # as (2,2)<->(3,3) are deliberately deduplicated by _add_edge.
    for row, column in CAMP_COORDINATES:
        for row_delta in (-1, 1):
            for column_delta in (-1, 1):
                _add_edge(
                    edges,
                    ArmPoint(seat, row, column),
                    ArmPoint(seat, row + row_delta, column + column_delta),
                    PathKind.ROAD,
                )


def _rotate_center(
    row: int, column: int, counter_clockwise_quarter_turns: int
) -> tuple[int, int]:
    """Rotate center coordinates around (2, 2).

    The mathematical y axis points from SOUTH toward NORTH, so row 1 has
    y=-1.  A WEST viewer needs one counter-clockwise turn to occupy SOUTH's
    canonical viewing position.
    """

    x = column - 2
    y = row - 2
    for _ in range(counter_clockwise_quarter_turns % 4):
        x, y = -y, x
    return y + 2, x + 2


class _EncodedBoard:
    point_count: int

    def _finish(self, physical_edges: PhysicalEdgeMap) -> None:
        records: list[PathRecord] = []
        for physical_pair, kind in physical_edges.items():
            first, second = tuple(physical_pair)
            first_code = self.encode(first)
            second_code = self.encode(second)
            start, end = sorted((first_code, second_code))
            records.append(PathRecord(start, end, kind))

        self.path_records = tuple(
            sorted(records, key=lambda record: (record.start, record.end))
        )
        self.paths = tuple(record.pair for record in self.path_records)
        self.directed_paths = tuple(
            sorted(
                directed
                for record in self.path_records
                for directed in ((record.start, record.end), (record.end, record.start))
            )
        )
        self.points = tuple(self.point(code) for code in range(self.point_count))
        # Runtime games call encode/decode hundreds of thousands of times.
        # The geometry is immutable, so turn those operations into table
        # lookups after construction instead of repeatedly rotating points and
        # instantiating Enum values in Python.
        self._decoded_points = tuple(record.physical for record in self.points)
        self._encoded_points = {
            point: code for code, point in enumerate(self._decoded_points)
        }

        adjacency: dict[int, list[tuple[int, PathKind]]] = {
            code: [] for code in range(self.point_count)
        }
        for record in self.path_records:
            adjacency[record.start].append((record.end, record.kind))
            adjacency[record.end].append((record.start, record.kind))
        self._adjacency = {
            code: tuple(sorted(neighbors, key=lambda item: item[0]))
            for code, neighbors in adjacency.items()
        }
        self._neighbors_all = tuple(
            tuple(neighbor for neighbor, _kind in self._adjacency[code])
            for code in range(self.point_count)
        )
        self._neighbors_road = tuple(
            tuple(
                neighbor
                for neighbor, edge_kind in self._adjacency[code]
                if edge_kind is PathKind.ROAD
            )
            for code in range(self.point_count)
        )
        self._neighbors_railway = tuple(
            tuple(
                neighbor
                for neighbor, edge_kind in self._adjacency[code]
                if edge_kind is PathKind.RAILWAY
            )
            for code in range(self.point_count)
        )
        self._path_kind_lookup = {
            (code, neighbor): edge_kind
            for code, neighbors in self._adjacency.items()
            for neighbor, edge_kind in neighbors
        }
        self._railway_directions: dict[
            tuple[int, int], tuple[RailwayDirection, RailwayDirection]
        ] = {}
        for record in self.path_records:
            if record.kind is not PathKind.RAILWAY:
                continue
            first = self.decode(record.start)
            second = self.decode(record.end)
            self._railway_directions[(record.start, record.end)] = (
                self._railway_travel_directions(first, second)
            )
            self._railway_directions[(record.end, record.start)] = (
                self._railway_travel_directions(second, first)
            )
        self.action_space = _build_static_action_space(self.points, self._adjacency)
        # The source-major tuple is the canonical, stable action list used by
        # flat policy heads and serialized training samples.
        self.actions = self.action_space.actions

    def encode(self, point: PhysicalPoint) -> int:
        raise NotImplementedError

    def decode(self, code: int) -> PhysicalPoint:
        raise NotImplementedError

    def _relative_arm(self, point: ArmPoint) -> RelativeArm:
        raise NotImplementedError

    def _seat_back_direction(self, seat: int) -> RailwayDirection:
        raise NotImplementedError

    def _railway_travel_directions(
        self, first: PhysicalPoint, second: PhysicalPoint
    ) -> tuple[RailwayDirection, RailwayDirection]:
        """Return movement tangents at both ends of a directed rail segment."""

        if isinstance(first, CenterPoint) and isinstance(second, CenterPoint):
            if first.row == second.row:
                direction = (
                    RailwayDirection.EAST
                    if second.column > first.column
                    else RailwayDirection.WEST
                )
            elif first.column == second.column:
                direction = (
                    RailwayDirection.NORTH
                    if second.row > first.row
                    else RailwayDirection.SOUTH
                )
            else:
                raise BoardEncodingError(
                    f"central railway segment is not orthogonal: {first}, {second}"
                )
            return (direction, direction)

        if isinstance(first, ArmPoint) and isinstance(second, CenterPoint):
            direction = self._seat_back_direction(first.seat).opposite
            return (direction, direction)

        if isinstance(first, CenterPoint) and isinstance(second, ArmPoint):
            direction = self._seat_back_direction(second.seat)
            return (direction, direction)

        if not isinstance(first, ArmPoint) or not isinstance(second, ArmPoint):
            raise BoardEncodingError(
                f"unsupported railway endpoints: {first!r}, {second!r}"
            )

        first_back = self._seat_back_direction(first.seat)
        second_back = self._seat_back_direction(second.seat)
        if first.seat != second.seat:
            # Four-player side links are printed curves.  Two-player front
            # links are straight, for which these two directions are equal.
            return (first_back.opposite, second_back)

        if first.row == second.row:
            facing = first_back.opposite
            local_right = RailwayDirection((int(facing) + 1) % 4)
            direction = (
                local_right
                if second.column > first.column
                else local_right.opposite
            )
            return (direction, direction)

        if first.column == second.column:
            direction = (
                first_back if second.row > first.row else first_back.opposite
            )
            return (direction, direction)

        raise BoardEncodingError(
            f"arm railway segment is neither horizontal nor vertical: "
            f"{first}, {second}"
        )

    def _validate_code(self, code: int) -> None:
        if isinstance(code, bool) or not isinstance(code, int):
            raise BoardEncodingError(f"point code must be an integer: {code!r}")
        if not 0 <= code < self.point_count:
            raise BoardEncodingError(
                f"point code must be in 0..{self.point_count - 1}: {code}"
            )

    def point(self, code: int) -> PointRecord:
        cached = getattr(self, "points", None)
        if cached is not None:
            self._validate_code(code)
            return cached[code]
        physical = self.decode(code)
        relative_arm = (
            None if isinstance(physical, CenterPoint) else self._relative_arm(physical)
        )
        return PointRecord(code, physical, _point_kind(physical), relative_arm)

    def neighbors(
        self, code: int, kind: PathKind | None = None
    ) -> tuple[int, ...]:
        """Return adjacent point codes, optionally filtered by path kind."""

        self._validate_code(code)
        if kind is None:
            return self._neighbors_all[code]
        if kind is PathKind.ROAD:
            return self._neighbors_road[code]
        if kind is PathKind.RAILWAY:
            return self._neighbors_railway[code]
        return ()

    def path_kind(self, first: int, second: int) -> PathKind | None:
        """Return the printed segment kind, or None when points are not adjacent."""

        self._validate_code(first)
        self._validate_code(second)
        return self._path_kind_lookup.get((first, second))

    def railway_directions(
        self, first: int, second: int
    ) -> tuple[RailwayDirection, RailwayDirection] | None:
        """Return directed rail tangents, or ``None`` for a non-rail segment.

        A non-engineer may chain two railway segments only when the arrival
        direction of the first equals the departure direction of the second.
        """

        self._validate_code(first)
        self._validate_code(second)
        return self._railway_directions.get((first, second))

    def action(self, start: int, end: int) -> Action:
        """Validate and return an ordered endpoint pair for a move action.

        This validates the representation only.  Use ``action_space`` or
        ``action_index`` to require membership in the static vocabulary, and a
        dynamic mask to establish legality in a particular game state.
        """

        self._validate_code(start)
        self._validate_code(end)
        if start == end:
            raise BoardEncodingError("a move action requires different endpoints")
        return (start, end)

    def action_targets(self, start: int) -> tuple[int, ...]:
        """Return every destination that is ever possible from ``start``."""

        self._validate_code(start)
        return self.action_space.targets_from(start)

    def action_index(self, start: int, end: int) -> int:
        """Return the stable flat index for a statically possible action."""

        return self.action_space.action_index(start, end)

    def decode_action(self, index: int) -> Action:
        """Return the endpoint pair stored at a flat action-list index."""

        return self.action_space.action_at(index)


class FourPlayerBoard(_EncodedBoard):
    """The 129-point four-player board from one player's perspective."""

    point_count = 4 * POINTS_PER_ARM + CENTER_SIZE * CENTER_SIZE

    def __init__(self, viewer: FourPlayerSeat = FourPlayerSeat.SOUTH) -> None:
        try:
            self.viewer = FourPlayerSeat(viewer)
        except ValueError as error:
            raise BoardEncodingError(f"invalid four-player viewer: {viewer!r}") from error
        self._finish(_four_player_edges())

    def encode(self, point: PhysicalPoint) -> int:
        cached = getattr(self, "_encoded_points", None)
        if cached is not None:
            try:
                return cached[point]
            except (KeyError, TypeError):
                pass
        if isinstance(point, ArmPoint):
            try:
                seat = FourPlayerSeat(point.seat)
            except ValueError as error:
                raise BoardEncodingError(
                    f"invalid four-player arm seat: {point.seat!r}"
                ) from error
            relative_arm = (int(seat) - int(self.viewer)) % 4
            return relative_arm * POINTS_PER_ARM + _arm_offset(
                point.row, point.column
            )

        if isinstance(point, CenterPoint):
            row, column = _rotate_center(
                point.row, point.column, int(self.viewer)
            )
            return 4 * POINTS_PER_ARM + (row - 1) * CENTER_SIZE + (column - 1)

        raise BoardEncodingError(f"unsupported physical point: {point!r}")

    def decode(self, code: int) -> PhysicalPoint:
        self._validate_code(code)
        cached = getattr(self, "_decoded_points", None)
        if cached is not None:
            return cached[code]
        if code < 4 * POINTS_PER_ARM:
            relative_arm, offset = divmod(code, POINTS_PER_ARM)
            seat = FourPlayerSeat((int(self.viewer) + relative_arm) % 4)
            row, column = _row_column(offset)
            return ArmPoint(seat, row, column)

        center_offset = code - 4 * POINTS_PER_ARM
        relative_row, relative_column = (
            center_offset // CENTER_SIZE + 1,
            center_offset % CENTER_SIZE + 1,
        )
        absolute_row, absolute_column = _rotate_center(
            relative_row, relative_column, -int(self.viewer)
        )
        return CenterPoint(absolute_row, absolute_column)

    def _relative_arm(self, point: ArmPoint) -> FourPlayerRelativeArm:
        try:
            seat = FourPlayerSeat(point.seat)
        except ValueError as error:
            raise BoardEncodingError(
                f"invalid four-player arm seat: {point.seat!r}"
            ) from error
        return FourPlayerRelativeArm((int(seat) - int(self.viewer)) % 4)

    def _seat_back_direction(self, seat: int) -> RailwayDirection:
        try:
            return {
                FourPlayerSeat.SOUTH: RailwayDirection.SOUTH,
                FourPlayerSeat.WEST: RailwayDirection.WEST,
                FourPlayerSeat.NORTH: RailwayDirection.NORTH,
                FourPlayerSeat.EAST: RailwayDirection.EAST,
            }[FourPlayerSeat(seat)]
        except ValueError as error:
            raise BoardEncodingError(
                f"invalid four-player arm seat: {seat!r}"
            ) from error


class TwoPlayerBoard(_EncodedBoard):
    """The standard 60-point two-player Luzhanqi board."""

    point_count = 2 * POINTS_PER_ARM

    def __init__(self, viewer: TwoPlayerSeat = TwoPlayerSeat.SOUTH) -> None:
        try:
            self.viewer = TwoPlayerSeat(viewer)
        except ValueError as error:
            raise BoardEncodingError(f"invalid two-player viewer: {viewer!r}") from error
        self._finish(_two_player_edges())

    def encode(self, point: PhysicalPoint) -> int:
        cached = getattr(self, "_encoded_points", None)
        if cached is not None:
            try:
                return cached[point]
            except (KeyError, TypeError):
                pass
        if isinstance(point, CenterPoint):
            raise BoardEncodingError("the two-player board has no center points")
        if not isinstance(point, ArmPoint):
            raise BoardEncodingError(f"unsupported physical point: {point!r}")
        try:
            seat = TwoPlayerSeat(point.seat)
        except ValueError as error:
            raise BoardEncodingError(
                f"invalid two-player arm seat: {point.seat!r}"
            ) from error
        relative_arm = 0 if seat == self.viewer else 1
        return relative_arm * POINTS_PER_ARM + _arm_offset(point.row, point.column)

    def decode(self, code: int) -> ArmPoint:
        self._validate_code(code)
        cached = getattr(self, "_decoded_points", None)
        if cached is not None:
            point = cached[code]
            if not isinstance(point, ArmPoint):
                raise BoardEncodingError("two-player code decoded outside an arm")
            return point
        relative_arm, offset = divmod(code, POINTS_PER_ARM)
        seat = self.viewer if relative_arm == 0 else TwoPlayerSeat(1 - int(self.viewer))
        row, column = _row_column(offset)
        return ArmPoint(seat, row, column)

    def _relative_arm(self, point: ArmPoint) -> TwoPlayerRelativeArm:
        try:
            seat = TwoPlayerSeat(point.seat)
        except ValueError as error:
            raise BoardEncodingError(
                f"invalid two-player arm seat: {point.seat!r}"
            ) from error
        return (
            TwoPlayerRelativeArm.SELF
            if seat == self.viewer
            else TwoPlayerRelativeArm.OPPONENT
        )

    def _seat_back_direction(self, seat: int) -> RailwayDirection:
        try:
            return {
                TwoPlayerSeat.SOUTH: RailwayDirection.SOUTH,
                TwoPlayerSeat.NORTH: RailwayDirection.NORTH,
            }[TwoPlayerSeat(seat)]
        except ValueError as error:
            raise BoardEncodingError(
                f"invalid two-player arm seat: {seat!r}"
            ) from error


def _build_static_action_space(
    points: tuple[PointRecord, ...],
    adjacency: dict[int, tuple[tuple[int, PathKind], ...]],
) -> StaticActionSpace:
    """Build the union of moves possible for any movable piece and state."""

    # Every printed segment is traversable in one move in at least one state.
    targets = [
        {neighbor for neighbor, _kind in adjacency[code]}
        for code in range(len(points))
    ]

    # An engineer can turn at railway junctions.  Therefore every ordered pair
    # within one connected railway component is potentially a one-move action.
    railway_nodes = {
        code
        for code, neighbors in adjacency.items()
        if any(kind is PathKind.RAILWAY for _neighbor, kind in neighbors)
    }
    unseen = set(railway_nodes)
    while unseen:
        seed = unseen.pop()
        component = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor, kind in adjacency[current]:
                if kind is not PathKind.RAILWAY or neighbor in component:
                    continue
                component.add(neighbor)
                unseen.discard(neighbor)
                stack.append(neighbor)

        for start in component:
            targets[start].update(component - {start})

    # Headquarters lock is positional: no piece can ever move out, regardless
    # of its identity or of the rest of the board state.
    for point in points:
        if point.kind is PointKind.HEADQUARTERS:
            targets[point.code].clear()

    return StaticActionSpace(targets)


def _four_player_edges() -> PhysicalEdgeMap:
    edges: PhysicalEdgeMap = {}
    for seat in FourPlayerSeat:
        _add_arm_edges(edges, seat)

    # The central nine railway points form an orthogonal 3 x 3 grid.
    for row in range(1, CENTER_SIZE + 1):
        for column in range(1, CENTER_SIZE):
            _add_edge(
                edges,
                CenterPoint(row, column),
                CenterPoint(row, column + 1),
                PathKind.RAILWAY,
            )
    for row in range(1, CENTER_SIZE):
        for column in range(1, CENTER_SIZE + 1):
            _add_edge(
                edges,
                CenterPoint(row, column),
                CenterPoint(row + 1, column),
                PathKind.RAILWAY,
            )

    # Three railway entries from each arm: its two outer front points connect
    # diagonally to center corners, and its middle front point connects to the
    # middle point on that side of the central grid.
    center_entries: dict[FourPlayerSeat, dict[int, tuple[int, int]]] = {
        FourPlayerSeat.SOUTH: {1: (1, 1), 3: (1, 2), 5: (1, 3)},
        FourPlayerSeat.WEST: {1: (3, 1), 3: (2, 1), 5: (1, 1)},
        FourPlayerSeat.NORTH: {1: (3, 3), 3: (3, 2), 5: (3, 1)},
        FourPlayerSeat.EAST: {1: (1, 3), 3: (2, 3), 5: (3, 3)},
    }
    for seat, entries in center_entries.items():
        for arm_column, (center_row, center_column) in entries.items():
            _add_edge(
                edges,
                ArmPoint(seat, 1, arm_column),
                CenterPoint(center_row, center_column),
                PathKind.RAILWAY,
            )

    # At each corner of the central square, the adjacent arms' outer front
    # points also have a direct printed railway link.
    side_links = (
        (ArmPoint(FourPlayerSeat.SOUTH, 1, 1), ArmPoint(FourPlayerSeat.WEST, 1, 5)),
        (ArmPoint(FourPlayerSeat.WEST, 1, 1), ArmPoint(FourPlayerSeat.NORTH, 1, 5)),
        (ArmPoint(FourPlayerSeat.NORTH, 1, 1), ArmPoint(FourPlayerSeat.EAST, 1, 5)),
        (ArmPoint(FourPlayerSeat.EAST, 1, 1), ArmPoint(FourPlayerSeat.SOUTH, 1, 5)),
    )
    for start, end in side_links:
        _add_edge(edges, start, end, PathKind.RAILWAY)

    return edges


def _two_player_edges() -> PhysicalEdgeMap:
    edges: PhysicalEdgeMap = {}
    for seat in TwoPlayerSeat:
        _add_arm_edges(edges, seat)

    # The owners face one another, so physical left/right columns are reversed.
    for south_column in (1, 3, 5):
        _add_edge(
            edges,
            ArmPoint(TwoPlayerSeat.SOUTH, 1, south_column),
            ArmPoint(TwoPlayerSeat.NORTH, 1, ARM_COLUMNS + 1 - south_column),
            PathKind.RAILWAY,
        )
    return edges


def reencode_four_player(
    code: int,
    from_viewer: FourPlayerSeat,
    to_viewer: FourPlayerSeat,
) -> int:
    """Translate a four-player code between two player perspectives."""

    try:
        source = FourPlayerSeat(from_viewer)
        target = FourPlayerSeat(to_viewer)
    except ValueError as error:
        raise BoardEncodingError(
            f"invalid four-player viewer pair: {from_viewer!r}, {to_viewer!r}"
        ) from error
    if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code < 129:
        raise BoardEncodingError(f"point code must be in 0..128: {code!r}")

    if code < 4 * POINTS_PER_ARM:
        source_relative_arm, arm_offset = divmod(code, POINTS_PER_ARM)
        physical_seat = (int(source) + source_relative_arm) % 4
        target_relative_arm = (physical_seat - int(target)) % 4
        return target_relative_arm * POINTS_PER_ARM + arm_offset

    center_offset = code - 4 * POINTS_PER_ARM
    source_row = center_offset // CENTER_SIZE + 1
    source_column = center_offset % CENTER_SIZE + 1
    absolute_row, absolute_column = _rotate_center(
        source_row, source_column, -int(source)
    )
    target_row, target_column = _rotate_center(
        absolute_row, absolute_column, int(target)
    )
    return (
        4 * POINTS_PER_ARM
        + (target_row - 1) * CENTER_SIZE
        + (target_column - 1)
    )


def reencode_two_player(
    code: int,
    from_viewer: TwoPlayerSeat,
    to_viewer: TwoPlayerSeat,
) -> int:
    """Translate a two-player code between the two player perspectives."""

    try:
        source = TwoPlayerSeat(from_viewer)
        target = TwoPlayerSeat(to_viewer)
    except ValueError as error:
        raise BoardEncodingError(
            f"invalid two-player viewer pair: {from_viewer!r}, {to_viewer!r}"
        ) from error
    if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code < 60:
        raise BoardEncodingError(f"point code must be in 0..59: {code!r}")

    source_relative_arm, arm_offset = divmod(code, POINTS_PER_ARM)
    physical_seat = (
        int(source) if source_relative_arm == 0 else 1 - int(source)
    )
    target_relative_arm = 0 if physical_seat == int(target) else 1
    return target_relative_arm * POINTS_PER_ARM + arm_offset
