from __future__ import annotations

import sys
import unittest
from collections import Counter, deque
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from junqi.board import (  # noqa: E402
    ArmPoint,
    BoardEncodingError,
    CenterPoint,
    FourPlayerBoard,
    FourPlayerRelativeArm,
    FourPlayerSeat,
    PathKind,
    PointKind,
    TwoPlayerBoard,
    TwoPlayerRelativeArm,
    TwoPlayerSeat,
    reencode_four_player,
    reencode_two_player,
)


def assert_connected(test_case: unittest.TestCase, board: object) -> None:
    seen = {0}
    queue = deque([0])
    while queue:
        current = queue.popleft()
        for neighbor in board.neighbors(current):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    test_case.assertEqual(len(seen), board.point_count)


class FourPlayerBoardTests(unittest.TestCase):
    def test_point_count_kinds_and_round_trip_for_every_viewer(self) -> None:
        expected_kinds = Counter(
            {
                PointKind.STATION: 92,
                PointKind.CAMP: 20,
                PointKind.HEADQUARTERS: 8,
                PointKind.CENTRAL_STATION: 9,
            }
        )
        for viewer in FourPlayerSeat:
            with self.subTest(viewer=viewer):
                board = FourPlayerBoard(viewer)
                self.assertEqual(board.point_count, 129)
                self.assertEqual([point.code for point in board.points], list(range(129)))
                self.assertEqual(Counter(point.kind for point in board.points), expected_kinds)
                for point in board.points:
                    self.assertEqual(board.decode(board.encode(point.physical)), point.physical)

    def test_spatial_relative_arm_ranges(self) -> None:
        board = FourPlayerBoard(FourPlayerSeat.SOUTH)
        self.assertEqual(board.encode(ArmPoint(FourPlayerSeat.SOUTH, 1, 1)), 0)
        self.assertEqual(board.encode(ArmPoint(FourPlayerSeat.WEST, 1, 1)), 30)
        self.assertEqual(board.encode(ArmPoint(FourPlayerSeat.NORTH, 1, 1)), 60)
        self.assertEqual(board.encode(ArmPoint(FourPlayerSeat.EAST, 1, 1)), 90)
        self.assertEqual(board.point(30).relative_arm, FourPlayerRelativeArm.LEFT_OPPONENT)
        self.assertEqual(board.point(60).relative_arm, FourPlayerRelativeArm.PARTNER)
        self.assertEqual(board.point(90).relative_arm, FourPlayerRelativeArm.RIGHT_OPPONENT)

    def test_center_rotates_with_viewer(self) -> None:
        south_west_center = CenterPoint(1, 1)
        expected_codes = {
            FourPlayerSeat.SOUTH: 120,
            FourPlayerSeat.WEST: 122,
            FourPlayerSeat.NORTH: 128,
            FourPlayerSeat.EAST: 126,
        }
        for viewer, expected in expected_codes.items():
            with self.subTest(viewer=viewer):
                board = FourPlayerBoard(viewer)
                self.assertEqual(board.encode(south_west_center), expected)
                self.assertEqual(board.decode(expected), south_west_center)

    def test_paths_are_unique_typed_and_connected(self) -> None:
        board = FourPlayerBoard()
        self.assertEqual(len(board.paths), 288)
        self.assertEqual(len(board.directed_paths), 576)
        self.assertEqual(len(set(board.paths)), len(board.paths))
        self.assertTrue(all(start < end for start, end in board.paths))
        self.assertTrue(
            all(0 <= start < 129 and 0 <= end < 129 for start, end in board.paths)
        )
        self.assertEqual(
            Counter(record.kind for record in board.path_records),
            Counter({PathKind.ROAD: 196, PathKind.RAILWAY: 92}),
        )
        self.assertEqual(board.path_kind(0, 1), PathKind.RAILWAY)
        self.assertEqual(board.path_kind(0, 120), PathKind.RAILWAY)
        self.assertEqual(board.path_kind(0, 34), PathKind.RAILWAY)
        self.assertEqual(board.path_kind(120, 121), PathKind.RAILWAY)
        assert_connected(self, board)

    def test_reencoding_is_reversible(self) -> None:
        for source in FourPlayerSeat:
            for target in FourPlayerSeat:
                for code in range(129):
                    translated = reencode_four_player(code, source, target)
                    restored = reencode_four_player(translated, target, source)
                    self.assertEqual(restored, code)

    def test_static_action_space_filters_impossible_endpoint_pairs(self) -> None:
        board = FourPlayerBoard()
        space = board.action_space
        headquarters = {
            point.code
            for point in board.points
            if point.kind is PointKind.HEADQUARTERS
        }

        self.assertEqual(space.action_count, 5625)
        self.assertEqual(space.origin_count, 121)
        self.assertEqual(space.max_destination_count, 76)
        self.assertEqual(len(board.actions), 5625)
        self.assertEqual(
            set(space.origin_codes), set(range(board.point_count)) - headquarters
        )
        self.assertTrue(all(start not in headquarters for start, _end in board.actions))

        # A camp is not a railway point, so even an engineer can only use one
        # of its adjacent printed road segments.
        camp = 6
        self.assertEqual(board.action_targets(camp), board.neighbors(camp))
        self.assertNotIn(120, board.action_targets(camp))

        # A railway engineer can cross and turn through the connected network.
        self.assertIsNone(board.path_kind(0, 127))
        self.assertIn(127, board.action_targets(0))

        # Headquarters can be entered, but no piece can move back out.
        self.assertIn(26, board.action_targets(21))
        self.assertEqual(board.action_targets(26), ())

    def test_static_action_indices_are_stable_and_reversible(self) -> None:
        board = FourPlayerBoard()
        space = board.action_space
        self.assertEqual(board.actions, tuple(sorted(board.actions)))

        for index, action in enumerate(board.actions):
            start, end = action
            self.assertEqual(board.action_index(start, end), index)
            self.assertEqual(board.decode_action(index), action)
            destination_slot = space.destination_slot(start, end)
            self.assertEqual(space.destination_at(start, destination_slot), end)

        for origin_index, code in enumerate(space.origin_codes):
            self.assertEqual(space.origin_index(code), origin_index)
            self.assertEqual(space.origin_at(origin_index), code)

    def test_static_action_codes_are_the_same_in_every_player_view(self) -> None:
        expected = FourPlayerBoard(FourPlayerSeat.SOUTH).actions
        for viewer in FourPlayerSeat:
            with self.subTest(viewer=viewer):
                self.assertEqual(FourPlayerBoard(viewer).actions, expected)


class TwoPlayerBoardTests(unittest.TestCase):
    def test_point_count_kinds_and_round_trip_for_both_viewers(self) -> None:
        expected_kinds = Counter(
            {
                PointKind.STATION: 46,
                PointKind.CAMP: 10,
                PointKind.HEADQUARTERS: 4,
            }
        )
        for viewer in TwoPlayerSeat:
            with self.subTest(viewer=viewer):
                board = TwoPlayerBoard(viewer)
                self.assertEqual(board.point_count, 60)
                self.assertEqual([point.code for point in board.points], list(range(60)))
                self.assertEqual(Counter(point.kind for point in board.points), expected_kinds)
                for point in board.points:
                    self.assertEqual(board.decode(board.encode(point.physical)), point.physical)

    def test_relative_ranges_and_front_connections(self) -> None:
        board = TwoPlayerBoard(TwoPlayerSeat.SOUTH)
        self.assertEqual(board.encode(ArmPoint(TwoPlayerSeat.SOUTH, 1, 1)), 0)
        self.assertEqual(board.encode(ArmPoint(TwoPlayerSeat.NORTH, 1, 1)), 30)
        self.assertEqual(board.point(0).relative_arm, TwoPlayerRelativeArm.SELF)
        self.assertEqual(board.point(30).relative_arm, TwoPlayerRelativeArm.OPPONENT)
        self.assertEqual(board.path_kind(0, 34), PathKind.RAILWAY)
        self.assertEqual(board.path_kind(2, 32), PathKind.RAILWAY)
        self.assertEqual(board.path_kind(4, 30), PathKind.RAILWAY)

    def test_paths_are_unique_typed_and_connected(self) -> None:
        board = TwoPlayerBoard()
        self.assertEqual(len(board.paths), 133)
        self.assertEqual(len(board.directed_paths), 266)
        self.assertEqual(len(set(board.paths)), len(board.paths))
        self.assertTrue(all(start < end for start, end in board.paths))
        self.assertEqual(
            Counter(record.kind for record in board.path_records),
            Counter({PathKind.ROAD: 98, PathKind.RAILWAY: 35}),
        )
        assert_connected(self, board)

    def test_reencoding_is_reversible(self) -> None:
        for source in TwoPlayerSeat:
            for target in TwoPlayerSeat:
                for code in range(60):
                    translated = reencode_two_player(code, source, target)
                    restored = reencode_two_player(translated, target, source)
                    self.assertEqual(restored, code)

    def test_center_is_rejected(self) -> None:
        with self.assertRaises(BoardEncodingError):
            TwoPlayerBoard().encode(CenterPoint(2, 2))

    def test_static_action_space_counts_and_boundaries(self) -> None:
        board = TwoPlayerBoard()
        space = board.action_space

        self.assertEqual(space.action_count, 1177)
        self.assertEqual(space.origin_count, 56)
        self.assertEqual(space.max_destination_count, 35)
        self.assertEqual(board.action_targets(6), board.neighbors(6))
        self.assertIn(50, board.action_targets(0))
        self.assertEqual(board.action_targets(26), ())
        self.assertEqual(
            TwoPlayerBoard(TwoPlayerSeat.NORTH).actions,
            board.actions,
        )


class ActionPairTests(unittest.TestCase):
    def test_action_preserves_direction(self) -> None:
        board = FourPlayerBoard()
        self.assertEqual(board.action(7, 120), (7, 120))
        self.assertEqual(board.action(120, 7), (120, 7))

    def test_action_rejects_self_move_and_out_of_range(self) -> None:
        board = TwoPlayerBoard()
        with self.assertRaises(BoardEncodingError):
            board.action(3, 3)
        with self.assertRaises(BoardEncodingError):
            board.action(0, 60)

    def test_static_action_encoding_rejects_impossible_moves(self) -> None:
        board = TwoPlayerBoard()
        with self.assertRaises(BoardEncodingError):
            board.action_index(7, 32)
        with self.assertRaises(BoardEncodingError):
            board.action_index(26, 21)
        with self.assertRaises(BoardEncodingError):
            board.action_space.origin_index(26)


if __name__ == "__main__":
    unittest.main()
