from __future__ import annotations

import random
import sys
import unittest
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from junqi.pieces import (  # noqa: E402
    CAMP_COORDINATES,
    HEADQUARTERS_COORDINATES,
    LAYOUT_ORDER,
    PIECE_COUNTS,
    PIECE_RANKS,
    PIECE_TYPE_INDICES,
    PIECE_TYPE_ORDER,
    SETUP_COORDINATES,
    LayoutBuilder,
    Piece,
    PieceType,
    PlayerSetup,
    SetupError,
)


class PieceTests(unittest.TestCase):
    def test_inventory_has_25_pieces_and_expected_mobility(self) -> None:
        self.assertEqual(sum(PIECE_COUNTS.values()), 25)
        self.assertFalse(Piece(0, PieceType.FLAG).movable)
        self.assertFalse(Piece(0, PieceType.MINE).movable)
        self.assertTrue(Piece(0, PieceType.BOMB).movable)
        self.assertEqual(PIECE_RANKS[PieceType.COMMANDER], 9)
        self.assertEqual(PIECE_RANKS[PieceType.ENGINEER], 1)

    def test_moved_returns_an_immutable_updated_piece(self) -> None:
        piece = Piece(2, PieceType.ENGINEER)
        moved = piece.moved()
        self.assertFalse(piece.has_moved)
        self.assertTrue(moved.has_moved)
        self.assertIs(moved.moved(), moved)


class PlayerSetupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.setup = PlayerSetup.random(random.Random(17))

    def test_random_setups_obey_every_fixed_constraint(self) -> None:
        for seed in range(100):
            with self.subTest(seed=seed):
                setup = PlayerSetup.random(random.Random(seed))
                self.assertEqual(set(setup), SETUP_COORDINATES)
                self.assertEqual(Counter(setup.values()), Counter(PIECE_COUNTS))
                self.assertTrue(
                    all(coordinate not in setup for coordinate in CAMP_COORDINATES)
                )
                flag = next(
                    coordinate
                    for coordinate, kind in setup.items()
                    if kind is PieceType.FLAG
                )
                self.assertIn(flag, HEADQUARTERS_COORDINATES)
                self.assertTrue(
                    all(
                        coordinate[0] in (5, 6)
                        for coordinate, kind in setup.items()
                        if kind is PieceType.MINE
                    )
                )
                self.assertTrue(
                    all(
                        coordinate[0] != 1
                        for coordinate, kind in setup.items()
                        if kind is PieceType.BOMB
                    )
                )

    def test_rows_round_trip_and_require_empty_camps(self) -> None:
        restored = PlayerSetup.from_rows(self.setup.to_rows())
        self.assertEqual(dict(restored), dict(self.setup))

        rows = [list(row) for row in self.setup.to_rows()]
        rows[1][1] = PieceType.ENGINEER
        with self.assertRaises(SetupError):
            PlayerSetup.from_rows(rows)

    def test_rejects_flag_outside_headquarters(self) -> None:
        placements = dict(self.setup)
        flag = next(point for point, kind in placements.items() if kind is PieceType.FLAG)
        replacement = next(
            point
            for point, kind in placements.items()
            if point not in HEADQUARTERS_COORDINATES and kind is not PieceType.FLAG
        )
        placements[flag], placements[replacement] = (
            placements[replacement],
            placements[flag],
        )
        with self.assertRaisesRegex(SetupError, "flag must be placed"):
            PlayerSetup(placements)

    def test_rejects_mine_outside_last_two_rows(self) -> None:
        placements = dict(self.setup)
        mine = next(point for point, kind in placements.items() if kind is PieceType.MINE)
        replacement = next(
            point
            for point, kind in placements.items()
            if point[0] in (2, 3, 4)
            and kind not in (PieceType.MINE, PieceType.FLAG)
        )
        placements[mine], placements[replacement] = (
            placements[replacement],
            placements[mine],
        )
        with self.assertRaisesRegex(SetupError, "mines must be placed"):
            PlayerSetup(placements)

    def test_rejects_bomb_on_front_row(self) -> None:
        placements = dict(self.setup)
        bomb = next(point for point, kind in placements.items() if kind is PieceType.BOMB)
        replacement = next(
            point
            for point, kind in placements.items()
            if point[0] == 1 and kind is not PieceType.BOMB
        )
        placements[bomb], placements[replacement] = (
            placements[replacement],
            placements[bomb],
        )
        with self.assertRaisesRegex(SetupError, "bombs cannot be placed"):
            PlayerSetup(placements)

    def test_rejects_wrong_inventory_and_camp_placement(self) -> None:
        placements = dict(self.setup)
        commander = next(
            point for point, kind in placements.items() if kind is PieceType.COMMANDER
        )
        placements[commander] = PieceType.ENGINEER
        with self.assertRaisesRegex(SetupError, "incorrect piece inventory"):
            PlayerSetup(placements)

        placements = dict(self.setup)
        placements[(2, 2)] = placements.pop((1, 1))
        with self.assertRaisesRegex(SetupError, "25 non-camp points"):
            PlayerSetup(placements)


class LayoutBuilderTests(unittest.TestCase):
    def test_layout_order_covers_every_setup_point_once(self) -> None:
        self.assertEqual(len(LAYOUT_ORDER), 25)
        self.assertEqual(len(set(LAYOUT_ORDER)), 25)
        self.assertEqual(set(LAYOUT_ORDER), SETUP_COORDINATES)
        self.assertEqual(LAYOUT_ORDER[:2], ((6, 2), (6, 4)))

    def test_future_feasibility_forces_flag_into_second_headquarters(self) -> None:
        builder = LayoutBuilder()
        builder.place(PieceType.COMMANDER)
        self.assertEqual(builder.next_coordinate, (6, 4))
        self.assertEqual(builder.legal_piece_types(), (PieceType.FLAG,))
        with self.assertRaises(SetupError):
            builder.place(PieceType.ENGINEER)

    def test_front_row_mask_excludes_flag_mine_and_bomb(self) -> None:
        builder = LayoutBuilder()
        builder.place(PieceType.FLAG)
        builder.place(PieceType.COMMANDER)
        self.assertEqual(builder.next_coordinate, (1, 1))
        legal = set(builder.legal_piece_types())
        self.assertNotIn(PieceType.FLAG, legal)
        self.assertNotIn(PieceType.MINE, legal)
        self.assertNotIn(PieceType.BOMB, legal)
        mask = builder.legal_piece_mask()
        self.assertEqual(len(mask), 12)
        self.assertFalse(mask[PIECE_TYPE_INDICES[PieceType.BOMB]])

    def test_piece_index_interface_uses_stable_enum_order(self) -> None:
        builder = LayoutBuilder()
        flag_index = PIECE_TYPE_INDICES[PieceType.FLAG]
        self.assertEqual(PIECE_TYPE_ORDER[flag_index], PieceType.FLAG)
        self.assertEqual(builder.place_index(flag_index), (6, 2))

    def test_masked_sampling_always_completes_a_legal_layout(self) -> None:
        for seed in range(10):
            with self.subTest(seed=seed):
                setup = LayoutBuilder.sample(random.Random(seed))
                self.assertEqual(Counter(setup.values()), Counter(PIECE_COUNTS))

    def test_clone_is_independent_and_incomplete_layout_cannot_build(self) -> None:
        builder = LayoutBuilder()
        with self.assertRaisesRegex(SetupError, "incomplete"):
            builder.build()
        clone = builder.clone()
        clone.place(PieceType.FLAG)
        self.assertEqual(builder.step_index, 0)
        self.assertEqual(clone.step_index, 1)


if __name__ == "__main__":
    unittest.main()
