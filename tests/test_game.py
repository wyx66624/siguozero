from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from junqi import (  # noqa: E402
    ArmPoint,
    CombatOutcome,
    FourPlayerBoard,
    GameConfig,
    GameResult,
    GameRuleError,
    GameVariant,
    IllegalActionError,
    InformationMode,
    JunqiGame,
    PIECE_COUNTS,
    Piece,
    PieceType,
    RailwayDirection,
    TerminalGameError,
    TerminationReason,
)


def flags(player_count: int, *, first_flag_column: int = 2) -> dict[ArmPoint, Piece]:
    return {
        ArmPoint(owner, 6, first_flag_column if owner == 0 else 2): Piece(
            owner, PieceType.FLAG
        )
        for owner in range(player_count)
    }


def two_config(
    *,
    draw_plies: int = 60,
    max_plies: int | None = None,
    mode: InformationMode = InformationMode.OPEN,
    dead_rules_enabled: bool = True,
) -> GameConfig:
    return GameConfig(
        variant=GameVariant.TWO_PLAYER,
        information_mode=mode,
        dead_rules_enabled=dead_rules_enabled,
        no_interaction_draw_plies=draw_plies,
        max_plies=max_plies,
    )


def four_config(
    *, mode: InformationMode = InformationMode.FULL_OPEN
) -> GameConfig:
    return GameConfig(
        variant=GameVariant.FOUR_PLAYER,
        information_mode=mode,
    )


class BoardRailwayDirectionTests(unittest.TestCase):
    def test_straight_and_curved_railway_tangents(self) -> None:
        board = FourPlayerBoard()
        self.assertEqual(
            board.railway_directions(5, 0),
            (RailwayDirection.NORTH, RailwayDirection.NORTH),
        )
        self.assertEqual(
            board.railway_directions(0, 120),
            (RailwayDirection.NORTH, RailwayDirection.NORTH),
        )
        self.assertEqual(
            board.railway_directions(0, 34),
            (RailwayDirection.NORTH, RailwayDirection.WEST),
        )
        self.assertEqual(
            board.railway_directions(34, 0),
            (RailwayDirection.EAST, RailwayDirection.SOUTH),
        )
        self.assertIsNone(board.railway_directions(6, 7))

    def test_reverse_railway_tangents_are_opposites_for_every_edge(self) -> None:
        for board in (FourPlayerBoard(), JunqiGame.new_two_player(seed=1).board_for(0)):
            for record in board.path_records:
                if record.kind.value != "railway":
                    continue
                forward = board.railway_directions(record.start, record.end)
                reverse = board.railway_directions(record.end, record.start)
                self.assertEqual(
                    reverse,
                    (forward[1].opposite, forward[0].opposite),
                )


class GameInitializationTests(unittest.TestCase):
    def test_seeded_random_games_are_ready_for_training(self) -> None:
        four = JunqiGame.new_four_player(seed=11)
        two = JunqiGame.new_two_player(seed=11)

        self.assertEqual(len(four.pieces), 100)
        self.assertEqual(len(two.pieces), 50)
        self.assertGreater(len(four.legal_actions()), 0)
        self.assertGreater(len(two.legal_actions()), 0)
        self.assertEqual(len(four.legal_action_mask()), 5624)
        self.assertEqual(len(two.legal_action_mask()), 1176)
        self.assertEqual(len(four.legal_origin_mask()), 121)
        self.assertEqual(len(two.legal_origin_mask()), 56)
        self.assertEqual(
            sum(four.legal_action_mask()), len(four.legal_actions())
        )
        self.assertEqual(sum(two.legal_action_mask()), len(two.legal_actions()))

        start = two.legal_actions()[0][0]
        expected_targets = {
            end for action_start, end in two.legal_actions() if action_start == start
        }
        self.assertEqual(sum(two.legal_destination_mask(start)), len(expected_targets))
        self.assertEqual(len(two.legal_destination_mask(start)), 35)

    def test_seeded_games_are_reproducible(self) -> None:
        first = JunqiGame.new_four_player(seed=23)
        second = JunqiGame.new_four_player(seed=23)
        self.assertEqual(first.state_key(), second.state_key())

    def test_restored_position_requires_each_active_players_flag(self) -> None:
        pieces = {ArmPoint(0, 6, 2): Piece(0, PieceType.FLAG)}
        with self.assertRaisesRegex(GameRuleError, "exactly one flag"):
            JunqiGame.from_position(two_config(), pieces)

    def test_unrevealed_flag_requires_a_living_commander(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(0, 2, 3)] = Piece(0, PieceType.ENGINEER)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.COMMANDER)
        with self.assertRaisesRegex(GameRuleError, "flag must be revealed"):
            JunqiGame.from_position(
                two_config(),
                pieces,
                revealed_flags=(False, False),
            )


class MovementRuleTests(unittest.TestCase):
    def test_road_piece_moves_one_printed_segment_only(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(0, 2, 3)] = Piece(0, PieceType.COMPANY_COMMANDER)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.COMPANY_COMMANDER)
        game = JunqiGame.from_position(two_config(), pieces)

        self.assertTrue(game.is_legal_action((7, 6)))
        self.assertTrue(game.is_legal_action((7, 2)))
        self.assertFalse(game.is_legal_action((7, 17)))

    def test_camp_can_only_reach_adjacent_points_and_cannot_be_attacked(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(0, 2, 3)] = Piece(0, PieceType.ENGINEER)
        pieces[ArmPoint(0, 2, 2)] = Piece(1, PieceType.COMPANY_COMMANDER)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.ENGINEER)
        game = JunqiGame.from_position(two_config(), pieces)

        self.assertFalse(game.is_legal_action((7, 6)))
        self.assertFalse(game.is_legal_action((6, 32)))

        empty_camp_pieces = dict(pieces)
        del empty_camp_pieces[ArmPoint(0, 2, 2)]
        game = JunqiGame.from_position(two_config(), empty_camp_pieces)
        self.assertTrue(game.is_legal_action((7, 6)))

    def test_headquarters_can_be_entered_but_never_left(self) -> None:
        pieces = flags(2, first_flag_column=4)
        pieces[ArmPoint(0, 5, 2)] = Piece(0, PieceType.ENGINEER)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.ENGINEER)
        game = JunqiGame.from_position(two_config(), pieces)

        self.assertTrue(game.is_legal_action((21, 26)))
        outcome = game.step((21, 26))
        self.assertEqual(outcome.combat, CombatOutcome.MOVE)
        entered_piece = game.piece_at(ArmPoint(0, 6, 2))
        self.assertIsNotNone(entered_piece)
        self.assertTrue(entered_piece.has_moved)
        self.assertNotIn((26, 21), game.legal_actions(0))

    def test_non_engineer_uses_straight_and_curved_rail_but_cannot_turn(self) -> None:
        pieces = flags(4)
        pieces[ArmPoint(0, 5, 1)] = Piece(0, PieceType.COMPANY_COMMANDER)
        for owner in (1, 2, 3):
            pieces[ArmPoint(owner, 2, 3)] = Piece(
                owner, PieceType.COMPANY_COMMANDER
            )
        game = JunqiGame.from_position(four_config(), pieces)

        self.assertTrue(game.is_legal_action((20, 126)))
        self.assertTrue(game.is_legal_action((20, 39)))
        self.assertFalse(game.is_legal_action((20, 2)))

    def test_engineer_can_turn_anywhere_on_an_unblocked_railway(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(0, 5, 1)] = Piece(0, PieceType.ENGINEER)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.COMPANY_COMMANDER)
        game = JunqiGame.from_position(two_config(), pieces)

        self.assertTrue(game.is_legal_action((20, 2)))
        self.assertTrue(game.is_legal_action((20, 50)))

    def test_occupied_railway_points_block_passage(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(0, 5, 1)] = Piece(0, PieceType.COMPANY_COMMANDER)
        pieces[ArmPoint(0, 3, 1)] = Piece(0, PieceType.PLATOON_COMMANDER)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.COMPANY_COMMANDER)
        game = JunqiGame.from_position(two_config(), pieces)

        self.assertFalse(game.is_legal_action((20, 5)))
        self.assertFalse(game.is_legal_action((20, 10)))

    def test_four_player_cannot_attack_partner_piece(self) -> None:
        pieces = flags(4)
        pieces[ArmPoint(0, 2, 3)] = Piece(0, PieceType.COMMANDER)
        pieces[ArmPoint(0, 1, 3)] = Piece(2, PieceType.ENGINEER)
        for owner in (1, 3):
            pieces[ArmPoint(owner, 2, 3)] = Piece(owner, PieceType.ENGINEER)
        game = JunqiGame.from_position(four_config(), pieces)

        self.assertTrue(game.are_allies(0, 2))
        self.assertFalse(game.is_legal_action((7, 2)))


class CombatRuleTests(unittest.TestCase):
    def _game_for_adjacent_combat(
        self, attacker: PieceType, defender: PieceType
    ) -> JunqiGame:
        pieces = flags(2)
        pieces[ArmPoint(0, 2, 3)] = Piece(0, attacker)
        pieces[ArmPoint(0, 1, 3)] = Piece(1, defender)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.ENGINEER)
        return JunqiGame.from_position(two_config(), pieces)

    def test_higher_rank_wins_and_advances(self) -> None:
        game = self._game_for_adjacent_combat(
            PieceType.COMMANDER, PieceType.ARMY_COMMANDER
        )
        result = game.step((7, 2))
        self.assertEqual(result.combat, CombatOutcome.ATTACKER_WINS)
        survivor = game.piece_at(ArmPoint(0, 1, 3))
        self.assertEqual(survivor.kind, PieceType.COMMANDER)
        self.assertTrue(survivor.has_moved)

    def test_lower_rank_loses_and_equal_ranks_both_leave(self) -> None:
        game = self._game_for_adjacent_combat(
            PieceType.ENGINEER, PieceType.COMMANDER
        )
        result = game.step((7, 2))
        self.assertEqual(result.combat, CombatOutcome.DEFENDER_WINS)
        self.assertEqual(
            game.piece_at(ArmPoint(0, 1, 3)).kind, PieceType.COMMANDER
        )

        game = self._game_for_adjacent_combat(
            PieceType.COMPANY_COMMANDER, PieceType.COMPANY_COMMANDER
        )
        result = game.step((7, 2))
        self.assertEqual(result.combat, CombatOutcome.BOTH_REMOVED)
        self.assertIsNone(game.piece_at(ArmPoint(0, 1, 3)))

    def test_engineer_clears_mine_and_normal_piece_dies_to_mine(self) -> None:
        for attacker, expected in (
            (PieceType.ENGINEER, CombatOutcome.ATTACKER_WINS),
            (PieceType.COMPANY_COMMANDER, CombatOutcome.DEFENDER_WINS),
        ):
            with self.subTest(attacker=attacker):
                pieces = flags(2)
                pieces[ArmPoint(1, 5, 1)] = Piece(0, attacker)
                pieces[ArmPoint(1, 5, 2)] = Piece(1, PieceType.MINE)
                pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.ENGINEER)
                game = JunqiGame.from_position(two_config(), pieces)
                result = game.step((50, 51))
                self.assertEqual(result.combat, expected)

    def test_bomb_removes_both_sides_and_commander_death_reveals_flag(self) -> None:
        game = self._game_for_adjacent_combat(
            PieceType.COMMANDER, PieceType.BOMB
        )
        result = game.step((7, 2))
        self.assertEqual(result.combat, CombatOutcome.BOTH_REMOVED)
        self.assertTrue(game.revealed_flags[0])
        self.assertEqual(result.newly_revealed_flags, (0,))
        self.assertEqual(game.public_history[-1].newly_revealed_flags, (0,))

    def test_capturing_flag_eliminates_player_and_ends_two_player_game(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(1, 5, 2)] = Piece(0, PieceType.ENGINEER)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.COMMANDER)
        game = JunqiGame.from_position(two_config(), pieces)

        result = game.step((51, 56))
        self.assertEqual(result.combat, CombatOutcome.ATTACKER_WINS)
        self.assertEqual(result.flag_captured_owner, 1)
        self.assertEqual(result.eliminated_players, (1,))
        self.assertEqual(
            game.result,
            GameResult(TerminationReason.TEAM_ELIMINATED, winner_team=0),
        )
        self.assertEqual(game.rewards(), (1.0, -1.0))
        self.assertFalse(game.active_players[1])
        self.assertTrue(all(piece.owner != 1 for piece in game.pieces.values()))

    def test_bomb_and_flag_are_both_removed_but_flag_owner_still_loses(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(1, 5, 2)] = Piece(0, PieceType.BOMB)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.ENGINEER)
        game = JunqiGame.from_position(two_config(), pieces)

        result = game.step((51, 56))
        self.assertEqual(result.combat, CombatOutcome.BOTH_REMOVED)
        self.assertEqual(result.flag_captured_owner, 1)
        self.assertEqual(game.result.winner_team, 0)
        self.assertIsNone(game.piece_at(ArmPoint(1, 6, 2)))

    def test_one_four_player_flag_capture_does_not_end_match_if_partner_lives(self) -> None:
        pieces = flags(4)
        pieces[ArmPoint(1, 5, 2)] = Piece(0, PieceType.ENGINEER)
        for owner in (1, 2, 3):
            pieces[ArmPoint(owner, 2, 3)] = Piece(owner, PieceType.COMMANDER)
        game = JunqiGame.from_position(four_config(), pieces)

        result = game.step((51, 56))
        self.assertEqual(result.flag_captured_owner, 1)
        self.assertFalse(game.active_players[1])
        self.assertTrue(game.active_players[3])
        self.assertIsNone(game.result)
        self.assertEqual(game.current_player, 3)


class LifecycleAndTrainingTests(unittest.TestCase):
    def test_turn_order_is_counterclockwise_for_four_players(self) -> None:
        pieces = flags(4)
        for owner in range(4):
            pieces[ArmPoint(owner, 2, 3)] = Piece(owner, PieceType.ENGINEER)
        game = JunqiGame.from_position(four_config(), pieces)

        game.step((7, 6))
        self.assertEqual(game.current_player, 3)
        self.assertIn((7, 6), game.legal_actions())

    def test_player_with_no_move_is_eliminated_when_turn_arrives(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(0, 2, 3)] = Piece(0, PieceType.ENGINEER)
        game = JunqiGame.from_position(two_config(), pieces)

        result = game.step((7, 6))
        self.assertEqual(result.eliminated_players, (1,))
        self.assertTrue(game.is_terminal)
        self.assertEqual(game.result.winner_team, 0)

    def test_restoring_one_surviving_team_is_immediately_terminal(self) -> None:
        pieces = {
            ArmPoint(0, 6, 2): Piece(0, PieceType.FLAG),
            ArmPoint(0, 2, 3): Piece(0, PieceType.ENGINEER),
        }
        game = JunqiGame.from_position(
            two_config(),
            pieces,
            active_players=(True, False),
        )
        self.assertTrue(game.is_terminal)
        self.assertEqual(game.result.winner_team, 0)

    def test_no_interaction_and_max_ply_draws(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(0, 2, 3)] = Piece(0, PieceType.ENGINEER)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.ENGINEER)

        game = JunqiGame.from_position(
            two_config(draw_plies=60),
            pieces,
            no_interaction_plies=59,
        )
        game.step((7, 6))
        self.assertEqual(
            game.result.reason, TerminationReason.NO_INTERACTION_DRAW
        )
        self.assertEqual(game.rewards(), (0.0, 0.0))

        game = JunqiGame.from_position(
            two_config(max_plies=1),
            pieces,
        )
        game.step((7, 6))
        self.assertEqual(game.result.reason, TerminationReason.MAX_PLIES_DRAW)

    def test_combat_resets_no_interaction_counter(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(0, 2, 3)] = Piece(0, PieceType.COMMANDER)
        pieces[ArmPoint(0, 1, 3)] = Piece(1, PieceType.ENGINEER)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.COMMANDER)
        game = JunqiGame.from_position(
            two_config(), pieces, no_interaction_plies=59
        )

        game.step((7, 2))
        self.assertEqual(game.no_interaction_plies, 0)
        self.assertIsNone(game.result)

    def test_step_accepts_flat_action_index_and_clone_is_independent(self) -> None:
        game = JunqiGame.new_two_player(seed=5)
        original_key = game.state_key()
        clone = game.clone()
        action_index = clone.legal_action_indices()[0]
        clone.step(action_index)

        self.assertEqual(game.state_key(), original_key)
        self.assertNotEqual(clone.state_key(), original_key)

    def test_illegal_and_terminal_actions_raise_clear_errors(self) -> None:
        game = JunqiGame.new_two_player(seed=7, max_plies=1)
        with self.assertRaises(IllegalActionError):
            game.step((26, 21))
        game.step(game.legal_action_indices()[0])
        with self.assertRaises(TerminalGameError):
            game.step(0)


class ObservationTests(unittest.TestCase):
    def test_dark_observation_hides_opponents_but_not_self(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(0, 2, 3)] = Piece(0, PieceType.COMMANDER)
        pieces[ArmPoint(0, 1, 3)] = Piece(1, PieceType.ARMY_COMMANDER)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.COMMANDER)
        game = JunqiGame.from_position(
            two_config(mode=InformationMode.DARK),
            pieces,
            revealed_flags=(False, False),
        )

        observation = game.observe(0)
        self.assertEqual(observation.points[7].kind, PieceType.COMMANDER)
        self.assertTrue(observation.points[7].identity_visible)
        self.assertIsNone(observation.points[2].kind)
        self.assertFalse(observation.points[2].identity_visible)
        self.assertEqual(
            sum(observation.legal_action_mask), len(game.legal_actions())
        )

        non_current = game.observe(1)
        self.assertFalse(any(non_current.legal_action_mask))

    def test_revealed_flag_is_visible_in_dark_mode(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(0, 2, 3)] = Piece(0, PieceType.ENGINEER)
        pieces[ArmPoint(0, 3, 2)] = Piece(0, PieceType.COMMANDER)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.ENGINEER)
        game = JunqiGame.from_position(
            two_config(mode=InformationMode.DARK),
            pieces,
            revealed_flags=(False, True),
        )
        observation = game.observe(0)
        self.assertEqual(observation.points[56].kind, PieceType.FLAG)
        self.assertTrue(observation.points[56].identity_visible)

    def test_double_open_mode_shows_partner_only(self) -> None:
        pieces = flags(4)
        pieces[ArmPoint(0, 2, 3)] = Piece(0, PieceType.ENGINEER)
        pieces[ArmPoint(0, 3, 2)] = Piece(0, PieceType.COMMANDER)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.COMMANDER)
        pieces[ArmPoint(2, 2, 3)] = Piece(2, PieceType.ARMY_COMMANDER)
        pieces[ArmPoint(2, 3, 2)] = Piece(2, PieceType.COMMANDER)
        pieces[ArmPoint(3, 2, 3)] = Piece(3, PieceType.COMMANDER)
        game = JunqiGame.from_position(
            four_config(mode=InformationMode.DOUBLE_OPEN),
            pieces,
            revealed_flags=(False, False, False, False),
        )
        observation = game.observe(0)

        self.assertEqual(observation.points[67].kind, PieceType.ARMY_COMMANDER)
        self.assertIsNone(observation.points[37].kind)
        self.assertIsNone(observation.points[97].kind)

    def test_hidden_identity_swap_does_not_change_observation_or_legal_mask(self) -> None:
        base = flags(2)
        base[ArmPoint(0, 2, 3)] = Piece(0, PieceType.ENGINEER)
        base[ArmPoint(0, 3, 2)] = Piece(0, PieceType.COMMANDER)
        base[ArmPoint(1, 2, 3)] = Piece(1, PieceType.COMMANDER)
        base[ArmPoint(1, 3, 3)] = Piece(1, PieceType.ARMY_COMMANDER)
        swapped = dict(base)
        swapped[ArmPoint(1, 2, 3)], swapped[ArmPoint(1, 3, 3)] = (
            swapped[ArmPoint(1, 3, 3)],
            swapped[ArmPoint(1, 2, 3)],
        )
        first = JunqiGame.from_position(
            two_config(mode=InformationMode.DARK),
            base,
            revealed_flags=(False, False),
        )
        second = JunqiGame.from_position(
            two_config(mode=InformationMode.DARK),
            swapped,
            revealed_flags=(False, False),
        )
        self.assertEqual(first.observe(0), second.observe(0))

    def test_public_candidate_mask_tracks_moves_and_engineer_turns(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(0, 5, 1)] = Piece(0, PieceType.ENGINEER)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.ENGINEER)

        straight = JunqiGame.from_position(
            two_config(mode=InformationMode.DARK), pieces
        )
        straight.step((20, 0))
        straight_candidates = straight.public_candidates[ArmPoint(0, 1, 1)]
        self.assertNotIn(PieceType.MINE, straight_candidates)
        self.assertNotIn(PieceType.FLAG, straight_candidates)
        self.assertIn(PieceType.COMMANDER, straight_candidates)

        turning = JunqiGame.from_position(
            two_config(mode=InformationMode.DARK), pieces
        )
        turning.step((20, 2))
        self.assertEqual(
            turning.public_candidates[ArmPoint(0, 1, 3)],
            frozenset({PieceType.ENGINEER}),
        )
        observer_view = turning.observe(1, history_limit=0)
        turned_point = turning.board_for(1).encode(ArmPoint(0, 1, 3))
        self.assertEqual(
            observer_view.points[turned_point].kind,
            PieceType.ENGINEER,
        )
        self.assertTrue(observer_view.points[turned_point].identity_visible)

    def test_army_commander_capture_reveals_surviving_commander(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(0, 2, 3)] = Piece(0, PieceType.COMMANDER)
        pieces[ArmPoint(0, 1, 3)] = Piece(1, PieceType.ARMY_COMMANDER)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.COMMANDER)
        game = JunqiGame.from_position(
            two_config(mode=InformationMode.DARK),
            pieces,
            revealed_flags=(False, False),
        )

        game.step((7, 2))
        survivor = ArmPoint(0, 1, 3)
        observer_view = game.observe(1, history_limit=0)
        point_code = game.board_for(1).encode(survivor)
        self.assertEqual(
            observer_view.points[point_code].kind,
            PieceType.COMMANDER,
        )
        self.assertEqual(
            game.known_identities[1][survivor],
            PieceType.COMMANDER,
        )

        # Knowledge follows the piece after later moves and does not depend on
        # retaining the original battle in the bounded action history.
        waiting_action = next(
            action
            for action in game.legal_actions()
            if game.piece_at(game.board_for(1).decode(action[1])) is None
        )
        game.step(waiting_action)
        start_code = game.board_for(0).encode(survivor)
        follow_action = next(
            action
            for action in game.legal_actions()
            if action[0] == start_code
            and game.piece_at(game.board_for(0).decode(action[1])) is None
        )
        destination = game.board_for(0).decode(follow_action[1])
        game.step(follow_action)
        moved_view = game.observe(1, history_limit=0)
        moved_code = game.board_for(1).encode(destination)
        self.assertEqual(
            moved_view.points[moved_code].kind,
            PieceType.COMMANDER,
        )
        self.assertNotIn(survivor, game.known_identities[1])
        self.assertEqual(
            game.known_identities[1][destination],
            PieceType.COMMANDER,
        )

        clone = game.clone()
        self.assertEqual(
            clone.known_identities[1][destination],
            PieceType.COMMANDER,
        )
        restored = JunqiGame.from_position(
            game.config,
            game.pieces,
            current_player=game.current_player,
            active_players=game.active_players,
            revealed_flags=game.revealed_flags,
            ply_count=game.ply_count,
            no_interaction_plies=game.no_interaction_plies,
            public_candidates=game.public_candidates,
            known_identities=game.known_identities,
            known_casualties=game.known_casualties,
            public_history=game.public_history,
        )
        self.assertEqual(restored.state_key(), game.state_key())

    def test_initial_exact_casualty_matrix_shapes_for_all_modes(self) -> None:
        four_dark = JunqiGame.new_four_player(
            seed=3, information_mode=InformationMode.FOUR_DARK
        ).observe(0, history_limit=0)
        double_open = JunqiGame.new_four_player(
            seed=3, information_mode=InformationMode.DOUBLE_OPEN
        ).observe(0, history_limit=0)
        two_player = JunqiGame.new_two_player(
            seed=3, information_mode=InformationMode.DARK
        ).observe(0, history_limit=0)

        self.assertEqual(four_dark.casualty_players, (1, 2, 3))
        self.assertEqual(double_open.casualty_players, (1, 3))
        self.assertEqual(two_player.casualty_players, (1,))
        for observation, rows in (
            (four_dark, 3),
            (double_open, 2),
            (two_player, 1),
        ):
            self.assertEqual(len(observation.known_casualties), rows)
            self.assertTrue(
                all(len(row) == 25 and not any(row)
                    for row in observation.known_casualties)
            )

    def test_engineer_capture_marks_exact_mine_casualty(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(1, 5, 1)] = Piece(0, PieceType.ENGINEER)
        pieces[ArmPoint(0, 3, 2)] = Piece(0, PieceType.COMMANDER)
        pieces[ArmPoint(1, 5, 2)] = Piece(1, PieceType.MINE)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.COMMANDER)
        game = JunqiGame.from_position(
            two_config(mode=InformationMode.DARK),
            pieces,
            revealed_flags=(False, False),
        )

        result = game.step((50, 51))
        self.assertEqual(result.combat, CombatOutcome.ATTACKER_WINS)
        self.assertEqual(game.known_casualties[0][1][PieceType.MINE], 1)
        self.assertEqual(
            game.known_identities[1][ArmPoint(1, 5, 2)],
            PieceType.ENGINEER,
        )
        casualty_row = game.observe(0, history_limit=0).known_casualties[0]
        self.assertTrue(casualty_row[1])
        self.assertFalse(casualty_row[2])

    def test_dead_rule_switch_removes_deductions_and_casualty_rows(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(1, 5, 1)] = Piece(0, PieceType.ENGINEER)
        pieces[ArmPoint(0, 3, 2)] = Piece(0, PieceType.COMMANDER)
        pieces[ArmPoint(1, 5, 2)] = Piece(1, PieceType.MINE)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.COMMANDER)
        game = JunqiGame.from_position(
            two_config(
                mode=InformationMode.DARK,
                dead_rules_enabled=False,
            ),
            pieces,
            revealed_flags=(False, False),
        )

        result = game.step((50, 51))
        self.assertEqual(result.combat, CombatOutcome.ATTACKER_WINS)
        self.assertFalse(game.known_identities[1])
        self.assertFalse(game.known_casualties[0][1])
        observation = game.observe(1, history_limit=0)
        self.assertFalse(observation.dead_rules_enabled)
        self.assertEqual(observation.casualty_players, ())
        self.assertEqual(observation.known_casualties, ())
        destination = game.board_for(1).encode(ArmPoint(1, 5, 2))
        self.assertIsNone(observation.points[destination].kind)

    def test_one_flag_reveal_forces_commander_and_opposing_bomb_deaths(self) -> None:
        pieces = flags(4)
        pieces[ArmPoint(0, 2, 3)] = Piece(0, PieceType.COMMANDER)
        pieces[ArmPoint(0, 1, 3)] = Piece(1, PieceType.BOMB)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.COMMANDER)
        pieces[ArmPoint(2, 2, 3)] = Piece(2, PieceType.COMMANDER)
        pieces[ArmPoint(3, 2, 3)] = Piece(3, PieceType.COMMANDER)
        game = JunqiGame.from_position(
            four_config(mode=InformationMode.FOUR_DARK),
            pieces,
            revealed_flags=(False, False, False, False),
        )

        result = game.step((7, 2))
        self.assertEqual(result.combat, CombatOutcome.BOTH_REMOVED)
        self.assertEqual(result.newly_revealed_flags, (0,))
        for viewer in range(4):
            self.assertEqual(
                game.known_casualties[viewer][0][PieceType.COMMANDER], 1
            )
            self.assertEqual(
                game.known_casualties[viewer][1][PieceType.BOMB], 1
            )

    def test_two_flag_reveals_force_commander_against_commander(self) -> None:
        pieces = flags(4)
        pieces[ArmPoint(0, 2, 3)] = Piece(0, PieceType.COMMANDER)
        pieces[ArmPoint(0, 1, 3)] = Piece(1, PieceType.COMMANDER)
        pieces[ArmPoint(2, 2, 3)] = Piece(2, PieceType.COMMANDER)
        pieces[ArmPoint(3, 2, 3)] = Piece(3, PieceType.COMMANDER)
        game = JunqiGame.from_position(
            four_config(mode=InformationMode.FOUR_DARK),
            pieces,
            revealed_flags=(False, False, False, False),
        )

        result = game.step((7, 2))
        self.assertEqual(result.newly_revealed_flags, (0, 1))
        for viewer in range(4):
            for owner in (0, 1):
                self.assertEqual(
                    game.known_casualties[viewer][owner][PieceType.COMMANDER],
                    1,
                )

    def test_unmoved_first_row_excludes_bomb_and_forces_equal_rank_death(self) -> None:
        pieces = flags(2)
        start_point = ArmPoint(1, 2, 3)
        end_point = ArmPoint(1, 1, 3)
        pieces[start_point] = Piece(0, PieceType.DIVISION_COMMANDER)
        pieces[end_point] = Piece(1, PieceType.DIVISION_COMMANDER)
        pieces[ArmPoint(0, 3, 2)] = Piece(0, PieceType.COMMANDER)
        pieces[ArmPoint(1, 3, 2)] = Piece(1, PieceType.COMMANDER)
        game = JunqiGame.from_position(
            two_config(mode=InformationMode.DARK),
            pieces,
            revealed_flags=(False, False),
        )
        board = game.board_for(0)

        result = game.step((board.encode(start_point), board.encode(end_point)))
        self.assertEqual(result.combat, CombatOutcome.BOTH_REMOVED)
        self.assertEqual(
            game.known_casualties[0][1][PieceType.DIVISION_COMMANDER], 1
        )

    def test_revealed_flag_division_capture_forces_army_commander(self) -> None:
        pieces = flags(2)
        start_point = ArmPoint(1, 2, 3)
        end_point = ArmPoint(1, 1, 3)
        pieces[start_point] = Piece(0, PieceType.ARMY_COMMANDER)
        pieces[end_point] = Piece(1, PieceType.DIVISION_COMMANDER)
        pieces[ArmPoint(1, 3, 2)] = Piece(1, PieceType.COMMANDER)
        game = JunqiGame.from_position(
            two_config(mode=InformationMode.DARK),
            pieces,
            revealed_flags=(True, False),
        )
        board = game.board_for(0)

        result = game.step((board.encode(start_point), board.encode(end_point)))
        self.assertEqual(result.combat, CombatOutcome.ATTACKER_WINS)
        survivor = game.observe(1, history_limit=0).points[
            game.board_for(1).encode(end_point)
        ]
        self.assertEqual(survivor.kind, PieceType.ARMY_COMMANDER)

    def test_flag_capture_marks_complete_enemy_inventory_dead(self) -> None:
        pieces = flags(2)
        start_point = ArmPoint(1, 5, 2)
        flag_point = ArmPoint(1, 6, 2)
        pieces[start_point] = Piece(0, PieceType.ENGINEER)
        pieces[ArmPoint(0, 3, 2)] = Piece(0, PieceType.COMMANDER)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.COMMANDER)
        game = JunqiGame.from_position(
            two_config(mode=InformationMode.DARK),
            pieces,
            revealed_flags=(False, False),
        )
        board = game.board_for(0)

        game.step((board.encode(start_point), board.encode(flag_point)))
        self.assertTrue(
            all(
                game.known_casualties[0][1][kind] == count
                for kind, count in PIECE_COUNTS.items()
            )
        )
        self.assertTrue(all(game.observe(0).known_casualties[0]))

    def test_double_open_allies_share_exact_enemy_casualties(self) -> None:
        pieces = flags(4)
        start_point = ArmPoint(1, 5, 1)
        end_point = ArmPoint(1, 5, 2)
        pieces[start_point] = Piece(0, PieceType.ENGINEER)
        pieces[end_point] = Piece(1, PieceType.MINE)
        for owner in range(4):
            pieces[ArmPoint(owner, 2, 3)] = Piece(owner, PieceType.COMMANDER)
        game = JunqiGame.from_position(
            four_config(mode=InformationMode.DOUBLE_OPEN),
            pieces,
            revealed_flags=(False, False, False, False),
        )
        board = game.board_for(0)

        game.step((board.encode(start_point), board.encode(end_point)))
        self.assertEqual(game.known_casualties[0][1][PieceType.MINE], 1)
        self.assertEqual(game.known_casualties[2][1][PieceType.MINE], 1)
        ally_view = game.observe(2, history_limit=0)
        self.assertEqual(ally_view.casualty_players, (1, 3))
        self.assertTrue(ally_view.known_casualties[1][1])

    def test_commander_losing_as_attacker_reveals_surviving_mine(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(1, 5, 1)] = Piece(0, PieceType.COMMANDER)
        pieces[ArmPoint(0, 3, 2)] = Piece(0, PieceType.ENGINEER)
        pieces[ArmPoint(1, 5, 2)] = Piece(1, PieceType.MINE)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.COMMANDER)
        game = JunqiGame.from_position(
            two_config(mode=InformationMode.DARK),
            pieces,
            revealed_flags=(False, False),
        )

        game.step((50, 51))
        mine_point = ArmPoint(1, 5, 2)
        observer_view = game.observe(0, history_limit=0)
        point_code = game.board_for(0).encode(mine_point)
        self.assertEqual(observer_view.points[point_code].kind, PieceType.MINE)
        self.assertEqual(
            game.known_identities[0][mine_point],
            PieceType.MINE,
        )

    def test_combat_identity_deduction_is_viewer_local_in_four_dark(self) -> None:
        pieces = flags(4)
        pieces[ArmPoint(0, 2, 3)] = Piece(0, PieceType.COMMANDER)
        pieces[ArmPoint(0, 1, 3)] = Piece(1, PieceType.ARMY_COMMANDER)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.COMMANDER)
        pieces[ArmPoint(2, 2, 3)] = Piece(2, PieceType.COMMANDER)
        pieces[ArmPoint(3, 2, 3)] = Piece(3, PieceType.COMMANDER)
        game = JunqiGame.from_position(
            four_config(mode=InformationMode.FOUR_DARK),
            pieces,
            revealed_flags=(False, False, False, False),
        )

        game.step((7, 2))
        survivor = ArmPoint(0, 1, 3)
        informed = game.observe(1, history_limit=0)
        uninformed = game.observe(2, history_limit=0)
        informed_code = game.board_for(1).encode(survivor)
        uninformed_code = game.board_for(2).encode(survivor)
        self.assertEqual(
            informed.points[informed_code].kind,
            PieceType.COMMANDER,
        )
        self.assertIsNone(uninformed.points[uninformed_code].kind)
        self.assertIn(survivor, game.known_identities[1])
        self.assertNotIn(survivor, game.known_identities[2])

    def test_public_history_is_rotated_without_hidden_piece_identities(self) -> None:
        pieces = flags(2)
        pieces[ArmPoint(0, 2, 3)] = Piece(0, PieceType.ENGINEER)
        pieces[ArmPoint(1, 2, 3)] = Piece(1, PieceType.ENGINEER)
        game = JunqiGame.from_position(two_config(), pieces)
        game.step((7, 6))

        self.assertEqual(len(game.public_history), 1)
        south_event = game.observe(0).history[0]
        north_event = game.observe(1).history[0]
        self.assertEqual(south_event.action, (7, 6))
        self.assertEqual(south_event.actor, 0)
        self.assertEqual(north_event.action, (37, 36))
        self.assertEqual(north_event.actor, 1)
        self.assertFalse(south_event.was_attack)
        self.assertFalse(any(game.observe(0, history_limit=0).history))

    def test_candidate_mask_has_one_entry_per_piece_type(self) -> None:
        game = JunqiGame.new_two_player(seed=31)
        observation = game.observe(0)
        occupied = [piece for piece in observation.points if piece is not None]
        self.assertTrue(occupied)
        self.assertTrue(
            all(len(piece.candidate_mask) == len(PieceType) for piece in occupied)
        )


if __name__ == "__main__":
    unittest.main()
