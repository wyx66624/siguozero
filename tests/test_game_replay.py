"""Completed-game disclosure and exact replay of captures, passes and rotations."""
import unittest
from unittest.mock import patch

import test_local_console as fixtures


class GameReplayTests(unittest.TestCase):
    def test_hidden_identities_stay_private_until_the_whole_game_ends(self):
        for mode in ("four_dark", "double_open"):
            with self.subTest(mode=mode):
                cpu = fixtures.SpectatorTests.make_game(mode)
                initial = cpu.command({"op": "state"})
                self.assertTrue(initial["review_supported"])
                self.assertTrue(any(p and p["kind"] is None for p in initial["pieces"]))
                self.assertNotIn("review", cpu.command({"op": "replay"}))
                eliminated = cpu.command({"op": "move", "action": [0, 0], "expected_ply": 0})
                self.assertFalse(eliminated["active_players"][0])
                self.assertIsNone(eliminated["result"])
                self.assertNotIn("review", cpu.command({"op": "replay", "reveal": True}))
                finished = cpu.command({"op": "advance"})
                self.assertEqual(finished["result"]["outcome"], "win")
                with patch.object(cpu.engine, "step", side_effect=AssertionError("replay must not infer")):
                    review = cpu.command({"op": "replay"})["review"]
                self.assertFalse(review["ended_early"])
                self.assertEqual(len(review["steps"]), finished["ply"])
                pieces = list(review["initial"]["pieces"])
                self.assertTrue(all(p["visible"] and p["kind"] for p in pieces if p))
                for ply, step in enumerate(review["steps"], 1):
                    self.assertEqual(step["event"]["ply"], ply)
                    for code, piece in step["changes"]:
                        pieces[code] = piece
                board = cpu.game.board_for(cpu.human)
                for code, recorded in enumerate(pieces):
                    actual = cpu.game.piece_at(board.decode(code))
                    self.assertEqual(recorded["kind"] if recorded else None,
                                     actual.kind.value if actual else None)
                self.assertEqual(review["steps"][0]["event"]["combat"], "pass")
                self.assertEqual(review["steps"][0]["changes"], [])
                self.assertTrue(any(s["event"]["eliminated"] for s in review["steps"]))
                self.assertEqual(review["result"], finished["result"])

    def test_explicit_finish_unlocks_review_but_cannot_resume_play(self):
        cpu = fixtures.SpectatorTests.make_game()
        review = cpu.command({"op": "replay", "finish": True})["review"]
        self.assertTrue(review["ended_early"])
        self.assertIsNone(review["result"], "ending manually must not invent a win or draw")
        self.assertFalse(cpu.game.is_terminal)
        self.assertFalse(cpu.command({"op": "state"})["your_turn"])
        self.assertEqual(cpu.command({"op": "state"})["legal_actions"], [])
        for op in ("advance", "move"):
            with self.assertRaisesRegex(ValueError, "已经结束"):
                cpu.command({"op": op, "action": [0, 0], "expected_ply": 0})

    def test_rotated_view_and_compulsory_passes_are_recorded(self):
        cpu = fixtures.SpectatorTests.make_immobile_game(human=2)
        first = cpu.command({"op": "state"})
        cpu.command({"op": "advance"})
        review = cpu.command({"op": "replay", "finish": True})["review"]
        self.assertEqual(review["initial"]["human_seat"], 2)
        for visible, actual in zip(first["pieces"], review["initial"]["pieces"]):
            if visible:
                self.assertEqual(visible["owner"], actual["owner"])
        self.assertEqual(review["steps"][0]["event"]["actor"], 0)
        self.assertEqual(review["steps"][0]["event"]["combat"], "pass")
        self.assertEqual(review["steps"][0]["state"]["passes_remaining"][0], 3)
        self.assertEqual(len(review["steps"]), cpu.game.ply_count)


if __name__ == "__main__":
    unittest.main()
