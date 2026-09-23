"""Numeric contracts for entropy trends reconstructed from existing logs."""
import math
import unittest

from junqi.web.exploration import entropy_history, normalized_action_entropy


def record(update, *, opening=.2, other=.6, steps=100):
    # Ten percent opening, eighty percent later play, ten percent forced moves.
    return {"update": update, "rollout/environment_plies": steps,
            "policy/opening_entropy_ratio_mass": .1 * opening,
            "policy/opening_entropy_fraction": .1,
            "policy/other_entropy_ratio_mass": .8 * other,
            "policy/other_entropy_fraction": .8}


class ExplorationHistoryTests(unittest.TestCase):
    def test_phase_sample_weights_and_forced_moves(self):
        self.assertAlmostEqual(normalized_action_entropy(record(1)), 5 / 9)
        # An absent phase is valid when its moments explicitly record zero.
        row = record(1)
        row["policy/opening_entropy_ratio_mass"] = 0
        row["policy/opening_entropy_fraction"] = 0
        self.assertAlmostEqual(normalized_action_entropy(row), .6)

    def test_window_uses_environment_steps_not_epochs_and_expires_old_rows(self):
        rows = [record(1, opening=.9, other=.9),
                record(2, opening=.2, other=.2, steps=100),
                record(3, opening=.6, other=.6, steps=300)]
        rows[1]["optimizer/policy_epochs"] = 3
        result = entropy_history(rows, window_updates=2)
        self.assertAlmostEqual(result["moving_average"], .5)
        self.assertEqual(result["environment_plies"], 400)
        self.assertEqual((result["start_update"], result["end_update"]), (2, 3))
        self.assertEqual(result["covered_updates"], 2)

    def test_missing_nonfinite_and_forced_only_are_not_zero_observations(self):
        forced = record(2)
        for key in list(forced):
            if key.startswith("policy/"):
                forced[key] = 0
        invalid = record(3)
        invalid["policy/other_entropy_ratio_mass"] = math.nan
        result = entropy_history([record(1), forced, invalid], window_updates=2)
        self.assertIsNone(result["moving_average"])
        self.assertEqual(result["covered_updates"], 0)
        self.assertIsNone(normalized_action_entropy({}))
        zero = entropy_history([record(4, opening=0, other=0)])
        self.assertEqual(zero["moving_average"], 0)
        self.assertEqual(zero["covered_updates"], 1)


if __name__ == "__main__":
    unittest.main()
