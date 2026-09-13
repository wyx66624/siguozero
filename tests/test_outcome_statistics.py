"""Completed-game accounting across long logs, service restarts and training resumes."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from junqi.web.monitor import Monitor
from junqi.web.outcomes import OutcomeStore, evaluation_outcomes


class OutcomeStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "four_dark"
        self.run.mkdir()
        self.log = self.run / "metrics.jsonl"
        self.store = OutcomeStore(self.root / "statistics.sqlite3")

    def row(self, update, *, wins=2, draws=1, losses=1, cumulative=None):
        return {"update": update, "timestamp_unix": 1000 + update, "algorithm": "ppo", "mode": "four_dark",
                "rollout/wins": wins, "rollout/draws": draws, "rollout/losses": losses,
                "rollout/base_games_completed": wins + draws + losses,
                "cumulative/base_games": 4 * update if cumulative is None else cumulative,
                "rollout/environment_plies": 16, "cumulative/environment_plies": 16 * update}

    def append(self, *records):
        with self.log.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")

    def snapshot(self):
        return self.store.snapshot(self.run, algorithm="ppo", mode="four_dark")

    def test_resume_filters_abandoned_games_before_first_new_update(self):
        self.append(*(self.row(i) for i in range(1, 5)))
        self.assertEqual(self.snapshot()['totals']['games'], 16)
        result = self.store.snapshot(self.run, algorithm='ppo', mode='four_dark',
                                     max_update=2, session_started=2000, resumed_update=2)
        self.assertEqual(result['totals']['games'], 8)
        self.assertEqual(result['last_update'], 2)
        self.assertTrue(result['coverage_complete'])
        self.append({**self.row(3, wins=3, cumulative=13), 'timestamp_unix': 2001})
        result = self.store.snapshot(self.run, algorithm='ppo', mode='four_dark',
                                     max_update=3, session_started=2000, resumed_update=2)
        self.assertEqual(result['totals']['games'], 13)
        self.assertTrue(result['coverage_complete'])

    def test_all_history_beyond_chart_window_survives_service_restart_without_recounting(self):
        self.append(*(self.row(i) for i in range(1, 151)))
        result = self.snapshot()
        self.assertEqual(result["totals"]["games"], 600)
        self.assertEqual(result["totals"]["wins"], 300)
        self.assertEqual(len(result["recent"]), 12)
        self.assertTrue(result["coverage_complete"])
        self.store = OutcomeStore(self.store.path)
        with patch("junqi.web.outcomes.training_record", side_effect=AssertionError("Already consumed log was parsed again")):
            self.assertEqual(self.snapshot()["totals"], result["totals"])
        self.append(self.row(151), self.row(151), self.row(2))
        result = self.snapshot()
        self.assertEqual(result["totals"]["games"], 604)
        self.assertEqual(result["last_update"], 151)

    def test_incomplete_line_is_not_consumed_and_malformed_line_cannot_fabricate_results(self):
        self.append(self.row(1))
        with self.log.open("a") as stream:
            stream.write(json.dumps(self.row(2)))
        self.assertEqual(self.snapshot()["totals"]["games"], 4)
        self.assertTrue(self.snapshot()["catching_up"])
        with self.log.open("a") as stream:
            stream.write("\n{invalid json}\n")
        self.assertEqual(self.snapshot()["totals"]["games"], 8)
        self.assertFalse(self.snapshot()["catching_up"])

    def test_resume_replaces_abandoned_future_updates(self):
        self.append(*(self.row(i) for i in range(1, 5)))
        self.assertEqual(self.snapshot()["totals"]["games"], 16)
        self.append(self.row(3, wins=0, draws=1, losses=0, cumulative=9))
        result = self.snapshot()
        self.assertEqual(result["totals"]["games"], 9)
        self.assertEqual([row["update"] for row in result["recent"]], [3, 2, 1])
        self.assertTrue(result["coverage_complete"])

    def test_rewritten_or_truncated_source_rebuilds_its_index(self):
        self.append(self.row(1))
        self.snapshot()
        before_size = self.log.stat().st_size
        self.log.write_text(json.dumps(self.row(1, wins=1, draws=2, losses=1)) + "\n")
        self.assertEqual(self.log.stat().st_size, before_size)
        self.assertEqual(self.snapshot()["totals"]["wins"], 1)
        self.log.write_text("")
        self.assertFalse(self.snapshot()["available"])
        self.append(self.row(1, wins=0, draws=1, losses=0, cumulative=1))
        self.assertEqual(self.snapshot()["totals"]["games"], 1)

    def test_backfill_has_a_bounded_work_budget_and_reports_incomplete_coverage(self):
        self.append(*(self.row(i) for i in range(1, 5)))
        self.store = OutcomeStore(self.store.path, batch_bytes=1)
        first = self.snapshot()
        self.assertEqual(first["totals"]["games"], 4)
        self.assertTrue(first["catching_up"])
        self.assertFalse(first["coverage_complete"])
        for _ in range(3):
            last = self.snapshot()
        self.assertEqual(last["totals"]["games"], 16)
        self.assertTrue(last["coverage_complete"])

    def test_missing_or_inconsistent_outcomes_are_unknown_not_zero(self):
        self.append(self.row(10))
        result = self.snapshot()
        self.assertFalse(result["coverage_complete"])
        self.assertEqual(result["unrecorded_games"], 36)
        incomplete = self.row(11)
        incomplete.pop("rollout/draws")
        mismatch = self.row(12)
        mismatch["rollout/base_games_completed"] = 100
        self.append(incomplete, mismatch)
        result = self.snapshot()
        self.assertIsNone(result["latest"]["games"])
        self.assertEqual(result["totals"]["games"], 4)
        self.assertEqual(result["recorded_updates"], 1)
        self.assertFalse(result["coverage_complete"])

    def test_grpo_branch_results_are_separate_from_ppo_games_and_base_games(self):
        self.append(self.row(1))
        original = self.snapshot()
        other = self.root / "two_player"
        other.mkdir()
        row = {"update": 1, "algorithm": "grpo", "mode": "two_player",
               "rollout/wins": 3, "rollout/draws": 4, "rollout/losses": 1,
               "rollout/base_games_completed": 2, "rollout/terminal_continuations": 8,
               "cumulative/base_games": 2, "cumulative/terminal_continuations": 8}
        (other / "metrics.jsonl").write_text(json.dumps(row) + "\n")
        result = self.store.snapshot(other, algorithm="grpo", mode="two_player")
        self.assertEqual(result["unit"], "branches")
        self.assertEqual(result["perspective"], "root_player")
        self.assertEqual(result["totals"]["games"], 8)
        self.assertTrue(result["coverage_complete"])
        self.assertEqual(self.snapshot()["totals"], original["totals"])

    def test_arena_only_metrics_do_not_replace_training_progress_or_game_results(self):
        self.append(self.row(1), self.row(2))
        arena = {"update": 2, "timestamp_unix": 2000, "arena/games": 100, "arena/wins": 60,
                 "arena/draws": 20, "arena/losses": 20}
        self.append(arena)
        (self.run / "latest_metrics.json").write_text(json.dumps(arena))
        monitor = Monitor({"runs": [{"id": "live", "name": "test", "mode": "four_dark", "path": "four_dark"}]}, self.root)
        result = monitor.run_status(monitor.specs[0])
        self.assertEqual(result["steps"], 32)
        self.assertEqual(result["metrics"]["rollout/wins"], 2)
        self.assertEqual(result["outcomes"]["totals"]["games"], 8)
        self.assertEqual(result["evaluation_outcomes"]["rounds"], 0)

    def test_only_complete_committed_mode_matched_evaluations_are_counted(self):
        directory = self.run / "model_selection"
        directory.mkdir()
        report = {"mode": "four_dark", "games": 100, "wins": 60, "draws": 30, "losses": 10,
                  "candidate_update": 500, "opponent_update": 0}
        (directory / "round1.json").write_text(json.dumps(report))
        (directory / "uncommitted.json").write_text(json.dumps(report))
        (directory / "invalid.json").write_text(json.dumps({**report, "games": 101}))
        (directory / "wrong_mode.json").write_text(json.dumps({**report, "mode": "two_player"}))
        (directory / "panel.json").write_text(json.dumps({**report, "evaluation_type": "historical_only"}))
        self.assertEqual(evaluation_outcomes(self.run, {}, "four_dark")["totals"]["games"], 0)
        state = {"rounds": ["round1.json", "round1.json", "invalid.json", "wrong_mode.json", "../../outside.json", "panel.json"]}
        result = evaluation_outcomes(self.run, state, "four_dark")
        self.assertEqual(result["rounds"], 1)
        self.assertEqual(result["totals"]["games"], 100)
        self.assertEqual(result["latest"]["score"], 0.75)
        self.assertEqual(result["unavailable_rounds"], 3)


if __name__ == "__main__":
    unittest.main()
