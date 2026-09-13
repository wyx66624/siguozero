"""Behavioral coverage for frozen allies and observational model evaluation."""
from collections import Counter
from dataclasses import replace
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_arena import ArenaFixtures
from test_model_selection import training_settings, match_result, file_sha256, torch
from junqi.training.arena import MatchSettings, play_groups
from junqi.training.arena_four_player import prepare_group, play_group, summarize_games
from junqi.training.distributed import DistributedContext
from junqi.training.model_selection import ModelSelection
from junqi.training.models import GamePolicyTransformer, PieceConditionedLayoutPointerDecoder


def balanced_result(games, *, win=False):
    one = match_result(games // 2 if win else 0, 0, 0 if win else games // 2)
    return {**match_result(games if win else 0, 0, 0 if win else games),
            "teammate_results": {key: dict(one) for key in ("current", "historical")}}


class BalancedArenaTests(ArenaFixtures, unittest.TestCase):
    def test_500_games_have_exact_half_teammates_and_balanced_complete_groups(self):
        current, old = self.engines("four_dark")
        settings = MatchSettings(mode="four_dark", pairs=125, historical_teammate_fraction=.5)
        records, roles = [], Counter()
        for group in range(125):
            specs = prepare_group(current, old, group, settings)
            self.assertEqual({s["metadata"]["rotation"] for s in specs}, set(range(4)))
            for version in ("current", "historical"):
                legs = [s for s in specs if s["metadata"]["teammate_version"] == version]
                self.assertEqual(len(legs), 2)
                self.assertEqual({s["candidate_seats"][0] % 2 for s in legs}, {0, 1})
            for spec in specs:
                meta = spec["metadata"]
                roles[meta["teammate_version"], meta["focal_seat"]] += 1
                focal = meta["focal_seat"]
                reward = 1 if meta["teammate_version"] == "current" else -1
                records.append({**meta, "mode": "four_dark", "candidate_team": focal % 2,
                    "candidate_seats": sorted(spec["candidate_seats"]), "candidate_reward": reward,
                    "candidate_score": (reward + 1) / 2,
                    "player_rewards": [reward if seat % 2 == focal % 2 else -reward for seat in range(4)],
                    "winner_team": focal % 2 if reward == 1 else 1 - focal % 2,
                    "plies": 50, "terminal_reason": "team_eliminated"})
        self.assertLessEqual(max(roles.values()) - min(roles.values()), 1)
        summary = summarize_games(records, settings, alpha=.05)
        self.assertEqual((summary["games"], summary["rotation_groups"]), (500, 125))
        self.assertEqual(summary["score"], .5)
        for version, score in (("current", 1.), ("historical", 0.)):
            split = summary["teammate_results"][version]
            self.assertEqual((split["games"], split["score"]), (250, score))
        self.assertAlmostEqual(summary["score_alpha"], .05 / 3)
        for bad in (records[:-1], records[:-1] + [records[0]]):
            with self.assertRaisesRegex(ValueError, "unpaired"):
                summarize_games(bad, settings, alpha=.05)
        bad = copy.deepcopy(records)
        bad[0]["teammate_version"] = "historical"
        with self.assertRaisesRegex(ValueError, "teammate assignment"):
            summarize_games(bad, settings, alpha=.05)

    def test_frozen_ally_uses_old_actor_and_each_seats_own_private_observation(self):
        for mode in ("four_dark", "double_open"):
            current, old = self.engines(mode)
            calls = []
            def sample(label, states, **kwargs):
                state = states[0]
                codes = state.records[0].board_codes
                self.assertFalse(any(62 <= code <= 73 or 126 <= code <= 137 for code in codes))
                self.assertEqual(sum(94 <= code <= 105 for code in codes), 0 if mode == "four_dark" else 25)
                calls.append(label)
                return [[state.legal_actions[0]]], []
            settings = MatchSettings(mode=mode, pairs=1, max_plies=4, historical_teammate_fraction=.5)
            with patch.object(current.actor, "sample", side_effect=lambda s, **k: sample("new", s, **k)), \
                    patch.object(old.actor, "sample", side_effect=lambda s, **k: sample("old", s, **k)):
                records = play_group(current, old, 0, settings)
            expected = ["new" if action["player"] in record["candidate_seats"] else "old"
                        for record in records for action in record["actions"]]
            self.assertEqual(calls, expected)
            self.assertEqual(Counter(calls), {"new": 6, "old": 10})

    def test_parallel_reordering_is_reproducible_and_reuses_two_model_copies(self):
        current, old = self.engines("four_dark")
        originals = [(e.policy, e.layout, e.policy.config) for e in (current, old)]
        settings = MatchSettings(mode="four_dark", pairs=2, max_plies=20,
                                 historical_teammate_fraction=.5, smoke_test=True)
        expected = [r for group in range(2) for r in play_group(current, old, group, settings)]
        actual = play_groups(current, old, [1, 0], replace(settings, parallel_games=8,
                            inference_batch_size=3, environment_workers=4))
        self.assertEqual(actual, expected)
        for engine, before in zip((current, old), originals):
            for value, original in zip((engine.policy, engine.layout, engine.policy.config), before):
                self.assertIs(value, original)
            self.assertFalse(engine.policy._inference_temporal_cache)
        summary = summarize_games(actual, settings, alpha=.05)
        self.assertEqual([r["games"] for r in summary["teammate_results"].values()], [4, 4])
        self.assertEqual(summary["verdict"], "smoke_test_not_strength_evidence")

    def test_settings_do_not_accept_unbalanced_or_two_player_allies(self):
        for fraction in (True, .2, 1., -.5):
            with self.assertRaises(ValueError):
                MatchSettings(mode="four_dark", historical_teammate_fraction=fraction)
        with self.assertRaises(ValueError):
            MatchSettings(mode="two_player", historical_teammate_fraction=.5)

    def test_eliminated_current_player_gets_its_frozen_teammates_team_win(self):
        from junqi import ArmPoint, GameConfig, GameVariant, InformationMode, JunqiGame, Piece, PieceType
        current, old = self.engines("four_dark")
        def position(_mode=None, **kwargs):
            return JunqiGame.from_position(GameConfig(variant=GameVariant.FOUR_PLAYER,
                information_mode=InformationMode.FOUR_DARK, max_plies=4), {
                ArmPoint(2,6,2): Piece(2,PieceType.FLAG),
                ArmPoint(3,6,2): Piece(3,PieceType.FLAG),
                ArmPoint(3,5,2): Piece(2,PieceType.ENGINEER),
                ArmPoint(3,2,3): Piece(3,PieceType.COMMANDER),
            }, current_player=2, active_players=(False,False,True,True))
        game = position()
        capture = next(a for a in game.legal_actions() if game.clone().step(a).flag_captured_owner == 3)
        settings = MatchSettings(mode="four_dark", pairs=3, max_plies=4, historical_teammate_fraction=.5)
        with patch("junqi.training.arena.new_game", side_effect=position), \
                patch.object(current.actor, "sample", return_value=([[capture]], [])), \
                patch.object(old.actor, "sample", return_value=([[capture]], [])):
            records = play_group(current, old, 2, settings)
        focal = records[0]
        self.assertEqual(focal["candidate_seats"], [0])
        self.assertEqual(focal["teammate_version"], "historical")
        self.assertFalse(focal["active_players"][0])
        self.assertEqual(focal["actions"][0]["player"], 2)
        self.assertEqual(focal["candidate_reward"], 1)
        self.assertEqual(focal["player_rewards"], [1,-1,1,-1])


class ObservationalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.context = DistributedContext(0, 1, 0, torch.device("cpu"))

    def selection(self, settings, update=0, steps=0):
        policy = GamePolicyTransformer(settings.model)
        layout = PieceConditionedLayoutPointerDecoder(settings.model)
        selection = ModelSelection(settings, self.root, self.context)
        selection.initialize(policy, layout, update=update, cumulative={"environment_plies": steps})
        return selection, policy, layout

    def test_migration_keeps_old_reports_reference_alpha_and_always_retains_new_candidates(self):
        old = training_settings(target_environment_plies=1000, arena_interval_environment_plies=100)
        selection, policy, layout = self.selection(old)
        with patch("junqi.training.model_selection.run_match", return_value=match_result()):
            selection.evaluate(policy, layout, update=1, cumulative={"environment_plies": 100})
        before = selection.state_dict()
        old_report = (selection.directory / before["rounds"][0]).read_bytes()
        alpha = selection._round_alpha()
        updated = replace(old, arena_observational_only=True, arena_historical_teammate_fraction=.5)
        selection, policy, layout = self.selection(updated, update=1, steps=120)
        self.assertEqual(selection._round_alpha(), alpha)
        self.assertEqual(selection.state["best_sha256"], before["best_sha256"])
        self.assertGreater(selection.state["schedule_seed_base"], old.arena_seed)
        for update, win in ((2, False), (3, True)):
            with torch.no_grad():
                next(policy.parameters()).add_(.01)
            with patch("junqi.training.model_selection.run_match", return_value=balanced_result(4, win=win)) as run:
                report = selection.evaluate(policy, layout, update=update, cumulative={"environment_plies": update * 100})
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[1], selection.directory / before["best_snapshot"])
            self.assertEqual(run.call_args.args[2].historical_teammate_fraction, .5)
            self.assertEqual(report["decision"], "use_latest")
            self.assertEqual(report["evaluation_type"], "fixed_reference")
            self.assertNotIn("promoted", report)
            self.assertFalse(report["champion_evaluated"])
            self.assertEqual(report["opponent_update"], 1)
            self.assertEqual(selection.state["best_sha256"], before["best_sha256"])
            self.assertEqual(file_sha256(selection.best_path), before["best_sha256"])
            self.assertEqual(selection.state["last_evaluated_update"], update)
            self.assertTrue((selection.directory / report["candidate_snapshot"]).is_file())
        self.assertEqual((selection.directory / before["rounds"][0]).read_bytes(), old_report)
        view = json.loads((self.root / "historical_opponents/latest_evaluation.json").read_text())
        self.assertEqual(view["evaluation_type"], "fixed_reference")
        self.assertEqual(view["results"][0]["score_change"], 1.)
        restarted, _, _ = self.selection(updated, update=3, steps=300)
        self.assertEqual(len(restarted.state["observational_evaluation_migrations"]), 1)
        with self.assertRaisesRegex(RuntimeError, "schedule/budget changed"):
            self.selection(old, update=3, steps=300)

    def test_pending_evaluation_cannot_change_protocol(self):
        old = training_settings(target_environment_plies=1000, arena_interval_environment_plies=100)
        selection, _, _ = self.selection(old)
        before = selection.state_path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "finish the pending evaluation"):
            self.selection(replace(old, arena_observational_only=True, arena_historical_teammate_fraction=.5),
                           update=1, steps=100)
        self.assertEqual(selection.state_path.read_bytes(), before)

    def test_deferred_reference_gets_a_real_hash_before_match(self):
        settings = training_settings(checkpoint_policy="evaluation", target_environment_plies=1000,
            arena_interval_environment_plies=100, arena_observational_only=True, arena_historical_teammate_fraction=.5)
        selection, policy, layout = self.selection(settings)
        self.assertIsNone(selection.state["best_sha256"])
        with patch("junqi.training.model_selection.run_match", return_value=balanced_result(4)) as run:
            report = selection.evaluate(policy, layout, update=1, cumulative={"environment_plies": 100})
        self.assertEqual(run.call_args.kwargs["opponent_sha256"], file_sha256(run.call_args.args[1]))
        self.assertEqual(report["historical_panel"]["contract"]["total_games"], 4)

    def test_real_cpu_training_evaluates_and_resumes_latest_even_without_a_win(self):
        from junqi.training.trainer import SelfPlayTrainer
        settings = training_settings(total_updates=2, anchor_batch=4, target_environment_plies=8,
            arena_interval_environment_plies=4, arena_observational_only=True, arena_historical_teammate_fraction=.5)
        trainer = SelfPlayTrainer(settings, run_directory=self.root)
        trainer.train()
        reports = [json.loads((trainer.model_selection.directory / name).read_text())
                   for name in trainer.model_selection.state["rounds"]]
        self.assertEqual([r["decision"] for r in reports], ["use_latest"] * 2)
        self.assertEqual([r["draws"] for r in reports], [4, 4])
        self.assertEqual(trainer.model_selection.state["best_update"], 0)
        payload = torch.load(trainer.checkpoints.latest_path, weights_only=False)
        self.assertEqual(payload["update"], 2)
        self.assertEqual(payload["trainer_state"]["model_selection"]["last_evaluated_update"], 2)
        resumed = SelfPlayTrainer(settings, run_directory=self.root)
        self.addCleanup(resumed.logger.close)
        for name, value in resumed.policy.state_dict().items():
            torch.testing.assert_close(value, payload["policy"][name], rtol=0, atol=0)
        with patch("junqi.training.model_selection.run_match") as match:
            resumed.train()
        match.assert_not_called()

    def test_historical_500_budget_keeps_half_teammates_within_every_opponent(self):
        from test_historical_opponents import settings as history_settings
        from junqi.training.historical_opponents import HistoricalOpponents
        settings = replace(history_settings(), arena_games=500, historical_eval_total_games=500,
            historical_snapshot_fractions=(.05,.1,.2,.3,.4), arena_after_half_historical_only=True,
            arena_observational_only=True, arena_historical_teammate_fraction=.5)
        selection, policy, layout = self.selection(settings)
        league = HistoricalOpponents(settings, self.root, self.context)
        for steps in (0, 50, 100, 200, 300, 400, 500):
            league.update_progress(policy, layout, environment_plies=steps, update=steps // 10)
        with patch("junqi.training.model_selection.run_match", side_effect=lambda *a, **k: balanced_result(a[2].pairs * 4)) as match:
            report = selection.evaluate(policy, layout, update=50, cumulative={"environment_plies": 500}, historical=league)
        counts = [c.args[2].pairs * 4 for c in match.call_args_list]
        self.assertEqual(counts, [84,84,84,84,84,80])
        self.assertEqual(report["games"], 500)
        self.assertEqual(report["decision"], "use_latest")
        self.assertEqual(report["evaluation_type"], "historical_only")
        self.assertEqual(selection.state["best_update"], 0)
        for role in ("current", "historical"):
            self.assertEqual(sum(r["teammate_results"][role]["games"] for r in report["historical_panel"]["results"]), 250)

    def test_split_validation_rejects_wrong_totals_and_missing_teammates(self):
        settings = training_settings(arena_observational_only=True, arena_historical_teammate_fraction=.5)
        selection, _, _ = self.selection(settings)
        for bad in (match_result(), {**balanced_result(4), "teammate_results": {"current": match_result()}}):
            with self.assertRaises(ValueError):
                selection._validate_historical_result(bad, games=4)

    def test_interrupted_panel_reuses_finished_balanced_matches_only(self):
        from test_historical_opponents import settings as history_settings
        from junqi.training.historical_opponents import HistoricalOpponents
        settings = replace(history_settings(), historical_eval_total_games=12,
            arena_after_half_historical_only=True, arena_observational_only=True,
            arena_historical_teammate_fraction=.5)
        selection, policy, layout = self.selection(settings)
        league = HistoricalOpponents(settings, self.root, self.context)
        for steps in (0,100,200,500):
            league.update_progress(policy, layout, environment_plies=steps, update=steps//10)
        with patch("junqi.training.model_selection.run_match", side_effect=[balanced_result(4), InterruptedError()]):
            with self.assertRaises(InterruptedError):
                selection.evaluate(policy, layout, update=50, cumulative={"environment_plies":500}, historical=league)
        self.assertFalse(selection.state["rounds"])
        saved = list(selection.directory.glob("historical_env_*.json"))
        self.assertEqual(len(saved), 1)
        original = saved[0].read_bytes()
        with patch("junqi.training.model_selection.run_match", return_value=balanced_result(4)) as match:
            report = selection.evaluate(policy, layout, update=50, cumulative={"environment_plies":500}, historical=league)
        self.assertEqual(match.call_count, 2)
        self.assertEqual(saved[0].read_bytes(), original)
        self.assertEqual(report["games"], 12)
        self.assertEqual(report["decision"], "use_latest")
        self.assertEqual(len(selection.state["rounds"]), 1)
