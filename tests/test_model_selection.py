from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("PyTorch training extra is not installed") from error

from junqi.training.distributed import DistributedContext
from junqi.training.models import (
    GamePolicyTransformer,
    PieceConditionedLayoutPointerDecoder,
)
from junqi.training.modes import TrainingMode
from junqi.training.settings import TrainingSettings


CONFIG = Path(__file__).parents[1] / "configs" / "bootstrap.yaml"


def training_settings(mode=TrainingMode.FOUR_DARK, **overrides):
    return TrainingSettings.from_yaml(
        CONFIG,
        mode,
        tiny=True,
        overrides={
            "device": "cpu",
            "total_updates": 100,
            "arena_enabled": True,
            "arena_games": 4,
            "arena_max_plies": 4,
            "arena_temporal_cache_entries": 2,
            **overrides,
        },
    )


def match_result(wins=3, draws=0, losses=1):
    games = wins + draws + losses
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": (wins + draws / 2) / games,
        "score_ci": [0.0, 1.0],
        "verdict": "inconclusive",
        "draw_rate": draws / games,
        "warnings": [],
        "terminal_reasons": {"team_eliminated": games},
    }


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ModelSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.context = DistributedContext(0, 1, 0, torch.device("cpu"))

    def selection(self, *, settings=None, update=0, cumulative=None):
        from junqi.training.model_selection import ModelSelection

        settings = settings or training_settings()
        policy = GamePolicyTransformer(settings.model)
        layout = PieceConditionedLayoutPointerDecoder(settings.model)
        selection = ModelSelection(settings, self.root, self.context)
        selection.initialize(policy, layout, update=update, cumulative=cumulative or {})
        return selection, policy, layout

    def test_settings_enable_full_matches_by_default_and_keep_smoke_runs_small(self):
        for mode in TrainingMode:
            with self.subTest(mode=mode):
                settings = TrainingSettings.from_yaml(CONFIG, mode)
                self.assertTrue(settings.arena_enabled)
                self.assertEqual(settings.arena_start_percent, 30)
                self.assertEqual(settings.arena_interval_percent, 5)
                self.assertEqual(settings.arena_games, 1000 if mode is TrainingMode.TWO_PLAYER else 100)
                self.assertEqual(settings.arena_interval_environment_plies,
                                 None if mode is TrainingMode.TWO_PLAYER else 50_000_000)
                self.assertEqual(settings.checkpoint_policy,
                                 "periodic")
                self.assertEqual(settings.checkpoint_interval_environment_plies,
                                 None if mode is TrainingMode.TWO_PLAYER else 25_000_000)
                self.assertIsNone(settings.max_game_plies)
                self.assertIsNone(settings.arena_max_plies)
                self.assertEqual(settings.no_capture_draw_plies, 70)
                self.assertEqual(TrainingSettings.from_yaml(CONFIG, mode, tiny=True).checkpoint_policy,
                                 "periodic")
                self.assertFalse(TrainingSettings.from_yaml(CONFIG, mode, tiny=True).arena_enabled)
                self.assertTrue(training_settings(mode).arena_enabled)

    def test_settings_reject_invalid_or_unbalanced_match_protocols(self):
        for overrides in (
            {"arena_enabled": "false"}, {"arena_start_percent": 30.5},
            {"arena_start_percent": 0}, {"arena_interval_percent": True},
            {"arena_interval_percent": 0}, {"arena_games": 1002},
            {"arena_games": 0}, {"arena_games": 4.0},
            {"arena_max_plies": 0}, {"arena_temporal_cache_entries": 0},
            {"arena_seed": -1}, {"arena_seed": 2**63 - 1},
            {"arena_interval_environment_plies": 0},
            {"arena_interval_environment_plies": True},
            {"arena_interval_environment_plies": 1.5},
            {"arena_interval_environment_plies": 50_000_000, "target_environment_plies": None},
            {"checkpoint_policy": "never"}, {"checkpoint_policy": None},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                training_settings(**overrides)
        self.assertEqual(training_settings(TrainingMode.TWO_PLAYER, arena_games=1002).arena_games, 1002)

    def test_cli_forwards_model_selection_overrides_into_training(self):
        from junqi.training.cli import main

        with patch("junqi.training.cli.SelfPlayTrainer") as trainer:
            main([
                "--config", str(CONFIG), "--mode", "four_dark", "--device", "cpu",
                "--no-model-selection", "--arena-games", "8",
                "--arena-start-percent", "35", "--arena-interval-percent", "10",
                "--arena-max-plies", "64", "--run-directory", str(self.root),
            ])
        settings = trainer.call_args.args[0]
        self.assertFalse(settings.arena_enabled)
        self.assertEqual(settings.arena_games, 8)
        self.assertEqual(settings.checkpoint_policy, "periodic")
        self.assertEqual(settings.arena_start_percent, 35)
        self.assertEqual(settings.arena_interval_percent, 10)
        self.assertEqual(settings.arena_max_plies, 64)
        trainer.return_value.train.assert_called_once_with()

    def test_cli_environment_interval_uses_absolute_steps(self):
        from junqi.training.cli import main
        with patch("junqi.training.cli.SelfPlayTrainer") as trainer:
            main(["--config", str(CONFIG), "--mode", "four_dark", "--device", "cpu",
                  "--arena-interval-environment-plies", "50000000", "--arena-games", "100",
                  "--run-directory", str(self.root)])
        settings = trainer.call_args.args[0]
        self.assertEqual(settings.arena_interval_environment_plies, 50_000_000)
        self.assertEqual(settings.arena_games, 100)
        self.assertEqual(settings.checkpoint_policy, "periodic")
        self.assertEqual(len(settings.arena_milestones), 60)
        self.assertEqual(settings.arena_milestones[-1], 3_000_000_000)
        with patch("junqi.training.cli.SelfPlayTrainer") as trainer, self.assertRaises(SystemExit):
            main(["--config", str(CONFIG), "--mode", "four_dark",
                  "--arena-interval-environment-plies", "50000000", "--arena-start-percent", "30"])
        trainer.assert_not_called()

    def test_cli_can_explicitly_restore_periodic_saves_without_model_selection(self):
        from junqi.training.cli import main
        with patch("junqi.training.cli.SelfPlayTrainer") as trainer:
            main(["--config", str(CONFIG), "--mode", "four_dark", "--device", "cpu",
                  "--no-model-selection", "--checkpoint-policy", "periodic", "--checkpoint-every", "7",
                  "--run-directory", str(self.root)])
        settings = trainer.call_args.args[0]
        self.assertFalse(settings.arena_enabled)
        self.assertEqual(settings.checkpoint_policy, "periodic")
        self.assertEqual(settings.checkpoint_every_updates, 7)

    def test_evaluation_only_defers_baseline_and_keeps_its_original_parameters(self):
        selection, policy, layout = self.selection(
            settings=training_settings(checkpoint_policy="evaluation"),
        )
        originals = {name: copy.deepcopy(model.state_dict())
                     for name, model in (("policy", policy), ("layout", layout))}
        self.assertFalse(selection.state_path.exists())
        self.assertEqual(list(self.root.rglob("*.pt")), [])
        with torch.no_grad():
            for model in (policy, layout):
                for parameter in model.parameters():
                    parameter.add_(1)
        self.assertIsNone(selection.evaluate(policy, layout, update=29, cumulative={}))
        self.assertEqual(list(self.root.rglob("*.pt")), [])
        with patch("junqi.training.model_selection.run_match", return_value=match_result()) as match:
            selection.evaluate(policy, layout, update=30, cumulative={})
        baseline = torch.load(match.call_args.args[1], map_location="cpu", weights_only=False)
        candidate = torch.load(match.call_args.args[0], map_location="cpu", weights_only=False)
        for name, model in (("policy", policy), ("layout", layout)):
            for key, expected in originals[name].items():
                torch.testing.assert_close(baseline[name][key], expected, rtol=0, atol=0)
                torch.testing.assert_close(candidate[name][key], model.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(baseline["update"], 0)
        self.assertEqual(candidate["update"], 30)
        self.assertIsNone(selection._pending_baseline)

    def test_evaluation_only_stop_before_first_match_writes_no_parameters(self):
        from junqi.training.trainer import SelfPlayTrainer
        settings = training_settings(
            checkpoint_policy="evaluation", total_updates=4,
            target_environment_plies=4, arena_interval_environment_plies=4,
        )
        trainer = SelfPlayTrainer(settings, run_directory=self.root)
        self.addCleanup(trainer.logger.close)
        self.assertEqual(list(self.root.rglob("*.pt")), [])
        evaluate = trainer._maybe_evaluate_model

        def stop_after_one_update():
            result = evaluate()
            if trainer.update == 1:
                trainer.stop_requested = True
            return result

        with patch.object(trainer, "_maybe_evaluate_model", side_effect=stop_after_one_update), \
                patch("junqi.training.model_selection.run_match") as match:
            trainer.train()
        self.assertEqual(trainer.update, 1)
        self.assertEqual(list(self.root.rglob("*.pt")), [])
        self.assertFalse(trainer.model_selection.state_path.exists())
        match.assert_not_called()
        # Logs alone must never be mistaken for a resumable training state.
        with self.assertRaisesRegex(RuntimeError, "latest.pt is missing"):
            SelfPlayTrainer(settings, run_directory=self.root)

    def test_evaluation_only_resumes_last_match_after_stopping_between_matches(self):
        from junqi.training.trainer import SelfPlayTrainer
        settings = training_settings(
            checkpoint_policy="evaluation", total_updates=4,
            target_environment_plies=4, arena_interval_environment_plies=2,
        )
        trainer = SelfPlayTrainer(settings, run_directory=self.root)
        self.addCleanup(trainer.logger.close)
        evaluate = trainer._maybe_evaluate_model

        def stop_after_third_update():
            result = evaluate()
            if trainer.update == 3:
                trainer.stop_requested = True
            return result

        with patch.object(trainer, "_maybe_evaluate_model", side_effect=stop_after_third_update), \
                patch.object(trainer, "save_checkpoint", wraps=trainer.save_checkpoint) as save, \
                patch("junqi.training.model_selection.run_match", return_value=match_result()) as match:
            trainer.train()
        self.assertEqual(trainer.update, 3)
        self.assertEqual([call.kwargs["reason"] for call in save.call_args_list],
                         ["before_model_selection", "after_model_selection"])
        match.assert_called_once()
        path = trainer.checkpoints.latest_path
        saved_hash = file_sha256(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["update"], 2)
        self.assertEqual(payload["trainer_state"]["cumulative"]["environment_plies"], 2)
        self.assertEqual(payload["trainer_state"]["model_selection"]["best_update"], 2)
        self.assertEqual(list(path.parent.glob("update_*.pt")), [])
        resumed = SelfPlayTrainer(settings, run_directory=self.root)
        self.addCleanup(resumed.logger.close)
        self.assertEqual(file_sha256(path), saved_hash)
        self.assertEqual(resumed.update, 2)
        for name in ("policy", "layout", "critic"):
            for key, tensor in getattr(resumed, name).state_dict().items():
                torch.testing.assert_close(tensor, payload[name][key], rtol=0, atol=0)
        for name in ("policy_optimizer", "critic_optimizer"):
            restored = getattr(resumed, name).state_dict()
            self.assertEqual(restored["param_groups"], payload[name]["param_groups"])
            self.assertTrue(restored["state"])
            for key, values in restored["state"].items():
                for field, tensor in values.items():
                    torch.testing.assert_close(tensor, payload[name]["state"][key][field], rtol=0, atol=0)
        with patch("junqi.training.model_selection.run_match", return_value=match_result(1, 0, 3)) as match:
            resumed.train()
        match.assert_called_once()
        self.assertEqual(resumed.update, 4)
        self.assertEqual(resumed.cumulative["environment_plies"], 4)
        self.assertEqual(resumed.model_selection.state["best_update"], 2)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["reason"], "after_model_selection")
        self.assertEqual(payload["trainer_state"]["model_selection"]["last_completed_environment_plies"], 4)

    def test_evaluation_only_retries_interrupted_final_match_from_saved_candidate(self):
        from junqi.training.trainer import SelfPlayTrainer
        settings = training_settings(
            checkpoint_policy="evaluation", total_updates=2,
            target_environment_plies=2, arena_interval_environment_plies=2,
        )
        trainer = SelfPlayTrainer(settings, run_directory=self.root)
        self.addCleanup(trainer.logger.close)
        with patch("junqi.training.model_selection.run_match", side_effect=InterruptedError), \
                patch.object(trainer, "save_checkpoint", wraps=trainer.save_checkpoint) as save:
            trainer.train()
        self.assertEqual([call.kwargs["reason"] for call in save.call_args_list], ["before_model_selection"])
        payload = torch.load(trainer.checkpoints.latest_path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["update"], 2)
        self.assertEqual(payload["trainer_state"]["model_selection"]["last_completed_environment_plies"], 0)
        self.assertEqual(payload["trainer_state"]["model_selection"]["best_update"], 0)
        resumed = SelfPlayTrainer(settings, run_directory=self.root)
        self.addCleanup(resumed.logger.close)
        with patch("junqi.training.model_selection.run_match", return_value=match_result()) as match:
            resumed.train()
        match.assert_called_once()
        self.assertEqual(resumed.update, 2)
        self.assertEqual(resumed.cumulative["environment_plies"], 2)
        self.assertEqual(resumed.model_selection.state["last_completed_environment_plies"], 2)

    def test_switching_existing_run_to_evaluation_only_keeps_old_champion_and_checkpoint(self):
        from junqi.training.trainer import SelfPlayTrainer
        settings = training_settings()
        trainer = SelfPlayTrainer(settings, run_directory=self.root)
        trainer.logger.close()
        old_latest = file_sha256(trainer.checkpoints.latest_path)
        old_best = file_sha256(trainer.model_selection.best_path)
        resumed = SelfPlayTrainer(replace(settings, checkpoint_policy="evaluation"), run_directory=self.root)
        self.addCleanup(resumed.logger.close)
        self.assertEqual(file_sha256(resumed.checkpoints.latest_path), old_latest)
        self.assertEqual(file_sha256(resumed.model_selection.best_path), old_best)
        self.assertIsNone(resumed.model_selection._pending_baseline)

    def test_environment_threshold_ignores_update_percentage_and_counts_all_branches(self):
        settings = training_settings(arena_interval_environment_plies=50_000_000,
                                     target_environment_plies=3_000_000_000, arena_games=100)
        selection, policy, layout = self.selection(settings=settings)
        self.assertEqual(selection.state["best_update"], 0)
        self.assertIsNone(selection.due_milestone(update=100, cumulative={"environment_plies": 49_999_999}))
        self.assertIsNone(selection.due_milestone(update=0, cumulative={}))
        cumulative = {"environment_plies": 50_000_000, "base_plies": 10_000_000,
                      "continuation_plies": 40_000_000}
        self.assertEqual(selection.due_milestone(update=1, cumulative=cumulative), 50_000_000)
        with patch("junqi.training.model_selection.run_match", return_value=match_result(60, 0, 40)) as run:
            report = selection.evaluate(policy, layout, update=1, cumulative=cumulative)
        self.assertEqual(report["milestone_environment_plies"], 50_000_000)
        self.assertNotIn("milestone_percent", report)
        self.assertEqual(report["environment_plies"], 50_000_000)
        self.assertAlmostEqual(run.call_args.kwargs["alpha"], .05 / 60)
        self.assertEqual(run.call_args.args[2].pairs, 25)  # 25 whole groups = 100 games globally.
        self.assertEqual(report["games"], 100)
        self.assertIsNone(selection.due_milestone(update=2, cumulative={"environment_plies": 99_999_999}))
        self.assertEqual(selection.due_milestone(update=2, cumulative={"environment_plies": 100_000_000}), 100_000_000)
        self.assertEqual(selection.due_milestone(update=2, cumulative={"environment_plies": 3_000_000_005}), 3_000_000_000)

    def test_local_after_half_schedule_boundaries_and_save_interval(self):
        config = CONFIG.with_name("local_4090_training.yaml")
        settings = TrainingSettings.from_yaml(config, "four_dark", model_scale="main")
        self.assertEqual(settings.arena_after_half_interval_environment_plies, 50_000_000)
        self.assertFalse(settings.arena_after_half_historical_only)
        self.assertTrue(settings.arena_champion_only)
        self.assertEqual(settings.arena_games, 500)
        self.assertEqual(settings.historical_eval_total_games, 500)
        for total in (True, 0, -1, 500.0, 502, 20):
            with self.subTest(total=total), self.assertRaises(ValueError):
                replace(settings, historical_eval_total_games=total).validate()
        self.assertEqual(settings.checkpoint_interval_environment_plies, 10_000_000)
        self.assertEqual(len(settings.arena_milestones), 60)
        self.assertEqual(settings.arena_milestones[29:33],
                         (1_500_000_000, 1_550_000_000, 1_600_000_000, 1_650_000_000))
        self.assertEqual(settings.arena_milestones[-1], 3_000_000_000)
        from junqi.training.model_selection import ModelSelection
        selection = ModelSelection(settings, self.root, self.context)
        for progress, expected in ((49_999_999, None), (50_000_000, 50_000_000),
                (1_499_999_999, 1_450_000_000), (1_500_000_000, 1_500_000_000),
                (1_524_999_999, 1_500_000_000), (1_525_000_000, 1_500_000_000),
                (1_549_999_999, 1_500_000_000), (1_550_000_000, 1_550_000_000),
                (3_000_000_099, 3_000_000_000)):
            self.assertEqual(selection.due_milestone(update=10**9, cumulative={"environment_plies": progress}), expected)
        for invalid in (True, 0, -1, 1.5, 50_000_001):
            with self.assertRaises(ValueError):
                replace(settings, arena_after_half_interval_environment_plies=invalid).validate()
        self.assertIsNone(TrainingSettings.from_yaml(config, "two_player").arena_after_half_interval_environment_plies)
        self.assertFalse(TrainingSettings.from_yaml(config, "two_player").arena_after_half_historical_only)
        self.assertFalse(TrainingSettings.from_yaml(config, "four_dark", tiny=True).arena_after_half_historical_only)
        for changes in ({"arena_after_half_historical_only": "true"}, {"historical_enabled": False},
                        {"historical_start_fraction": .6}):
            with self.assertRaises(ValueError):
                replace(settings, **{"arena_champion_only": False, "arena_observational_only": True,
                                    "arena_after_half_historical_only": True, **changes}).validate()

    def test_cli_after_half_interval_and_explicit_percentage_override(self):
        from junqi.training.cli import main
        config = str(CONFIG.with_name("local_4090_training.yaml"))
        for options, expected in (([], 50_000_000),
                (["--arena-after-half-interval-environment-plies", "10000000"], 10_000_000),
                (["--arena-after-half-interval-environment-plies", "0"], None),
                (["--arena-start-percent", "30"], None)):
            with patch("junqi.training.cli.SelfPlayTrainer") as trainer:
                main(["--config", config, "--mode", "four_dark", "--device", "cpu", *options])
            self.assertEqual(trainer.call_args.args[0].arena_after_half_interval_environment_plies, expected)
            self.assertFalse(trainer.call_args.args[0].arena_after_half_historical_only)
            self.assertTrue(trainer.call_args.args[0].arena_champion_only)
        with patch("junqi.training.cli.SelfPlayTrainer") as trainer:
            main(["--config", config, "--mode", "four_dark", "--device", "cpu", "--no-arena-after-half-historical-only"])
        self.assertFalse(trainer.call_args.args[0].arena_after_half_historical_only)

    def test_piecewise_matches_have_distinct_seeds_snapshots_and_resume_once(self):
        settings = training_settings(target_environment_plies=400, arena_interval_environment_plies=100,
                                     arena_after_half_interval_environment_plies=50)
        selection, policy, layout = self.selection(settings=settings)
        self.assertEqual(settings.arena_milestones, (100, 200, 250, 300, 350, 400))
        with patch("junqi.training.model_selection.run_match", side_effect=lambda *a, **k: match_result()) as run:
            for update, progress in enumerate(settings.arena_milestones, 1):
                report = selection.evaluate(policy, layout, update=update, cumulative={"environment_plies": progress})
                self.assertEqual(report["milestone_environment_plies"], progress)
            seeds = [call.args[2].seed for call in run.call_args_list]
            self.assertEqual(len(set(seeds)), 6)
            self.assertTrue(all(call.kwargs["alpha"] == .05 / 6 for call in run.call_args_list))
        self.assertEqual(len(list((selection.directory / "snapshots").glob("candidate_*.pt"))), 6)
        resumed, _, _ = self.selection(settings=settings, update=6, cumulative={"environment_plies": 400})
        self.assertIsNone(resumed.due_milestone(update=7, cumulative={"environment_plies": 400}))
        self.assertEqual(resumed.state["rounds"], selection.state["rounds"])

    def test_all_four_player_profiles_keep_fifty_million_step_evaluation_intervals(self):
        for name in ("bootstrap.yaml", "local_4090_training.yaml", "ppo_4090_throughput.yaml"):
            for mode in ("four_dark", "double_open"):
                with self.subTest(config=name, mode=mode):
                    settings = TrainingSettings.from_yaml(CONFIG.with_name(name), mode)
                    self.assertEqual(tuple(settings.arena_milestones),
                                     tuple(range(50_000_000, 3_000_000_001, 50_000_000)))

    def test_reduced_schedule_preserves_completed_rounds_alpha_and_seed_slots(self):
        old = training_settings(target_environment_plies=400, arena_interval_environment_plies=100,
                                arena_after_half_interval_environment_plies=50)
        selection, policy, layout = self.selection(settings=old)
        with patch("junqi.training.model_selection.run_match", return_value=match_result()) as run:
            for update, progress in enumerate((100, 200, 250), 1):
                selection.evaluate(policy, layout, update=update, cumulative={"environment_plies": progress})
        prior_seeds = [call.args[2].seed for call in run.call_args_list]
        prior_state = copy.deepcopy(selection.state)
        report_hashes = {name: file_sha256(selection.directory / name) for name in prior_state["rounds"]}
        settings = replace(old, arena_after_half_interval_environment_plies=100)
        resumed, policy, layout = self.selection(settings=settings, update=3, cumulative={"environment_plies": 250})
        self.assertEqual(resumed.state["rounds"], prior_state["rounds"])
        self.assertEqual(file_sha256(resumed.best_path), prior_state["best_sha256"])
        self.assertEqual(resumed.state["last_completed_environment_plies"], 250)
        self.assertEqual(resumed._round_alpha(), .05 / 6)
        self.assertEqual(resumed.state["evaluation_seed_milestones"], list(old.arena_milestones))
        self.assertEqual(report_hashes, {name: file_sha256(resumed.directory / name) for name in report_hashes})
        self.assertIsNone(resumed.due_milestone(update=4, cumulative={"environment_plies": 299}))
        with patch("junqi.training.model_selection.run_match", return_value=match_result()) as run:
            resumed.evaluate(policy, layout, update=4, cumulative={"environment_plies": 300})
            self.assertIsNone(resumed.evaluate(policy, layout, update=5, cumulative={"environment_plies": 350}))
            resumed.evaluate(policy, layout, update=6, cumulative={"environment_plies": 400})
        new_seeds = [call.args[2].seed for call in run.call_args_list]
        self.assertTrue(set(prior_seeds).isdisjoint(new_seeds))
        self.assertEqual(new_seeds, [old.arena_seed + 4 * 6, old.arena_seed + 6 * 6])
        restarted, _, _ = self.selection(settings=settings, update=6, cumulative={"environment_plies": 400})
        self.assertEqual(len(restarted.state["evaluation_schedule_migrations"]), 1)
        self.assertEqual(restarted._round_alpha(), .05 / 6)
        self.assertIsNone(restarted.due_milestone(update=7, cumulative={"environment_plies": 400}))

    def test_reduced_schedule_does_not_allow_unrelated_protocol_changes(self):
        old = training_settings(target_environment_plies=400, arena_interval_environment_plies=100,
                                arena_after_half_interval_environment_plies=50)
        selection, _, _ = self.selection(settings=old)
        before = selection.state_path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "schedule/budget changed"):
            self.selection(settings=replace(old, arena_after_half_interval_environment_plies=100, arena_games=8))
        self.assertEqual(selection.state_path.read_bytes(), before)

    def test_game_budget_migration_keeps_reports_champion_alpha_and_uses_new_seeds(self):
        old = training_settings(target_environment_plies=400, arena_interval_environment_plies=100,
                                arena_games=100)
        selection, policy, layout = self.selection(settings=old)
        with patch("junqi.training.model_selection.run_match", return_value=match_result(58, 28, 14)):
            selection.evaluate(policy, layout, update=1, cumulative={"environment_plies": 100})
        state = selection.state_dict()
        reports = {name: file_sha256(selection.directory / name) for name in state["rounds"]}
        updated = replace(old, arena_games=500, historical_eval_total_games=500)
        resumed, policy, layout = self.selection(settings=updated, update=1, cumulative={"environment_plies": 120})
        for key in ("best_snapshot", "best_sha256", "best_update", "rounds", "last_evaluated_update"):
            self.assertEqual(resumed.state[key], state[key])
        self.assertEqual(resumed._round_alpha(), selection._round_alpha())
        self.assertEqual(reports, {name: file_sha256(resumed.directory / name) for name in reports})
        self.assertEqual(resumed.state["contract"]["games"], 500)
        with patch("junqi.training.model_selection.run_match", return_value=match_result(300, 100, 100)) as run:
            report = resumed.evaluate(policy, layout, update=2, cumulative={"environment_plies": 200})
        self.assertEqual(run.call_args.args[2].pairs, 125)
        old_reserved_end = old.arena_seed + (len(old.arena_milestones) + 1) * 6 * (old.historical_eval_games // 4)
        self.assertGreater(run.call_args.args[2].seed, old_reserved_end)
        self.assertEqual(report["games"], 500)
        restarted, _, _ = self.selection(settings=updated, update=2, cumulative={"environment_plies": 200})
        self.assertEqual(len(restarted.state["evaluation_game_budget_migrations"]), 1)
        self.assertIsNone(restarted.due_milestone(update=3, cumulative={"environment_plies": 250}))
        self.assertEqual(reports, {name: file_sha256(restarted.directory / name) for name in reports})

    def test_game_budget_cannot_change_an_unfinished_evaluation(self):
        old = training_settings(target_environment_plies=400, arena_interval_environment_plies=100)
        selection, _, _ = self.selection(settings=old)
        before = selection.state_path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "finish the pending evaluation"):
            self.selection(settings=replace(old, arena_games=500), update=1,
                           cumulative={"environment_plies": 100})
        self.assertEqual(selection.state_path.read_bytes(), before)

    def test_frequency_migration_preserves_champion_and_spent_alpha_budget(self):
        old = training_settings(target_environment_plies=400, arena_interval_environment_plies=100)
        selection, policy, layout = self.selection(settings=old)
        with patch("junqi.training.model_selection.run_match", return_value=match_result()):
            selection.evaluate(policy, layout, update=1, cumulative={"environment_plies": 100})
        best_hash, reports = file_sha256(selection.best_path), list(selection.state["rounds"])
        settings = replace(old, arena_after_half_interval_environment_plies=50)
        resumed, policy, layout = self.selection(settings=settings, update=1, cumulative={"environment_plies": 100})
        self.assertEqual(file_sha256(resumed.best_path), best_hash)
        self.assertEqual(resumed.state["rounds"], reports)
        self.assertEqual(resumed._round_alpha(), (.05 - .05 / 4) / 5)
        self.assertIsNone(resumed.due_milestone(update=100, cumulative={"environment_plies": 199}))
        with patch("junqi.training.model_selection.run_match", side_effect=lambda *a, **k: match_result()) as run:
            resumed.evaluate(policy, layout, update=2, cumulative={"environment_plies": 200})
            resumed.evaluate(policy, layout, update=3, cumulative={"environment_plies": 250})
        self.assertNotEqual(run.call_args_list[0].args[2].seed, run.call_args_list[1].args[2].seed)
        self.assertEqual(len(resumed.state["rounds"]), 3)

    def test_uniform_schedule_keeps_evaluation_and_checkpoint_boundaries_separate(self):
        from junqi.training.trainer import SelfPlayTrainer
        settings = training_settings(target_environment_plies=16, total_updates=8, anchor_batch=2,
            checkpoint_interval_environment_plies=2, arena_interval_environment_plies=4,
            arena_after_half_interval_environment_plies=4)
        trainer = SelfPlayTrainer(settings, run_directory=self.root)
        saved = []
        save = trainer.save_checkpoint
        def remember(**kwargs):
            saved.append((trainer.cumulative["environment_plies"], kwargs["reason"]))
            return save(**kwargs)
        with patch.object(trainer, "save_checkpoint", side_effect=remember), \
             patch("junqi.training.model_selection.run_match", side_effect=lambda *a, **k: match_result()):
            trainer.train()
        reports = [json.loads((trainer.model_selection.directory / path).read_text())
                   for path in trainer.model_selection.state["rounds"]]
        self.assertEqual([r["milestone_environment_plies"] for r in reports], [4, 8, 12, 16])
        for progress in (4, 8, 12, 16):
            self.assertIn((progress, "before_model_selection"), saved)
            self.assertIn((progress, "after_model_selection"), saved)
        self.assertIn((2, "periodic"), saved)
        self.assertIn((6, "periodic"), saved)
        self.assertIn((10, "periodic"), saved)
        self.assertIn((14, "periodic"), saved)

    def test_environment_round_skips_missing_snapshots_and_is_not_repeated_on_resume(self):
        settings = training_settings(arena_interval_environment_plies=50_000_000,
                                     target_environment_plies=3_000_000_000)
        selection, policy, layout = self.selection(settings=settings)
        cumulative = {"environment_plies": 150_000_007}
        with patch("junqi.training.model_selection.run_match", return_value=match_result()):
            report = selection.evaluate(policy, layout, update=3, cumulative=cumulative)
        self.assertEqual(report["skipped_milestones"], [50_000_000, 100_000_000])
        resumed, policy, layout = self.selection(settings=settings, update=3, cumulative=cumulative)
        with patch("junqi.training.model_selection.run_match") as run:
            self.assertIsNone(resumed.evaluate(policy, layout, update=3, cumulative=cumulative))
        run.assert_not_called()
        self.assertEqual(resumed.state["last_completed_environment_plies"], 150_000_000)

    def test_percentage_to_environment_migration_keeps_champion_history_and_separates_seeds(self):
        old_settings = training_settings(target_environment_plies=3_000_000_000, arena_games=8)
        selection, policy, layout = self.selection(settings=old_settings)
        with patch("junqi.training.model_selection.run_match", return_value=match_result(6, 0, 2)) as old_run:
            selection.evaluate(policy, layout, update=30, cumulative={"environment_plies": 900_000_000})
        best_hash = file_sha256(selection.best_path)
        old_rounds = list(selection.state["rounds"])
        settings = training_settings(target_environment_plies=3_000_000_000,
                                     arena_interval_environment_plies=50_000_000, arena_games=4)
        resumed, policy, layout = self.selection(settings=settings, update=30,
                                                cumulative={"environment_plies": 900_000_000})
        self.assertEqual(file_sha256(resumed.best_path), best_hash)
        self.assertEqual(resumed.state["rounds"], old_rounds)
        self.assertEqual(resumed.state["previous_schedule"]["contract"]["games"], 8)
        self.assertEqual(resumed.due_milestone(update=30, cumulative={"environment_plies": 900_000_000}), 900_000_000)
        with patch("junqi.training.model_selection.run_match", return_value=match_result()) as new_run:
            report = resumed.evaluate(policy, layout, update=30, cumulative={"environment_plies": 900_000_000})
        old_reserved_end = old_settings.arena_seed + 101 * 6 * 2
        self.assertGreater(new_run.call_args.args[2].seed, old_reserved_end)
        self.assertGreater(new_run.call_args.args[2].seed, old_run.call_args.args[2].seed)
        self.assertEqual(len(resumed.state["rounds"]), 2)
        self.assertEqual(report["statistical_family"], "environment_schedule_after_percentage")
        self.assertTrue((resumed.directory / old_rounds[0]).exists())

    def test_environment_training_selects_and_resumes_at_exact_counter(self):
        from junqi.training.trainer import SelfPlayTrainer
        for mode in (TrainingMode.FOUR_DARK, TrainingMode.DOUBLE_OPEN):
            with self.subTest(mode=mode):
                settings = training_settings(mode, total_updates=2, anchor_batch=4,
                                             target_environment_plies=8,
                                             arena_interval_environment_plies=4)
                trainer = SelfPlayTrainer(settings, run_directory=self.root / mode.value)
                self.addCleanup(trainer.logger.close)
                trainer.train()
                state = trainer.model_selection.state
                reports = [json.loads((trainer.model_selection.directory / path).read_text())
                           for path in state["rounds"]]
                self.assertEqual([report["milestone_environment_plies"] for report in reports], [4, 8])
                self.assertEqual([report["games"] for report in reports], [4, 4])
                self.assertEqual(state["last_completed_environment_plies"], 8)
                resumed = SelfPlayTrainer(settings, run_directory=self.root / mode.value)
                self.addCleanup(resumed.logger.close)
                with patch("junqi.training.model_selection.run_match") as run:
                    resumed.train()
                run.assert_not_called()

    def test_schedule_starts_at_30_and_runs_once_per_five_percent(self):
        selection, policy, layout = self.selection()
        with patch("junqi.training.model_selection.run_match", return_value=match_result()) as run:
            for update in (0, 1, 29):
                self.assertIsNone(selection.due_milestone(update=update, cumulative={}))
                self.assertIsNone(selection.evaluate(policy, layout, update=update, cumulative={}))
            run.assert_not_called()
            self.assertEqual(selection.due_milestone(update=30, cumulative={}), 30)
            selection.evaluate(policy, layout, update=30, cumulative={})
            self.assertEqual(selection.state["last_completed_percent"], 30)
            for update in (30, 31, 34):
                self.assertIsNone(selection.due_milestone(update=update, cumulative={}))
            self.assertEqual(selection.due_milestone(update=35, cumulative={}), 35)
            selection.evaluate(policy, layout, update=35, cumulative={})
            self.assertEqual(run.call_count, 2)
            self.assertEqual(selection.state["last_completed_percent"], 35)

    def test_progress_uses_the_counter_that_stops_each_training_algorithm(self):
        for mode, counter, irrelevant in (
            (TrainingMode.FOUR_DARK, "environment_plies", "continuation_plies"),
            (TrainingMode.TWO_PLAYER, "continuation_plies", "environment_plies"),
        ):
            with self.subTest(mode=mode):
                # Avoid sharing durable state between incompatible modes.
                self.root = Path(self.temporary.name) / mode.value
                selection, _, _ = self.selection(
                    settings=training_settings(mode, total_updates=1000, target_continuation_plies=1000)
                )
                self.assertIsNone(selection.due_milestone(update=1, cumulative={counter: 299, irrelevant: 900}))
                self.assertEqual(selection.due_milestone(update=1, cumulative={counter: 300}), 30)
                self.assertEqual(selection.due_milestone(update=350, cumulative={counter: 1}), 35)

    def test_total_environment_grpo_budget_and_old_ppo_contract_resume(self):
        settings = training_settings(TrainingMode.TWO_PLAYER, total_updates=1000,
                                     target_environment_plies=1000)
        selection, _, _ = self.selection(settings=settings)
        self.assertEqual(selection.due_milestone(update=1, cumulative={
            "base_plies": 60, "continuation_plies": 240, "environment_plies": 300,
        }), 30)

        self.root = Path(self.temporary.name) / "old_ppo_contract"
        settings = training_settings(total_updates=1000, target_environment_plies=1000)
        selection, _, _ = self.selection(settings=settings)
        saved = copy.deepcopy(selection.state)
        best_hash = file_sha256(selection.best_path)
        contract = saved["contract"]
        contract["target_continuation_plies"] = contract.pop("step_budget_target")
        contract.pop("step_budget_counter")
        selection.state_path.write_text(json.dumps(saved), encoding="utf-8")
        resumed, _, _ = self.selection(settings=settings)
        self.assertEqual(file_sha256(resumed.best_path), best_hash)
        self.assertEqual(resumed.state["contract"]["step_budget_counter"], "environment_plies")
        self.assertEqual(resumed.state["rounds"], saved["rounds"])
        self.assertNotIn("target_continuation_plies", json.loads(resumed.state_path.read_text())["contract"])

    def test_crossed_thresholds_evaluate_latest_snapshot_once_and_legacy_resume_waits(self):
        selection, policy, layout = self.selection()
        self.assertEqual(selection.due_milestone(update=42, cumulative={}), 40)
        with patch("junqi.training.model_selection.run_match", return_value=match_result()) as run:
            report = selection.evaluate(policy, layout, update=42, cumulative={})
            run.assert_called_once()
        self.assertEqual(report["skipped_milestones"], [30, 35])
        self.assertEqual(selection.state["last_completed_percent"], 40)
        self.assertIsNone(selection.due_milestone(update=44, cumulative={}))
        self.assertEqual(selection.due_milestone(update=45, cumulative={}), 45)

        self.root = Path(self.temporary.name) / "legacy"
        resumed, _, _ = self.selection(update=67)
        self.assertEqual(resumed.state["best_update"], 67)
        self.assertEqual(resumed.state["baseline_skipped_milestones"], list(range(30, 66, 5)))
        self.assertIsNone(resumed.due_milestone(update=67, cumulative={}))
        self.assertIsNone(resumed.due_milestone(update=69, cumulative={}))
        self.assertEqual(resumed.due_milestone(update=70, cumulative={}), 70)

    def test_win_promotes_both_networks_while_tie_and_loss_retain_champion(self):
        selection, policy, layout = self.selection()
        alias = self.root / "checkpoints" / "best.pt"
        previous_sha = file_sha256(alias)
        for update, result, promotes in (
            (30, match_result(3, 0, 1), True),
            (35, match_result(0, 4, 0), False),
            (40, match_result(1, 0, 3), False),
        ):
            with self.subTest(update=update):
                with torch.no_grad():
                    next(policy.parameters()).add_(1)
                    next(layout.parameters()).add_(2)
                with patch("junqi.training.model_selection.run_match", return_value=result):
                    report = selection.evaluate(policy, layout, update=update, cumulative={})
                self.assertEqual(report["promoted"], promotes)
                self.assertEqual(report["decision"], "promote" if promotes else "retain_best")
                self.assertEqual(report["score_delta_vs_best"], result["score"] - 0.5)
                self.assertEqual(report["strength_verdict"], "inconclusive")
                self.assertEqual(selection.state["best_update"], 30)
                self.assertEqual(selection.state["last_completed_percent"], update)
                self.assertEqual(file_sha256(alias), selection.state["best_sha256"])
                if promotes:
                    self.assertNotEqual(file_sha256(alias), previous_sha)
                    payload = torch.load(alias, map_location="cpu", weights_only=False)
                    for name, value in policy.state_dict().items():
                        self.assertTrue(value.equal(payload["policy"][name]))
                    for name, value in layout.state_dict().items():
                        self.assertTrue(value.equal(payload["layout"][name]))
                    previous_sha = file_sha256(alias)
                else:
                    self.assertEqual(file_sha256(alias), previous_sha)

    def test_resume_recovers_best_alias_and_does_not_repeat_completed_match(self):
        selection, policy, layout = self.selection()
        with patch("junqi.training.model_selection.run_match", return_value=match_result()):
            selection.evaluate(policy, layout, update=30, cumulative={})
        expected = selection.state_dict()
        alias = self.root / "checkpoints" / "best.pt"
        alias.write_bytes(b"interrupted alias publication")
        resumed, policy, layout = self.selection(update=30)
        self.assertEqual(resumed.state_dict(), expected)
        self.assertEqual(file_sha256(alias), expected["best_sha256"])
        with patch("junqi.training.model_selection.run_match") as run:
            self.assertIsNone(resumed.evaluate(policy, layout, update=30, cumulative={}))
            run.assert_not_called()
        self.assertEqual(resumed.due_milestone(update=35, cumulative={}), 35)
        durable = json.loads((self.root / "model_selection" / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(durable["best_update"], 30)
        self.assertEqual(durable["last_completed_percent"], 30)

    def test_interrupted_state_commit_reuses_only_a_matching_finished_report(self):
        from junqi.training.arena import atomic_json
        from junqi.training.model_selection import ModelSelection

        selection, policy, layout = self.selection()
        initial = selection.state_dict()
        best_sha = file_sha256(self.root / "checkpoints" / "best.pt")

        def interrupted_write(path, payload):
            if Path(path) == selection.state_path:
                raise OSError("interrupted state commit")
            return atomic_json(path, payload)

        with patch("junqi.training.model_selection.run_match", return_value=match_result()) as run, \
                patch("junqi.training.model_selection.atomic_json", side_effect=interrupted_write):
            with self.assertRaisesRegex(RuntimeError, "interrupted state commit"):
                selection.evaluate(policy, layout, update=30, cumulative={})
            run.assert_called_once()
        self.assertEqual(selection.state_dict(), initial)
        self.assertEqual(file_sha256(self.root / "checkpoints" / "best.pt"), best_sha)
        self.assertEqual(selection.due_milestone(update=30, cumulative={}), 30)

        resumed = ModelSelection(selection.settings, self.root, self.context)
        resumed.initialize(policy, layout, update=30, cumulative={})
        with patch("junqi.training.model_selection.run_match") as run:
            report = resumed.evaluate(policy, layout, update=30, cumulative={})
            run.assert_not_called()
        self.assertTrue(report["promoted"])
        self.assertEqual(resumed.state["last_completed_percent"], 30)
        history = (self.root / "model_selection" / "history.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(history), 1)
        self.assertEqual(json.loads(history[0])["candidate_update"], 30)

    def test_modified_champion_snapshot_fails_resume_instead_of_replacing_history(self):
        from junqi.training.model_selection import ModelSelection

        selection, policy, layout = self.selection()
        expected = selection.state_dict()
        snapshot = selection.directory / expected["best_snapshot"]
        snapshot.write_bytes(b"damaged immutable champion")
        resumed = ModelSelection(selection.settings, self.root, self.context)
        with self.assertRaisesRegex(RuntimeError, "snapshot is missing or has changed"):
            resumed.initialize(policy, layout, update=0, cumulative={})
        durable = json.loads(selection.state_path.read_text(encoding="utf-8"))
        self.assertEqual(durable, expected)

    def test_failed_or_invalid_match_preserves_champion_and_retries_milestone(self):
        selection, policy, layout = self.selection()
        before = selection.state_dict()
        alias = self.root / "checkpoints" / "best.pt"
        best_sha = file_sha256(alias)
        invalid_results = (
            {**match_result(), "games": 3},
            {**match_result(), "score": float("nan")},
            {**match_result(), "score": float("inf")},
            {**match_result(), "score": 0.5},
            {**match_result(), "wins": 4},
            {**match_result(), "losses": -1},
        )
        for invalid in invalid_results:
            with self.subTest(result=invalid):
                with patch("junqi.training.model_selection.run_match", return_value=invalid):
                    with self.assertRaises((ValueError, RuntimeError)):
                        selection.evaluate(policy, layout, update=30, cumulative={})
                self.assertEqual(selection.state_dict(), before)
                self.assertEqual(file_sha256(alias), best_sha)
                self.assertEqual(selection.due_milestone(update=30, cumulative={}), 30)
        with patch("junqi.training.model_selection.run_match", side_effect=RuntimeError("worker failed")):
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                selection.evaluate(policy, layout, update=30, cumulative={})
        self.assertEqual(selection.state_dict(), before)
        self.assertEqual(file_sha256(alias), best_sha)

    def test_real_four_player_match_preserves_training_rng_parameters_and_flags(self):
        selection, policy, layout = self.selection()
        policy.train()
        policy.board_encoder.eval()
        layout.eval()
        next(policy.parameters()).requires_grad_(False)
        modules = (policy, layout)
        model_states = [copy.deepcopy(model.state_dict()) for model in modules]
        training_flags = [[module.training for module in model.modules()] for model in modules]
        gradient_flags = [[parameter.requires_grad for parameter in model.parameters()] for model in modules]
        torch.manual_seed(174)
        random.seed(923)
        torch_state, python_state = torch.get_rng_state().clone(), random.getstate()

        result = selection.evaluate(policy, layout, update=30, cumulative={})

        self.assertIsInstance(result, dict)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(random.getstate(), python_state)
        self.assertEqual(selection.state["best_update"], 0)  # Four short games are draws.
        self.assertEqual(selection.state["last_completed_percent"], 30)
        for model, expected, training, flags in zip(modules, model_states, training_flags, gradient_flags):
            self.assertEqual([module.training for module in model.modules()], training)
            self.assertEqual([parameter.requires_grad for parameter in model.parameters()], flags)
            for name, value in model.state_dict().items():
                self.assertTrue(value.equal(expected[name]))
        shards = list((self.root / "model_selection").rglob("*.jsonl"))
        records = [json.loads(line) for shard in shards for line in shard.read_text(encoding="utf-8").splitlines()]
        game_records = [record for record in records if "rotation" in record]
        self.assertEqual(len(game_records), 4)
        self.assertEqual([record["rotation"] for record in game_records], [0, 1, 2, 3])

    def test_trainer_saves_candidate_before_match_and_completed_state_after(self):
        from junqi.training.trainer import SelfPlayTrainer

        trainer = SelfPlayTrainer(training_settings(), run_directory=self.root)
        self.addCleanup(trainer.logger.close)
        trainer.update = 30
        events = []

        def save(**kwargs):
            events.append(("save", trainer.model_selection.state["last_completed_percent"]))
            return trainer.checkpoints.latest_path

        def match(*args, **kwargs):
            events.append(("match", trainer.model_selection.state["last_completed_percent"]))
            return match_result()

        with patch.object(trainer, "save_checkpoint", side_effect=save), \
                patch("junqi.training.model_selection.run_match", side_effect=match):
            trainer._maybe_evaluate_model()
            self.assertEqual([event[0] for event in events], ["save", "match", "save"])
            self.assertLess(events[0][1], 30)
            self.assertLess(events[1][1], 30)
            self.assertEqual(events[2][1], 30)
            trainer._maybe_evaluate_model()
            self.assertEqual(len(events), 3)
        state = trainer._trainer_state()
        self.assertEqual(state["model_selection"]["last_completed_percent"], 30)

    def test_training_loop_evaluates_and_resumes_without_replaying_completed_round(self):
        from junqi.training.trainer import SelfPlayTrainer

        for mode in (TrainingMode.TWO_PLAYER, TrainingMode.FOUR_DARK):
            with self.subTest(mode=mode):
                settings = training_settings(
                    mode, total_updates=2, arena_start_percent=50, arena_interval_percent=50,
                )
                directory = self.root / mode.value
                trainer = SelfPlayTrainer(settings, run_directory=directory)
                self.addCleanup(trainer.logger.close)
                trainer.train()
                state = trainer.model_selection.state
                reports = [json.loads((trainer.model_selection.directory / name).read_text(encoding="utf-8"))
                           for name in state["rounds"]]
                self.assertEqual([report["milestone_percent"] for report in reports], [50, 100])
                self.assertEqual([report["games"] for report in reports], [4, 4])
                self.assertEqual([report["draws"] for report in reports], [4, 4])
                self.assertTrue(all(report["decision"] == "retain_best" for report in reports))
                self.assertEqual(state["best_update"], 0)
                payload = torch.load(trainer.checkpoints.latest_path, map_location="cpu", weights_only=False)
                self.assertEqual(payload["trainer_state"]["model_selection"]["last_completed_percent"], 100)
                self.assertEqual(payload["trainer_state"]["model_selection"]["best_update"], 0)
                resumed = SelfPlayTrainer(settings, run_directory=directory)
                self.addCleanup(resumed.logger.close)
                self.assertEqual(resumed.update, 2)
                self.assertEqual(resumed.model_selection.state["last_completed_percent"], 100)
                with patch("junqi.training.model_selection.run_match") as run:
                    resumed.train()
                    run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
