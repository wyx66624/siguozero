from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_arena import ArenaFixtures, TORCH_AVAILABLE


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch training extra is not installed")
class FourPlayerArenaTests(ArenaFixtures, unittest.TestCase):
    def test_four_rotations_in_each_information_mode_are_reproducible(self):
        from junqi.training.arena import MatchSettings
        from junqi.training.arena_four_player import play_group, summarize_games
        for mode in ("four_dark", "double_open"):
            with self.subTest(mode=mode):
                candidate, opponent = self.engines(mode)
                settings = MatchSettings(mode=mode, pairs=1, max_plies=4, smoke_test=True)
                before = [p.clone() for p in candidate.policy.parameters()]
                records = play_group(candidate, opponent, 0, settings)
                self.assertEqual(records, play_group(candidate, opponent, 0, settings))
                self.assertEqual([r["rotation"] for r in records], [0, 1, 2, 3])
                self.assertEqual([r["candidate_seats"] for r in records], [[0, 2], [1, 3], [0, 2], [1, 3]])
                self.assertEqual([r["action_seed"] for r in records], list(range(settings.seed + 2, settings.seed + 6)))
                summary = summarize_games(records, settings, alpha=0.05)
                self.assertEqual(summary["rotation_groups"], 1)
                self.assertEqual(summary["games"], 4)
                self.assertNotIn("pairs", summary)
                self.assertEqual(summary["statistical_unit"], "four_game_rotation_group")
                self.assertEqual(summary["result_unit"], "team_game")
                self.assertEqual(summary["verdict"], "smoke_test_not_strength_evidence")
                self.assertEqual(summary["team_scores"], {"0": 0.5, "1": 0.5})
                for old, parameter in zip(before, candidate.policy.parameters()):
                    self.assertTrue(old.equal(parameter))
                    self.assertIsNone(parameter.grad)
                    self.assertFalse(parameter.requires_grad)

    def test_each_fixed_layout_visits_all_four_physical_seats(self):
        from junqi.training.arena import MatchSettings, new_game
        from junqi.training.arena_four_player import play_group
        candidate, opponent = self.engines("four_dark")
        with patch("junqi.training.arena.new_game", wraps=new_game) as create, \
                patch.object(candidate, "sample_layouts", wraps=candidate.sample_layouts) as new_layouts, \
                patch.object(opponent, "sample_layouts", wraps=opponent.sample_layouts) as old_layouts:
            play_group(candidate, opponent, 0, MatchSettings(mode="four_dark", pairs=1, max_plies=4))
        new_layouts.assert_called_once_with(2, temperature=0.7)
        old_layouts.assert_called_once_with(2, temperature=0.7)
        initial = create.call_args_list[0].kwargs["setups"]
        for rotation, call in enumerate(create.call_args_list):
            for seat, setup in enumerate(call.kwargs["setups"]):
                self.assertIs(setup, initial[(seat - rotation) % 4])

    def test_team_dispatch_keeps_four_dark_and_double_open_information_isolated(self):
        from junqi.training.arena import MatchSettings
        from junqi.training.arena_four_player import play_group
        from junqi.training.encoding import PolicyState
        for mode in ("four_dark", "double_open"):
            candidate, opponent = self.engines(mode)
            calls = []

            def sample(label, states, **kwargs):
                self.assertEqual(len(states), 1)
                state = states[0]
                self.assertIsInstance(state, PolicyState)
                self.assertEqual(state.mode.value, mode)
                initial = state.records[0].board_codes
                self.assertEqual(len(initial), 129)
                self.assertFalse(any(62 <= code <= 73 or 126 <= code <= 137 for code in initial))
                if mode == "four_dark":
                    self.assertFalse(any(94 <= code <= 105 for code in initial))
                    self.assertIn(2, initial)  # ally is hidden, too
                else:
                    self.assertEqual(sum(94 <= code <= 105 for code in initial), 25)
                calls.append(label)
                return [[state.legal_actions[0]]], []

            with patch.object(candidate.actor, "sample", side_effect=lambda states, **kw: sample("new", states, **kw)), \
                    patch.object(opponent.actor, "sample", side_effect=lambda states, **kw: sample("old", states, **kw)):
                records = play_group(candidate, opponent, 0, MatchSettings(mode=mode, pairs=1, max_plies=4))
            self.assertEqual(calls, ["new", "old", "new", "old", "old", "new", "old", "new"] * 2)
            self.assertEqual([a["player"] for a in records[0]["actions"]], [0, 3, 2, 1])

    def test_first_flag_loss_does_not_end_team_game(self):
        from junqi import ArmPoint, GameConfig, GameVariant, InformationMode, JunqiGame, Piece, PieceType
        from junqi.training.arena import MatchSettings
        from junqi.training.arena_four_player import play_group
        candidate, opponent = self.engines("four_dark")

        def position(_mode, **kwargs):
            pieces = {ArmPoint(owner, 6, 2): Piece(owner, PieceType.FLAG) for owner in range(4)}
            pieces[ArmPoint(1, 5, 2)] = Piece(0, PieceType.ENGINEER)
            for owner in (1, 2, 3):
                pieces[ArmPoint(owner, 2, 3)] = Piece(owner, PieceType.COMMANDER)
            return JunqiGame.from_position(GameConfig(
                variant=GameVariant.FOUR_PLAYER, information_mode=InformationMode.FOUR_DARK,
                max_plies=2,
            ), pieces)

        def sample(states, **kwargs):
            legal = states[0].legal_actions
            return [[(51, 56) if (51, 56) in legal else legal[0]]], []

        with patch("junqi.training.arena.new_game", side_effect=position), \
                patch.object(candidate.actor, "sample", side_effect=sample), \
                patch.object(opponent.actor, "sample", side_effect=sample):
            records = play_group(candidate, opponent, 0, MatchSettings(mode="four_dark", pairs=1, max_plies=2))
        for record in records:
            self.assertEqual([a["player"] for a in record["actions"]], [0, 3])
            self.assertFalse(record["active_players"][1])
            self.assertTrue(record["active_players"][3])
            self.assertEqual(record["terminal_reason"], "max_plies_draw")
            self.assertEqual(record["candidate_reward"], 0)

    def test_eliminated_ally_still_receives_final_team_win(self):
        from junqi import ArmPoint, GameConfig, GameVariant, InformationMode, JunqiGame, Piece, PieceType
        from junqi.training.arena import MatchSettings
        from junqi.training.arena_four_player import play_group, summarize_games
        candidate, opponent = self.engines("four_dark")

        def position(_mode=None, **kwargs):
            return JunqiGame.from_position(GameConfig(
                variant=GameVariant.FOUR_PLAYER, information_mode=InformationMode.FOUR_DARK,
                max_plies=4,
            ), {
                ArmPoint(2, 6, 2): Piece(2, PieceType.FLAG),
                ArmPoint(3, 6, 2): Piece(3, PieceType.FLAG),
                ArmPoint(3, 5, 2): Piece(2, PieceType.ENGINEER),
                ArmPoint(3, 2, 3): Piece(3, PieceType.COMMANDER),
            }, current_player=2, active_players=(False, False, True, True))

        game = position()
        capture = next(action for action in game.legal_actions()
                       if game.clone().step(action).flag_captured_owner == 3)
        with patch("junqi.training.arena.new_game", side_effect=position), \
                patch.object(candidate.actor, "sample", return_value=([[capture]], [])), \
                patch.object(opponent.actor, "sample", return_value=([[capture]], [])):
            records = play_group(candidate, opponent, 0, MatchSettings(mode="four_dark", pairs=1, max_plies=4))
        self.assertEqual([r["candidate_reward"] for r in records], [1, -1, 1, -1])
        for record in records:
            self.assertFalse(record["active_players"][0])
            self.assertEqual(record["winner_team"], 0)
            self.assertEqual(record["player_rewards"], [1, -1, 1, -1])
        summary = summarize_games(records, MatchSettings(mode="four_dark", pairs=1), alpha=0.05)
        self.assertEqual((summary["wins"], summary["draws"], summary["losses"]), (2, 0, 2))

    def test_ci_uses_rotation_groups_and_rejects_wrong_teams_or_partial_groups(self):
        from junqi.training.arena import MatchSettings
        from junqi.training.arena_four_player import summarize_games
        settings = MatchSettings(mode="four_dark", pairs=3)
        records = [{
            "group_index": group, "rotation": rotation, "mode": settings.mode,
            "candidate_team": rotation % 2, "candidate_seats": [rotation % 2, rotation % 2 + 2],
            "winner_team": 0, "player_rewards": [1, -1, 1, -1],
            "candidate_reward": 1 if rotation % 2 == 0 else -1,
            "candidate_score": 1 if rotation % 2 == 0 else 0,
            "plies": 100, "terminal_reason": "team_eliminated",
        } for group in range(3) for rotation in range(4)]
        stats = summarize_games(records, settings, alpha=0.05)
        self.assertEqual((stats["rotation_groups"], stats["games"]), (3, 12))
        self.assertEqual(stats["bootstrap_ci95_descriptive"], [0.5, 0.5])
        for bad in (records[:-1], records[:-1] + [records[0]]):
            with self.assertRaisesRegex(ValueError, "unpaired"):
                summarize_games(bad, settings, alpha=0.05)
        for field, value in (("candidate_seats", [0, 1]), ("mode", "double_open"),
                             ("candidate_reward", -1), ("player_rewards", [1, -1, 0, -1])):
            bad = copy.deepcopy(records)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                summarize_games(bad, settings, alpha=0.05)

    def test_four_and_two_protocols_cannot_call_each_other(self):
        from junqi.training.arena import MatchSettings
        from junqi.training.arena_four_player import play_group as four
        from junqi.training.arena_two_player import play_group as two
        new, old = self.engines("four_dark")
        with self.assertRaises(ValueError):
            two(new, old, 0, MatchSettings(mode="four_dark", pairs=1))
        with self.assertRaises(ValueError):
            four(new, old, 0, MatchSettings(pairs=1))
        from junqi.training.modes import TrainingMode
        old.mode = TrainingMode.DOUBLE_OPEN
        with self.assertRaisesRegex(ValueError, "mix modes"):
            four(new, old, 0, MatchSettings(mode="four_dark", pairs=1))

    def test_mode_specific_manifests_snapshots_and_stores(self):
        from junqi.training.evaluate_history import ArenaStore, pin_checkpoint, read_manifest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode in ("four_dark", "double_open"):
                source = self.checkpoint(root / mode, 10, latest=True, mode=mode)
                self.assertEqual(read_manifest(source.parent, mode=mode)["mode"], mode)
                with self.assertRaises(ValueError):
                    read_manifest(source.parent)  # default two-player must reject
                with self.assertRaises(ValueError):
                    pin_checkpoint(source, root / "bad")
                store = ArenaStore(root / f"eval_{mode}", self.config(mode=mode), source)
                self.assertEqual(store.state["baseline"]["mode"], mode)
                with self.assertRaisesRegex(ValueError, "settings changed"):
                    ArenaStore(store.root, self.config(mode="two_player"), None)
                other = "double_open" if mode == "four_dark" else "four_dark"
                self.checkpoint(source.parent, 20, latest=True, mode=other)
                with self.assertRaises(ValueError):
                    store.prepare(source.parent)

    def test_independent_entrypoints_validate_modes_before_creating_outputs(self):
        from junqi.training.evaluate_two_player import evaluate as two
        from junqi.training.evaluate_four_player import evaluate as four
        with tempfile.TemporaryDirectory() as tmp, redirect_stderr(io.StringIO()):
            root = Path(tmp)
            common = ["--checkpoint-dir", str(root / "checkpoints"), "--output-dir", str(root / "eval")]
            for entry, arguments in ((two, ["--mode", "four_dark"]),
                                     (four, ["--mode", "two_player"]), (four, ["--pairs", "1"]),
                                     (two, ["--groups", "1"])):
                with self.assertRaises(SystemExit):
                    entry(common + arguments)
            self.assertFalse((root / "eval").exists())

    def test_four_player_cli_round_resume_and_seed_block_separation(self):
        from junqi.training.evaluate_four_player import evaluate
        from junqi.training.evaluate_history import sha256_file
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            for mode in ("four_dark", "double_open"):
                checkpoints, output = root / mode / "checkpoints", root / mode / "eval"
                baseline = self.checkpoint(checkpoints, 10, mode=mode)
                latest = self.checkpoint(checkpoints, 20, latest=True, mode=mode)
                digest = sha256_file(latest)
                args = ["--checkpoint-dir", str(checkpoints), "--output-dir", str(output),
                        "--baseline", str(baseline), "--mode", mode, "--device", "cpu",
                        "--groups", "1", "--max-plies", "4", "--smoke-test", "--every-updates", "5"]
                result = evaluate(args)
                match = result["matches"][0]
                self.assertEqual(result["mode"], mode)
                self.assertEqual(match["evaluation_type"], "historical_team_rotations_four_player")
                self.assertEqual((match["rotation_groups"], match["games"]), (1, 4))
                self.assertEqual(sha256_file(latest), digest)
                self.assertIn("四局整队轮转", (output / "report.md").read_text())
                with patch("junqi.training.evaluate_history.run_match") as run:
                    self.assertIn("idle", evaluate(args))
                    run.assert_not_called()
                self.checkpoint(checkpoints, 25, latest=True, mode=mode)
                second = evaluate(args)
                self.assertEqual(len(second["matches"]), 2)
                self.assertEqual(second["matches"][0]["settings"]["seed"], match["settings"]["seed"] + 6)
                # Completed matchup recovery must expect 4*groups, not 2*pairs.
                state = json.loads((output / "state.json").read_text())
                state["rounds"][-1]["status"] = "pending"
                from junqi.training.arena import atomic_json
                atomic_json(output / "state.json", state)
                with patch("junqi.training.evaluate_history.run_match") as run:
                    evaluate(args)
                    run.assert_not_called()
                self.assertEqual(len(json.loads((output / "history.json").read_text())), 2)

    @unittest.skipIf(os.name == "nt", "distributed CPU smoke is exercised under WSL/Linux")
    def test_four_player_gloo_whole_group_sharding_matches_single_process(self):
        from junqi.training.evaluate_four_player import evaluate
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            baseline = self.checkpoint(root / "checkpoints", 10, mode="four_dark")
            self.checkpoint(baseline.parent, 20, latest=True, mode="four_dark")
            args = ["--checkpoint-dir", str(baseline.parent), "--baseline", str(baseline),
                    "--device", "cpu", "--groups", "3", "--max-plies", "4", "--smoke-test",
                    "--every-updates", "5"]
            single = evaluate(args + ["--output-dir", str(root / "single")])
            process = subprocess.run([
                sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2",
                "--module", "junqi.training.evaluate_four_player", *args,
                "--output-dir", str(root / "distributed"),
            ], env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"),
                capture_output=True, text=True, timeout=90)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            distributed = json.loads((root / "distributed/latest.json").read_text())
            self.assertEqual(distributed["matches"][0]["world_size"], 2)
            self.assertEqual(distributed["matches"][0]["games"], 12)
            for key in ("score", "score_ci", "rotation_groups", "wins", "draws", "losses"):
                self.assertEqual(single["matches"][0][key], distributed["matches"][0][key])
            def records(report):
                return sorted([json.loads(line) for name in report["matches"][0]["game_shards"]
                               for line in Path(name).read_text().splitlines()],
                              key=lambda r: (r["group_index"], r["rotation"]))
            self.assertEqual(records(single), records(distributed))

    @unittest.skipIf(os.name == "nt", "Bash launcher is exercised under WSL/Linux")
    def test_separate_npu_launchers_dispatch_correct_units_and_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cann, capture = root / "cann.sh", root / "capture.sh"
            cann.write_text(":\n")
            capture.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
            capture.chmod(0o700)
            scripts = Path(__file__).resolve().parents[1] / "scripts"
            env = dict(os.environ, CANN_ENV_FILE=str(cann), PYTHON_BIN=str(capture),
                       NNODES="1", NPROC_PER_NODE="1", EVAL_OUTPUT_DIR=str(root / "eval"))
            for mode, script, module, count_flag in (
                ("two_player", "evaluate_two_player_npu.sh", "evaluate_two_player", "--pairs"),
                ("four_dark", "evaluate_four_player_npu.sh", "evaluate_four_player", "--groups"),
                ("double_open", "evaluate_four_player_npu.sh", "evaluate_four_player", "--groups"),
            ):
                process = subprocess.run(["bash", str(scripts / script), str(root / "latest.pt"), "--watch"],
                                         env={**env, "EVAL_GAME_MODE": mode}, capture_output=True, text=True, timeout=15)
                self.assertEqual(process.returncode, 0, process.stderr)
                arguments = process.stdout.splitlines()
                self.assertEqual(arguments[arguments.index("--module") + 1], f"junqi.training.{module}")
                self.assertEqual(arguments[arguments.index("--mode") + 1], mode)
                self.assertIn(count_flag, arguments)
                self.assertIn("--watch", arguments)
            bad = subprocess.run(["bash", str(scripts / "evaluate_four_player_npu.sh"), str(root / "latest.pt")],
                                 env={**env, "EVAL_GAME_MODE": "two_player"}, capture_output=True, text=True, timeout=15)
            self.assertEqual(bad.returncode, 2)


if __name__ == "__main__":
    unittest.main()
