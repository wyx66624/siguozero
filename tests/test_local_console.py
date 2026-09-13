"""Real rules/CPU-worker tests and read-only monitor contracts."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import select
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

import torch

from junqi.training.inference import InferenceEngine
from junqi.training.checkpoint_format import CHECKPOINT_FORMAT_VERSION
from junqi.training.inference_snapshot import export_inference_snapshot
from junqi.training.models import GamePolicyTransformer, PieceConditionedLayoutPointerDecoder
from junqi.training.modes import new_game
from junqi.training.settings import TrainingSettings
from junqi.web.monitor import Monitor, tail_records, training_process
from junqi.web.server import Application, Handler, ThreadingHTTPServer, Worker

ROOT = Path(__file__).resolve().parents[1]


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.spec = {"id": "test", "name": "测试", "mode": "four_dark", "path": "run/four_dark/with_dead_rules"}
        self.monitor = Monitor({"runs": [self.spec]}, self.root)
        self.run = self.monitor.specs[0]["path"]
        self.run.mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, value):
        path = self.run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_stopped_process_cannot_look_running_from_old_heartbeat(self):
        self.write("resource_latest.json", {"process/pid": 999999999, "timestamp_unix": time.time(), "training/update": 50})
        result = self.monitor.run_status(self.monitor.specs[0])
        self.assertEqual(result["status"], "stopped")
        self.assertIsNone(result["cycle_rate"])
        self.assertEqual(result["heartbeat_count"], 0)

    def test_pid_reuse_and_unrelated_process_are_rejected(self):
        with patch("junqi.web.monitor.process_info", return_value={"pid": 12, "started": 100, "command": f"python -m junqi.training.cli --run-dir {self.run}"}):
            self.assertIsNone(training_process(12, self.run, 70))
            self.assertIsNotNone(training_process(12, self.run, 101))
        with patch("junqi.web.monitor.process_info", return_value={"pid": 12, "started": 100, "command": f"python -m junqi.web.cpu_game {self.run}"}):
            self.assertIsNone(training_process(12, self.run, 101))

    def test_restart_immediately_replaces_abandoned_log_progress_and_losses(self):
        now = time.time()
        self.write('checkpoints/manifest.json', {'update': 10, 'environment_plies': 1_010_000})
        (self.run/'checkpoints/latest.pt').touch()
        self.write('resolved_config.json', {'target_environment_plies': 3_000_000_000,
            'checkpoint_policy': 'periodic', 'checkpoint_interval_environment_plies': 10_000_000})
        rows = [{'update': i, 'timestamp_unix': now-500+i, 'cumulative/environment_plies': i*101000,
                 'rollout/environment_plies': 101000, 'loss/policy_total': i} for i in (10, 11, 12)]
        self.write('latest_metrics.json', rows[-1])
        (self.run/'metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        self.write('training_progress.json', {'pid': 42, 'timestamp_unix': now, 'session_started_unix': now-10,
            'update': 10, 'cumulative': {'environment_plies': 1_010_000}, 'resumed_from_update': 10})
        self.write('resource_latest.json', {'process/pid': 42, 'timestamp_unix': now,
            'training/update': 10, 'training/environment_plies': 1_010_000})
        with patch('junqi.web.monitor.training_process', side_effect=lambda pid,*a:
                   {'pid':42,'started':now-11} if pid==42 else None):
            result=self.monitor.run_status(self.monitor.specs[0])
        self.assertEqual(result['steps'], 1_010_000)
        self.assertEqual(result['update'], 10)
        self.assertEqual(result['metrics']['loss/policy_total'], 10)
        self.assertEqual(result['unsaved_steps'], 0)
        self.assertEqual(result['next_checkpoint_steps'], 10_000_000)
        self.assertEqual([r['update'] for r in result['chart']], [10])
        self.assertIsNone(result['cycle_rate'])

    def test_crash_shows_saved_progress_and_normal_stop_can_save_between_milestones(self):
        self.write('latest_metrics.json', {'update': 12, 'timestamp_unix': time.time(),
            'cumulative/environment_plies': 1_212_000, 'rollout/environment_plies': 101000})
        self.write('checkpoints/manifest.json', {'update': 10, 'environment_plies': 1_010_000})
        (self.run/'checkpoints/latest.pt').touch()
        result = self.monitor.run_status(self.monitor.specs[0])
        self.assertEqual(result['steps'], 1_010_000)
        self.assertEqual(result['discarded_steps'], 202000)
        self.assertEqual(result['update'], 10)
        self.write('checkpoints/manifest.json', {'update': 12, 'environment_plies': 1_212_000,
                                               'reason': 'completed_or_stopped'})
        result = self.monitor.run_status(self.monitor.specs[0])
        self.assertEqual(result['steps'], 1_212_000)
        self.assertEqual(result['discarded_steps'], 0)

    def test_live_unsaved_progress_is_visible_but_not_counted_as_recoverable(self):
        now=time.time()
        self.write('checkpoints/manifest.json', {'update': 10, 'environment_plies': 1_010_000})
        (self.run/'checkpoints/latest.pt').touch()
        self.write('resource_latest.json', {'process/pid': 42, 'timestamp_unix': now,
            'training/update': 12, 'training/environment_plies': 1_212_000})
        with patch('junqi.web.monitor.training_process', side_effect=lambda pid,*a:
                   {'pid':42,'started':now-100} if pid==42 else None):
            result=self.monitor.run_status(self.monitor.specs[0])
        self.assertEqual(result['steps'], 1_212_000)
        self.assertEqual(result['checkpoint_steps'], 1_010_000)
        self.assertEqual(result['unsaved_steps'], 202000)
        self.assertEqual(result['discarded_steps'], 0)

    def test_complete_cycle_rate_includes_gaps_and_uses_environment_counter(self):
        now = time.time()
        self.write("resolved_config.json", {"target_environment_plies": 3000000000, "step_budget_counter": "environment_plies"})
        rows = [{"update": i, "timestamp_unix": now - (3-i)*20,
                 "cumulative/environment_plies": i*1000, "cumulative/continuation_plies": 999999,
                 "rollout/environment_plies": 1000, "timing/update_seconds": 10} for i in (1,2,3)]
        self.write("latest_metrics.json", rows[-1])
        (self.run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        self.write("resource_latest.json", {"process/pid": 12, "timestamp_unix": now})
        with patch("junqi.web.monitor.training_process", side_effect=lambda pid,*a: {"pid":12,"started":now-100} if pid else None):
            result = self.monitor.run_status(self.monitor.specs[0])
        self.assertEqual(result["cycle_rate"], 50)
        self.assertEqual(result["inner_rate"], 100)
        self.assertEqual(result["steps"], 3000)
        self.assertEqual(result["rate_updates"], 2)

    def test_partial_rank_heartbeat_is_not_healthy(self):
        now = time.time()
        self.write("resource_latest.json", {"process/pid": 12, "timestamp_unix": now, "distributed/world_size": 2})
        with patch("junqi.web.monitor.training_process", side_effect=lambda pid,*a: {"pid":12,"started":now-100} if pid else None):
            result = self.monitor.run_status(self.monitor.specs[0])
        self.assertEqual(result["status"], "stale")
        self.assertIsNone(result["cycle_rate"])

    def test_tail_ignores_partial_record(self):
        path = self.run / "metrics.jsonl"
        path.write_text('{"update":1}\n{"update":2}\n{"update":')
        self.assertEqual([x["update"] for x in tail_records(path)], [1,2])

    def test_best_requires_evaluation_and_path_cannot_escape(self):
        snapshot = self.run / "model_selection/snapshots/base.pt"
        snapshot.parent.mkdir(parents=True)
        snapshot.touch()
        state = {"best_snapshot": "snapshots/base.pt", "best_update": 0, "rounds": []}
        self.write("model_selection/state.json", state)
        self.assertFalse(self.monitor.public_catalog()[0]["evaluated"])
        state["rounds"] = ["round.json"]
        self.write("model_selection/state.json", state)
        self.assertTrue(self.monitor.public_catalog()[0]["evaluated"])
        state["best_snapshot"] = "../../../../outside.pt"
        (self.root / "outside.pt").touch()
        self.write("model_selection/state.json", state)
        self.assertEqual(self.monitor.public_catalog(), [])
        with self.assertRaises(ValueError):
            self.monitor.resolve_checkpoint("../../outside.pt")

    def test_legacy_architecture_is_visible_but_not_playable(self):
        self.write("checkpoints/manifest.json", {"format_version":4,"update":212})
        (self.run / "checkpoints/latest.pt").touch()
        entry = self.monitor.public_catalog()[0]
        self.assertFalse(entry["available"])
        self.assertIn("v4", entry["unavailable_reason"])
        with self.assertRaises(ValueError):
            self.monitor.resolve_checkpoint(entry["id"])

    def test_after_half_catalog_prefers_latest_and_labels_old_champion_as_archived(self):
        self.write("resolved_config.json", {"arena_after_half_historical_only": True,
                   "target_environment_plies": 1000})
        self.write("checkpoints/manifest.json", {"format_version": CHECKPOINT_FORMAT_VERSION,
                   "update": 50, "environment_plies": 500})
        self.write("inference/manifest.json", {"format_version": CHECKPOINT_FORMAT_VERSION,
                   "update": 55, "file": "live.pt"})
        self.write("model_selection/state.json", {"best_snapshot": "snapshots/best.pt", "best_update": 20,
                   "rounds": ["old.json", "new.json"], "last_evaluated_update": 50,
                   "latest_evaluated_snapshot": "snapshots/evaluated.pt"})
        for name in ("checkpoints/latest.pt", "inference/live.pt", "model_selection/snapshots/best.pt",
                     "model_selection/snapshots/evaluated.pt"):
            self.write(name, {})
        entries = {e["kind"]: e for e in self.monitor.public_catalog()}
        self.assertTrue(entries["live"]["preferred"])
        self.assertNotIn("preferred", entries["best"])
        self.assertIn("归档", entries["best"]["label"])
        self.assertEqual(entries["evaluated"]["update"], 50)
        self.assertEqual(self.monitor.resolve_checkpoint(entries["live"]["id"])["update"], 55)
        self.write("latest_metrics.json", {"update": 55, "cumulative/environment_plies": 550,
                   "rollout/environment_plies": 10, "timing/update_seconds": 1, "timestamp_unix": time.time()})
        status = self.monitor.run_status(self.monitor.specs[0])
        self.assertEqual(status["evaluation_type"], "historical_only")
        self.assertTrue(status["after_half_historical_only"])

    def test_observational_first_half_uses_latest_with_immutable_old_reference(self):
        self.test_after_half_catalog_prefers_latest_and_labels_old_champion_as_archived()
        self.write("resolved_config.json", {"arena_observational_only": True,
                   "arena_historical_teammate_fraction": .5, "target_environment_plies": 3000})
        entries = {e["kind"]: e for e in self.monitor.public_catalog()}
        self.assertTrue(entries["live"]["preferred"])
        self.assertEqual(entries["best"]["label"], "固定旧基准（归档）")
        self.assertEqual(entries["best"]["update"], 20)
        self.assertIn("evaluated", entries)
        status = self.monitor.run_status(self.monitor.specs[0])
        self.assertEqual(status["evaluation_type"], "fixed_reference")
        self.assertTrue(status["observational_only"])
        self.assertEqual(status["evaluation_historical_teammate_fraction"], .5)


class CpuWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        cls.directory = Path(cls.temporary.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def snapshot(self, mode, update=7):
        settings = TrainingSettings.from_yaml(ROOT / "configs/bootstrap.yaml", mode, tiny=True,
                                              overrides={"device":"cpu"})
        policy = GamePolicyTransformer(settings.model)
        layout = PieceConditionedLayoutPointerDecoder(settings.model)
        path = export_inference_snapshot(self.directory / mode, policy, layout, settings, update)
        return settings, policy, layout, path

    def test_snapshot_is_inference_only_atomic_and_does_not_change_rng_or_weights(self):
        settings, policy, layout, path = self.snapshot("four_dark")
        original = {k:v.clone() for k,v in policy.state_dict().items()}
        rng = torch.get_rng_state().clone()
        next_path = export_inference_snapshot(self.directory / "four_dark", policy, layout, settings, 8)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        payload = torch.load(next_path, weights_only=False)
        self.assertNotIn("policy_optimizer", payload)
        self.assertNotIn("trainer_state", payload)
        self.assertEqual(payload["update"], 8)
        self.assertTrue(path.exists())
        for key,value in original.items():
            self.assertTrue(torch.equal(value, payload["policy"][key]))
        engine = InferenceEngine.from_checkpoint(next_path, device="cpu", mode="four_dark")
        self.assertEqual(engine.checkpoint_update, 8)

    def test_worker_cpu_isolation_visibility_legal_move_and_frozen_version(self):
        for mode in ("two_player", "four_dark", "double_open"):
            with self.subTest(mode=mode):
                settings, policy, layout, path = self.snapshot(mode)
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES":"0", "WORLD_SIZE":"8", "RANK":"4"}):
                    worker = Worker(ROOT, 1, self.directory)
                try:
                    state = worker.request({"op":"new", "checkpoint":str(path), "mode":mode,
                        "dead_rules_enabled":True, "threads":1, "seat":0, "seed":42})
                    self.assertEqual(state["model"]["device"], "cpu")
                    self.assertFalse(state["model"]["cuda_initialized"])
                    self.assertEqual(state["model"]["cuda_visible_devices"], "")
                    self.assertNotEqual(state["model"]["pid"], os.getpid())
                    for piece in filter(None, state["pieces"]):
                        visible = piece["owner"] == 0 or (mode == "double_open" and piece["owner"] == 2)
                        self.assertEqual(piece["visible"], visible)
                        if not visible:
                            self.assertIsNone(piece["kind"])
                            self.assertEqual(piece["name"], "暗棋")
                    before = state["ply"]
                    with self.assertRaises(ValueError):
                        worker.request({"op":"move", "action":[1,1], "expected_ply":before})
                    unchanged = worker.request({"op":"state"})
                    self.assertEqual(unchanged["ply"], before)
                    moved = worker.request({"op":"move", "action":0, "expected_ply":before})
                    self.assertGreater(moved["ply"], before)
                    self.assertTrue(moved["your_turn"] or moved["result"])
                    frames = moved["frames"]
                    self.assertEqual(frames[0]["history"][-1]["action"], [0, 0])
                    self.assertEqual(frames[0]["history"][-1]["combat"], "pass")
                    self.assertEqual(frames[0]["pieces"], state["pieces"])
                    self.assertEqual(frames[0]["passes_remaining"][0], 3)
                    self.assertEqual([frame["ply"] for frame in frames],
                                     list(range(before + 1, moved["ply"] + 1)))
                    self.assertEqual(frames[0]["history"][-1]["actor"], 0)
                    self.assertEqual(frames[-1]["pieces"], moved["pieces"])
                    for frame in frames:
                        self.assertNotIn("frames", frame)
                        self.assertEqual(frame["history"][-1]["ply"], frame["ply"])
                        for piece in filter(None, frame["pieces"]):
                            if not piece["visible"]:
                                self.assertIsNone(piece["kind"])
                                self.assertEqual(piece["name"], "暗棋")
                    self.assertNotIn("frames", worker.request({"op":"state"}))
                    with self.assertRaises(ValueError):
                        worker.request({"op":"move", "action":state["legal_actions"][0], "expected_ply":before})
                    export_inference_snapshot(self.directory / mode, policy, layout, settings, 99)
                    self.assertEqual(worker.request({"op":"state"})["model"]["update"], 7)
                    self.assertTrue(worker.request({"op":"replay"})["history"])
                finally:
                    worker.close()
                self.assertIsNotNone(worker.process.poll())

    def test_completed_game_and_rotated_human_seat(self):
        _, _, _, path = self.snapshot("two_player")
        worker = Worker(ROOT, 1, self.directory)
        try:
            state = worker.request({"op":"new", "checkpoint":str(path), "mode":"two_player",
                "dead_rules_enabled":True, "threads":1, "seat":1, "seed":123})
            self.assertTrue(state["your_turn"])
            self.assertEqual(state["current_player"], 0)
            self.assertEqual([frame["ply"] for frame in state["frames"]],
                             list(range(state["ply"] + 1)))
            self.assertIsNone(state["frames"][0]["result"])
            for _ in range(1000):
                if state["result"]:
                    break
                state = worker.request({"op":"move", "action":state["legal_actions"][0], "expected_ply":state["ply"]})
            self.assertIsNotNone(state["result"])
            self.assertEqual(state["legal_actions"], [])
            self.assertIn(state["result"]["outcome"], ("win","loss","draw"))
            self.assertEqual(state["frames"][-1]["result"], state["result"])
            review = worker.request({"op": "replay"})["review"]
            self.assertEqual(len(review["steps"]), state["ply"])
            self.assertEqual(review["initial"]["human_seat"], 1)
            self.assertTrue(all(p["visible"] and p["kind"] for p in review["initial"]["pieces"] if p))
            self.assertFalse(review["initial"]["model"]["cuda_initialized"])
        finally:
            worker.close()

    @unittest.skipUnless(os.name == "posix", "local console launcher uses WSL/Linux")
    def test_service_shutdown_reaps_its_cpu_worker(self):
        _, _, _, path = self.snapshot("four_dark")
        config = {"runs":[{"id":"test", "name":"测试", "mode":"four_dark",
                  "path":str(path.parent.parent.relative_to(ROOT))}], "cpu_threads":1}
        config_path = self.directory / "console.json"
        config_path.write_text(json.dumps(config))
        process = subprocess.Popen([sys.executable,"-m","junqi.web.server","--root",str(ROOT),
            "--config",str(config_path),"--port","0","--no-service-metadata"], cwd=ROOT,
            env={**os.environ,"PYTHONPATH":str(ROOT/"src"),"CUDA_VISIBLE_DEVICES":""},
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        try:
            self.assertTrue(select.select([process.stdout],[],[],10)[0])
            url = process.stdout.readline().strip().split()[-1]
            request = urllib.request.Request(url+"/api/game/new", data=json.dumps({"checkpoint_id":"test:live"}).encode(),
                headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                state = json.load(response)
            cpu_pid = state["model"]["pid"]
            process.terminate()
            self.assertEqual(process.wait(timeout=10), 0)
            from junqi.web.monitor import process_info
            self.assertIsNone(process_info(cpu_pid))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()


class SpectatorTests(unittest.TestCase):
    @staticmethod
    def make_game(mode="four_dark"):
        from junqi import ArmPoint, GameConfig, GameVariant, InformationMode, JunqiGame, Piece, PieceType
        from junqi.training.encoding import GameHistory
        # Import the CPU entry point without changing this test process's environment.
        with patch.dict(os.environ):
            from junqi.web.cpu_game import CpuGame

        pieces = {ArmPoint(owner, 6, 2): Piece(owner, PieceType.FLAG) for owner in range(4)}
        pieces.update({ArmPoint(owner, 2, 3): Piece(owner, PieceType.COMMANDER) for owner in range(4)})
        pieces[ArmPoint(0, 5, 2)] = Piece(3, PieceType.ENGINEER)
        pieces[ArmPoint(3, 5, 2)] = Piece(2, PieceType.ENGINEER)
        pieces[ArmPoint(1, 5, 2)] = Piece(2, PieceType.ENGINEER)
        game = JunqiGame.from_position(GameConfig(variant=GameVariant.FOUR_PLAYER,
            information_mode=InformationMode(mode)), pieces, current_player=0)
        # The human loses their flag, then their partner wins on a later round.
        script = iter([(3, 0), (2, 3), (1, None), (2, None), (1, None), (2, 1)])

        def step(game, history, **kwargs):
            actor, target = next(script)
            assert game.current_player == actor
            board = game.board_for(actor)
            action = (0, 0) if target is None else (
                board.encode(ArmPoint(target, 5, 2)), board.encode(ArmPoint(target, 6, 2)))
            game.step(action)
            history.append_after_step(game)

        cpu = CpuGame.__new__(CpuGame)
        cpu.human, cpu.temperature, cpu.seed = 0, 0.7, 42
        cpu.game, cpu.history = game, GameHistory.initialize(game, mode)
        cpu.engine = SimpleNamespace(step=step)
        cpu.metadata = {"mode": mode, "update": 0, "threads": 1, "pid": os.getpid(),
                        "seed": 42, "device": "cpu", "cuda_initialized": False, "dead_rules_enabled": True}
        cpu.inference_seconds, cpu.ai_moves = 0.0, 0
        return cpu

    def test_captured_human_flag_can_be_followed_to_partner_victory(self):
        for mode in ("four_dark", "double_open"):
            with self.subTest(mode=mode):
                cpu = self.make_game(mode)
                eliminated = cpu.command({"op": "move", "action": [0, 0], "expected_ply": 0})
                self.assertFalse(eliminated["active_players"][0])
                self.assertTrue(eliminated["active_players"][2])
                self.assertFalse(eliminated["your_turn"])
                self.assertEqual(eliminated["legal_actions"], [])
                self.assertIsNone(eliminated["result"])
                self.assertEqual([frame["ply"] for frame in eliminated["frames"]], [1, 2, 3, 4, 5])
                self.assertEqual(eliminated["frames"][1]["history"][-1]["flag_captured_owner"], 0)
                finished = cpu.command({"op": "advance"})
                self.assertEqual([frame["ply"] for frame in finished["frames"]], [5, 6, 7])
                self.assertEqual(finished["result"], {"reason": "team_eliminated", "outcome": "win"})
                self.assertFalse(finished["active_players"][0])
                for frame in eliminated["frames"] + finished["frames"]:
                    for piece in filter(None, frame["pieces"]):
                        if not piece["visible"]:
                            self.assertIsNone(piece["kind"])
                            self.assertEqual(piece["name"], "暗棋")

    @staticmethod
    def make_immobile_game(mode="four_dark", *, human=0, unblock=False):
        from junqi import ArmPoint, GameConfig, GameVariant, InformationMode, JunqiGame, Piece, PieceType
        from junqi.training.encoding import GameHistory
        cpu = SpectatorTests.make_game(mode)
        point = lambda arm, row, column: ArmPoint((arm + human) % 4, row, column)
        pieces = {ArmPoint(owner, 6, 2): Piece(owner, PieceType.FLAG) for owner in range(4)}
        pieces.update({ArmPoint(owner, 2, 3): Piece(owner, PieceType.COMMANDER)
                       for owner in range(4) if owner != human})
        pieces[point(0, 2, 2) if unblock else point(3, 6, 4)] = Piece(human, PieceType.COMMANDER)
        for row, column in ((5, 1), (6, 1), (6, 5)):
            pieces[point(0, row, column)] = Piece(human, PieceType.MINE)
        if unblock:
            neighbors = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]
            kinds = ([PieceType.ENGINEER] + [PieceType.PLATOON_COMMANDER] * 3
                     + [PieceType.COMPANY_COMMANDER] * 3 + [PieceType.BATTALION_COMMANDER])
            for (row, column), kind in zip(neighbors, kinds):
                pieces[point(0, row, column)] = Piece((human + 2) % 4, kind)
        game = JunqiGame.from_position(GameConfig(variant=GameVariant.FOUR_PLAYER,
            information_mode=InformationMode(mode)), pieces, current_player=human)

        def step(game, history, **kwargs):
            actor = game.current_player
            assert actor != human, "the model must not choose the human's compulsory pass"
            legal = game.legal_actions()
            if unblock and actor == (human + 2) % 4:
                board = game.board_for(actor)
                action = (board.encode(point(0, 3, 1)), board.encode(point(0, 4, 1)))
            else:
                action = next(action for action in ((0, 0), (7, 6), (6, 7)) if action in legal)
            game.step(action)
            history.append_after_step(game)

        cpu.human = human
        cpu.game, cpu.history = game, GameHistory.initialize(game, mode)
        cpu.engine = SimpleNamespace(step=step)
        return cpu

    def test_immobile_seat_spectates_before_pass_budget_is_exhausted(self):
        for mode in ("four_dark", "double_open"):
            for human in (0, 2):
                with self.subTest(mode=mode, human=human):
                    cpu = self.make_immobile_game(mode, human=human)
                    before = cpu.command({"op": "state"})
                    self.assertTrue(before["active_players"][0])
                    self.assertEqual(before["passes_remaining"][0], 4)
                    self.assertEqual(before["spectator_reason"], "no_legal_moves")
                    self.assertFalse(before["your_turn"])
                    self.assertEqual(before["legal_actions"], [])
                    pieces = before["pieces"]
                    state = cpu.command({"op": "advance"})
                    self.assertEqual([frame["ply"] for frame in state["frames"]], [0, 1, 2, 3, 4])
                    first = state["frames"][1]
                    self.assertEqual(first["history"][-1]["actor"], 0)
                    self.assertEqual(first["history"][-1]["combat"], "pass")
                    self.assertEqual(first["pieces"], pieces)
                    self.assertEqual(first["passes_remaining"][0], 3)
                    self.assertEqual(first["no_capture_plies"], 1)
                    self.assertEqual(state["ai_moves"], 3)
                    for _ in range(24):
                        if state["result"]:
                            break
                        self.assertFalse(state["your_turn"])
                        state = cpu.command({"op": "advance"})
                    self.assertEqual(state["result"], {"reason": "no_capture_draw", "outcome": "draw"})
                    self.assertEqual(state["ply"], 70)
                    self.assertIsNone(state["spectator_reason"])

    def test_temporarily_blocked_human_regains_control_when_a_move_opens(self):
        cpu = self.make_immobile_game(unblock=True)
        self.assertEqual(cpu.state()["spectator_reason"], "no_legal_moves")
        state = cpu.command({"op": "advance"})
        self.assertEqual(state["ply"], 4)
        self.assertTrue(state["your_turn"])
        self.assertIsNone(state["spectator_reason"])
        self.assertIn((6, 10), state["legal_actions"])
        self.assertEqual(state["passes_remaining"][0], 3)
        self.assertEqual(cpu.command({"op": "advance"})["ply"], 4)


class HttpTests(unittest.TestCase):
    def test_http_origin_and_static_path_boundary(self):
        app = Application({"runs": []}, ROOT)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.app = app
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(base+"/api/health") as response:
                self.assertFalse(json.load(response)["gpu_inference"])
            for path in ("/", "/play", "/style.css", "/dashboard.js", "/play.js"):
                with urllib.request.urlopen(base+path) as response:
                    self.assertEqual(response.status, 200)
            for path,headers in (("/api/status", {"Host":"attacker.test"}),
                                  ("/api/status", {"Origin":"https://attacker.test"}),
                                  ("/../pyproject.toml", {})):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(urllib.request.Request(base+path, headers=headers))
                self.assertIn(error.exception.code, (403,404))
            request = urllib.request.Request(base+"/api/game/new", data=b'{}', headers={"Content-Type":"application/json", "Origin":"https://attacker.test"})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, 403)
        finally:
            server.shutdown()
            server.server_close()
            app.close()


if __name__ == "__main__":
    unittest.main()
