"""Real rules/CPU-worker tests and read-only monitor contracts."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
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
                        worker.request({"op":"move", "action":[0,0], "expected_ply":before})
                    unchanged = worker.request({"op":"state"})
                    self.assertEqual(unchanged["ply"], before)
                    moved = worker.request({"op":"move", "action":state["legal_actions"][0], "expected_ply":before})
                    self.assertGreater(moved["ply"], before)
                    self.assertTrue(moved["your_turn"] or moved["result"])
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
            for _ in range(1000):
                if state["result"]:
                    break
                state = worker.request({"op":"move", "action":state["legal_actions"][0], "expected_ply":state["ply"]})
            self.assertIsNotNone(state["result"])
            self.assertEqual(state["legal_actions"], [])
            self.assertIn(state["result"]["outcome"], ("win","loss","draw"))
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
