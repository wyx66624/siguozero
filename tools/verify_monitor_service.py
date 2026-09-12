"""Check the installed monitor; optionally exercise recovery using a test CPU game."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from junqi.web.monitor import process_info, read_json
from junqi.web.service_control import UNIT, unit_properties, wait_ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise-restart", action="store_true",
                        help="kill only the monitor main process to verify systemd recovery; requires no existing games")
    parser.add_argument("--run-id", default="four-dark-local")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", default="docs/benchmarks/monitor_service_validation_20260911.json")
    args = parser.parse_args()

    def request(path, data=None):
        req = urllib.request.Request(f"http://127.0.0.1:{args.port}" + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={} if data is None else {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=45) as response:
            return json.load(response)

    def training(status):
        run = next(r for r in status["runs"] if r["id"] == args.run_id)
        return {k: run[k] for k in ("id", "status", "update", "steps", "processes", "cycle_rate", "eta_seconds")}

    def cgroup(pid):
        return Path(f"/proc/{pid}/cgroup").read_text().strip()

    before_properties = unit_properties()
    assert before_properties["ActiveState"] == "active" and before_properties["UnitFileState"] == "enabled"
    assert before_properties["WorkingDirectory"] == str(ROOT)
    before_health = wait_ready(args.port, expected_pid=int(before_properties["MainPID"]))
    assert before_health["managed_by"] == UNIT
    status = request("/api/status")
    before = training(status)
    assert before["status"] == "running" and before["processes"]
    trainer_groups = {str(p["pid"]): cgroup(p["pid"]) for p in before["processes"]}
    assert all(UNIT not in group for group in trainer_groups.values())
    evidence = {"timestamp_unix": time.time(), "before": before,
                "properties_before": before_properties, "trainer_cgroups": trainer_groups}

    if args.exercise_restart:
        if status["cpu_play"]["sessions"]:
            raise RuntimeError("Finish existing CPU games before the recovery exercise")
        state = request("/api/game/new", {"checkpoint_id": args.run_id + ":live", "seat": 0, "seed": 20260911})
        killed = False
        try:
            model = state["model"]
            assert model["device"] == "cpu" and not model["cuda_initialized"] and model["cuda_visible_devices"] == ""
            cpu_group = cgroup(model["pid"])
            assert UNIT in cpu_group
            moved = request("/api/game/move", {"session_id": state["session_id"],
                "expected_ply": state["ply"], "action": state["legal_actions"][0]})
            assert moved["ply"] > state["ply"]
            sessions = request("/api/status")["cpu_play"]["sessions"]
            if len(sessions) != 1 or sessions[0]["pid"] != model["pid"]:
                raise RuntimeError("Another CPU game appeared; the recovery exercise was cancelled")
            print("CPU test game is isolated; exercising monitor-only crash recovery", flush=True)
            started = time.monotonic()
            subprocess.run(["systemctl", "kill", "--kill-whom=main", "--signal=SIGKILL", UNIT], check=True, timeout=10)
            killed = True
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                properties = unit_properties()
                new_pid = int(properties.get("MainPID", 0))
                if properties.get("ActiveState") == "active" and new_pid not in (0, before_health["pid"]):
                    break
                time.sleep(0.2)
            else:
                raise RuntimeError("systemd did not restart the monitor")
            health = wait_ready(args.port, expected_pid=new_pid)
            assert int(properties["NRestarts"]) > int(before_properties["NRestarts"])
            assert process_info(model["pid"]) is None
            assert read_json(ROOT / "output/local_console/service.json")["pid"] == new_pid
            assert all(process_info(p["pid"]) for p in before["processes"])
            evidence["crash_recovery"] = {"seconds": time.monotonic() - started,
                "old_monitor_pid": before_health["pid"], "new_monitor_pid": new_pid,
                "health": health, "cpu_model": model, "cpu_cgroup": cpu_group,
                "cpu_game_plies": moved["ply"], "cpu_worker_reaped": True}
        finally:
            if not killed:
                request("/api/game/close", {"session_id": state["session_id"]})

    print("Monitor is healthy; waiting for the original training process to finish another update", flush=True)
    deadline = time.monotonic() + 180
    while True:
        after = training(request("/api/status"))
        assert after["status"] == "running" and after["processes"] == before["processes"]
        if after["update"] > before["update"] and after["steps"] > before["steps"]:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Training is alive but no completed update arrived within the verification window")
        time.sleep(5)
    evidence.update(complete=True, after=after, properties_after=unit_properties(),
                    training_process_unchanged=True, training_progress_advanced=True)
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
