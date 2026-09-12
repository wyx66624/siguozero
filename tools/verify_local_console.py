"""Bounded API smoke test using a real frozen checkpoint alongside training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--minimum-update", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", default="docs/benchmarks/local_console_validation_20260911.json")
    args = parser.parse_args()

    def request(path, data=None):
        value = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(args.url+path, data=value,
            headers={} if data is None else {"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)

    deadline, last = time.monotonic() + args.timeout, None
    while True:
        status = request("/api/status")
        run = next(r for r in status["runs"] if r["id"] == "four-dark-local")
        entry = next((m for m in request("/api/models")["models"] if m["id"] == "four-dark-local:live"), None)
        if entry and entry["update"] >= args.minimum_update:
            break
        if run["status"] not in ("running","starting") or time.monotonic() > deadline:
            raise RuntimeError("Training stopped or the requested snapshot was not published in time")
        if run["update"] != last:
            last = run["update"]
            print(f"Training update {last}, steps {run['steps']}; waiting for CPU snapshot update {args.minimum_update}", flush=True)
        time.sleep(15)

    started = time.perf_counter()
    state = request("/api/game/new", {"checkpoint_id":entry["id"], "seed":20260911, "seat":0})
    load_seconds = time.perf_counter() - started
    session = state["session_id"]
    try:
        assert state["model"]["device"] == "cpu"
        assert state["model"]["cuda_visible_devices"] == ""
        assert not state["model"]["cuda_initialized"]
        assert state["model"]["update"] >= args.minimum_update
        during = request("/api/status")
        assert any(s["pid"] == state["model"]["pid"] and s["alive"] for s in during["cpu_play"]["sessions"])
        live = next(r for r in during["runs"] if r["id"] == run["id"])
        assert live["status"] == "running" and live["processes"][0]["pid"] != state["model"]["pid"]
        times = []
        for _ in range(3):
            assert state["your_turn"]
            for piece in filter(None, state["pieces"]):
                if not piece["visible"]:
                    assert piece["kind"] is None and piece["name"] == "暗棋"
            before = state["ply"]
            state = request("/api/game/move", {"session_id":session, "expected_ply":before, "action":state["legal_actions"][0]})
            assert state["ply"] > before
            times.append(state["inference_seconds"])
        try:
            request("/api/game/move", {"session_id":session, "expected_ply":state["ply"], "action":[0,0]})
        except urllib.error.HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("Illegal action was accepted")
        restored = request("/api/game/state", {"session_id":session})
        assert restored["ply"] == state["ply"]
        assert all(not isinstance(v, float) or math.isfinite(v) for v in live["metrics"].values())
        assert live["metrics"]["optimizer/policy_lr"] > 0
        snapshot = ROOT / entry["path"]
        with snapshot.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        source_paths = [*sorted((ROOT/"src/junqi/web").glob("*.py")),
                        *sorted((ROOT/"src/junqi/web/static").glob("*")),
                        ROOT/"src/junqi/training/inference_snapshot.py", ROOT/"tools/local_console.py"]
        tests = {}
        for name in ("local_console_full_tests.log", "local_console_tests.log"):
            content = (ROOT/"output"/name).read_text()
            tests[name] = {"passed":bool(re.search(r"\nOK(?: \([^\n]*\))?\s*$", content)),
                           "count":int(re.findall(r"Ran (\d+) tests",content)[-1])}
        report = {"complete":True, "timestamp_unix":time.time(), "snapshot":entry,
                  "snapshot_sha256":sha, "cpu_model":state["model"], "game_plies":state["ply"],
                  "model_load_and_layout_seconds":load_seconds, "cpu_round_seconds":times,
                  "same_time_training":{"pid":live["processes"][0]["pid"], "update":live["update"],
                    "steps":live["steps"], "cycle_rate":live["cycle_rate"],
                    "eta_days":live["eta_seconds"]/86400, "required_rate_20d":live["required_rate_20d"],
                    "policy_lr":live["metrics"]["optimizer/policy_lr"], "metrics_finite":True},
                  "tests":tests, "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
        destination = ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({k:v for k,v in report.items() if k not in ("source_sha256","snapshot")},ensure_ascii=False,indent=2), flush=True)
    finally:
        request("/api/game/close", {"session_id":session})


if __name__ == "__main__":
    main()
