"""Start persistent local console and, explicitly, a single GPU training run.

Run with the CUDA-enabled Python in WSL from the project root.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from junqi.web.monitor import process_info, read_json, training_process
from junqi.web.service_control import ensure_managed_console


def launch(command, log, environment):
    with log.open("a") as stream:
        return subprocess.Popen(command, cwd=ROOT, env=environment,
                                stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                                start_new_session=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-training", action="store_true")
    parser.add_argument("--config", default="configs/local_console.json")
    parser.add_argument("--training-config", default="configs/local_4090_training.yaml")
    parser.add_argument("--run-id", default="four-dark-local")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    output = ROOT / "output/local_console"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "launcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = (ROOT / args.config).resolve()
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}
        service = ensure_managed_console(ROOT, config, args.port) or read_json(output / "service.json")
        info = process_info(service.get("pid"))
        alive = info and "junqi.web.server" in info["command"] and str(ROOT) in info["command"]
        if not alive:
            process = launch([sys.executable, "-m", "junqi.web.server", "--root", str(ROOT),
                "--config", str(config), "--port", str(args.port)], output / "server.log", environment)
            service = {"pid": process.pid, "port": args.port, "started": time.time(), "config": str(config)}
            (output / "service.json").write_text(json.dumps(service, indent=2))
            for _ in range(60):
                if process.poll() is not None:
                    raise RuntimeError("Console startup failed; see output/local_console/server.log")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/api/health", timeout=1) as response:
                        if json.load(response)["pid"] == process.pid:
                            break
                except OSError:
                    time.sleep(0.2)
            else:
                raise RuntimeError("Console startup timed out")
        elif service.get("port") != args.port or service.get("config") != str(config):
            raise RuntimeError("A console with different settings is already running")
        result = {"console": service, "url": f"http://localhost:{args.port}"}
        if args.start_training:
            spec = next(s for s in json.loads(config.read_text())["runs"] if s["id"] == args.run_id)
            run = (ROOT / spec["path"]).resolve()
            if not run.is_relative_to(ROOT):
                raise ValueError("Run must be inside the project")
            metadata_path = output / f"training-{args.run_id}.json"
            metadata = read_json(metadata_path)
            if training_process(metadata.get("pid"), run):
                result["training"] = {**metadata, "already_running": True}
            else:
                for entry in Path("/proc").iterdir():
                    if entry.name.isdigit():
                        running = process_info(entry.name)
                        if running and any(m in running["command"] for m in ("-m junqi.training.train_", "-m junqi.training.cli")):
                            raise RuntimeError(f"Another trainer is already running: PID {running['pid']}")
                if run.exists() and any(run.iterdir()) and not (run / "checkpoints/latest.pt").is_file():
                    raise RuntimeError("Existing training artifacts have no resumable checkpoint; choose a new run directory")
                train_environment = {**environment, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "4",
                                     "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"}
                command = [sys.executable, "-m", "junqi.training.cli", "--mode", spec["mode"],
                    "--config", str((ROOT / args.training_config).resolve()), "--model-scale", "main",
                    "--device", "cuda", "--run-dir", str(run)]
                process = launch(command, output / f"training-{args.run_id}.log", train_environment)
                metadata = {"pid": process.pid, "started": time.time(), "command": command, "run": str(run)}
                metadata_path.write_text(json.dumps(metadata, indent=2))
                time.sleep(1)
                if process.poll() is not None:
                    raise RuntimeError("Training startup failed; inspect its console log")
                result["training"] = metadata
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
