"""Install the monitor as a WSL systemd service without stopping training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from junqi.web.monitor import process_info, read_json
from junqi.web.service_control import MARKER, UNIT, ensure_managed_console, render_unit, unit_properties, wait_ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local_console.json")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--render", help="write a reviewable unit file only; do not install")
    args = parser.parse_args()
    config = (ROOT / args.config).resolve()
    content = render_unit(ROOT, config, sys.executable, args.port)
    if args.render:
        Path(args.render).write_text(content, encoding="utf-8")
        print(args.render)
        return
    if os.geteuid() != 0 or Path("/proc/1/comm").read_text().strip() != "systemd":
        raise RuntimeError("Installation requires root in a WSL instance already using systemd")
    destination = Path("/etc/systemd/system") / UNIT
    if destination.exists() and not destination.read_text().startswith(MARKER):
        raise RuntimeError("A service with this name exists and is not owned by this installer")
    properties = unit_properties()
    # An unchanged, healthy installation is a no-op, including for active CPU games.
    if (destination.exists() and destination.read_text() == content
            and properties.get("ActiveState") == "active"):
        service = ensure_managed_console(ROOT, config, args.port)
        subprocess.run(["systemctl", "enable", UNIT], check=True, timeout=20)
        print(json.dumps({"unit": str(destination), "unchanged": True, "service": service}, indent=2))
        return
    metadata = read_json(ROOT / "output/local_console/service.json")
    old = process_info(metadata.get("pid"))
    if old:
        if ("-m junqi.web.server" not in old["command"] or str(ROOT) not in old["command"]
                or metadata.get("config") != str(config) or metadata.get("port") != args.port):
            raise RuntimeError("Existing process is not this project's monitor; refusing to stop it")
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/api/health", timeout=5) as response:
            if json.load(response).get("pid") != old["pid"]:
                raise RuntimeError("Monitor process and listening port do not match")
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/api/status", timeout=10) as response:
            if json.load(response)["cpu_play"]["sessions"]:
                raise RuntimeError("Finish active CPU games before migrating their monitor process")
    # Validate the concrete file before installing or touching the running server.
    preview = ROOT / "output/local_console" / UNIT
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_text(content, encoding="utf-8")
    # DrvFS projects can report 0777 for every file. Validate a Linux-side copy
    # with the same contents and the permissions used for the installed unit.
    with tempfile.TemporaryDirectory(prefix="siguozero-monitor-unit-") as directory:
        validation = Path(directory) / UNIT
        validation.write_text(content, encoding="utf-8")
        validation.chmod(0o644)
        subprocess.run(["systemd-analyze", "verify", str(validation)], check=True, timeout=20)
    if old and properties.get("ActiveState") != "active":
        os.kill(old["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 20
        while process_info(old["pid"]) and time.monotonic() < deadline:
            time.sleep(0.2)
        if process_info(old["pid"]):
            raise RuntimeError("Monitor did not stop cleanly; GPU training has not been touched")
    temporary = destination.with_suffix(".service.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(0o644)
    os.replace(temporary, destination)
    subprocess.run(["systemctl", "daemon-reload"], check=True, timeout=20)
    subprocess.run(["systemctl", "enable", UNIT], check=True, timeout=20)
    subprocess.run(["systemctl", "restart", UNIT], check=True, timeout=30)
    properties = unit_properties()
    health = wait_ready(args.port, expected_pid=int(properties["MainPID"]))
    print(json.dumps({"unit":str(destination), "properties":properties, "health":health}, indent=2))


if __name__ == "__main__":
    main()
