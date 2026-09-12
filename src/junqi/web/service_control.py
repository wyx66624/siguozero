"""systemd integration for the local monitor, independent of GPU training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

UNIT = "siguozero-monitor.service"
MARKER = "# Managed by SiguoZero monitor service installer."


def unit_properties():
    try:
        result = subprocess.run(["systemctl", "show", UNIT,
            "--property=LoadState,ActiveState,SubState,MainPID,WorkingDirectory,ExecStart,UnitFileState,NRestarts,FragmentPath"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode:
        return {}
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def unit_quote(value, *, command=False):
    value = str(value)
    if any(c in value for c in "\r\n\0"):
        raise ValueError("service paths cannot contain control characters")
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if command:
        value = value.replace("$", "$$")
    return '"' + value + '"'


def working_directory(value):
    value = str(value)
    if not Path(value).is_absolute() or value != value.strip() or any(c in value for c in "\r\n\0"):
        raise ValueError("working directory must be an absolute path without boundary whitespace")
    # Unlike ExecStart's argv parser, WorkingDirectory treats quotes literally.
    return value.replace("%", "%%")


def render_unit(root, config, python, port):
    return f"""{MARKER}
[Unit]
Description=SiguoZero training monitor and CPU play
After=local-fs.target network.target
RequiresMountsFor={unit_quote(root)}
StartLimitIntervalSec=60
StartLimitBurst=10

[Service]
Type=exec
WorkingDirectory={working_directory(root)}
ExecStart={unit_quote(python, command=True)} -m junqi.web.server --root {unit_quote(root, command=True)} --config {unit_quote(config, command=True)} --port {int(port)}
Environment={unit_quote('PYTHONPATH=' + str(Path(root) / 'src'))}
Environment=PYTHONUNBUFFERED=1
Environment=CUDA_VISIBLE_DEVICES=
Environment=SIGUOZERO_MONITOR_SERVICE={UNIT}
Restart=always
RestartSec=3
TimeoutStopSec=20
KillMode=control-group
StandardOutput=journal
StandardError=journal
SyslogIdentifier=siguozero-monitor

[Install]
WantedBy=multi-user.target
"""


def write_service_metadata(root, config, port, *, managed_by=None):
    directory = Path(root) / "output/local_console"
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"pid": os.getpid(), "port": port, "started": time.time(),
                "config": str(Path(config).resolve()), "managed_by": managed_by}
    temporary = directory / f".service.{os.getpid()}.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, directory / "service.json")
    return metadata


def wait_ready(port, *, expected_pid=None, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as response:
                health = json.load(response)
            if health.get("ok") and (expected_pid is None or health.get("pid") == expected_pid):
                return health
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise RuntimeError("Monitor did not become healthy; inspect journalctl -u " + UNIT)


def ensure_managed_console(root, config, port):
    properties = unit_properties()
    if properties.get("LoadState") != "loaded":
        return None
    if (properties.get("WorkingDirectory") != str(Path(root).resolve())
            or str(Path(config).resolve()) not in properties.get("ExecStart", "")
            or f"--port {int(port)}" not in properties.get("ExecStart", "")):
        raise RuntimeError("Installed monitor service uses another project/configuration/port")
    subprocess.run(["systemctl", "start", UNIT], check=True, timeout=30)
    health = wait_ready(port)
    current = unit_properties()
    if health["pid"] != int(current.get("MainPID", 0)) or current.get("ActiveState") != "active":
        raise RuntimeError("Port owner is not the installed monitor service")
    return {"pid": health["pid"], "port": port, "config": str(Path(config).resolve()),
            "managed_by": UNIT, "active": True, "enabled": current.get("UnitFileState") == "enabled"}
