"""Foreground WSL lifetime anchor used by the hidden Windows login task."""
from __future__ import annotations

import fcntl
from pathlib import Path
import subprocess
import time

UNIT = "siguozero-monitor.service"


def main():
    with Path("/run/siguozero-monitor-keepalive.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Monitor WSL lifetime anchor already running", flush=True)
            return
        subprocess.run(["systemctl", "start", UNIT], check=True, timeout=60)
        print("Monitor service started; keeping this WSL invocation alive", flush=True)
        while subprocess.run(["systemctl", "is-enabled", "--quiet", UNIT], timeout=10).returncode == 0:
            time.sleep(30)
        print("Monitor service disabled; releasing WSL lifetime anchor", flush=True)


if __name__ == "__main__":
    main()
