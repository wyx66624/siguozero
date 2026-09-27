"""Serve the dedicated two-player monitor and CPU play pages on port 8767.

This entry point only starts the web console when explicitly invoked. It never
starts a trainer or installs/restarts the existing four-player monitor service.
"""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from junqi.web.server import main


if __name__ == "__main__":
    main(["--root", str(ROOT), "--config", str(ROOT / "configs/two_player_console.json"),
          "--port", "8767", "--no-service-metadata", *sys.argv[1:]])
