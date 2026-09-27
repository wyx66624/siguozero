#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-/root/anaconda3/envs/siguozero/bin/python}"
export PYTHONUNBUFFERED=1

# Dedicated web console only. Training has its own explicit launch command.
exec "$python_bin" tools/two_player_console.py "$@"
