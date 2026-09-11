#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EVAL_MODE=history
export EVAL_GAME_MODE="${EVAL_GAME_MODE:-four_dark}"
case "$EVAL_GAME_MODE" in
  four_dark|double_open) ;;
  *) echo "four-player entry accepts only four_dark or double_open" >&2; exit 2 ;;
esac
exec bash "$repo_dir/scripts/evaluate_npu_cluster.sh" "$@"
