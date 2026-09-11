#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EVAL_MODE=history
export EVAL_GAME_MODE=two_player
exec bash "$repo_dir/scripts/evaluate_npu_cluster.sh" "$@"
