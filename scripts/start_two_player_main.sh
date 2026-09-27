#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${PYTHON_BIN:-/root/anaconda3/envs/siguozero/bin/python}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Share the expanded four-player ACPO/MAPPO profile, with its two-player
# terminal-only reward and opponent-only history overrides. The old GRPO
# checkpoint is architecturally incompatible; use the separate mode run root.
exec "$python_bin" -m junqi.training.train_two_player \
  --config configs/expand_mappo_training.yaml \
  --device cuda \
  --model-scale main \
  --dead-rules \
  --run-dir "${RUN_DIR:-runs_expand_mappo}" \
  "$@"
