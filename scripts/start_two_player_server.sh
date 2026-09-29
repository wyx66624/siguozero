#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
# Optional host placement; leave unset on machines with a different NUMA layout.
launcher=()
if [[ -n "${NUMA_CPU_LIST:-}" ]]; then
  command -v numactl >/dev/null
  launcher=(numactl "--physcpubind=$NUMA_CPU_LIST")
  if [[ -n "${NUMA_MEMORY_NODE:-}" ]]; then
    launcher+=("--preferred=$NUMA_MEMORY_NODE")
  fi
fi
exec "${launcher[@]}" "$python_bin" -m junqi.training.train_two_player \
  --config "$repo_dir/configs/expand_mappo_training.yaml" \
  --device cuda --model-scale main --dead-rules \
  --run-dir "$repo_dir/runs_expand_mappo" \
  --base-game-pool "${BASE_GAME_POOL:-768}" \
  --actor-batch "${ACTOR_BATCH:-768}" \
  --environment-workers "${ENVIRONMENT_WORKERS:-32}" \
  --microbatch "${MICROBATCH:-64}" \
  --resource-monitor-seconds 15 "$@"
