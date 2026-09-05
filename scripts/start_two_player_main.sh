#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

source /root/anaconda3/etc/profile.d/conda.sh
conda activate siguozero
export PYTHONUNBUFFERED=1

exec python -m junqi.training.train_two_player \
  --config configs/bootstrap.yaml \
  --device cuda \
  --model-scale main \
  --microbatch 24 \
  --actor-batch 64 \
  --rollout-anchor-wave 8 \
  --environment-workers 2 \
  --temporal-cache-entries 192 \
  --target-continuation-plies 3000000000 \
  --checkpoint-every 1 \
  --archive-every 100 \
  --keep-checkpoint-archives 4 \
  --resource-monitor-seconds 15 \
  --dead-rules \
  --run-dir runs_optimized_v14 \
  "$@"
