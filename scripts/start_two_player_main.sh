#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

source /root/anaconda3/etc/profile.d/conda.sh
conda activate siguozero
export PYTHONUNBUFFERED=1

# Use the fastest checkpoint-replay profile that stays clear of WDDM paging.
# The 160/20/480 profile reserves 11.71875 GiB for paged KV, used about
# 18.3 GiB device memory in the update-101 long-history benchmark, and leaves
# roughly 5.7 GiB for learner transients and the Windows display driver.
exec python -m junqi.training.train_two_player \
  --config configs/bootstrap.yaml \
  --device cuda \
  --model-scale main \
  --microbatch 24 \
  --actor-batch 160 \
  --rollout-anchor-wave 20 \
  --environment-workers 2 \
  --temporal-cache-entries 480 \
  --target-continuation-plies 3000000000 \
  --checkpoint-every 1 \
  --archive-every 100 \
  --keep-checkpoint-archives 4 \
  --resource-monitor-seconds 15 \
  --dead-rules \
  --run-dir runs_optimized_v14 \
  "$@"
