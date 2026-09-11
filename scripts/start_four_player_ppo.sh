#!/usr/bin/env bash
set -euo pipefail

mode="${1:-four_dark}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  four_dark|double_open) ;;
  *) echo "mode must be four_dark or double_open" >&2; exit 2 ;;
esac

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-/root/anaconda3/envs/siguozero/bin/python}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
dead_rules_flag="--dead-rules"
if [[ "${DEAD_RULES:-1}" == "0" ]]; then dead_rules_flag="--no-dead-rules"; fi

# RTX 4090: 20 rollout games; learner microbatch 8 exact-prefix sequences.
# Default budget is total training environment transitions, including any
# actually executed simulated branches. Network passes/epochs add no steps.
budget_args=(--target-environment-plies "${TARGET_ENVIRONMENT_PLIES:-3000000000}")
for arg in "$@"; do
  case "$arg" in
    --target-environment-plies|--target-environment-plies=*|--target-continuation-plies|--target-continuation-plies=*|--grpo-equivalent-plies|--grpo-equivalent-plies=*)
      budget_args=() ;;
  esac
done
exec "$python_bin" -m "junqi.training.train_${mode}" \
  --config configs/bootstrap.yaml \
  --device cuda --model-scale main "$dead_rules_flag" \
  --microbatch "${MICROBATCH:-8}" \
  --actor-batch "${ACTOR_BATCH:-20}" \
  --base-game-pool "${BASE_GAME_POOL:-20}" \
  --temporal-cache-entries "${TEMPORAL_CACHE_ENTRIES:-240}" \
  --transition-batch "${TRANSITION_BATCH:-5120}" \
  --ppo-minibatch "${PPO_MINIBATCH:-512}" \
  --arena-parallel-games "${EVAL_PARALLEL_GAMES:-32}" \
  --arena-inference-batch "${EVAL_INFERENCE_BATCH_SIZE:-32}" \
  --arena-environment-workers "${EVAL_ENVIRONMENT_WORKERS:-4}" \
  "${budget_args[@]}" \
  --checkpoint-every 1 --archive-every 100 --keep-checkpoint-archives 4 \
  --resource-monitor-seconds 15 \
  --run-dir "${RUN_DIR:-runs_four_player_ppo_3b}" \
  "$@"
