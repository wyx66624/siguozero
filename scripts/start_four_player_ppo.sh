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

# Four-player Policy/Critic: 128+128 input, width 256, FFN 1024.
# Use a separate run root: old width-512 checkpoints cannot resume this model.
# Rollout concurrency is measured separately from learner microbatch size.
# NUM_GPUS enables single-host DDP. Pool and transition counts are global;
# actor batch, temporal cache and learner microbatch are per rank.
num_gpus="${NUM_GPUS:-1}"
games_per_gpu="${GAMES_PER_GPU:-48}"
if ! [[ "$num_gpus" =~ ^[1-9][0-9]*$ && "$games_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS and GAMES_PER_GPU must be positive integers" >&2
  exit 2
fi
global_pool="${BASE_GAME_POOL:-$((games_per_gpu * num_gpus))}"
global_transitions="${TRANSITION_BATCH:-$((256 * games_per_gpu * num_gpus))}"
launcher=("$python_bin" -m)
default_run_dir=runs_four_player_ppo_128_3b
if (( num_gpus > 1 )); then
  launcher=("$python_bin" -m torch.distributed.run --standalone "--nproc-per-node=$num_gpus" -m)
  default_run_dir="${default_run_dir}_${num_gpus}gpu"
fi
# Default budget is total training environment transitions, including any
# actually executed simulated branches. Network passes/epochs add no steps.
budget_args=(--target-environment-plies "${TARGET_ENVIRONMENT_PLIES:-3000000000}")
for arg in "$@"; do
  case "$arg" in
    --target-environment-plies|--target-environment-plies=*|--target-continuation-plies|--target-continuation-plies=*|--grpo-equivalent-plies|--grpo-equivalent-plies=*)
      budget_args=() ;;
  esac
done
exec "${launcher[@]}" "junqi.training.train_${mode}" \
  --config configs/bootstrap.yaml \
  --device cuda --model-scale main "$dead_rules_flag" \
  --microbatch "${MICROBATCH:-32}" \
  --environment-workers "${ENVIRONMENT_WORKERS:-4}" \
  --actor-batch "${ACTOR_BATCH:-$games_per_gpu}" \
  --base-game-pool "$global_pool" \
  --temporal-cache-entries "${TEMPORAL_CACHE_ENTRIES:-$((12 * games_per_gpu))}" \
  --transition-batch "$global_transitions" \
  --ppo-minibatch "${PPO_MINIBATCH:-512}" \
  --arena-parallel-games "${EVAL_PARALLEL_GAMES:-32}" \
  --arena-inference-batch "${EVAL_INFERENCE_BATCH_SIZE:-32}" \
  --arena-environment-workers "${EVAL_ENVIRONMENT_WORKERS:-4}" \
  "${budget_args[@]}" \
  --checkpoint-policy evaluation --archive-every 100 --keep-checkpoint-archives 4 \
  --resource-monitor-seconds 15 \
  --run-dir "${RUN_DIR:-$default_run_dir}" \
  "$@"
