#!/usr/bin/env bash
set -euo pipefail

# Usage: MODE is the first argument; remaining arguments are forwarded to the
# SiguoZero trainer.  All nodes must use the same values and a shared RUN_DIR.
mode="${1:-two_player}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$mode" in
  two_player)
    train_module="junqi.training.train_two_player"
    ;;
  four_dark)
    train_module="junqi.training.train_four_dark"
    ;;
  double_open)
    train_module="junqi.training.train_double_open"
    ;;
  *)
    echo "mode must be two_player, four_dark, or double_open" >&2
    exit 2
    ;;
esac

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

cann_env_file="${CANN_ENV_FILE:-}"
if [[ -z "$cann_env_file" ]]; then
  if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    cann_env_file=/usr/local/Ascend/cann/set_env.sh
  elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    cann_env_file=/usr/local/Ascend/ascend-toolkit/set_env.sh
  else
    echo "CANN environment was not found; set CANN_ENV_FILE" >&2
    exit 2
  fi
fi
# shellcheck disable=SC1090
source "$cann_env_file"

python_bin="${PYTHON_BIN:-python}"
nnodes="${NNODES:-1}"
node_rank="${NODE_RANK:-0}"
nproc_per_node="${NPROC_PER_NODE:-8}"
master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-29500}"
world_size=$((nnodes * nproc_per_node))

if (( nnodes > 1 )) && [[ "$master_addr" == "127.0.0.1" ]]; then
  echo "multi-node training requires MASTER_ADDR to be the rank-0 host IP" >&2
  exit 2
fi

if [[ "$mode" == "two_player" ]]; then
  default_batch=$((world_size * 8))
  default_pool=$((world_size * 4))
  default_actor=64
  default_cache=192
  default_run_dir=runs_npu_910b
else
  # PPO batches are real transitions; each rank keeps 8 games and two KV arenas.
  default_batch=$((world_size * 512))
  default_pool=$((world_size * 8))
  default_actor=8
  default_cache=96
  default_run_dir=runs_npu_910b_ppo_3b
fi
anchor_batch="${ANCHOR_BATCH:-$default_batch}"
base_game_pool="${BASE_GAME_POOL:-$default_pool}"
if (( anchor_batch % world_size != 0 )); then
  echo "ANCHOR_BATCH=$anchor_batch must be divisible by world size $world_size" >&2
  exit 2
fi
if (( base_game_pool < world_size )); then
  echo "BASE_GAME_POOL must be at least world size $world_size" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# torchrun normally supplies GROUP_RANK; keeping it explicit also enables the
# HCCL zero-copy path on torch_npu releases that inspect this variable.
export GROUP_RANK="${GROUP_RANK:-$node_rank}"

dead_rules_flag="--dead-rules"
if [[ "${DEAD_RULES:-1}" == "0" ]]; then
  dead_rules_flag="--no-dead-rules"
fi

trainer_args=(
  --config "${CONFIG:-configs/bootstrap.yaml}"
  --device npu
  --model-scale "${MODEL_SCALE:-main}"
  --anchor-batch "$anchor_batch"
  --base-game-pool "$base_game_pool"
  --microbatch "${MICROBATCH:-8}"
  --actor-batch "${ACTOR_BATCH:-$default_actor}"
  --rollout-anchor-wave "${ROLLOUT_ANCHOR_WAVE:-8}"
  --environment-workers "${ENVIRONMENT_WORKERS:-2}"
  --temporal-cache-entries "${TEMPORAL_CACHE_ENTRIES:-$default_cache}"
  --arena-parallel-games "${EVAL_PARALLEL_GAMES:-32}"
  --arena-inference-batch "${EVAL_INFERENCE_BATCH_SIZE:-32}"
  --arena-environment-workers "${EVAL_ENVIRONMENT_WORKERS:-4}"
  --checkpoint-every "${CHECKPOINT_EVERY:-10}"
  --archive-every "${ARCHIVE_EVERY:-500}"
  --keep-checkpoint-archives "${KEEP_CHECKPOINT_ARCHIVES:-10}"
  --resource-monitor-seconds "${RESOURCE_MONITOR_SECONDS:-30}"
  "$dead_rules_flag"
  --run-dir "${RUN_DIR:-$default_run_dir}"
)
if [[ -n "${UPDATES:-}" ]]; then
  trainer_args+=(--updates "$UPDATES")
elif [[ "$mode" == "two_player" ]]; then
  trainer_args+=(--updates 200000)
fi
# Four-player PPO derives its update limit from the real-step target in YAML.
if [[ "$mode" != "two_player" && -n "${TARGET_ENVIRONMENT_PLIES:-}" ]]; then
  trainer_args+=(--target-environment-plies "$TARGET_ENVIRONMENT_PLIES")
elif [[ -n "${TARGET_CONTINUATION_PLIES:-}" ]]; then
  trainer_args+=(--target-continuation-plies "$TARGET_CONTINUATION_PLIES")
fi

exec "$python_bin" -m torch.distributed.run \
  --nnodes "$nnodes" \
  --nproc-per-node "$nproc_per_node" \
  --node-rank "$node_rank" \
  --master-addr "$master_addr" \
  --master-port "$master_port" \
  --module "$train_module" \
  "${trainer_args[@]}" \
  "$@"
