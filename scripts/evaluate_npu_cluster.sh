#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CHECKPOINT [extra evaluator arguments...]" >&2
  exit 2
fi
checkpoint="$1"
shift

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
evaluation_mode="${EVAL_MODE:-self_play}"
default_processes=8
if [[ "$evaluation_mode" == "history" ]]; then
  default_processes=1
fi
nproc_per_node="${NPROC_PER_NODE:-$default_processes}"
master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-29501}"
if (( nnodes > 1 )) && [[ "$master_addr" == "127.0.0.1" ]]; then
  echo "multi-node evaluation requires MASTER_ADDR to be the rank-0 host IP" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export GROUP_RANK="${GROUP_RANK:-$node_rank}"

if [[ "$evaluation_mode" == "history" ]]; then
  if [[ "$(basename "$checkpoint")" != "latest.pt" ]]; then
    echo "history mode requires the training latest.pt path; set EVAL_BASELINE separately" >&2
    exit 2
  fi
  game_mode="${EVAL_GAME_MODE:-two_player}"
  case "$game_mode" in
    two_player)
      evaluator_module=junqi.training.evaluate_two_player
      count_args=(--pairs "${EVAL_PAIRS:-200}")
      if [[ -n "${EVAL_GROUPS:-}" ]]; then
        echo "two_player uses EVAL_PAIRS, not EVAL_GROUPS" >&2
        exit 2
      fi
      ;;
    four_dark|double_open)
      evaluator_module=junqi.training.evaluate_four_player
      count_args=(--groups "${EVAL_GROUPS:-200}")
      if [[ -n "${EVAL_PAIRS:-}" ]]; then
        echo "four-player evaluation uses EVAL_GROUPS (four games each), not EVAL_PAIRS" >&2
        exit 2
      fi
      ;;
    *)
      echo "EVAL_GAME_MODE must be two_player, four_dark, or double_open" >&2
      exit 2
      ;;
  esac
  evaluator_args=(
    --mode "$game_mode"
    --checkpoint-dir "$(dirname "$checkpoint")"
    --output-dir "${EVAL_OUTPUT_DIR:?history mode requires EVAL_OUTPUT_DIR}"
    --device npu
    "${count_args[@]}"
    --every-updates "${EVAL_EVERY_UPDATES:-25}"
    --recent-opponents "${EVAL_RECENT_OPPONENTS:-2}"
    --seed "${EVAL_SEED:-20260908}"
    --temperature "${TEMPERATURE:-1.0}"
    --parallel-games "${EVAL_PARALLEL_GAMES:-32}"
    --inference-batch-size "${EVAL_INFERENCE_BATCH_SIZE:-32}"
    --environment-workers "${EVAL_ENVIRONMENT_WORKERS:-4}"
  )
  if [[ -n "${EVAL_BASELINE:-}" ]]; then
    evaluator_args+=(--baseline "$EVAL_BASELINE")
  fi
elif [[ "$evaluation_mode" == "self_play" ]]; then
  evaluator_module=junqi.training.evaluate
  evaluator_args=(
    --checkpoint "$checkpoint"
    --device npu
    --games "${GAMES:-100}"
    --seed "${EVAL_SEED:-20260902}"
    --temperature "${TEMPERATURE:-1.0}"
  )
  if [[ -n "${EVAL_SUMMARY:-}" ]]; then
    evaluator_args+=(--summary "$EVAL_SUMMARY")
  fi
  if [[ -n "${EVAL_GAMES_JSONL:-}" ]]; then
    evaluator_args+=(--games-jsonl "$EVAL_GAMES_JSONL")
  fi
else
  echo "EVAL_MODE must be self_play or history" >&2
  exit 2
fi

exec "$python_bin" -m torch.distributed.run \
  --nnodes "$nnodes" \
  --nproc-per-node "$nproc_per_node" \
  --node-rank "$node_rank" \
  --master-addr "$master_addr" \
  --master-port "$master_port" \
  --module "$evaluator_module" \
  "${evaluator_args[@]}" \
  "$@"
