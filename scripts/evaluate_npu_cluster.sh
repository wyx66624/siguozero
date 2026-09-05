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
nproc_per_node="${NPROC_PER_NODE:-8}"
master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-29501}"
if (( nnodes > 1 )) && [[ "$master_addr" == "127.0.0.1" ]]; then
  echo "multi-node evaluation requires MASTER_ADDR to be the rank-0 host IP" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export GROUP_RANK="${GROUP_RANK:-$node_rank}"

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

exec "$python_bin" -m torch.distributed.run \
  --nnodes "$nnodes" \
  --nproc-per-node "$nproc_per_node" \
  --node-rank "$node_rank" \
  --master-addr "$master_addr" \
  --master-port "$master_port" \
  --module junqi.training.evaluate \
  "${evaluator_args[@]}" \
  "$@"
