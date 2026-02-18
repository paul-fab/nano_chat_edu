#!/bin/bash
set -euo pipefail

# Run random-subset arm sourced from Hugging Face dataset.

ARM="rand16"
HF_DATASET="airtrain-ai/fineweb-edu-fortified"
HF_CONFIG=""
HF_SPLIT="train"
RATIO="0.165"
SEED="42"
PREPARE_MODE="prepare"
DEPTH="26"
RUN_NAME="exp-rand16-d26-8gpu"
DATA_DIR="${HOME}/.cache/nanochat/base_data"
TOTAL_BATCH_SIZE="524288"
TOKEN_COL="token_count"
NANOCHAT_DIR="${NANOCHAT_DIR:-$HOME/nanochat}"
NUM_GPUS="8"
DEVICE_BATCH_SIZE="32"
MASTER_PORT="29500"
ROWS_PER_SHARD="14000"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) HF_DATASET="$2"; shift 2 ;;
    --config) HF_CONFIG="$2"; shift 2 ;;
    --split) HF_SPLIT="$2"; shift 2 ;;
    --ratio) RATIO="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --prepare-mode) PREPARE_MODE="$2"; shift 2 ;;
    --depth) DEPTH="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --total-batch-size) TOTAL_BATCH_SIZE="$2"; shift 2 ;;
    --token-col) TOKEN_COL="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --device-batch-size) DEVICE_BATCH_SIZE="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --rows-per-shard) ROWS_PER_SHARD="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "$PREPARE_MODE" != "prepare" && "$PREPARE_MODE" != "skip" ]]; then
  echo "ERROR: --prepare-mode must be 'prepare' or 'skip'"
  exit 1
fi

cd "$NANOCHAT_DIR"
source .venv/bin/activate

mkdir -p "$DATA_DIR"

HF_CONFIG_ARGS=()
if [[ -n "$HF_CONFIG" ]]; then
  HF_CONFIG_ARGS=(--config "$HF_CONFIG")
fi

if [[ "$PREPARE_MODE" == "prepare" ]]; then
  if [[ "$NUM_GPUS" -gt 1 ]]; then
    echo "WARNING: Running HF sharding on a multi-GPU node is expensive. Prefer CPU prep + --prepare-mode skip."
  fi
  find "$DATA_DIR" -maxdepth 1 -type f -name 'shard_*.parquet' -delete
  rm -rf "$DATA_DIR/_duckdb_tmp" "$DATA_DIR/_top_shards_tmp"

  python prepare_hf_random_subset.py \
    --dataset "$HF_DATASET" \
    "${HF_CONFIG_ARGS[@]}" \
    --split "$HF_SPLIT" \
    --ratio "$RATIO" \
    --seed "$SEED" \
    --data-dir "$DATA_DIR" \
    --rows-per-shard "$ROWS_PER_SHARD"
else
  SHARD_COUNT=$(ls "$DATA_DIR"/shard_*.parquet 2>/dev/null | wc -l)
  if [[ "$SHARD_COUNT" -eq 0 ]]; then
    echo "ERROR: --prepare-mode skip requested, but no shards found in $DATA_DIR"
    exit 1
  fi
  echo "Skipping HF prepare. Using existing shards in $DATA_DIR (count=$SHARD_COUNT)."
fi

CALC_JSON="$(python calc_subset_iterations.py \
  --data-dir "$DATA_DIR" \
  --token-col "$TOKEN_COL" \
  --total-batch-size "$TOTAL_BATCH_SIZE" || true)"
echo "$CALC_JSON"

NUM_ITERATIONS="$(echo "$CALC_JSON" | python -c "import sys,json; s=sys.stdin.read().strip(); d=json.loads(s) if s else {}; print(d.get('num_iterations',''))" 2>/dev/null || true)"

python patch_nanochat.py --data-dir "$DATA_DIR"

# nanochat expects shards in ~/.cache/nanochat/base_data. Keep a symlink when using custom data-dir.
DEFAULT_DATA_DIR="${HOME}/.cache/nanochat/base_data"
if [[ "$DATA_DIR" != "$DEFAULT_DATA_DIR" ]]; then
  mkdir -p "$(dirname "$DEFAULT_DATA_DIR")"
  ln -sfn "$DATA_DIR" "$DEFAULT_DATA_DIR"
fi

if [[ -n "$NUM_ITERATIONS" ]]; then
  export TRAIN_EXTRA_ARGS="--target-param-data-ratio -1 --total-batch-size ${TOTAL_BATCH_SIZE} --num-iterations ${NUM_ITERATIONS}"
  echo "Using full-subset horizon: num_iterations=${NUM_ITERATIONS}, total_batch_size=${TOTAL_BATCH_SIZE}"
else
  export TRAIN_EXTRA_ARGS="--total-batch-size ${TOTAL_BATCH_SIZE}"
  echo "WARNING: token column not found, falling back to default horizon with fixed total_batch_size=${TOTAL_BATCH_SIZE}"
fi

export NUM_GPUS="$NUM_GPUS"
export DEVICE_BATCH_SIZE="$DEVICE_BATCH_SIZE"
export MASTER_PORT="$MASTER_PORT"

bash train.sh "$DEPTH" "$RUN_NAME"
