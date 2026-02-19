#!/bin/bash
set -euo pipefail

# Run one experiment arm from a prebuilt Azure subset.
#
# Example:
#   bash run_experiment_arm.sh \
#     --arm top16 \
#     --container quratingfiltered \
#     --prefix prebuilt/top16_5/ \
#     --depth 20 \
#     --run-name exp-top16-d20 \
#     --workers 16

ARM=""
CONTAINER="${AZURE_CONTAINER:-quratingfiltered}"
PREFIX=""
DEPTH="20"
RUN_NAME=""
WORKERS="16"
DATA_DIR="${HOME}/.cache/nanochat/base_data"
TOTAL_BATCH_SIZE="524288"
TOKEN_COL="token_count"
NANOCHAT_DIR="${NANOCHAT_DIR:-$HOME/nanochat}"
NUM_GPUS="1"
DEVICE_BATCH_SIZE="32"
MASTER_PORT="29500"
SKIP_DOWNLOAD="0"
DOWNLOAD_MODE="bulk"
MODEL_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) ARM="$2"; shift 2 ;;
    --container) CONTAINER="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --depth) DEPTH="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --total-batch-size) TOTAL_BATCH_SIZE="$2"; shift 2 ;;
    --token-col) TOKEN_COL="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --device-batch-size) DEVICE_BATCH_SIZE="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD="1"; shift 1 ;;
    --download-mode) DOWNLOAD_MODE="$2"; shift 2 ;;
    --model-tag) MODEL_TAG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$ARM" ]]; then
  echo "ERROR: required arg: --arm"
  exit 1
fi

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="exp-${ARM}-d${DEPTH}"
fi
if [[ -z "$MODEL_TAG" ]]; then
  MODEL_TAG="$RUN_NAME"
fi

cd "$NANOCHAT_DIR"
source .venv/bin/activate

mkdir -p "$DATA_DIR"

if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
  SHARD_COUNT=$(ls "$DATA_DIR"/shard_*.parquet 2>/dev/null | wc -l)
  if [[ "$SHARD_COUNT" -eq 0 ]]; then
    echo "ERROR: --skip-download set but no shards found in $DATA_DIR"
    exit 1
  fi
  echo "Skipping download. Using existing shards in $DATA_DIR (count=$SHARD_COUNT)."
else
  find "$DATA_DIR" -maxdepth 1 -type f -name 'shard_*.parquet' -delete
  rm -rf "$DATA_DIR/_duckdb_tmp" "$DATA_DIR/_top_shards_tmp" "$DATA_DIR/_azcopy_raw"

  python download_azure_data.py \
    --container "$CONTAINER" \
    --blob-prefix "$PREFIX" \
    --workers "$WORKERS" \
    --download-mode "$DOWNLOAD_MODE" \
    --skip-sort \
    --data-dir "$DATA_DIR"
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
export MODEL_TAG="$MODEL_TAG"

echo "Checkpoint namespace (model_tag): $MODEL_TAG"

bash train.sh "$DEPTH" "$RUN_NAME"
