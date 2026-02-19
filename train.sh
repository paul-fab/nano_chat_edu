#!/bin/bash
set -euo pipefail

# =============================================================================
# train.sh -- Train tokenizer and pretrain model on education data
# Run from inside the nanochat directory with venv activated.
#
# Usage:
#   bash train.sh              # default: d20 (124M params), ~3-4 hours on H100
#   bash train.sh 26           # d26 (215M params), ~8-10 hours on H100
#   bash train.sh 20 myrun     # d20 with custom wandb run name
# =============================================================================

DEPTH="${1:-20}"
RUN_NAME="${2:-edu-d${DEPTH}}"
NANOCHAT_DIR="${NANOCHAT_DIR:-$HOME/nanochat}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
MODEL_TAG="${MODEL_TAG:-}"

cd "$NANOCHAT_DIR"
source .venv/bin/activate

# Verify data exists
SHARD_COUNT=$(ls ~/.cache/nanochat/base_data/shard_*.parquet 2>/dev/null | wc -l)
if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "ERROR: No data shards found. Run download_azure_data.py first."
    exit 1
fi
echo "Found $SHARD_COUNT data shards"

# Validate requested GPU count before torchrun launch.
if [ "$NUM_GPUS" -gt 1 ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: NUM_GPUS=$NUM_GPUS requested but nvidia-smi not found."
        exit 1
    fi
    VISIBLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$VISIBLE_GPUS" -lt "$NUM_GPUS" ]; then
        echo "ERROR: NUM_GPUS=$NUM_GPUS requested, but only $VISIBLE_GPUS GPU(s) are visible."
        exit 1
    fi
fi

# Convert TRAIN_EXTRA_ARGS string to an array for safe expansion.
EXTRA_ARGS=()
if [[ -n "$TRAIN_EXTRA_ARGS" ]]; then
    read -r -a EXTRA_ARGS <<< "$TRAIN_EXTRA_ARGS"
fi

# =============================================================================
# Step 1: Train tokenizer
# =============================================================================
echo ""
echo "=== Training tokenizer ==="
echo "This reads a sample of the data and trains a BPE tokenizer."
echo ""

python -m scripts.tok_train \
    --vocab-size 32768 \
    --max-chars 2000000000 \
    --doc-cap 10000

echo "Tokenizer training complete."

# =============================================================================
# Step 2: Pretrain model
# =============================================================================
echo ""
echo "=== Starting pretraining ==="
PARAM_EST=$(python -c "d=$DEPTH; print(f'{d * 64 * d * 4 * 12 / 1e6:.0f}M params approx')")
echo "  Model: d${DEPTH} (~${PARAM_EST})"
echo "  Run name: $RUN_NAME"
echo "  Device batch size: $DEVICE_BATCH_SIZE"
echo "  Num GPUs: $NUM_GPUS"
if [[ -n "$MODEL_TAG" ]]; then
    echo "  Model tag: $MODEL_TAG"
fi
if [[ "$TRAIN_EXTRA_ARGS" == *"--target-param-data-ratio -1"* ]]; then
    echo "  Using fixed training horizon from provided --num-iterations"
else
    echo "  Using Chinchilla-optimal token count (10.5x params)"
fi
echo ""

MODEL_TAG_ARGS=()
if [[ -n "$MODEL_TAG" ]]; then
    MODEL_TAG_ARGS=(--model-tag "$MODEL_TAG")
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    # Single-node multi-GPU launch.
    torchrun --standalone --nproc_per_node "$NUM_GPUS" --master_port "$MASTER_PORT" -m scripts.base_train \
        --run "$RUN_NAME" \
        --depth "$DEPTH" \
        --device-batch-size "$DEVICE_BATCH_SIZE" \
        --max-seq-len 2048 \
        --eval-every 250 \
        --core-metric-every 1000 \
        --save-every 1000 \
        --sample-every 1000 \
        "${MODEL_TAG_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
else
    python -m scripts.base_train \
        --run "$RUN_NAME" \
        --depth "$DEPTH" \
        --device-batch-size "$DEVICE_BATCH_SIZE" \
        --max-seq-len 2048 \
        --eval-every 250 \
        --core-metric-every 1000 \
        --save-every 1000 \
        --sample-every 1000 \
        "${MODEL_TAG_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
fi

echo ""
echo "=== Training complete ==="
echo "Checkpoints saved to: logs/$RUN_NAME/"
