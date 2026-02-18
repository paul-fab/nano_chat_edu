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

cd "$NANOCHAT_DIR"
source .venv/bin/activate

# Verify data exists
SHARD_COUNT=$(ls ~/.cache/nanochat/base_data/shard_*.parquet 2>/dev/null | wc -l)
if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "ERROR: No data shards found. Run download_azure_data.py first."
    exit 1
fi
echo "Found $SHARD_COUNT data shards"

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
echo "  Device batch size: 32"
echo "  Using chinchilla-optimal token count (10.5x params)"
echo ""

# Single GPU -- no torchrun needed
python -m scripts.base_train \
    --run "$RUN_NAME" \
    --depth "$DEPTH" \
    --device-batch-size 32 \
    --max-seq-len 2048 \
    --eval-every 250 \
    --core-metric-every 1000 \
    --save-every 1000 \
    --sample-every 1000

echo ""
echo "=== Training complete ==="
echo "Checkpoints saved to: logs/$RUN_NAME/"
