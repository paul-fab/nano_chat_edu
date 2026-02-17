#!/bin/bash
set -euo pipefail

# =============================================================================
# setup_droplet.sh -- One-shot provisioning for DigitalOcean H100 GPU Droplet
# Run: bash setup_droplet.sh
# Expects: .env file in same directory with AZURE_STORAGE_ACCOUNT and AZURE_STORAGE_KEY
# =============================================================================

NANOCHAT_DIR="$HOME/nanochat"
DATA_DIR="$HOME/.cache/nanochat/base_data"

echo "=== [1/6] System dependencies ==="
apt-get update
apt-get install -y git build-essential curl wget

echo "=== [2/6] Installing uv (Python package manager) ==="
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

echo "=== [3/6] Cloning nanochat ==="
if [ -d "$NANOCHAT_DIR" ]; then
    echo "nanochat already cloned at $NANOCHAT_DIR, pulling latest..."
    cd "$NANOCHAT_DIR" && git pull && cd -
else
    git clone https://github.com/karpathy/nanochat.git "$NANOCHAT_DIR"
fi

echo "=== [4/6] Creating Python venv and installing dependencies ==="
cd "$NANOCHAT_DIR"
uv venv --python 3.12
source .venv/bin/activate

# Install nanochat and its GPU dependencies
uv pip install -e .

# Install our additional dependencies for Azure download pipeline
uv pip install azure-storage-blob python-dotenv pyarrow

echo "=== [5/6] Creating data directory ==="
mkdir -p "$DATA_DIR"

echo "=== [6/6] Copying scripts ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Copy our scripts into nanochat directory for easy access
cp "$SCRIPT_DIR/download_azure_data.py" "$NANOCHAT_DIR/"
cp "$SCRIPT_DIR/patch_nanochat.py" "$NANOCHAT_DIR/"
cp "$SCRIPT_DIR/train.sh" "$NANOCHAT_DIR/"

# Copy .env if it exists alongside this script
if [ -f "$SCRIPT_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env" "$NANOCHAT_DIR/.env"
    echo "Copied .env to $NANOCHAT_DIR/.env"
else
    echo "WARNING: No .env file found at $SCRIPT_DIR/.env"
    echo "You must create $NANOCHAT_DIR/.env with:"
    echo "  AZURE_STORAGE_ACCOUNT=quratingscoressa"
    echo "  AZURE_STORAGE_KEY=<your-key>"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  cd $NANOCHAT_DIR"
echo "  source .venv/bin/activate"
echo "  python download_azure_data.py        # ~15-30 min, downloads ~49GB"
echo "  python patch_nanochat.py             # patches dataset.py"
echo "  bash train.sh                        # trains tokenizer + model"
