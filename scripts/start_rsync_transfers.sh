#!/usr/bin/env bash
set -euo pipefail

BASE="/mnt/c/Users/ather/data/hf_random_base_data_20260218"
LOG_DIR="$BASE/_transfer_logs"

mkdir -p "$LOG_DIR"
chmod +x /mnt/c/Users/ather/code/nano_chat_edu/scripts/run_rsync_transfer.sh

pkill -f "run_rsync_transfer.sh hf_to_local" || true
pkill -f "run_rsync_transfer.sh local_to_8x" || true

nohup /mnt/c/Users/ather/code/nano_chat_edu/scripts/run_rsync_transfer.sh hf_to_local \
  > "$LOG_DIR/rsync_hf_to_local.out.log" \
  2> "$LOG_DIR/rsync_hf_to_local.err.log" < /dev/null &
echo $! > "$LOG_DIR/rsync_hf_to_local.pid"

nohup /mnt/c/Users/ather/code/nano_chat_edu/scripts/run_rsync_transfer.sh local_to_8x \
  > "$LOG_DIR/rsync_local_to_8x.out.log" \
  2> "$LOG_DIR/rsync_local_to_8x.err.log" < /dev/null &
echo $! > "$LOG_DIR/rsync_local_to_8x.pid"

echo "hf_to_local_pid=$(cat "$LOG_DIR/rsync_hf_to_local.pid")"
echo "local_to_8x_pid=$(cat "$LOG_DIR/rsync_local_to_8x.pid")"
