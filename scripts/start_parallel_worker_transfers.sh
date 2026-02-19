#!/usr/bin/env bash
set -euo pipefail

DOWN_WORKERS="${1:-6}"
UP_WORKERS="${2:-6}"

BASE="/mnt/c/Users/ather/data/hf_random_base_data_20260218"
LOG_DIR="$BASE/_transfer_logs"
LOCAL_DIR="$BASE/base_data"
SRC_HOST="ubuntu@129.158.35.217"
SRC_DIR="/home/ubuntu/.cache/nanochat/base_data"
DST_HOST="ubuntu@192.222.53.38"
DST_DIR="/home/ubuntu/.cache/nanochat/hf_random_data_staging/base_data"
RSH="/mnt/c/Users/ather/code/nano_chat_edu/scripts/wsl_ssh_wrapper.sh"

mkdir -p "$LOG_DIR" "$LOCAL_DIR"

chmod +x /mnt/c/Users/ather/code/nano_chat_edu/scripts/wsl_ssh_wrapper.sh

# Stop previous transfer workers and single-rsync jobs.
pkill -f "rsync -av --partial --append-verify --inplace --info=progress2" || true
pkill -f "worker_down_" || true
pkill -f "worker_up_" || true

ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$DST_HOST" "mkdir -p $DST_DIR"

# Snapshot source file list. Re-run this script later to include newly created shards.
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$SRC_HOST" \
  "cd $SRC_DIR && ls shard_*.parquet 2>/dev/null" > "$LOG_DIR/source_files.txt"

if [[ ! -s "$LOG_DIR/source_files.txt" ]]; then
  echo "No source shard files found."
  exit 1
fi

# Split list into per-worker chunks (mod-based striping).
rm -f "$LOG_DIR"/chunk_*.txt
for i in $(seq 0 $((DOWN_WORKERS - 1))); do
  : > "$LOG_DIR/chunk_$i.txt"
done

awk -v n="$DOWN_WORKERS" '{
  idx = (NR - 1) % n
  print $0 >> "'"$LOG_DIR"'/chunk_" idx ".txt"
}' "$LOG_DIR/source_files.txt"

# Start download workers.
for i in $(seq 0 $((DOWN_WORKERS - 1))); do
  chunk="$LOG_DIR/chunk_$i.txt"
  nohup bash -lc "
    while IFS= read -r f; do
      rsync -a --partial --append-verify --inplace -e '$RSH' \
        '$SRC_HOST:$SRC_DIR/'\"\$f\" '$LOCAL_DIR/'\"\$f\"
    done < '$chunk'
  " > "$LOG_DIR/worker_down_$i.out.log" 2> "$LOG_DIR/worker_down_$i.err.log" < /dev/null &
  echo $! > "$LOG_DIR/worker_down_$i.pid"
done

# Start upload workers. They wait for local files to appear, then push.
for i in $(seq 0 $((UP_WORKERS - 1))); do
  chunk="$LOG_DIR/chunk_$(( i % DOWN_WORKERS )).txt"
  nohup bash -lc "
    while IFS= read -r f; do
      while [[ ! -f '$LOCAL_DIR/'\"\$f\" ]]; do
        sleep 2
      done
      rsync -a --ignore-existing --partial --append-verify --inplace -e '$RSH' \
        '$LOCAL_DIR/'\"\$f\" '$DST_HOST:$DST_DIR/'\"\$f\"
    done < '$chunk'
  " > "$LOG_DIR/worker_up_$i.out.log" 2> "$LOG_DIR/worker_up_$i.err.log" < /dev/null &
  echo $! > "$LOG_DIR/worker_up_$i.pid"
done

echo "started_down_workers=$DOWN_WORKERS"
echo "started_up_workers=$UP_WORKERS"
echo "source_files=$(wc -l < "$LOG_DIR/source_files.txt")"
