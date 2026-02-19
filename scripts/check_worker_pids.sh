#!/usr/bin/env bash
set -euo pipefail

LOG=/mnt/c/Users/ather/data/hf_random_base_data_20260218/_transfer_logs

count=$(ls -1 "$LOG"/worker_*.pid 2>/dev/null | wc -l || true)
echo "pid_files=$count"

for p in "$LOG"/worker_*.pid; do
  [[ -f "$p" ]] || continue
  pid="$(cat "$p")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "alive $pid $p"
  else
    echo "dead $pid $p"
  fi
done
