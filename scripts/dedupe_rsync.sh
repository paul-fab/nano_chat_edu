#!/usr/bin/env bash
set -euo pipefail

# Keep only one process for each transfer direction.

hf_pids=$(pgrep -f "rsync -av --partial --append-verify --inplace --info=progress2 -e /mnt/c/Users/ather/code/nano_chat_edu/scripts/wsl_ssh_wrapper.sh ubuntu@129.158.35.217:/home/ubuntu/.cache/nanochat/base_data/ /mnt/c/Users/ather/data/hf_random_base_data_20260218/base_data/" || true)
if [[ -n "${hf_pids}" ]]; then
  keep=$(echo "$hf_pids" | head -n1)
  kill_list=$(echo "$hf_pids" | tail -n +2 || true)
  if [[ -n "${kill_list}" ]]; then
    echo "$kill_list" | xargs -r kill -9
  fi
  echo "hf_to_local_keep=$keep"
fi

up_pids=$(pgrep -f "rsync -av --partial --append-verify --inplace --info=progress2 -e /mnt/c/Users/ather/code/nano_chat_edu/scripts/wsl_ssh_wrapper.sh /mnt/c/Users/ather/data/hf_random_base_data_20260218/base_data/ ubuntu@192.222.53.38:/home/ubuntu/.cache/nanochat/hf_random_data_staging/base_data/" || true)
if [[ -n "${up_pids}" ]]; then
  keep=$(echo "$up_pids" | head -n1)
  kill_list=$(echo "$up_pids" | tail -n +2 || true)
  if [[ -n "${kill_list}" ]]; then
    echo "$kill_list" | xargs -r kill -9
  fi
  echo "local_to_8x_keep=$keep"
fi
