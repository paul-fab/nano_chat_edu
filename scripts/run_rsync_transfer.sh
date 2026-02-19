#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
BASE="/mnt/c/Users/ather/data/hf_random_base_data_20260218"
RSH="/mnt/c/Users/ather/code/nano_chat_edu/scripts/wsl_ssh_wrapper.sh"

mkdir -p "$BASE/base_data" "$BASE/_transfer_logs"

case "$MODE" in
  hf_to_local)
    exec rsync -av --partial --append-verify --inplace --info=progress2 \
      -e "$RSH" \
      ubuntu@129.158.35.217:/home/ubuntu/.cache/nanochat/base_data/ \
      "$BASE/base_data/"
    ;;
  local_to_8x)
    exec rsync -av --partial --append-verify --inplace --info=progress2 \
      -e "$RSH" \
      "$BASE/base_data/" \
      ubuntu@192.222.53.38:/home/ubuntu/.cache/nanochat/hf_random_data_staging/base_data/
    ;;
  *)
    echo "Usage: $0 {hf_to_local|local_to_8x}" >&2
    exit 2
    ;;
esac
