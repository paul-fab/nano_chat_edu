#!/usr/bin/env bash
set -euo pipefail

exec ssh \
  -i ~/.ssh/id_ed25519 \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  "$@"
