#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: scripts/merge_adapter.sh ADAPTER_DIR OUTPUT_DIR" >&2
  exit 2
fi

python merge_adapter.py --adapter "$1" --output "$2"
