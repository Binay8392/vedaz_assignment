#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/qwen2.5-lora.yaml}"
python train.py --config "${CONFIG_PATH}"
