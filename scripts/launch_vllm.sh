#!/usr/bin/env bash
set -euo pipefail

: "${VLLM_BASE_MODEL:=Qwen/Qwen2.5-3B-Instruct}"
: "${VLLM_ADAPTER_NAME:=vedaz}"
: "${VLLM_ADAPTER_PATH:=/models/adapter}"
: "${VLLM_HOST:=0.0.0.0}"
: "${VLLM_PORT:=8000}"
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.90}"
: "${VLLM_MAX_MODEL_LEN:=4096}"
: "${VLLM_MAX_LORA_RANK:=64}"

if [[ -z "${VLLM_API_KEY:-}" ]]; then
  echo "VLLM_API_KEY must be set" >&2
  exit 2
fi

if [[ ! -f "${VLLM_ADAPTER_PATH}/adapter_config.json" ]]; then
  echo "No PEFT adapter found at ${VLLM_ADAPTER_PATH}" >&2
  exit 2
fi

exec vllm serve "${VLLM_BASE_MODEL}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --api-key "${VLLM_API_KEY}" \
  --dtype auto \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --enable-prefix-caching \
  --enable-lora \
  --max-lora-rank "${VLLM_MAX_LORA_RANK}" \
  --lora-modules "${VLLM_ADAPTER_NAME}=${VLLM_ADAPTER_PATH}"
