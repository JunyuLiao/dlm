#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MODEL_ADAPTER=fast_dllm_v1_llada
export MODEL_PATH=${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}
export BLOCK_SIZE=${BLOCK_SIZE:-32}
export THRESHOLD=${THRESHOLD:-0.9}
exec "${script_dir}/eval_one.sh"

