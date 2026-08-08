#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MODEL_ADAPTER=fast_dllm_v2
export MODEL_PATH=${MODEL_PATH:-Efficient-Large-Model/Fast_dLLM_v2_7B}
export BLOCK_SIZE=${BLOCK_SIZE:-16}
export THRESHOLD=${THRESHOLD:-0.9}
exec "${script_dir}/eval_one.sh"

