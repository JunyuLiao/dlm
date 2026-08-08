#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MODEL_ADAPTER=llada2_1_mini
export MODEL_PATH=${MODEL_PATH:-inclusionAI/LLaDA2.1-mini}
export BLOCK_SIZE=${BLOCK_SIZE:-32}
export THRESHOLD=${THRESHOLD:-0.5}
exec "${script_dir}/eval_one.sh"

