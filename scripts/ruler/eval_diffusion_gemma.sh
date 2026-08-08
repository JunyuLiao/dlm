#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MODEL_ADAPTER=diffusion_gemma
export MODEL_PATH=${MODEL_PATH:-google/diffusiongemma-26B-A4B-it}
export BLOCK_SIZE=${BLOCK_SIZE:-256}
export PRECISION=bfloat16
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
# Leave STEPS unset for the checkpoint's native 48-step sampler. A small
# explicit value is useful only for smoke runs.
exec "${script_dir}/eval_one.sh"
