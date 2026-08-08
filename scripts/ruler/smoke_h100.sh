#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
script_dir=${repo_root}/scripts/ruler
export PYTHONPATH="${repo_root}/src:${PYTHONPATH:-}"

: "${RULER_ROOT:?set RULER_ROOT to the pinned NVIDIA RULER checkout}"
SMOKE_ROOT=${SMOKE_ROOT:-/tmp/dllm-ruler-smoke}
RULER_DEPENDENCY_PATH=${RULER_DEPENDENCY_PATH:-}
NLTK_DATA=${NLTK_DATA:-}

adapters=(
  fast_dllm_v1_llada
  fast_dllm_v1_dream
  fast_dllm_v2
  llada2_1_mini
  diffusion_gemma
)
models=(
  GSAI-ML/LLaDA-8B-Instruct
  Dream-org/Dream-v0-Base-7B
  Efficient-Large-Model/Fast_dLLM_v2_7B
  inclusionAI/LLaDA2.1-mini
  google/diffusiongemma-26B-A4B-it
)
blocks=(16 16 16 32 256)
thresholds=(0.9 0.9 0.9 0.5 0.9)

for index in "${!adapters[@]}"; do
  export MODEL_ADAPTER=${adapters[index]}
  export MODEL_PATH=${models[index]}
  export TOKENIZER_PATH=${MODEL_PATH}
  export CONTEXT_LENGTH=512
  export NUM_SAMPLES=1
  export TASKS=fwe
  export BLOCK_SIZE=${blocks[index]}
  export THRESHOLD=${thresholds[index]}
  export MAX_NEW_TOKENS=16
  unset STEPS GENERATION_CONFIG_JSON
  if [[ "${MODEL_ADAPTER}" == "diffusion_gemma" ]]; then
    export STEPS=${DIFFUSION_GEMMA_SMOKE_STEPS:-4}
    export VERIFY_DENSE_AFTER_BLASST=1
  else
    unset VERIFY_DENSE_AFTER_BLASST
  fi
  manifest_root=${SMOKE_ROOT}/${MODEL_ADAPTER}/manifest
  export OUTPUT_DIR=${manifest_root}
  "${script_dir}/prepare.sh"
  export MANIFEST=${manifest_root}/manifest.json
  for backend in dense blasst-reference; do
    export ATTENTION_BACKEND=${backend}
    export OUTPUT_DIR=${SMOKE_ROOT}/${MODEL_ADAPTER}/${backend}
    "${script_dir}/eval_one.sh"
  done
done
