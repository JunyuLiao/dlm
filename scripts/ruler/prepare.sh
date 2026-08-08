#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${repo_root}/src:${PYTHONPATH:-}"

: "${RULER_ROOT:?set RULER_ROOT}"
: "${TOKENIZER_PATH:?set TOKENIZER_PATH}"
: "${MODEL_ADAPTER:?set MODEL_ADAPTER}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

CONTEXT_LENGTH=${CONTEXT_LENGTH:-8192}
NUM_SAMPLES=${NUM_SAMPLES:-100}
SEED=${SEED:-42}

extra_args=()
if [[ -n "${RULER_DEPENDENCY_PATH:-}" ]]; then
  extra_args+=(--ruler-dependency-path "${RULER_DEPENDENCY_PATH}")
fi
if [[ -n "${NLTK_DATA:-}" ]]; then
  extra_args+=(--nltk-data "${NLTK_DATA}")
fi
if [[ -n "${TASKS:-}" ]]; then
  extra_args+=(--tasks "${TASKS}")
fi
if [[ -n "${GENERATION_CONFIG_JSON:-}" ]]; then
  extra_args+=(--generation-config-json "${GENERATION_CONFIG_JSON}")
fi

python -m dllm.cli.prepare_ruler \
  --ruler-root "${RULER_ROOT}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --model-adapter "${MODEL_ADAPTER}" \
  --context-length "${CONTEXT_LENGTH}" \
  --num-samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  "${extra_args[@]}"
