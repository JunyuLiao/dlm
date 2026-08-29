#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="${repo_root}/src:${PYTHONPATH:-}"

: "${MODEL_ADAPTER:?set MODEL_ADAPTER}"
: "${MODEL_PATH:?set MODEL_PATH}"
: "${RULER_ROOT:?set RULER_ROOT}"
: "${MANIFEST:?set MANIFEST}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

NUM_SAMPLES=${NUM_SAMPLES:-100}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-8192}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-dense}
BLASST_LAMBDA=${BLASST_LAMBDA:-0.003}
BLOCK_SIZE=${BLOCK_SIZE:-16}
THRESHOLD=${THRESHOLD:-0.9}
PRECISION=${PRECISION:-bfloat16}
STATS_LEVEL=${STATS_LEVEL:-summary}

extra_args=()
if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
  extra_args+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi
if [[ -n "${STEPS:-}" ]]; then
  extra_args+=(--steps "${STEPS}")
fi
if [[ -n "${GENERATION_CONFIG_JSON:-}" ]]; then
  extra_args+=(--generation-config-json "${GENERATION_CONFIG_JSON}")
fi
if [[ "${ATTENTION_BACKEND}" == "blasst-reference" ]]; then
  extra_args+=(--collect-attention-stats --stats-level "${STATS_LEVEL}")
fi
if [[ "${VERIFY_DENSE_AFTER_BLASST:-0}" == "1" ]]; then
  extra_args+=(--verify-dense-after-blasst)
fi

python -m dllm.cli.eval_ruler \
  --model-adapter "${MODEL_ADAPTER}" \
  --model-path "${MODEL_PATH}" \
  --manifest "${MANIFEST}" \
  --ruler-root "${RULER_ROOT}" \
  --context-length "${CONTEXT_LENGTH}" \
  --num-samples "${NUM_SAMPLES}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --blasst-lambda "${BLASST_LAMBDA}" \
  --block-size "${BLOCK_SIZE}" \
  --threshold "${THRESHOLD}" \
  --precision "${PRECISION}" \
  --output-dir "${OUTPUT_DIR}" \
  "${extra_args[@]}"
