#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
script_dir=${repo_root}/scripts/ruler/blasst
export PYTHONPATH="${repo_root}/src:${PYTHONPATH:-}"

: "${RULER_ROOT:?set RULER_ROOT to the pinned NVIDIA RULER checkout}"
SWEEP_ROOT=${SWEEP_ROOT:-${repo_root}/results/blasst/diffusion_gemma/context_tile_sweep_lambda_0p003}
MODEL_PATH=${MODEL_PATH:-google/diffusiongemma-26B-A4B-it}
MODEL_REVISION=${MODEL_REVISION:-f7f5b7f5fa82ffc52addd066915886d497f5517b}
RULER_DEPENDENCY_PATH=${RULER_DEPENDENCY_PATH:-}
NLTK_DATA=${NLTK_DATA:-}

contexts=(1024 2048 4096 8192 16384)
blocks=(32 64 128 256)

prepare_extra=()
if [[ -n "${RULER_DEPENDENCY_PATH}" ]]; then
  prepare_extra+=(--ruler-dependency-path "${RULER_DEPENDENCY_PATH}")
fi
if [[ -n "${NLTK_DATA}" ]]; then
  prepare_extra+=(--nltk-data "${NLTK_DATA}")
fi

for context in "${contexts[@]}"; do
  manifest_dir=${SWEEP_ROOT}/manifests/${context}
  prepare_tasks=()
  # The pinned RULER fwe generator has fewer than two tokenizer-valid prompts
  # at 8K and above. Keep the manifest exact-count by balancing over the four
  # long-context tasks that reliably yield enough official samples.
  if (( context >= 8192 )); then
    prepare_tasks+=(--tasks niah_multikey_1,niah_multivalue,niah_multiquery,vt)
  fi
  python -m dllm.cli.prepare_ruler \
    --ruler-root "${RULER_ROOT}" \
    --tokenizer-path "${MODEL_PATH}" \
    --model-adapter diffusion_gemma \
    --revision "${MODEL_REVISION}" \
    --context-length "${context}" \
    --num-samples 10 \
    --seed 42 \
    --output-dir "${manifest_dir}" \
    "${prepare_tasks[@]}" \
    "${prepare_extra[@]}"

  for block in "${blocks[@]}"; do
    output_dir=${SWEEP_ROOT}/runs/context_${context}/block_${block}
    if [[ -s "${output_dir}/summary.json" && -s "${output_dir}/predictions.jsonl" ]]; then
      echo "resume: keeping completed context=${context} block=${block}"
      continue
    fi
    python -m dllm.cli.eval_ruler \
      --model-adapter diffusion_gemma \
      --model-path "${MODEL_PATH}" \
      --revision "${MODEL_REVISION}" \
      --manifest "${manifest_dir}/manifest.json" \
      --ruler-root "${RULER_ROOT}" \
      --context-length "${context}" \
      --num-samples 10 \
      --attention-backend blasst-reference \
      --blasst-lambda 0.003 \
      --q-tile-size "${block}" \
      --kv-tile-size "${block}" \
      --block-size 256 \
      --precision bfloat16 \
      --collect-attention-stats \
      --stats-level summary \
      --output-dir "${output_dir}"
  done
done

python "${script_dir}/sweep_diffusion_gemma_blasst.py" "${SWEEP_ROOT}"
