#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
python_bin=${PYTHON_BIN:-/home/exouser/miniconda3/envs/ljy_dlm/bin/python}
: "${DIFFUSION_GEMMA_DEPENDENCY_PATH:=/tmp/diffusion-gemma-tf511-py312}"
export PATH="$(dirname -- "${python_bin}"):${PATH}"
export PYTHONPATH="${repo_root}/src:${DIFFUSION_GEMMA_DEPENDENCY_PATH}:${PYTHONPATH:-}"

: "${RULER_ROOT:=/tmp/NVIDIA-RULER}"
: "${RULER_DEPENDENCY_PATH:=/tmp/ruler-smoke-deps}"
: "${EXPERIMENT_ROOT:=${repo_root}/results/blasst/diffusion_gemma/corrected_lambda_0p003}"
: "${MODEL_PATH:=google/diffusiongemma-26B-A4B-it}"
: "${MODEL_REVISION:=f7f5b7f5fa82ffc52addd066915886d497f5517b}"

contexts=(1024 2048 4096 8192 16384)
tile_sizes=(32 64 128 256)

for context in "${contexts[@]}"; do
  manifest_dir=${EXPERIMENT_ROOT}/manifests/${context}
  prepare_tasks=()
  if (( context >= 8192 )); then
    prepare_tasks+=(--tasks niah_multikey_1,niah_multivalue,niah_multiquery,vt)
  fi
  "${python_bin}" -m dllm.cli.prepare_ruler \
    --ruler-root "${RULER_ROOT}" \
    --ruler-dependency-path "${RULER_DEPENDENCY_PATH}" \
    --tokenizer-path "${MODEL_PATH}" \
    --model-adapter diffusion_gemma \
    --revision "${MODEL_REVISION}" \
    --context-length "${context}" \
    --num-samples 20 \
    --seed 42 \
    --output-dir "${manifest_dir}" \
    "${prepare_tasks[@]}"

  dense_dir=${EXPERIMENT_ROOT}/runs/context_${context}/dense
  if [[ ! -s "${dense_dir}/summary.json" || ! -s "${dense_dir}/predictions.jsonl" ]]; then
    "${python_bin}" -m dllm.cli.eval_ruler \
      --model-adapter diffusion_gemma \
      --model-path "${MODEL_PATH}" \
      --revision "${MODEL_REVISION}" \
      --manifest "${manifest_dir}/manifest.json" \
      --ruler-root "${RULER_ROOT}" \
      --context-length "${context}" \
      --num-samples 20 \
      --attention-backend dense \
      --block-size 256 \
      --precision bfloat16 \
      --output-dir "${dense_dir}"
  else
    echo "resume: keeping completed dense context=${context}"
  fi

  for tile_size in "${tile_sizes[@]}"; do
    sparse_dir=${EXPERIMENT_ROOT}/runs/context_${context}/tile_${tile_size}
    if [[ -s "${sparse_dir}/summary.json" && -s "${sparse_dir}/predictions.jsonl" ]]; then
      echo "resume: keeping completed sparse context=${context} tile=${tile_size}"
      continue
    fi
    "${python_bin}" -m dllm.cli.eval_ruler \
      --model-adapter diffusion_gemma \
      --model-path "${MODEL_PATH}" \
      --revision "${MODEL_REVISION}" \
      --manifest "${manifest_dir}/manifest.json" \
      --ruler-root "${RULER_ROOT}" \
      --context-length "${context}" \
      --num-samples 20 \
      --attention-backend blasst-reference \
      --blasst-lambda 0.003 \
      --q-tile-size "${tile_size}" \
      --kv-tile-size "${tile_size}" \
      --block-size 256 \
      --precision bfloat16 \
      --collect-attention-stats \
      --include-masked-kv-tiles-in-physical-stats \
      --stats-level layer \
      --output-dir "${sparse_dir}"
  done
done

"${python_bin}" "${repo_root}/scripts/ruler/blasst/report_diffusion_gemma_blasst_corrected.py" \
  "${EXPERIMENT_ROOT}"
