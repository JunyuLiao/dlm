#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
python_bin=${PYTHON_BIN:-/home/exouser/miniconda3/envs/ljy_dlm/bin/python}
: "${DIFFUSION_GEMMA_DEPENDENCY_PATH:=/tmp/diffusion-gemma-tf511-py312}"
export PATH="$(dirname -- "${python_bin}"):${PATH}"
export PYTHONPATH="${repo_root}/src:${DIFFUSION_GEMMA_DEPENDENCY_PATH}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

: "${RULER_ROOT:=/tmp/NVIDIA-RULER}"
: "${EXPERIMENT_ROOT:=${repo_root}/results/blasst/diffusion_gemma/sweeps_8k_n100}"
: "${MODEL_PATH:=google/diffusiongemma-26B-A4B-it}"
: "${MODEL_REVISION:=f7f5b7f5fa82ffc52addd066915886d497f5517b}"

manifest=${EXPERIMENT_ROOT}/manifest/manifest.json
common=(
  --model-adapter diffusion_gemma
  --model-path "${MODEL_PATH}"
  --revision "${MODEL_REVISION}"
  --manifest "${manifest}"
  --ruler-root "${RULER_ROOT}"
  --context-length 8192
  --num-samples 100
  --block-size 256
  --precision bfloat16
)

dense_dir=${EXPERIMENT_ROOT}/runs/dense
if [[ ! -s "${dense_dir}/summary.json" || ! -s "${dense_dir}/predictions.jsonl" ]]; then
  "${python_bin}" -m dllm.cli.eval_ruler \
    "${common[@]}" \
    --attention-backend dense \
    --output-dir "${dense_dir}"
fi

for q_tile in 32 64 128 256; do
  run_dir=${EXPERIMENT_ROOT}/runs/q_sweep/q_${q_tile}
  if [[ -s "${run_dir}/summary.json" && -s "${run_dir}/predictions.jsonl" ]]; then
    echo "resume: q=${q_tile} is complete"
    continue
  fi
  "${python_bin}" -m dllm.cli.eval_ruler \
    "${common[@]}" \
    --attention-backend blasst-reference \
    --blasst-lambda 0.003 \
    --q-tile-size "${q_tile}" \
    --kv-tile-size 64 \
    --collect-attention-stats \
    --include-masked-kv-tiles-in-physical-stats \
    --stats-level layer \
    --output-dir "${run_dir}"
done

for lambda_name in 0p001 0p01 0p03 0p1; do
  case "${lambda_name}" in
    0p001) lambda_value=0.001 ;;
    0p01) lambda_value=0.01 ;;
    0p03) lambda_value=0.03 ;;
    0p1) lambda_value=0.1 ;;
  esac
  run_dir=${EXPERIMENT_ROOT}/runs/lambda_sweep/lambda_${lambda_name}
  if [[ -s "${run_dir}/summary.json" && -s "${run_dir}/predictions.jsonl" ]]; then
    echo "resume: lambda=${lambda_value} is complete"
    continue
  fi
  "${python_bin}" -m dllm.cli.eval_ruler \
    "${common[@]}" \
    --attention-backend blasst-reference \
    --blasst-lambda "${lambda_value}" \
    --q-tile-size 128 \
    --kv-tile-size 64 \
    --collect-attention-stats \
    --include-masked-kv-tiles-in-physical-stats \
    --stats-level layer \
    --output-dir "${run_dir}"
done

"${python_bin}" "${repo_root}/scripts/ruler/blasst/report_diffusion_gemma_blasst_sweeps_8k.py" \
  "${EXPERIMENT_ROOT}"
