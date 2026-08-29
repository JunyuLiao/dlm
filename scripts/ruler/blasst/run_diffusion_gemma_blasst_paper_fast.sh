#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
python_bin=${PYTHON_BIN:-/home/exouser/miniconda3/envs/ljy_dlm/bin/python}
: "${DIFFUSION_GEMMA_DEPENDENCY_PATH:=/tmp/diffusion-gemma-tf511-py312}"
export PATH="$(dirname -- "${python_bin}"):${PATH}"
export PYTHONPATH="${repo_root}/src:${DIFFUSION_GEMMA_DEPENDENCY_PATH}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

: "${RULER_ROOT:=/tmp/NVIDIA-RULER}"
: "${EXPERIMENT_ROOT:=${repo_root}/results/blasst/diffusion_gemma/paper_8k_n650}"
: "${MODEL_PATH:=google/diffusiongemma-26B-A4B-it}"
: "${MODEL_REVISION:=f7f5b7f5fa82ffc52addd066915886d497f5517b}"

manifest=${EXPERIMENT_ROOT}/manifest/manifest.json
screen_manifest=${EXPERIMENT_ROOT}/screen_manifest/manifest.json
common=(
  --model-adapter diffusion_gemma
  --model-path "${MODEL_PATH}"
  --revision "${MODEL_REVISION}"
  --ruler-root "${RULER_ROOT}"
  --context-length 8192
  --block-size 256
  --precision bfloat16
)

"${python_bin}" "${repo_root}/scripts/ruler/diagnostics/make_balanced_subset.py" \
  "${manifest}" "${EXPERIMENT_ROOT}/screen_manifest" --samples-per-task 10

# The full dense pass doubles as Algorithm-2 calibration: every lambda is
# evaluated on the same dense QK tensors and cannot perturb accuracy.
dense_dir=${EXPERIMENT_ROOT}/runs/dense_eager_calibration
if [[ ! -s "${dense_dir}/summary.json" || ! -s "${dense_dir}/predictions.jsonl" ]]; then
  "${python_bin}" -m dllm.cli.eval_ruler "${common[@]}" \
    --manifest "${manifest}" --num-samples 650 \
    --attention-backend eager-dense \
    --blasst-calibration-lambdas 0.003,0.01,0.03,0.1,0.3,0.6,0.9 \
    --q-tile-size 128 --kv-tile-size 64 \
    --collect-attention-stats --stats-level layer \
    --output-dir "${dense_dir}"
fi

# Exploratory q/lambda sweeps use all 13 tasks but only 10 examples per task.
# Policies selected from this screen are re-run on the full 50/task manifest.
for q_tile in 32 64 128 256; do
  run_dir=${EXPERIMENT_ROOT}/screen/q_${q_tile}
  if [[ -s "${run_dir}/summary.json" && -s "${run_dir}/predictions.jsonl" ]]; then
    continue
  fi
  "${python_bin}" -m dllm.cli.eval_ruler "${common[@]}" \
    --manifest "${screen_manifest}" --num-samples 130 \
    --attention-backend blasst-reference --blasst-lambda 0.003 \
    --q-tile-size "${q_tile}" --kv-tile-size 64 \
    --collect-attention-stats --stats-level layer \
    --output-dir "${run_dir}"
done

for lambda_name in 0p001 0p003 0p01 0p03 0p1; do
  lambda_value=${lambda_name/p/.}
  run_dir=${EXPERIMENT_ROOT}/screen/lambda_${lambda_name}
  if [[ -s "${run_dir}/summary.json" && -s "${run_dir}/predictions.jsonl" ]]; then
    continue
  fi
  "${python_bin}" -m dllm.cli.eval_ruler "${common[@]}" \
    --manifest "${screen_manifest}" --num-samples 130 \
    --attention-backend blasst-reference --blasst-lambda "${lambda_value}" \
    --q-tile-size 128 --kv-tile-size 64 \
    --collect-attention-stats --stats-level layer \
    --output-dir "${run_dir}"
done
