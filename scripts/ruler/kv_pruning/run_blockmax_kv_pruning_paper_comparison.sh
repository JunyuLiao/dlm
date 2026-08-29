#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
result_root=${repo_root}/results/kv_pruning/blockmax_ruler_paper_8k_n650
ruler_root=/tmp/NVIDIA-RULER
dg_python=/home/exouser/miniconda3/envs/ljy_dlm/bin/python
fast_python=/home/exouser/miniconda3/bin/python
dg_dependency=/tmp/diffusion-gemma-tf511-py312
dg_samples=${repo_root}/results/blasst/diffusion_gemma/paper_8k_n650/manifest/samples.jsonl
fast_samples=${result_root}/fast_dllm_v2/manifest/samples.jsonl

wait_for_stable_idle_h100() {
  local idle_checks=0
  local gpu_used_mib
  while [[ ${idle_checks} -lt 4 ]]; do
    gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [[ ${gpu_used_mib} -lt 1024 ]]; then
      idle_checks=$((idle_checks + 1))
      echo "H100 idle check ${idle_checks}/4"
      sleep 15
    else
      idle_checks=0
      echo "waiting for H100: ${gpu_used_mib} MiB in use"
      sleep 30
    fi
  done
}

export HF_HUB_OFFLINE=1
export MPLCONFIGDIR=/tmp/matplotlib-blockmax-paper

while true; do
  wait_for_stable_idle_h100
  set +e
  PYTHONPATH=${repo_root}/src:${dg_dependency} "${dg_python}" \
    "${repo_root}/scripts/ruler/kv_pruning/quantile_kv_pruning_experiment.py" run \
    --model-adapter diffusion_gemma \
    --model-path google/diffusiongemma-26B-A4B-it \
    --revision f7f5b7f5fa82ffc52addd066915886d497f5517b \
    --samples "${dg_samples}" \
    --output-dir "${result_root}/diffusion_gemma/run" \
    --num-samples 650 --context-length 8192 --length-mode total \
    --q-tile-size 128 --kv-tile-size 64 --block-size 256 \
    --methods block_max random --k-values 0.10 0.25 0.50 0.75
  dg_status=$?
  set -e
  if [[ ${dg_status} -eq 0 ]]; then
    break
  fi
  gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ ${gpu_used_mib} -lt 1024 ]]; then
    exit "${dg_status}"
  fi
  echo "DiffusionGemma launch collided with another GPU job; retrying after it finishes"
done

PYTHONPATH=${repo_root}/src:${dg_dependency} "${dg_python}" \
  "${repo_root}/scripts/ruler/kv_pruning/quantile_kv_pruning_experiment.py" report \
  --output-dir "${result_root}/diffusion_gemma/run" \
  --ruler-root "${ruler_root}"

wait_for_stable_idle_h100
PYTHONPATH=${repo_root}/src:/tmp/dllm-ruler "${fast_python}" \
  "${repo_root}/scripts/ruler/kv_pruning/quantile_kv_pruning_experiment.py" run \
  --model-adapter fast_dllm_v2 \
  --model-path /tmp/fast_dllm_v2_7b \
  --samples "${fast_samples}" \
  --output-dir "${result_root}/fast_dllm_v2/run" \
  --num-samples 650 --context-length 8192 --length-mode total \
  --q-tile-size 128 --kv-tile-size 64 --block-size 256 \
  --methods block_max random --k-values 0.10 0.25 0.50 0.75

PYTHONPATH=${repo_root}/src:/tmp/dllm-ruler "${fast_python}" \
  "${repo_root}/scripts/ruler/kv_pruning/quantile_kv_pruning_experiment.py" report \
  --output-dir "${result_root}/fast_dllm_v2/run" \
  --ruler-root "${ruler_root}"

PYTHONPATH=${repo_root}/src "${dg_python}" \
  "${repo_root}/scripts/ruler/kv_pruning/compare_blockmax_kv_pruning_models.py" \
  --diffusion-gemma-dir "${result_root}/diffusion_gemma/run" \
  --fast-dllm-dir "${result_root}/fast_dllm_v2/run" \
  --output-dir "${result_root}/comparison"
