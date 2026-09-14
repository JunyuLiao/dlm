#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/exouser/ljy/dlm
python_bin=/home/exouser/miniconda3/envs/ljy_dlm/bin/python
output_dir="$repo_root/results/diffusion_gemma_solattn_math500_prefix_vs_all"
model_path=/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b
revision=f7f5b7f5fa82ffc52addd066915886d497f5517b

mkdir -p "$output_dir"
while true; do
  used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ "$used_mib" =~ ^[0-9]+$ ]] && (( used_mib < 5000 )); then
    break
  fi
  printf '%s waiting_for_gpu used_mib=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$used_mib" >> "$output_dir/launcher.log"
  sleep 60
done

printf '%s gpu_acquired\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$output_dir/launcher.log"
cd "$repo_root"
export PYTHONPATH="$repo_root/src:$repo_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"$python_bin" -m experiments.diffusion_gemma_solattn_math500 smoke \
  --output-dir "$output_dir" \
  --model-path "$model_path" \
  --revision "$revision" \
  >> "$output_dir/smoke.log" 2>&1

printf '%s smoke_passed_sweep_starting\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$output_dir/launcher.log"
"$python_bin" -m experiments.diffusion_gemma_solattn_math500 run \
  --output-dir "$output_dir" \
  --model-path "$model_path" \
  --revision "$revision" \
  >> "$output_dir/run.log" 2>&1

"$python_bin" -m experiments.diffusion_gemma_solattn_math500 report \
  --output-dir "$output_dir" \
  >> "$output_dir/report.log" 2>&1
printf '%s experiment_complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$output_dir/launcher.log"
