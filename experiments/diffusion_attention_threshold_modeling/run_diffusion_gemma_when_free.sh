#!/usr/bin/env bash
set -euo pipefail

# Wait for the shared H100 to have enough headroom for the pinned 48 GiB
# DiffusionGemma checkpoint. This never signals or modifies other processes.
while true; do
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 {print int($1)}' || true)"
  if [ "${free_mib:-0}" -ge 52000 ]; then
    break
  fi
  sleep 30
done

cd /home/exouser/ljy/dlm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/tmp/threshold-transformers-5.11:src
PYTHON=/home/exouser/miniconda3/envs/ljy_dlm/bin/python
BASE=("${PYTHON}" -m experiments.diffusion_attention_threshold_modeling)

echo "[$(date -u +%FT%TZ)] H100 headroom available; running DiffusionGemma smoke" >&2
"${BASE[@]}" \
  eval-ruler-routing \
  --adapter diffusion_gemma \
  --model-path /home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b \
  --revision f7f5b7f5fa82ffc52addd066915886d497f5517b \
  --manifest-path results/attention/routing/ruler16k/manifests/smoke_diffusion_gemma/manifest.json \
  --ruler-root /tmp/NVIDIA-RULER \
  --output-dir results/attention/routing/ruler16k/smoke_diffusion_gemma \
  --num-samples 2 \
  --conditions dense gaussian_rho75 gaussian_rho75_prefix_only \
  --device cuda --precision bfloat16 --temperature 0

echo "[$(date -u +%FT%TZ)] smoke completed; running corrected 50-example DiffusionGemma sweep" >&2
"${BASE[@]}" \
  eval-ruler-routing \
  --adapter diffusion_gemma \
  --model-path /home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b \
  --revision f7f5b7f5fa82ffc52addd066915886d497f5517b \
  --manifest-path results/attention/routing/ruler16k/manifests/diffusion_gemma/manifest.json \
  --ruler-root /tmp/NVIDIA-RULER \
  --output-dir results/attention/routing/ruler16k/diffusion_gemma \
  --num-samples 50 \
  --threshold-model results/attention/routing/ruler16k/profiled_threshold_model.json \
  --device cuda --precision bfloat16 --temperature 0

echo "[$(date -u +%FT%TZ)] logical sweep completed; rebuilding reports" >&2
"${BASE[@]}" report-ruler \
  --output-dir results/attention/routing/ruler16k \
  --ruler-root /tmp/NVIDIA-RULER
"${BASE[@]}" canonical-report \
  --bundle-root results/attention/routing/ruler16k
