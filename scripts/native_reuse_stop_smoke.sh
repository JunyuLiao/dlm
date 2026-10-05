#!/usr/bin/env bash
set -euo pipefail
root=/home/exouser/dyh/numerical_qk_reuse_native_20260924
python=/home/exouser/miniconda3/envs/ljy_dlm/bin/python
model=/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPU_BUSY: stop diagnostic not started' >&2
  exit 2
fi
if [[ -e "$root/runs/stop_diagnostic_01" ]]; then
  echo 'Existing stop diagnostic: inspect rather than duplicate' >&2
  exit 3
fi
mkdir -p "$root/runs/stop_diagnostic_01"
cd "$root/code_cp2"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$root/inductor_cache" TRITON_CACHE_DIR="$root/triton_cache"
exec /usr/bin/time -v "$python" ../analysis_cp1/scripts/native_reuse_stop_diagnostic.py \
  --manifest results/numerical_qk_reuse_20260924/private/smoke_manifest.json \
  --output ../runs/stop_diagnostic_01 --model "$model" \
  --revision f7f5b7f5fa82ffc52addd066915886d497f5517b \
  --id aime26/2 --conditions fresh_junyu_T M1 M3 \
  --library ../build_cp1/value_direction_db080045f7a5fbce.so \
  --torch-library ../build_cp1/torch_4c65c048754f9fb7/value_direction_torch_4c65c048754f9fb7.so \
  > "$root/runs/stop_diagnostic_01/process.log" 2>&1
