#!/usr/bin/env bash
set -euo pipefail

cd /home/exouser/ljy/dlm

while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; do
  echo "H100 is occupied; waiting 30 seconds before starting MATH-500 block-max sweep" >&2
  sleep 30
done

export PYTHONPATH=/home/exouser/ljy/dlm/src:/home/exouser/ljy/dlm
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR=/tmp/mpl-diffusion-gemma-math500

exec /home/exouser/miniconda3/envs/ljy_dlm/bin/python \
  benchmarks/diffusion_gemma_math500/run_blockmax_quantiles.py \
  --nemo-gym-root /tmp/nemo-gym-math500 \
  --output-dir results/math500/blockmax_k50
