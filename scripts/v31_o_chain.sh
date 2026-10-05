#!/bin/bash
# Panel o (dlm2, overlay11, REGROUP_DIAG=1): regroup potential of the threshold selectors on real trajectories
# (kept fraction at 128 rows vs 64-row halves vs regrouped halves vs per row). Writes a final ^done to status_o.
set -u
O=$W/overlay11
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
cd $W
env W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD DATASETS=longbench_v2_32k,longbench_v2_64k MEM=0.90 TAG=o FIX_51994=1 \
  LABEL_SUFFIX=_rgd REGROUP_DIAG=1 LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton \
  BENCH=$W/v31_vllm_paired_bench11.py OVERLAY=$O \
  ARMS='method:PIECEWISE method:PIECEWISE:m2c_r method:PIECEWISE:m2c_cgate_r' \
  bash $W/v31_paired_host6.sh > $W/host_o.log 2>&1
