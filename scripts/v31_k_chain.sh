#!/bin/bash
# Panel k: selection quality at a fixed budget (keep fraction k12 / k5 per head and query block), risk vs mass ranking,
# with and without the C gate; M2 compact mu, every efficiency switch. env: W PY MODEL ROOT SHARD
set -u
O=$W/overlay7
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
cd $W
env W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD DATASETS=longbench_v2_32k,longbench_v2_64k MEM=0.90 TAG=k FIX_51994=1 \
  LABEL_SUFFIX=_fast LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton \
  BENCH=$W/v31_vllm_paired_bench6.py OVERLAY=$O \
  ARMS='method:PIECEWISE:m2c_k12 method:PIECEWISE:m2c_k12_mass method:PIECEWISE:m2c_k12_cgate method:PIECEWISE:m2c_k12_mass_cgate method:PIECEWISE:m2c_k5 method:PIECEWISE:m2c_k5_mass method:PIECEWISE:m2c_k5_cgate method:PIECEWISE:m2c_k5_mass_cgate' \
  bash $W/v31_paired_host6.sh > $W/host_k.log 2>&1
