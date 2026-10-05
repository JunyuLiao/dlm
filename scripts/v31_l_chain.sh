#!/bin/bash
# Panel l (overlay8: realized kept-fraction receipts): adaptive budget (threshold +/- C gate) vs uniform budgets
# (k20, k30) on M2 compact mu, plus MAGE k=2048; every efficiency switch. Runs after panel k. env: W PY MODEL ROOT SHARD
set -u
O=$W/overlay8
until grep -q '^done' $W/status_k 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
cd $W
env W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD DATASETS=longbench_v2_32k,longbench_v2_64k MEM=0.90 TAG=l FIX_51994=1 \
  LABEL_SUFFIX=_fast LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4 \
  BENCH=$W/v31_vllm_paired_bench6.py OVERLAY=$O \
  ARMS='method:PIECEWISE:m2c_r method:PIECEWISE:m2c_cgate_r method:PIECEWISE:m2c_k20 method:PIECEWISE:m2c_k30 mage:PIECEWISE:2048' \
  bash $W/v31_paired_host6.sh > $W/host_l.log 2>&1
