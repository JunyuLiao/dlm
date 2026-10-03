#!/bin/bash
# Panel p (overlay11): q64 regrouping (chw/value_aware) on M2 compact, with and without the C gate (Junyu): stable and
# unstable query rows can share 64-row FA4 tiles with others of their kind. After panel n. env: W PY MODEL ROOT SHARD
set -u
O=$W/overlay11
until grep -q '^done ' $W/status_n 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
cd $W
env W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD DATASETS=longbench_v2_32k,longbench_v2_64k MEM=0.90 TAG=p FIX_51994=1 \
  LABEL_SUFFIX=_fast REGROUP_DIAG=1 LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton \
  BENCH=$W/v31_vllm_paired_bench11.py OVERLAY=$O \
  ARMS='method:PIECEWISE:m2c_q64r method:PIECEWISE:m2c_cgate_q64r' \
  bash $W/v31_paired_host6.sh > $W/host_p.log 2>&1
