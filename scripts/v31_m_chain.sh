#!/bin/bash
# Panel m (diagnostic, overlay9, TRACE=1): where do sparsity-induced extra denoising steps occur? Per-canvas step counts
# and mean-entropy trajectories for dense (through the adapter) and four sparse arms. After panel l. env: W PY MODEL ROOT SHARD
set -u
O=$W/overlay9
until grep -q '^done' $W/status_l 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
cd $W
env W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD DATASETS=longbench_v2_32k,longbench_v2_64k MEM=0.90 TAG=m FIX_51994=1 \
  LABEL_SUFFIX=_trace TRACE=1 LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4 \
  BENCH=$W/v31_vllm_paired_bench9.py OVERLAY=$O \
  ARMS='native:PIECEWISE method:PIECEWISE:m2c_r method:PIECEWISE:m2c_cgate_r method:PIECEWISE:m2c_k5_mass mage:PIECEWISE:1024' \
  bash $W/v31_paired_host6.sh > $W/host_m.log 2>&1
