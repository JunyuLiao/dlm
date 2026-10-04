#!/bin/bash
# Panel s (overlay12): two-level selection = MAGE k=4096 shared budget + per-query-head critical tiles
# (MAGE_CRIT = mass-share threshold), RULER 32K/64K v31 pool, accuracy screen. env: W PY MODEL ROOT SHARD
set -u
O=$W/overlay12
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
cd $W
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
run() {  # SUFFIX MAGE_CRIT ARMS
  env W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD DATASETS=ruler32k,ruler64k MEM=0.90 TAG=s FIX_51994=1 \
    CELLS=$W/cells_ruler31.json MAN_DIR=$W/manifests_ruler31 LABEL_SUFFIX=$1 MAGE_CRIT=$2 $FAST \
    BENCH=$W/v31_vllm_paired_bench12.py OVERLAY=$O ARMS="$3" bash $W/v31_paired_host8.sh > $W/host_s$1.log 2>&1
  sed -i "s/^done/done$1/" $W/status_s
}
run _c05 0.05 'mage:PIECEWISE:4096'
run _c02 0.02 'mage:PIECEWISE:4096'
echo "done $(date -u)" >> $W/status_s
