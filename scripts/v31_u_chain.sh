#!/bin/bash
# Panel u (overlay13): coverage-adaptive per-KV-head budget (MAGE_COV = p, floor MAGE_K = 4096 tokens), RULER accuracy
# screen, targeting the diffuse-attention (aggregation, cwe) failure. After panel t. env: W PY MODEL ROOT SHARD
set -u
O=$W/overlay13
until [ -f $W/status_t ] && grep -q '^done' $W/status_t 2>/dev/null; do sleep 20; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
cd $W
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
run() {  # SUFFIX MAGE_COV
  env W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD DATASETS=ruler32k,ruler64k MEM=0.90 TAG=u FIX_51994=1 \
    CELLS=$W/cells_ruler31.json MAN_DIR=$W/manifests_ruler31 LABEL_SUFFIX=$1 MAGE_COV=$2 $FAST \
    BENCH=$W/v31_vllm_paired_bench13.py OVERLAY=$O ARMS='mage:PIECEWISE:4096' bash $W/v31_paired_host8.sh > $W/host_u$1.log 2>&1
  sed -i "s/^done/done$1/" $W/status_u
}
run _cov90 0.90
run _cov95 0.95
echo "done $(date -u)" >> $W/status_u
