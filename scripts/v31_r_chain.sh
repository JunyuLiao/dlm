#!/bin/bash
# Panel r: RULER 32K / 64K accuracy validation on the new v31 pool (13 tasks x 10 samples per length, generator seed
# 4242, pinned RULER checkout), same arms as the confirmation set. RULER answers are short (one canvas), so the
# request time is prefill-dominated: accuracy only. After panel q. env: W PY MODEL ROOT SHARD
set -u
O=$W/overlay11
until grep -q '^done ' $W/status_q 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
cd $W
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
run() {  # SUFFIX DENSE_WHEN ARMS
  env W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD DATASETS=ruler32k,ruler64k MEM=0.90 TAG=r FIX_51994=1 \
    CELLS=$W/cells_ruler31.json MAN_DIR=$W/manifests_ruler31 LABEL_SUFFIX=$1 DENSE_WHEN=$2 $FAST \
    BENCH=$W/v31_vllm_paired_bench11.py OVERLAY=$O ARMS="$3" bash $W/v31_paired_host8.sh > $W/host_r$1.log 2>&1
  sed -i "s/^done/done$1/" $W/status_r
}
run _ref "" 'dense:default'
run _plain "" 'method:PIECEWISE:m2c_r mage:PIECEWISE:4096'
run _s20 step:20 'method:PIECEWISE:m2c_r mage:PIECEWISE:4096'
echo "done $(date -u)" >> $W/status_r
