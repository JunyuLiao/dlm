#!/bin/bash
# Panel n (overlay10): step-level dense fallback near canvas convergence (DENSE_WHEN=conv:THETA: dense GLOBAL attention
# on the next step once the canvas mean entropy < THETA x confidence threshold), with traces. After panel m.
set -u
O=$W/overlay10
until grep -q '^done' $W/status_m 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
cd $W
FAST="TRACE=1 LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton"
run() {  # SUFFIX DENSE_WHEN ARMS
  env W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD DATASETS=longbench_v2_32k,longbench_v2_64k MEM=0.90 TAG=n FIX_51994=1 \
    LABEL_SUFFIX=$1 DENSE_WHEN=$2 $FAST BENCH=$W/v31_vllm_paired_bench10.py OVERLAY=$O ARMS="$3" \
    bash $W/v31_paired_host6.sh > $W/host_n$1.log 2>&1
  sed -i "s/^done/done$1/" $W/status_n
}
run _dw4 conv:4 'method:PIECEWISE:m2c_r method:PIECEWISE:m2c_cgate_r'
run _dw20 conv:20 'method:PIECEWISE:m2c_r'
echo "done $(date -u)" >> $W/status_n
