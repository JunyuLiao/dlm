#!/bin/bash
# Panel n (overlay10): step-level dense fallback (DENSE_WHEN). step:S = dense GLOBAL attention from the S-th denoising
# step of a canvas on (rescues the few canvases that linger above the convergence threshold, the main source of
# sparsity-induced forward inflation in panel m); conv:THETA = dense once the canvas mean entropy < THETA x threshold.
# Method and MAGE arms, with traces. After panel m.
set -u
O=$W/overlay10
until grep -q '^done' $W/status_m 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
cd $W
FAST="TRACE=1 LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
run() {  # SUFFIX DENSE_WHEN ARMS
  env W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD DATASETS=longbench_v2_32k,longbench_v2_64k MEM=0.90 TAG=n FIX_51994=1 \
    LABEL_SUFFIX=$1 DENSE_WHEN=$2 $FAST BENCH=$W/v31_vllm_paired_bench10.py OVERLAY=$O ARMS="$3" \
    bash $W/v31_paired_host6.sh > $W/host_n$1.log 2>&1
  sed -i "s/^done/done$1/" $W/status_n
}
run _s20 step:20 'method:PIECEWISE:m2c_r method:PIECEWISE:m2c_k5_mass mage:PIECEWISE:1024'
run _s28 step:28 'method:PIECEWISE:m2c_r'
run _dw4 conv:4 'method:PIECEWISE:m2c_r'
echo "done $(date -u)" >> $W/status_n
