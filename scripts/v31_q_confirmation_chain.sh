#!/bin/bash
# Panel q = CONFIRMATION SET (pre-registered 2026-10-04 00:25 UTC before any confirmation result):
# same 24 / 24 / 11 items, NEW panel seeds 3,4,5 (32K, 64K) and 3,4 (96K), cells_confirm.json, sharded over 3 hosts.
# Arms: dense FULL+fix (reference); m2c + step-20 dense rescue; m2c; mass k5 + rescue; MAGE k=4096; MAGE k=4096 + rescue.
# Every efficiency switch on; no traces (they add per-step work). env: W PY MODEL ROOT SHARD WAITCMD
set -u
O=$W/overlay11
eval "$WAITCMD"
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
cd $W
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
run() {  # SUFFIX DENSE_WHEN ARMS MEM
  env W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD DATASETS=longbench_v2_32k,longbench_v2_64k,longbench_v2_96k \
    MEM=$4 TAG=q FIX_51994=1 CELLS=$W/cells_confirm.json LABEL_SUFFIX=$1 DENSE_WHEN=$2 $FAST \
    BENCH=$W/v31_vllm_paired_bench11.py OVERLAY=$O ARMS="$3" bash $W/v31_paired_host7.sh > $W/host_q$1.log 2>&1
  sed -i "s/^done/done$1/" $W/status_q
}
run _ref "" 'dense:default' 0.92
run _plain "" 'method:PIECEWISE:m2c_r mage:PIECEWISE:4096' 0.92
run _s20 step:20 'method:PIECEWISE:m2c_r method:PIECEWISE:m2c_k5_mass mage:PIECEWISE:4096' 0.92
echo "done $(date -u)" >> $W/status_q
