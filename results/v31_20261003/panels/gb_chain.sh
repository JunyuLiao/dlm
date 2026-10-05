#!/bin/bash
# Panel gb (overlay ov_gs; EXPLORATORY), the budget-matched comparison with MAGE on the RULER v31 pool (panel x cells,
# mpk 1/3, dlm2 0/3). Same token budget per unit as MAGE (equal tiles per (head, block): balanced CTAs, so the held
# call costs what MAGE's does), varying only the statistic and the selection time:
#   2 x 2 at 4096 tokens: {MAGE's mean mass per KV head, per-(query head, block) worst-row max share} x {step 0, step 1}
#   (MAGE 4096 at step 0 = panel s/x's mage4096 plain, not rerun), plus the max-share unit at step 1 with 2048 tokens.
# The carry is on for the step-1 arms (it only acts on multi-canvas outputs; RULER answers are one canvas).
# Every arm waits for an empty GPU. env: W PY MODEL ROOT RSHARD WAIT_STATUS
set -u
cd $W
until grep -q '^done ' $W/$WAIT_STATUS 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
RUL="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$RSHARD DATASETS=ruler32k,ruler64k MEM=0.90 TAG=gb FIX_51994=1
  CELLS=$W/cells_ruler31.json MAN_DIR=$W/manifests_ruler31 BENCH=$W/bench_ov_gs.py OVERLAY=$W/ov_gs"
arm() {  # BUDGET GRAN STEP CARRY
  env $RUL $FAST MAGE_GRAN=$2 MAGE_STEP=$3 MAGE_CARRY=$4 LABEL_SUFFIX=_$2_step$3 ARMS="mage:PIECEWISE:$1" \
    bash $W/v31_paired_host8.sh > $W/host_gb_$1_$2_$3.log 2>&1
  sed -i "s/^done /done_$1_$2_$3 /" $W/status_gb
}
arm 4096 qblock_max 1 1
arm 4096 qblock_max 0 0
arm 4096 kvhead 1 1
arm 2048 qblock_max 1 1
echo "done $(date -u)" >> $W/status_gb
