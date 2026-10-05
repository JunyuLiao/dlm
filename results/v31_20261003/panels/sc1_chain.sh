#!/bin/bash
# Screening round S1 (EXPLORATORY; overlay ov_pa3 = research/v31-progress-aware-20261004 round 3), after ak. Replaces
# the queued pa2 panel (its RULER part could not trigger: RULER canvases converge in ~4 steps). Every arm runs the
# whole screening suite (v31_screen_suite.py, sc_suite.json): RULER v34 (78 items), LongBench-v2 0shot_think
# exploration split (32), MRCR 2-needle (24), GraphWalks b32k (12), each item under sampler seeds 1 and 2; official
# pools with their pinned max_model_len. Arms:
#   dense FULL (official serving path) | lean 4096 | lean 8192 (budget reference)
#   T1  lean 4096 + progress trigger f = 0.5 (re-select once per canvas at the step after the sampler first accepts
#       >= half of the canvas rows), C-gate rows (accepted rows weigh 0)
#   T2  T1 with a 2048-token re-selection (the budget decays with denoising progress)
#   T3  lean 8192 until the trigger, then a 2048-token C-gate re-selection (front-loaded budget)
#   T4  T1 with margin weights (query sensitivity M = 1 + 3 / (top-2 margin + 1)) instead of the C gate
# C gate / query sensitivity: Junyu Liao's ideas (collaboration candidates). Every arm waits for an empty GPU.
# env: W PY MODEL ROOT SHARD P
set -u
cd $W
until grep -q '^done ' $W/status_akchain 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
O=$W/ov_pa3; S=$W/sc1
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
LEAN="MAGE_GRAN=qblock_max MAGE_STEP=1 MAGE_CARRY=1"
arm() {  # NAME BUDGET(dense|tokens) EXTRA...: one arm over the four parts
  local L=$1 K=$2; shift 2
  for part in ruler lbt mrcr gw; do
    case $part in
      ruler) D=ruler32k_v34ofc,ruler64k_v34ofc,ruler128k_v34ofc; M=$P/ruler_v34ofc; X=136192 ;;
      lbt)   D=longbench_v2_0shot_think; M=$P/longbench_v2_ofc; X=141312 ;;
      mrcr)  D=mrcr2_32k_ofc,mrcr2_64k_ofc,mrcr2_128k_ofc; M=$P/mrcr_ofc; X=143360 ;;
      gw)    D=graphwalks_22k_b32k,graphwalks_45k_b32k,graphwalks_90k_b32k; M=$P/graphwalks_b32k; X=126976 ;;
    esac
    B="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD FIX_51994=1 MEM=0.90 OVERLAY=$O BENCH=$W/bench_ov_pa3.py
      TAG=sc1$part DATASETS=$D CELLS=$S/cells_sc_$part.json MAN_DIR=$M MAX_MODEL_LEN=$X"
    if [ $K = dense ]; then
      env $B ARMS='dense:default' bash $W/v31_paired_host8.sh > $W/host_sc1${part}_$K.log 2>&1
    else
      env $B $FAST $LEAN "$@" LABEL_SUFFIX=_lean$L ARMS="mage:PIECEWISE:$K" bash $W/v31_paired_host8.sh \
        > $W/host_sc1${part}_$K$L.log 2>&1
    fi
    sed -i "s/^done /done_$K$L /" $W/status_sc1$part
  done
  echo "$K$L complete $(date -u)" >> $W/status_sc1chain
}
arm '' dense
arm '' 4096
arm _t50_cgate 4096 MAGE_RESELECT_TRIGGER=0.5 MAGE_ROWW=cgate
arm _t50_cgate_k2048 4096 MAGE_RESELECT_TRIGGER=0.5 MAGE_ROWW=cgate MAGE_RESELECT_K=2048
arm _t50_cgate_k2048 8192 MAGE_RESELECT_TRIGGER=0.5 MAGE_ROWW=cgate MAGE_RESELECT_K=2048
arm _t50_margin 4096 MAGE_RESELECT_TRIGGER=0.5 MAGE_ROWW=margin
arm '' 8192
echo "done $(date -u)" >> $W/status_sc1chain
