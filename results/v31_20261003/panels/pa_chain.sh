#!/bin/bash
# Panel pa (EXPLORATORY screening; overlay ov_pa = research/v31-progress-aware-20261004): progress-aware re-selection on
# the lean candidate (MAGE port, qblock_max worst-row max share, step-1 selection, carry, 4096 tokens per unit).
#   P0 dense FULL; P1 lean (baseline); P2 lean + re-selection at step 4 (all rows); P3 + C-gate rows (rows the
#   sampler accepted at step 3 weigh 0; Junyu Liao's C gate idea); P4 + confidence prior T (1 + 3 sqrt(1 - p_top));
#   P5 lean + re-selection at step 8 with C-gate rows.
# Data: RULER v34ofc screening subset (13 tasks x 6 items x 32K / 128K = 156 cells, sharded) and MRCR 2-needle (72,
# sharded). If FINISH_GW=1, first the last stage-A GraphWalks arm (lean 2048) of this host. Every arm waits for an
# empty GPU. env: W PY MODEL ROOT SHARD P FINISH_GW
set -u
cd $W
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
LEAN="MAGE_GRAN=qblock_max MAGE_STEP=1 MAGE_CARRY=1"
if [ "${FINISH_GW:-0}" = 1 ]; then
  env W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD FIX_51994=1 MEM=0.90 OVERLAY=$W/ov_int BENCH=$W/bench_ov_int.py \
    TAG=fagw DATASETS=graphwalks_22k_b32k,graphwalks_45k_b32k,graphwalks_90k_b32k \
    CELLS=$P/graphwalks_b32k/cells_graphwalks_b32k.json MAN_DIR=$P/graphwalks_b32k MAX_MODEL_LEN=126976 \
    $FAST $LEAN LABEL_SUFFIX=_lean ARMS='mage:PIECEWISE:2048' bash $W/v31_paired_host8.sh > $W/host_fagw_a3.log 2>&1
  sed -i "s/^done /done_a3 /" $W/status_fagw
  echo "fagw complete (finished by pa_chain) $(date -u)" >> $W/status_fachain
fi
O=$W/ov_pa
for part in ruler mrcr; do
  if [ $part = ruler ]; then
    B="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD FIX_51994=1 MEM=0.90 OVERLAY=$O BENCH=$W/bench_ov_pa.py TAG=paruler
      DATASETS=ruler32k_v34ofc,ruler128k_v34ofc CELLS=$W/cells_pa_v34.json MAN_DIR=$P/ruler_v34ofc MAX_MODEL_LEN=136192"
  else
    B="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD FIX_51994=1 MEM=0.90 OVERLAY=$O BENCH=$W/bench_ov_pa.py TAG=pamrcr
      DATASETS=mrcr2_32k_ofc,mrcr2_64k_ofc,mrcr2_128k_ofc CELLS=$P/mrcr_ofc/cells_mrcr_ofc.json MAN_DIR=$P/mrcr_ofc MAX_MODEL_LEN=143360"
  fi
  T=pa$part
  env $B ARMS='dense:default' bash $W/v31_paired_host8.sh > $W/host_${T}_p0.log 2>&1; sed -i "s/^done /done_p0 /" $W/status_$T
  env $B $FAST $LEAN LABEL_SUFFIX=_lean ARMS='mage:PIECEWISE:4096' bash $W/v31_paired_host8.sh > $W/host_${T}_p1.log 2>&1; sed -i "s/^done /done_p1 /" $W/status_$T
  env $B $FAST $LEAN MAGE_RESELECT=4 LABEL_SUFFIX=_lean_rs4 ARMS='mage:PIECEWISE:4096' bash $W/v31_paired_host8.sh > $W/host_${T}_p2.log 2>&1; sed -i "s/^done /done_p2 /" $W/status_$T
  env $B $FAST $LEAN MAGE_RESELECT=4 MAGE_ROWW=cgate LABEL_SUFFIX=_lean_rs4_cgate ARMS='mage:PIECEWISE:4096' bash $W/v31_paired_host8.sh > $W/host_${T}_p3.log 2>&1; sed -i "s/^done /done_p3 /" $W/status_$T
  env $B $FAST $LEAN MAGE_RESELECT=4 MAGE_ROWW=conf LABEL_SUFFIX=_lean_rs4_conf ARMS='mage:PIECEWISE:4096' bash $W/v31_paired_host8.sh > $W/host_${T}_p4.log 2>&1; sed -i "s/^done /done_p4 /" $W/status_$T
  env $B $FAST $LEAN MAGE_RESELECT=8 MAGE_ROWW=cgate LABEL_SUFFIX=_lean_rs8_cgate ARMS='mage:PIECEWISE:4096' bash $W/v31_paired_host8.sh > $W/host_${T}_p5.log 2>&1; sed -i "s/^done /done_p5 /" $W/status_$T
done
echo "done $(date -u)" >> $W/status_pachain
