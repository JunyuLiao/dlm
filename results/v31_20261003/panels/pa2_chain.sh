#!/bin/bash
# Panel pa2 (EXPLORATORY; overlay ov_pa2 = research/v31-progress-aware-20261004 round 2), after ak: the query-sensitivity
# family (Junyu Liao's M margin / C confidence / T temporal weights and the C gate; collaboration candidates) as row
# weights of a once-per-canvas re-selection on the lean candidate (4096 tokens), plus a budget schedule.
#  (1) accuracy, RULER v34 screening subset + MRCR: rs4 + margin, rs4 + temporal, rs4 + MT, rs4 + C gate with an 8192-
#      token re-selection;
#  (2) adaptive generation length on IDENTICAL canvases (forced mode, LongBench-v2 64K + 96K, fc's dense PIECEWISE
#      reference): lean, rs4, rs4 + C gate, rs4 + C, rs4 + M, rs4 + T -- denoising steps per canvas and token agreement.
# Every arm waits for an empty GPU. env: W PY MODEL ROOT SHARD LBSHARD P
set -u
cd $W
until grep -q '^done ' $W/status_akchain 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
O=$W/ov_pa2
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
LEAN="MAGE_GRAN=qblock_max MAGE_STEP=1 MAGE_CARRY=1"
arm() {  # TAG LABEL_SUFFIX EXTRA...
  local T=$1 L=$2; shift 2
  env $B $FAST $LEAN "$@" LABEL_SUFFIX=$L ARMS='mage:PIECEWISE:4096' bash $W/v31_paired_host8.sh > $W/host_${T}${L}.log 2>&1
  sed -i "s/^done /done$L /" $W/status_$T
}
for part in ruler mrcr; do
  if [ $part = ruler ]; then
    B="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD FIX_51994=1 MEM=0.90 OVERLAY=$O BENCH=$W/bench_ov_pa2.py TAG=paruler
      DATASETS=ruler32k_v34ofc,ruler128k_v34ofc CELLS=$W/cells_pa_v34.json MAN_DIR=$P/ruler_v34ofc MAX_MODEL_LEN=136192"
  else
    B="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD FIX_51994=1 MEM=0.90 OVERLAY=$O BENCH=$W/bench_ov_pa2.py TAG=pamrcr
      DATASETS=mrcr2_32k_ofc,mrcr2_64k_ofc,mrcr2_128k_ofc CELLS=$P/mrcr_ofc/cells_mrcr_ofc.json MAN_DIR=$P/mrcr_ofc MAX_MODEL_LEN=143360"
  fi
  arm pa$part _lean_rs4_margin MAGE_RESELECT=4 MAGE_ROWW=margin
  arm pa$part _lean_rs4_temporal MAGE_RESELECT=4 MAGE_ROWW=temporal
  arm pa$part _lean_rs4_mt MAGE_RESELECT=4 MAGE_ROWW=mt
  arm pa$part _lean_rs4_cgate_k8192 MAGE_RESELECT=4 MAGE_ROWW=cgate MAGE_RESELECT_K=8192
done
B="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$LBSHARD FIX_51994=1 MEM=0.90 OVERLAY=$O BENCH=$W/bench_ov_pa2.py TAG=pafc
  DATASETS=longbench_v2_64k,longbench_v2_96k CELLS=$W/cells_confirm.json FORCE_REF=$W/private/fc_dense_PIECEWISE_ref.tokens.jsonl"
arm pafc _lean
arm pafc _lean_rs4 MAGE_RESELECT=4
arm pafc _lean_rs4_cgate MAGE_RESELECT=4 MAGE_ROWW=cgate
arm pafc _lean_rs4_conf MAGE_RESELECT=4 MAGE_ROWW=conf
arm pafc _lean_rs4_margin MAGE_RESELECT=4 MAGE_ROWW=margin
arm pafc _lean_rs4_temporal MAGE_RESELECT=4 MAGE_ROWW=temporal
echo "done $(date -u)" >> $W/status_pa2chain
