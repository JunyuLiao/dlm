#!/bin/bash
# Screening round S1, re-planned at 22:30 UTC (replaces the rest of sc1_chain.sh; same overlay ov_pa3, same suite).
# Panel pa / sc1 so far: the absolute progress trigger fires at step ~1 on short answers (trailing end-of-text rows)
# and did not change LongBench-think accuracy, so T4 (trigger + margin weights) is dropped and the lean 8192 budget
# reference moves ahead of T2 / T3. Runs the job list in $JOBS (one job per line: PART NAME BUDGET [ENV=VAL ...]) after
# the arm in flight; a "done" line that arm leaves unrenamed in a status file becomes "done_inflight". Ends with
# "done" in status_sc1chain (sc1x waits on it). env: W PY MODEL ROOT SHARD P JOBS
set -u
cd $W
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
sleep 30
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
for f in status_sc1ruler status_sc1lbt status_sc1mrcr status_sc1gw; do [ -f $f ] && sed -i "s/^done /done_inflight /" $f; done
O=$W/ov_pa3; S=$W/sc1
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
LEAN="MAGE_GRAN=qblock_max MAGE_STEP=1 MAGE_CARRY=1"
run() {  # PART NAME BUDGET(dense|tokens) EXTRA...
  local part=$1 L=$2 K=$3; shift 3
  [ "$L" = - ] && L=""
  local X=""
  case $part in
    ruler) D=ruler32k_v34ofc,ruler64k_v34ofc,ruler128k_v34ofc; X="CELLS=$S/cells_sc_ruler.json MAN_DIR=$P/ruler_v34ofc MAX_MODEL_LEN=136192" ;;
    lbt)   D=longbench_v2_0shot_think; X="CELLS=$S/cells_sc_lbt.json MAN_DIR=$P/longbench_v2_ofc MAX_MODEL_LEN=141312" ;;
    mrcr)  D=mrcr2_32k_ofc,mrcr2_64k_ofc,mrcr2_128k_ofc; X="CELLS=$S/cells_sc_mrcr.json MAN_DIR=$P/mrcr_ofc MAX_MODEL_LEN=143360" ;;
    gw)    D=graphwalks_22k_b32k,graphwalks_45k_b32k,graphwalks_90k_b32k; X="CELLS=$S/cells_sc_gw.json MAN_DIR=$P/graphwalks_b32k MAX_MODEL_LEN=126976" ;;
  esac
  local T=sc1$part
  B="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD FIX_51994=1 MEM=0.90 OVERLAY=$O BENCH=$W/bench_ov_pa3.py TAG=$T DATASETS=$D $X"
  env $B $FAST $LEAN "$@" LABEL_SUFFIX=_lean$L ARMS="mage:PIECEWISE:$K" bash $W/v31_paired_host8.sh > $W/host_${T}_$K$L.log 2>&1
  sed -i "s/^done /done_$K$L /" $W/status_$T
  echo "$T $K$L complete $(date -u)" >> $W/status_sc1chain
}
mapfile -t JOBL < $JOBS                      # read the list first: the runs must not share its stdin
for line in "${JOBL[@]}"; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue ;; esac
  run $line < /dev/null
done
echo "done $(date -u)" >> $W/status_sc1chain
