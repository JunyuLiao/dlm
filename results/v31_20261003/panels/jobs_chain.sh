#!/bin/bash
# Generic screening job runner (the S1 suite: RULER v34 / LongBench-v2 0shot_think exploration / MRCR / GraphWalks
# b32k, seeds 1-2, every arm of a cell on this host; official pools with their pinned max_model_len). Waits until
# $WAITFILE has a line starting with "done ", then runs the job list $JOBS, one job per line:
#   PART NAME BUDGET [ENV=VAL ...]     PART ruler|lbt|mrcr|gw, NAME the label suffix after _lean ('-' = none),
#                                      BUDGET dense (vLLM FULL) or a MAGE token budget (lean execution, PIECEWISE)
# Records go to the S1 tags (sc1<part>) so every arm of a part is scored together; overlay $OV (default ov_pa4).
# Each job waits for an empty GPU (v31_paired_host8.sh). Appends progress and a final "done" to $CHAIN.
# env: W PY MODEL ROOT SHARD P JOBS WAITFILE CHAIN [OV]
set -u
cd $W
until grep -q '^done ' $W/$WAITFILE 2>/dev/null; do sleep 30; done
O=$W/${OV:-ov_pa4}; BENCHF=$W/bench_${OV:-ov_pa4}.py; S=$W/sc1
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
LEAN="MAGE_GRAN=qblock_max MAGE_STEP=1 MAGE_CARRY=1"
run() {  # PART NAME BUDGET EXTRA...
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
  B="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD FIX_51994=1 MEM=0.90 OVERLAY=$O BENCH=$BENCHF TAG=$T DATASETS=$D $X"
  if [ $K = dense ]; then
    env $B "$@" LABEL_SUFFIX=$L ARMS='dense:default' bash $W/v31_paired_host8.sh > $W/host_${T}_dense$L.log 2>&1
  elif [ $K = native ]; then                  # vLLM's dense attention through the adapter hooks (step-control rules)
    env $B "$@" LABEL_SUFFIX=$L ARMS='native:PIECEWISE' bash $W/v31_paired_host8.sh > $W/host_${T}_native$L.log 2>&1
  else
    env $B $FAST $LEAN "$@" LABEL_SUFFIX=_lean$L ARMS="mage:PIECEWISE:$K" bash $W/v31_paired_host8.sh > $W/host_${T}_$K$L.log 2>&1
  fi
  sed -i "s/^done /done_$K$L /" $W/status_$T
  echo "$T $K$L complete $(date -u)" >> $W/$CHAIN
}
mapfile -t JOBL < $JOBS                      # read the list first: the runs must not share its stdin
for line in "${JOBL[@]}"; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue ;; esac
  run $line < /dev/null
done
echo "done $(date -u)" >> $W/$CHAIN
