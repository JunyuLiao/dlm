#!/bin/bash
# After sc1 (EXPLORATORY; overlay ov_pa3). Replaces the queued sc1s chain; adds what panel pa showed to matter.
#  (1) T0 = lean 4096 + progress trigger 0.5 WITHOUT row weights, on the four long parts of S1 (pa: the unweighted
#      re-selection was the best MRCR arm; weights did not help). Runs on ov_pa4 (round 4, identical single-threshold
#      behaviour, CPU-tested) with MAGE_CLOCK_TRACE=1: the receipts carry each canvas's per-step accepted fraction and
#      C-gate settledness, which calibrate round 4's thresholds (panel sc2, queued after this chain).
#  (2) forced canvases (identical canvases from each host's own dense PIECEWISE reference, LongBench-v2 64K / 96K, 47
#      cells; fc protocol): denoising steps per canvas, i.e. whether a re-observation also cuts step inflation -- a speed
#      lever beyond the attention share. Arms: dense FULL, lean, re-select at step 4, T0, T1, T2, T3.
#  The S1s short part moves after sc2 and runs only the finalists (user: short tasks check that the method does not
#  hurt them).
# Every arm waits for an empty GPU. env: W PY MODEL ROOT SHARD P
set -u
cd $W
until grep -q '^done ' $W/status_sc1chain 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
O=$W/ov_pa4; S=$W/sc1
if [ "${POOLCHECK:-0}" = 1 ]; then          # round 5 GPU check (pool observation vs dense observation), ~3 min
  C=$W/cache
  ( cd $ROOT/deploy/v27_r17_6064109 && PYTHONPATH=src:. PYTHONNOUSERSITE=1 V27_ADAPTER_DIR=$O TRITON_CACHE_DIR=$C/triton     TVM_FFI_CACHE_DIR=$C/tvm CUDA_CACHE_PATH=$C/cuda CUTE_DSL_CACHE_DIR=$C/cute FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$C/facute     TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg timeout 1200 $PY $O/v31_pool_observe_check.py $W/pool_check.jsonl ) > $W/pool_check.log 2>&1
  echo "poolcheck rc=$? $(date -u)" >> $W/status_sc1xchain
  until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
fi
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
LEAN="MAGE_GRAN=qblock_max MAGE_STEP=1 MAGE_CARRY=1"
run() {  # PART NAME BUDGET(dense|tokens) EXTRA...
  local part=$1 L=$2 K=$3; shift 3
  local X=""
  case $part in
    ruler) D=ruler32k_v34ofc,ruler64k_v34ofc,ruler128k_v34ofc; X="CELLS=$S/cells_sc_ruler.json MAN_DIR=$P/ruler_v34ofc MAX_MODEL_LEN=136192" ;;
    lbt)   D=longbench_v2_0shot_think; X="CELLS=$S/cells_sc_lbt.json MAN_DIR=$P/longbench_v2_ofc MAX_MODEL_LEN=141312" ;;
    mrcr)  D=mrcr2_32k_ofc,mrcr2_64k_ofc,mrcr2_128k_ofc; X="CELLS=$S/cells_sc_mrcr.json MAN_DIR=$P/mrcr_ofc MAX_MODEL_LEN=143360" ;;
    gw)    D=graphwalks_22k_b32k,graphwalks_45k_b32k,graphwalks_90k_b32k; X="CELLS=$S/cells_sc_gw.json MAN_DIR=$P/graphwalks_b32k MAX_MODEL_LEN=126976" ;;
    aime)  D=aime26; X="CELLS=$S/cells_sc_aime.json MAN_DIR=$W/manifests_ae MAX_MODEL_LEN=13312" ;;
    he)    D=humaneval; X="CELLS=$S/cells_sc_he.json MAN_DIR=$W/manifests_ae MAX_MODEL_LEN=13312" ;;
    fc)    D=longbench_v2_64k,longbench_v2_96k; X="CELLS=$W/cells_confirm.json FORCE_REF=$W/private/fc_dense_PIECEWISE_ref.tokens.jsonl" ;;
  esac
  local T=sc1$part
  B="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD FIX_51994=1 MEM=0.90 OVERLAY=$O BENCH=$W/bench_ov_pa4.py TAG=$T DATASETS=$D $X"
  if [ $K = dense ]; then
    env $B ARMS='dense:default' bash $W/v31_paired_host8.sh > $W/host_${T}_$K.log 2>&1
  else
    env $B $FAST $LEAN "$@" LABEL_SUFFIX=_lean$L ARMS="mage:PIECEWISE:$K" bash $W/v31_paired_host8.sh > $W/host_${T}_$K$L.log 2>&1
  fi
  sed -i "s/^done /done_$K$L /" $W/status_$T
  echo "$T $K$L complete $(date -u)" >> $W/status_sc1xchain
}
T0="MAGE_RESELECT_TRIGGER=0.5"
for part in ruler lbt mrcr gw; do run $part _t50 4096 $T0 MAGE_CLOCK_TRACE=1; done
run fc '' dense
run fc '' 4096
run fc _rs4 4096 MAGE_RESELECT=4
run fc _t50 4096 $T0
run fc _t50_cgate 4096 $T0 MAGE_ROWW=cgate
run fc _t50_cgate_k2048 4096 $T0 MAGE_ROWW=cgate MAGE_RESELECT_K=2048
run fc _t50_cgate_k2048 8192 $T0 MAGE_ROWW=cgate MAGE_RESELECT_K=2048
echo "done $(date -u)" >> $W/status_sc1xchain
