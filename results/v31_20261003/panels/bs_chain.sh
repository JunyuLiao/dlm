#!/bin/bash
# Panel bs (EXPLORATORY budget sweep; overlay ov_int, bench bench_ov_int.py), after confirmation stage A: the
# accuracy-cost frontier of the lean candidate and MAGE on a FRESH RULER pool (ruler_v34ofc, seed 7575; its qa questions
# coincide with every earlier pool's -- RULER's qa generator takes questions in index order).
#  (1) RULER v34ofc 32K / 64K / 128K (this host's shard): dense FULL, and {lean, MAGE} x {4096, 8192, 16384} tokens;
#  (2) MRCR 2-needle {lean, MAGE} x {8192, 16384} (stage A has dense and 4096);
#  (3) LongBench-v2 128K (lb_long_v31_128k, gl128's cells and shard; dense reference = gl128's dense run): per-step cost
#      of {lean, MAGE} x {8192, 16384}.
# Every arm waits for an empty GPU. env: W PY MODEL ROOT SHARD LBSHARD P WAIT_STATUS
set -u
cd $W
until grep -q '^done ' $W/$WAIT_STATUS 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
O=$W/ov_int
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
LEAN="MAGE_GRAN=qblock_max MAGE_STEP=1 MAGE_CARRY=1"
B="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD FIX_51994=1 MEM=0.90 OVERLAY=$O BENCH=$W/bench_ov_int.py TAG=bsruler
  DATASETS=ruler32k_v34ofc,ruler64k_v34ofc,ruler128k_v34ofc CELLS=$P/ruler_v34ofc/cells_ruler_v34ofc.json
  MAN_DIR=$P/ruler_v34ofc MAX_MODEL_LEN=136192"
env $B ARMS='dense:default' bash $W/v31_paired_host8.sh > $W/host_bs_dense.log 2>&1
sed -i "s/^done /done_dense /" $W/status_bsruler
for k in 4096 8192 16384; do
  env $B $FAST LABEL_SUFFIX=_plain ARMS="mage:PIECEWISE:$k" bash $W/v31_paired_host8.sh > $W/host_bs_plain_$k.log 2>&1
  sed -i "s/^done /done_plain_$k /" $W/status_bsruler
  env $B $FAST $LEAN LABEL_SUFFIX=_lean ARMS="mage:PIECEWISE:$k" bash $W/v31_paired_host8.sh > $W/host_bs_lean_$k.log 2>&1
  sed -i "s/^done /done_lean_$k /" $W/status_bsruler
done
# MRCR 2-needle (stage A's pool, cells and pin; dense and the 4096 arms come from stage A's famrcr records, same cells and
# settings): does a larger budget close the verbatim-retrieval gap?
M="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD FIX_51994=1 MEM=0.90 OVERLAY=$O BENCH=$W/bench_ov_int.py TAG=famrcr
  DATASETS=mrcr2_32k_ofc,mrcr2_64k_ofc,mrcr2_128k_ofc CELLS=$P/mrcr_ofc/cells_mrcr_ofc.json MAN_DIR=$P/mrcr_ofc MAX_MODEL_LEN=143360"
for k in 8192 16384; do
  env $M $FAST LABEL_SUFFIX=_plain ARMS="mage:PIECEWISE:$k" bash $W/v31_paired_host8.sh > $W/host_bsmrcr_plain_$k.log 2>&1
  sed -i "s/^done /done_plain_$k /" $W/status_famrcr
  env $M $FAST $LEAN LABEL_SUFFIX=_lean ARMS="mage:PIECEWISE:$k" bash $W/v31_paired_host8.sh > $W/host_bsmrcr_lean_$k.log 2>&1
  sed -i "s/^done /done_lean_$k /" $W/status_famrcr
done
L="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$LBSHARD FIX_51994=1 MEM=0.90 OVERLAY=$W/ov_gs BENCH=$W/bench_ov_gs.py TAG=gl128
  DATASETS=longbench_v2_128k CELLS=$W/cells_lb128k.json MAN_DIR=$W/manifests_lb128k"
for k in 8192 16384; do
  env $L $FAST LABEL_SUFFIX=_plain ARMS="mage:PIECEWISE:$k" bash $W/v31_paired_host8.sh > $W/host_bs128_plain_$k.log 2>&1
  sed -i "s/^done /done_plain_$k /" $W/status_gl128
  env $L $FAST $LEAN LABEL_SUFFIX=_lean ARMS="mage:PIECEWISE:$k" bash $W/v31_paired_host8.sh > $W/host_bs128_lean_$k.log 2>&1
  sed -i "s/^done /done_lean_$k /" $W/status_gl128
done
echo "done $(date -u)" >> $W/status_bschain
