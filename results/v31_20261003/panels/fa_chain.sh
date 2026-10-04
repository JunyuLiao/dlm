#!/bin/bash
# Pre-registered confirmation, STAGE A (registered in docs/V31_BASELINE_ROOT_CAUSE_20261003.md on branch
# research/v31-integration-20261004 before any stage-A result). Overlay ov_int + bench bench_ov_int.py (integration
# branch: official binding + pinned MAX_MODEL_LEN in every record). Arms, on every cell of this host's shard:
#   A0 dense:default  (vLLM FULL + PR #51994 backport: the official serving path)
#   A1 mage 4096      (MAGE as published: step-0 observation, mean mass per KV head, 64 tiles per unit, held)
#   A2 lean 4096      (ours: step-1 observation, per-(query head, 128-row block) worst-row max share, 64 tiles per unit,
#                      held, first call of each canvas on the carried selection)
#   A3 lean 2048      (ours, 32 tiles per unit)
# Datasets (official pools, seed 1): RULER v33ofc 32K/64K/128K, LongBench-v2 0shot (thinking off, 128 tokens),
# MRCR 2-needle ofc, GraphWalks b32k (amended 13:50 UTC: 58% of the 90k bin capped at 16K in the dense pilot), HumanEval (thinking on, 8192). Sparse arms: PIECEWISE, the same execution flags,
# alias splits S=2. Every arm waits for an empty GPU. env: W PY MODEL ROOT SHARD (k/2) P (pool dir) WAIT_STATUS
set -u
cd $W
until grep -q '^done ' $W/$WAIT_STATUS 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
C=$W/cache; O=$W/ov_int
( cd $ROOT/deploy/v27_r17_6064109 && PYTHONPATH=src:. PYTHONNOUSERSITE=1 V27_ADAPTER_DIR=$O TRITON_CACHE_DIR=$C/triton \
  TVM_FFI_CACHE_DIR=$C/tvm CUDA_CACHE_PATH=$C/cuda CUTE_DSL_CACHE_DIR=$C/cute FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$C/facute \
  TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg timeout 2400 $PY $O/v31_run_tests.py $(ls $O/test_*.py) ) > $W/fa_tests.log 2>&1
rc=$?
if [ $rc != 0 ] || grep -q '^FAIL' $W/fa_tests.log; then
  echo "tests FAILED rc=$rc $(date -u)" >> $W/status_fachain; echo "done $(date -u)" >> $W/status_fachain; exit 1
fi
echo "tests passed $(grep -c '^PASS' $W/fa_tests.log) $(date -u)" >> $W/status_fachain
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
LEAN="MAGE_GRAN=qblock_max MAGE_STEP=1 MAGE_CARRY=1"
run() {  # TAG DATASETS CELLS MAN_DIR MAX_MODEL_LEN
  local B="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD FIX_51994=1 MEM=0.90 OVERLAY=$O BENCH=$W/bench_ov_int.py
    TAG=$1 DATASETS=$2 CELLS=$3 MAN_DIR=$4 MAX_MODEL_LEN=$5"
  env $B ARMS='dense:default' bash $W/v31_paired_host8.sh > $W/host_$1_a0.log 2>&1
  sed -i "s/^done /done_a0 /" $W/status_$1
  env $B $FAST LABEL_SUFFIX=_plain ARMS='mage:PIECEWISE:4096' bash $W/v31_paired_host8.sh > $W/host_$1_a1.log 2>&1
  sed -i "s/^done /done_a1 /" $W/status_$1
  env $B $FAST $LEAN LABEL_SUFFIX=_lean ARMS='mage:PIECEWISE:4096' bash $W/v31_paired_host8.sh > $W/host_$1_a2.log 2>&1
  sed -i "s/^done /done_a2 /" $W/status_$1
  env $B $FAST $LEAN LABEL_SUFFIX=_lean ARMS='mage:PIECEWISE:2048' bash $W/v31_paired_host8.sh > $W/host_$1_a3.log 2>&1
  sed -i "s/^done /done_a3 /" $W/status_$1
  echo "$1 complete $(date -u)" >> $W/status_fachain
}
run faruler ruler32k_v33ofc,ruler64k_v33ofc,ruler128k_v33ofc $P/ruler_v33ofc/cells_ruler_v33ofc.json $P/ruler_v33ofc 136192
run falb longbench_v2_0shot $P/longbench_v2_ofc/cells_longbench_v2_0shot.json $P/longbench_v2_ofc 124928
run famrcr mrcr2_32k_ofc,mrcr2_64k_ofc,mrcr2_128k_ofc $P/mrcr_ofc/cells_mrcr_ofc.json $P/mrcr_ofc 143360
run fagw graphwalks_22k_b32k,graphwalks_45k_b32k,graphwalks_90k_b32k $P/graphwalks_b32k/cells_graphwalks_b32k.json $P/graphwalks_b32k 126976
run fahe humaneval $W/cells_he164.json $W/manifests_ae 13312
echo "done $(date -u)" >> $W/status_fachain
