#!/bin/bash
# Panel gs (overlay ov_gs = branch research/v31-group-shared-select-20261004; EXPLORATORY), after panel vk:
#  (1) the overlay's GPU tests (v31 suite + test_v31_group_select.py); abort unless all pass;
#  (2) RULER v31 pool, the panel x cells (mpk shard 1/3, dlm2 0/3), keep 12% of prefix tiles per unit, selection at
#      step 1 (the x ladder's setting):
#        control  qblock_max            -- must reproduce x's records token for token (same code path, host, seeds)
#        kvblock_max, kvhead_max        -- group-shared units (same per-head budget = same work as qblock_max)
#        + MAGE_CARRY=1 on qblock_max / kvblock_max / kvhead (MAGE's mean) -- canvas call 0 on the previous selection
#  (3) per-call GPU cost (PROFILE_MODE=events, 3 LongBench-v2 64K confirmation cells): qblock_max vs kvblock_max,
#      both + carry (the per-head-list L2 hypothesis at the selection level);
#  (4) end-to-end on the LongBench-v2 64K + 96K confirmation cells (pf's cells and shards): dense FULL reference again,
#      qblock_max + carry, kvblock_max + carry.
# Every arm waits for an empty GPU (v31_paired_host8.sh), so a classmate's job is never preempted between steps.
# env: W PY MODEL ROOT RSHARD LBSHARD
set -u
cd $W
until grep -q '^done ' $W/status_vk 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
C=$W/cache; O=$W/ov_gs
( cd $ROOT/deploy/v27_r17_6064109 && PYTHONPATH=src:. PYTHONNOUSERSITE=1 V27_ADAPTER_DIR=$O TRITON_CACHE_DIR=$C/triton \
  TVM_FFI_CACHE_DIR=$C/tvm CUDA_CACHE_PATH=$C/cuda CUTE_DSL_CACHE_DIR=$C/cute FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$C/facute \
  TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg timeout 2400 $PY $O/v31_run_tests.py $(ls $O/test_*.py) ) > $W/gs_tests.log 2>&1
rc=$?
if [ $rc != 0 ] || grep -q '^FAIL' $W/gs_tests.log; then
  echo "tests FAILED rc=$rc $(date -u)" >> $W/status_gs; echo "done $(date -u)" >> $W/status_gs; exit 1
fi
echo "tests passed $(grep -c '^PASS' $W/gs_tests.log) $(date -u)" >> $W/status_gs
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
RUL="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$RSHARD DATASETS=ruler32k,ruler64k MEM=0.90 TAG=gs FIX_51994=1
  CELLS=$W/cells_ruler31.json MAN_DIR=$W/manifests_ruler31 BENCH=$W/bench_ov_gs.py OVERLAY=$O"
rung() {  # GRAN CARRY
  local sfx=_frac12_$1_step1; [ "$2" = 1 ] && sfx=${sfx}_carry
  env $RUL $FAST MAGE_FRAC=0.12 MAGE_GRAN=$1 MAGE_STEP=1 MAGE_CARRY=$2 LABEL_SUFFIX=$sfx ARMS='mage:PIECEWISE:0' \
    bash $W/v31_paired_host8.sh > $W/host_gs_$1_$2.log 2>&1
  sed -i "s/^done /done_$1_$2 /" $W/status_gs
}
rung qblock_max 0
rung kvblock_max 0
rung kvhead_max 0
rung qblock_max 1
rung kvblock_max 1
rung kvhead 1
LB="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$LBSHARD FIX_51994=1 MEM=0.90 CELLS=$W/cells_confirm.json OVERLAY=$O"
for g in qblock_max kvblock_max; do
  env $LB $FAST TAG=gsp DATASETS=longbench_v2_64k LIMIT=3 BENCH=$O/v31_step_profile.py PROFILE_MODE=events \
    PROFILE_OUT=$W/public/gsp_profile_mage_frac12_${g}_step1_carry.jsonl MAGE_FRAC=0.12 MAGE_GRAN=$g MAGE_STEP=1 \
    MAGE_CARRY=1 LABEL_SUFFIX=_frac12_${g}_step1_carry ARMS='mage:PIECEWISE:0' \
    bash $W/v31_paired_host8.sh > $W/host_gsp_$g.log 2>&1
  sed -i "s/^done /done_$g /" $W/status_gsp
done
env $LB TAG=gsl DATASETS=longbench_v2_64k,longbench_v2_96k BENCH=$W/bench_ov_gs.py ARMS='dense:default' \
  bash $W/v31_paired_host8.sh > $W/host_gsl_dense.log 2>&1
sed -i "s/^done /done_dense /" $W/status_gsl
for g in qblock_max kvblock_max; do
  env $LB $FAST TAG=gsl DATASETS=longbench_v2_64k,longbench_v2_96k BENCH=$W/bench_ov_gs.py MAGE_FRAC=0.12 MAGE_GRAN=$g \
    MAGE_STEP=1 MAGE_CARRY=1 LABEL_SUFFIX=_frac12_${g}_step1_carry ARMS='mage:PIECEWISE:0' \
    bash $W/v31_paired_host8.sh > $W/host_gsl_$g.log 2>&1
  sed -i "s/^done /done_$g /" $W/status_gsl
done
echo "done $(date -u)" >> $W/status_gs
