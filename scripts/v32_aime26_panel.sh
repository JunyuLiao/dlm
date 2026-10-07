#!/bin/bash
# Value-aware cross-step reuse: AIME26 frozen panel (30 problems x seeds 42/43/44 = 90 cells/arm).
# The pool, cells, seeds, budget, thinking setting and scorer are the ones the frozen 2026-10-06
# control panel used (manifest sha256 85ebd2dd...), so this panel is directly comparable to it.
# env: WORK PY ROOT
set -u
W=/home/exouser/dyh/ljy_value_20261007
cd /home/exouser/ljy/dlm
export WORK=$W MAN_DIR=$W/pools/aime26 CELLS=$W/cells_aime.json DATASETS=aime26 MEM=${MEM:-0.88} MAX_MODEL_LEN=9216 \
       MAGE_K=1728 MAGE_RESELECT_TRIGGER=0.15 MAGE_TRIGGER_SIGNAL=settle VALUE_SCAN=triton
mkdir -p $W/run_aime26

run () {  # TAG ARM [env...]
  local tag=$1; shift; local arm=$1; shift
  echo "=== $tag $(date -u)" >> $W/run_aime26/driver.log
  env "$@" TAG=$tag ARM="$arm" bash scripts/v32_value_arm.sh >> $W/run_aime26/driver.log 2>&1
  echo "=== $tag rc=$? $(date -u)" >> $W/run_aime26/driver.log
}

run aime_dense          'dense:default:dense_full_fix51994'
run aime_control        'mage:PIECEWISE:current_v31_control'
run aime_allkept        'mage:PIECEWISE:allkept_fa4' VALUE_SELECTOR= MAGE_K=1000000
run aime_v1             'mage:PIECEWISE:value_v1_online_discard_mass'   VALUE_SELECTOR=v1 VALUE_THRESH=0.01
run aime_v2             'mage:PIECEWISE:value_v2_online_preserve_mass'  VALUE_SELECTOR=v2 VALUE_THRESH=0.01
run aime_v3a            'mage:PIECEWISE:value_v3a_singleton_delete'     VALUE_SELECTOR=v3a
run aime_v3b            'mage:PIECEWISE:value_v3b_greedy_exact'        VALUE_SELECTOR=v3b
run aime_v3b_drop       'mage:PIECEWISE:value_v3b_drop_r025'           VALUE_SELECTOR=v3b_drop VALUE_DROP=0.25
run aime_v3b_short      'mage:PIECEWISE:value_v3b_shortlist64'         VALUE_SELECTOR=v3b_shortlist VALUE_SHORTLIST=64
echo "=== ALL DONE $(date -u)" >> $W/run_aime26/driver.log