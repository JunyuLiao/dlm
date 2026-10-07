#!/usr/bin/env bash
# Target panel for the v32 value-aware selectors, on the pre-registered 120-cell LongBench-v2
# subsample and the full AIME26 panel. Runs only after scripts/v32_lbdev_calibration.sh has fixed
# VALUE_THRESH; that value is passed in, never guessed here.
#
# The subsample was drawn before any generation: cells_lb2think_target120.json is 120 of the 488
# target cells, four equal-count prompt-length strata, seed 20261007. It reproduces the full
# target's prompt-length profile (median 109,820 vs 107,678 tokens; max 120,016 vs 120,017). It is
# a SUBSAMPLE, and every table must say so; the 488-cell target is not covered by this panel.
#
# usage: SUITE=aime26|longbench TH=<threshold> bash scripts/v32_target_panel.sh
set -u
W=${WORK:-/home/exouser/dyh/ljy_value_20261007}
ROOT=/home/exouser/ljy/dlm
SUITE=${SUITE:-aime26}
TH=${TH:?set TH to the threshold frozen by the dev calibration}
STAMP=$(date -u +%Y%m%d_%H%M%S)
OUT=$W/run_${SUITE}_target_${STAMP}
mkdir -p "$OUT"
TRY=${TRY:-4000} WAIT=${WAIT:-180}

if [ "$SUITE" = aime26 ]; then
  export MAN_DIR=$W/pools/aime26 CELLS=$W/cells_aime.json DATASETS=aime26 \
         MEM=${MEM:-0.80} MAX_MODEL_LEN=9216
else
  export MAN_DIR=$W/pools/longbench_v2_ofc CELLS=$W/cells_lb2think_target120.json \
         DATASETS=longbench_v2_0shot_think MEM=${MEM:-0.80} MAX_MODEL_LEN=136401
fi
export WORK=$W
VAL="VALUE_SELECTOR=$SEL VALUE_THRESH=$TH MAGE_TRIGGER_SIGNAL=settle MAGE_RESELECT_TRIGGER=0.15"

run () {
  local label=$1 armspec=$2; shift 2
  echo "=== $label $(date -u) ===" >> "$OUT/driver.log"
  ( env "$@" ARM="$armspec" TAG="$label" bash "$ROOT/scripts/v32_value_arm.sh" ) \
      >> "$OUT/driver.log" 2>&1
  rc=$?
  echo "=== $label rc=$rc $(date -u) ===" >> "$OUT/driver.log"
}

# matched references, then the control, then the selectors
run dense_full_fix51994 'dense:default:dense_full_fix51994'
run dense_piecewise       'dense:PIECEWISE:dense_piecewise'
run allkept_fa4           'allkept:PIECEWISE:allkept_fa4'
run current_v31_control   'mage:PIECEWISE:current_v31_control'
run value_v1              'mage:PIECEWISE:value_v1'                 VALUE_SELECTOR=v1  $VAL
run value_v2              'mage:PIECEWISE:value_v2'                 VALUE_SELECTOR=v2  $VAL
run value_v3a             'mage:PIECEWISE:value_v3a'                VALUE_SELECTOR=v3a $VAL
run value_v3b             'mage:PIECEWISE:value_v3b'                VALUE_SELECTOR=v3b $VAL
run value_v3b_drop025     'mage:PIECEWISE:value_v3b_drop025'        VALUE_SELECTOR=v3b_drop VALUE_DROP=0.25 $VAL
run value_v3b_shortlist16 'mage:PIECEWISE:value_v3b_shortlist16'    VALUE_SELECTOR=v3b_shortlist VALUE_SHORTLIST=16 $VAL
echo "done ${SUITE} target $STAMP" >> "$OUT/driver.log"