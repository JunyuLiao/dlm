#!/usr/bin/env bash
# Threshold calibration for the v32 value-aware selectors on the RESERVED dev subset.
#
# Run BEFORE any target generation. Uses only results/cells_lb2think_dev15.json (15 held-out
# LongBench-v2 0shot_think cells, disjoint from the 488-cell target), plus the matched dense and
# current-v31 control references on the same cells. Picks the V1/V2 objective threshold, then that
# value is FROZEN in scripts/v32_aime26_panel.sh / the target panel. Transductive calibration and
# target scoring are reported separately.
#
# Pins the frozen v31 substrate: FA4 local fix, native mask fix 51994, LOCAL native dense,
# GLOBAL layers 5/11/17/23/29, MAGE_K=1728, settle/0.15, sticky 1.386, and the per-suite block/
# chunk pins frozen in scripts/v32_value_arm.sh (LongBench BLOCK=32 CHUNK=16384, matching the
# earlier LongBench probe; AIME26 BLOCK=64 CHUNK=4096, matching the frozen v31 AIME26 panel).
set -u
W=${WORK:-/home/exouser/dyh/ljy_value_20261007}
ROOT=/home/exouser/ljy/dlm
STAMP=$(date -u +%Y%m%d_%H%M%S)
OUT=$W/run_lbdev_calib_${STAMP}
mkdir -p "$OUT"

export WORK=$W MAN_DIR=$W/pools/longbench_v2_ofc \
       CELLS=${CELLS:-$W/cells_lb2think_dev15.json} DATASETS=longbench_v2_0shot_think \
       MEM=${MEM:-0.90} MAX_MODEL_LEN=141312 BLOCK=32 CHUNK=16384

run () {
  local label=$1 armspec=$2; shift 2
  echo "=== $label $(date -u) ===" >> "$OUT/driver.log"
  ( env "$@" ARM="$armspec" TAG="$label" bash "$ROOT/scripts/v32_value_arm.sh" ) \
      >> "$OUT/driver.log" 2>&1
  echo "=== $label rc=$? $(date -u) ===" >> "$OUT/driver.log"
}

# matched references on the dev cells
run dev_dense        'dense:default:dense_full_fix51994'
run dev_dense_piece  'dense:PIECEWISE:dense_piecewise'
run dev_allkept      'allkept:PIECEWISE:allkept_fa4'
run dev_control      'mage:PIECEWISE:current_v31_control'

# V1 / V2 objective-threshold sweep, plus the other selectors at the probe threshold
for t in 0.005 0.01 0.02 0.05; do
  run "dev_v1_t${t}" 'mage:PIECEWISE:value_v1' VALUE_SELECTOR=v1 VALUE_THRESH=$t MAGE_TRIGGER_SIGNAL=settle MAGE_RESELECT_TRIGGER=0.15
  run "dev_v2_t${t}" 'mage:PIECEWISE:value_v2' VALUE_SELECTOR=v2 VALUE_THRESH=$t MAGE_TRIGGER_SIGNAL=settle MAGE_RESELECT_TRIGGER=0.15
done
run dev_v3a          'mage:PIECEWISE:value_v3a' VALUE_SELECTOR=v3a VALUE_THRESH=0.01 MAGE_TRIGGER_SIGNAL=settle MAGE_RESELECT_TRIGGER=0.15
run dev_v3b          'mage:PIECEWISE:value_v3b' VALUE_SELECTOR=v3b VALUE_THRESH=0.01 MAGE_TRIGGER_SIGNAL=settle MAGE_RESELECT_TRIGGER=0.15
run dev_v3b_drop     'mage:PIECEWISE:value_v3b_drop025' VALUE_SELECTOR=v3b_drop VALUE_DROP=0.25 VALUE_THRESH=0.01 MAGE_TRIGGER_SIGNAL=settle MAGE_RESELECT_TRIGGER=0.15
run dev_v3b_short    'mage:PIECEWISE:value_v3b_shortlist16' VALUE_SELECTOR=v3b_shortlist VALUE_SHORTLIST=16 VALUE_THRESH=0.01 MAGE_TRIGGER_SIGNAL=settle MAGE_RESELECT_TRIGGER=0.15
echo "done lbdev calib $STAMP" >> "$OUT/driver.log"