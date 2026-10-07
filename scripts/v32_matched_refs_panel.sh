#!/usr/bin/env bash
# Supplementary matched-reference panel for the v32 value-aware study.
#
# Separate file (not an edit of scripts/v32_aime26_panel.sh) because that script was
# already executing. This one adds the two matched references the first panel driver
# did not express correctly:
#   dense_piecewise : dense arm with cudagraph_mode=PIECEWISE, so the matched dense
#                     reference runs on the same capture mode as the sparse arms.
#   allkept_fa4     : the adapter's real all-kept sparse-consumer path (v27_fa4.dense
#                     over the same routed consumer), NOT "mage with an infinite budget".
# Both are required controls for a sparse-vs-dense comparison, because the dense path and
# the sparse path use different consumers.
#
# usage: SUITE=aime26|longbench bash scripts/v32_matched_refs_panel.sh
set -u
W=${WORK:-/home/exouser/dyh/ljy_value_20261007}
ROOT=/home/exouser/ljy/dlm
SUITE=${SUITE:-aime26}
STAMP=$(date -u +%Y%m%d_%H%M%S)
OUT=$W/run_${SUITE}_refs_${STAMP}
mkdir -p "$OUT"

if [ "$SUITE" = aime26 ]; then
  export MAN_DIR=$W/pools/aime26 CELLS=$W/cells_aime.json DATASETS=aime26 MEM=${MEM:-0.80}
else
  export MAN_DIR=$W/pools/longbench_v2_ofc CELLS=${CELLS:-$W/cells_lb2think_dev15.json} \
         DATASETS=longbench_v2_0shot_think MEM=${MEM:-0.80} MAX_MODEL_LEN=136401
fi
export WORK=$W

run () {  # run LABEL ARMSPEC
  local label=$1; shift
  echo "=== $label $(date -u) ===" >> "$OUT/driver.log"
  ( env "$@" ARM="$ARM" TAG="$label" bash "$ROOT/scripts/v32_value_arm.sh" ) \
      >> "$OUT/driver.log" 2>&1
  echo "=== $label rc=$? $(date -u) ===" >> "$OUT/driver.log"
}

for spec in "dense_piecewise|dense:PIECEWISE:dense_piecewise" \
            "allkept_fa4|allkept:PIECEWISE:allkept_fa4"; do
  label=${spec%%|*}; armspec=${spec#*|}
  run "$label" "$armspec"
done
echo "done $SUITE refs $STAMP" >> "$OUT/driver.log"