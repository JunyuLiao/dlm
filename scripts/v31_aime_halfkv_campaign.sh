#!/usr/bin/env bash
# Run the V31 AIME26 half-context-budget reproduction on an authorized H100 host.
# All private paths are supplied through the environment; nothing is hard-coded.
set -euo pipefail

: "${W:?set W to the private run directory on the GPU host}"
: "${PY:?set PY to the pinned vLLM Python interpreter}"
: "${MODEL:?set MODEL to the pinned DiffusionGemma model directory}"
: "${ROOT:?set ROOT to the private V31 deployment root}"
: "${OVERLAY:?set OVERLAY to the pinned V31 adapter overlay}"

BENCH=${BENCH:-$W/scripts/v31_vllm_paired_bench.py}
MAN_DIR=${MAN_DIR:-$W/manifests_ae}
CELLS=${CELLS:-$W/cells/cells_aime_halfkv_20261005.json}
OUT_ROOT=${OUT_ROOT:-$W/aime_halfkv_20261005}
SEEDS=${SEEDS:-42,43,44}
SEED_BASE=${SEED_BASE:-31}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-9216}
MEM=${MEM:-0.80}
BLOCK=${BLOCK:-32}
CHUNK=${CHUNK:-4096}

mkdir -p "$OUT_ROOT/public" "$OUT_ROOT/private" "$OUT_ROOT/cache"

if [[ ! -f "$CELLS" ]]; then
  : "${HE_CELLS:?set HE_CELLS when generating the AIME cell file}"
  "$PY" "$W/scripts/v31_screen_suite.py" --short "$MAN_DIR" "$HE_CELLS" "$OUT_ROOT/suite" \
    --seeds "$SEEDS" --sample-seed 20261005
  cp "$OUT_ROOT/suite/cells_sc_aime.json" "$CELLS"
fi

export VLLM_CACHE_ROOT="$OUT_ROOT/cache/vllm" XDG_CACHE_HOME="$OUT_ROOT/cache/xdg" \
  TMPDIR="$OUT_ROOT/cache/tmp" TORCHINDUCTOR_CACHE_DIR="$OUT_ROOT/cache/inductor" \
  TRITON_CACHE_DIR="$OUT_ROOT/cache/triton" CUDA_CACHE_PATH="$OUT_ROOT/cache/cuda" \
  HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONNOUSERSITE=1 \
  PYTHONPATH="$W/src:$W" V27_ADAPTER_DIR="$OVERLAY" SEED_BASE MEM BLOCK CHUNK \
  FIX_51994=1 FA4_LOCAL_FIX=1 DATASETS=aime26 CELLS MAN_DIR MAX_MODEL_LEN
mkdir -p "$TMPDIR"

run_arm() {
  local arm=$1 cg=$2 label=$3
  local out="$OUT_ROOT/public/aime_${label}.jsonl"
  local priv="$OUT_ROOT/private/aime_${label}.private.jsonl"
  echo "starting $label $(date -u +%FT%TZ)" | tee -a "$OUT_ROOT/status.log"
  if [[ "$arm" == mage ]]; then
    env LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 MAGE_SELECT=fa4 KV_COPY=triton MERGE=triton \
      MAGE_GRAN=qblock_max MAGE_STEP=1 MAGE_CARRY=1 \
      MAGE_RESELECT_TRIGGER=0.15 MAGE_TRIGGER_SIGNAL=settle MAGE_STICKY=1.386 \
      MAGE_K=4096 LOCAL_KV_BUDGET=512 \
      timeout 21600 "$PY" "$BENCH" "$MODEL" "$MAN_DIR" "$CELLS" "$out" "$priv" "$arm" "$cg"
  else
    timeout 21600 "$PY" "$BENCH" "$MODEL" "$MAN_DIR" "$CELLS" "$out" "$priv" "$arm" "$cg"
  fi
  echo "finished $label $(date -u +%FT%TZ)" | tee -a "$OUT_ROOT/status.log"
}

run_arm dense default dense
run_arm mage PIECEWISE mage4096_settle15_sticky

echo "complete $(date -u +%FT%TZ)" | tee -a "$OUT_ROOT/status.log"
