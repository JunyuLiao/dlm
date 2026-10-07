#!/bin/bash
# Value-aware cross-step reuse: one-arm launch wrapper. One worker at a time on the H100.
# env in: WORK TAG ARM [DATASETS] [CELLS] [MAN_DIR] [EXTRA_ENV...]
#   ARM is a comma list of  v27 arm : cudagraph : label
set -eu
W=${WORK:?}; TAG=${TAG:?}; ARM=${ARM:?}
PY=${PY:-/home/exouser/dyh/v31_env_20261005/bin/python}
ROOT=${ROOT:-/home/exouser/ljy/dlm}
MODEL=${MODEL:-/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b}
C=$W/cache
D=$W/run_$TAG
mkdir -p "$D/public" "$D/private" "$C"
cd "$ROOT"

IFS=',' read -ra ARMS <<< "$ARM"
for a in "${ARMS[@]}"; do
  v=${a%%:*}; rest=${a#*:}; cg=${rest%%:*}; label=${rest#*:}
  echo "starting $label ($v/$cg) $(date -u)" >> "$D/status.log"
  timeout 100000 env \
    PYTHONPATH="$ROOT/src:$ROOT" PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/home/exouser/.triton/cache} TVM_FFI_CACHE_DIR=$C/tvm CUDA_CACHE_PATH=$C/cuda \
    CUTE_DSL_CACHE_DIR=$C/cute FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$C/facute \
    TRITON_HOME=${TRITON_HOME:-/home/exouser/.triton} XDG_CACHE_HOME=$C/xdg TMPDIR=$C/tmp HOME=$W/home \
    FIX_51994=1 FA4_LOCAL_FIX=1 \
    MEM=${MEM:-0.80} MAX_MODEL_LEN=${MAX_MODEL_LEN:-} SEED_BASE=${SEED_BASE:-31} \
    REPEATS=${REPEATS:-1} LIMIT=${LIMIT:-} SHARD=${SHARD:-} \
    DATASETS=${DATASETS:-} CELLS=${CELLS:-$W/cells.json} MAN_DIR=${MAN_DIR:-$W/pool} \
    LOGIT_STATS=${LOGIT_STATS:-legacy} DP_BUILD=${DP_BUILD:-legacy} OBSERVE=${OBSERVE:-triton} \
    MAGE_SELECT=${MAGE_SELECT:-fa4} MAGE_K=${MAGE_K:-} MAGE_GRAN=${MAGE_GRAN:-qblock_max} \
    MAGE_STEP=${MAGE_STEP:-1} MAGE_CARRY=${MAGE_CARRY:-1} MAGE_STICKY=${MAGE_STICKY:-1.386} \
    MAGE_RESELECT_TRIGGER=${MAGE_RESELECT_TRIGGER:-} MAGE_TRIGGER_SIGNAL=${MAGE_TRIGGER_SIGNAL:-accept} \
    KV_COPY=${KV_COPY:-triton} MERGE=${MERGE:-triton} \
    VALUE_SELECTOR=${VALUE_SELECTOR:-} VALUE_THRESH=${VALUE_THRESH:-1.0} \
    VALUE_DROP=${VALUE_DROP:-0.0} VALUE_SHORTLIST=${VALUE_SHORTLIST:-0} \
    VALUE_EXACT_MAX=${VALUE_EXACT_MAX:-256} VALUE_SCAN=${VALUE_SCAN:-triton} \
    AUDIT=${AUDIT:-1} \
    "$PY" "$ROOT/scripts/v31_vllm_paired_bench.py" "$MODEL" "$MAN_DIR" "$CELLS" \
      "$D/public/${TAG}_${label}.jsonl" "$D/private/${TAG}_${label}.private.jsonl" "$v" "$cg" \
      >> "$D/log_${TAG}_${label}.out" 2>&1
  echo "finished $label $(date -u)" >> "$D/status.log"
done
echo "done $TAG $(date -u)" >> "$D/status.log"