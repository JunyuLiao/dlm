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

# Per-suite substrate pins, frozen once and reused by every arm so the panel is matched.
# AIME26  : the frozen v31 AIME26 substrate (per-cell records in
#           results/v31_20261006_aime_global_local/attempt001/full/ record chunk=4096, block=64).
# LongBench: no frozen v31 LongBench pin exists in this repository, so this study freezes the
#           bench defaults, which are also what the earlier LongBench probe used
#           (its prefill_steps=8 for a 120017-token prompt implies ceil(120017/16384)=8).
case "$DATASETS" in
  *aime26*)      V32_BLOCK=64  V32_CHUNK=4096  ;;
  *)             V32_BLOCK=32  V32_CHUNK=16384 ;;
esac

# The frozen v31 control settings. Defaults here, not at the call site, so that an arm which
# only sets VALUE_SELECTOR is the control plus that selector and nothing else.
: "${MAGE_K:=1728}"
: "${MAGE_SELECT:=fa4}"
: "${MAGE_GRAN:=qblock_max}"
: "${MAGE_STEP:=1}"
: "${MAGE_CARRY:=1}"
: "${MAGE_STICKY:=1.386}"
: "${MAGE_RESELECT_TRIGGER:=0.15}"
: "${MAGE_TRIGGER_SIGNAL:=settle}"
: "${MAGE_SINK:=0}"
: "${MAGE_RECENT:=0}"
: "${KV_COPY:=triton}"
: "${MERGE:=triton}"
: "${OBSERVE:=triton}"
: "${LOGIT_STATS:=legacy}"
: "${DP_BUILD:=legacy}"
: "${BLOCK:=$V32_BLOCK}"
: "${CHUNK:=$V32_CHUNK}"
# Memory footprint. On this shared H100 another tenant can claim tens of GiB inside a 30 s
# window, so ask for the least that still works: weights are 50.46 GiB and the observed KV rate
# is ~0.03234 MiB/token (MEM=0.90 gave 673,105 tokens in 21.27 GiB), so 141,312 tokens needs
# ~4.5 GiB of KV and MEM=0.75 leaves ~8.9 GiB. The KV assertion below is what actually decides.
: "${MEM:=0.75}"
: "${SEED_BASE:=31}"
: "${REPEATS:=1}"
: "${AUDIT:=1}"
: "${VALUE_THRESH:=1.0}"
: "${VALUE_DROP:=0.0}"
: "${VALUE_SHORTLIST:=0}"
: "${VALUE_EXACT_MAX:=256}"
: "${VALUE_SCAN:=triton}"

# Build the child environment from a table of NAME -> value, skipping anything unset or empty.
# scripts/v31_vllm_paired_bench.py parses several knobs with int(os.environ.get(NAME, D)) and
# no empty guard, so exporting NAME='' crashes the run instead of falling back to D. Dropping
# empty entries here keeps "unset" and "empty" identical for every knob.
ENVS=()
add () { if [ -n "${2:-}" ]; then ENVS+=("$1=$2"); fi; return 0; }
add FIX_51994 1
add FA4_LOCAL_FIX 1
add MEM "$MEM"
add MAX_MODEL_LEN "$MAX_MODEL_LEN"
add SEED_BASE "$SEED_BASE"
add REPEATS "$REPEATS"
add LIMIT "${LIMIT:-}"
add SHARD "${SHARD:-}"
add BLOCK "$BLOCK"
add CHUNK "$CHUNK"
add DATASETS "${DATASETS:-}"
add CELLS "${CELLS:-$W/cells.json}"
add MAN_DIR "${MAN_DIR:-$W/pool}"
add LOGIT_STATS "$LOGIT_STATS"
add DP_BUILD "$DP_BUILD"
add OBSERVE "$OBSERVE"
add MAGE_SELECT "$MAGE_SELECT"
add MAGE_K "$MAGE_K"
add MAGE_GRAN "$MAGE_GRAN"
add MAGE_STEP "$MAGE_STEP"
add MAGE_CARRY "$MAGE_CARRY"
add MAGE_STICKY "$MAGE_STICKY"
add MAGE_RESELECT_TRIGGER "$MAGE_RESELECT_TRIGGER"
add MAGE_TRIGGER_SIGNAL "$MAGE_TRIGGER_SIGNAL"
add MAGE_SINK "$MAGE_SINK"
add MAGE_RECENT "$MAGE_RECENT"
add KV_COPY "$KV_COPY"
add MERGE "$MERGE"
add VALUE_SELECTOR "${VALUE_SELECTOR:-}"
add VALUE_THRESH "$VALUE_THRESH"
add VALUE_DROP "$VALUE_DROP"
add VALUE_SHORTLIST "$VALUE_SHORTLIST"
add VALUE_EXACT_MAX "$VALUE_EXACT_MAX"
add VALUE_SCAN "$VALUE_SCAN"
add AUDIT "$AUDIT"

IFS=',' read -ra ARMS <<< "$ARM"
for a in "${ARMS[@]}"; do
  v=${a%%:*}; rest=${a#*:}; cg=${rest%%:*}; label=${rest#*:}
  echo "starting $label ($v/$cg) $(date -u)" >> "$D/status.log"
  # Retry while the device is too small. Another tenant's process is never touched; we only
  # wait for it to end. Retries go to a new attempt suffix so no run directory is reused.
  for attempt in $(seq 1 "${TRY:-8}"); do
    A_TAG=$TAG
    if [ "$attempt" -gt 1 ]; then A_TAG="${TAG}_retry${attempt}"; mkdir -p "$W/run_$A_TAG/public" "$W/run_$A_TAG/private"; fi
  # `if` protects the failing exit status from `set -e` so it can be inspected and retried.
  if timeout 100000 env \
    PYTHONPATH="$ROOT/src:$ROOT" PYTHONNOUSERSITE=1 \
    TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/exouser/.triton/cache}" \
    CUTE_DSL_CACHE_DIR="${CUTE_DSL_CACHE_DIR:-/home/exouser/.cache/cute}" \
    FLASH_ATTENTION_CUTE_DSL_CACHE_DIR="${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:-/home/exouser/.cache/facute}" \
    TMPDIR=$C/tmp HOME="${HOME:-/home/exouser}" \
    XDG_CACHE_HOME="${XDG_CACHE_HOME:-/home/exouser/.cache}" \
    FLASHINFER_WORKSPACE_BASE_PATH=$C/flashinfer \
    VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-$C/flashinfer_autotune} \
    "${ENVS[@]}" \
    "$PY" "$ROOT/scripts/v31_vllm_paired_bench.py" "$MODEL" "$MAN_DIR" "$CELLS" \
      "$W/run_$A_TAG/public/${TAG}_${label}.jsonl" "$W/run_$A_TAG/private/${TAG}_${label}.private.jsonl" "$v" "$cg" \
      >> "$D/log_${TAG}_${label}.out" 2>&1
  then rc=0; else rc=$?; fi
  if [ "$rc" -ne 0 ] && tail -40 "$D/log_${TAG}_${label}.out" | grep -q "Free memory on device"; then
    echo "contended $label attempt=$attempt rc=$rc, waiting ${WAIT:-120}s $(date -u)" >> "$D/status.log"
    sleep "${WAIT:-120}"; continue
  fi
  if [ "$rc" -ne 0 ]; then
    echo "FAILED $label attempt=$attempt rc=$rc $(date -u)" >> "$D/status.log"
  else
    # The engine must hold one full prompt plus the canvas. Verify the KV pool actually covers
    # max_model_len instead of trusting gpu_memory_utilization.
    kv=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" "$D/log_${TAG}_${label}.out" | tail -1 | grep -oE "[0-9,]+")
    if [ -n "${kv:-}" ] && [ -n "$MAX_MODEL_LEN" ] && [ "${kv//,/}" -lt "$MAX_MODEL_LEN" ]; then
      echo "KV_TOO_SMALL $label cache=${kv} < MAX_MODEL_LEN=$MAX_MODEL_LEN; raise MEM" >> "$D/status.log"
      rc=1
    else
      echo "finished $label attempt=$attempt kv_tokens=${kv:-?} $(date -u)" >> "$D/status.log"
    fi
  fi
  break
  done
done
echo "done $TAG $(date -u)" >> "$D/status.log"