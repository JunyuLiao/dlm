#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/home/exouser/ljy/dlm
ROOT=results/attention/routing/ruler16k
RULER_ROOT=/tmp/NVIDIA-RULER
PY=/home/exouser/miniconda3/envs/ljy_dlm/bin/python
DG_MODEL=/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b
DG_REV=f7f5b7f5fa82ffc52addd066915886d497f5517b
FAST_MODEL=/home/exouser/.cache/huggingface/hub/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974
FAST_REV=0661abf5f9f0ee338970d091052a26c8efa51974
LOG_DIR=${WORKDIR}/${ROOT}/logs
mkdir -p "${LOG_DIR}"
LOG=${LOG_DIR}/remaining_staged_$(date -u +%Y%m%dT%H%M%SZ).log
exec >>"${LOG}" 2>&1

cd "${WORKDIR}"
echo "log=${LOG}"
echo "started=$(date -u --iso-8601=seconds)"

wait_for_pid() {
  local pid=${1:-}
  if [[ -n "${pid}" ]]; then
    while ps -p "${pid}" >/dev/null 2>&1; do
      echo "waiting_for_active_pid=${pid} at=$(date -u --iso-8601=seconds)"
      sleep 60
    done
  fi
}

wait_for_h100_idle() {
  while true; do
    local apps
    if ! apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); then
      echo "h100_query_failed at=$(date -u --iso-8601=seconds)"
      sleep 60
      continue
    fi
    apps=$(printf '%s\n' "${apps}" | sed '/^$/d')
    if [[ -z "${apps}" ]]; then
      nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu --format=csv,noheader
      return 0
    fi
    echo "h100_busy_pids=${apps//$'\n'/,} at=$(date -u --iso-8601=seconds)"
    sleep 60
  done
}

condition_complete() {
  local dir=$1
  local condition=$2
  local expected=$3
  local predictions=${dir}/${condition}/predictions.jsonl
  local summary=${dir}/${condition}/summary.json
  [[ -f "${predictions}" && -f "${summary}" ]] || return 1
  local lines
  lines=$(wc -l <"${predictions}")
  [[ "${lines}" -ge "${expected}" ]]
}

conditions_complete() {
  local dir=$1
  local expected=$2
  shift 2
  local condition
  for condition in "$@"; do
    condition_complete "${dir}" "${condition}" "${expected}" || return 1
  done
}

run_fast_missing_logical() {
  local out=${ROOT}/fast_dllm_v2_decision_revised
  local conditions=(
    profiled_rho75 profiled_rho50 profiled_rho25
    gaussian_rho75_prefix_only gaussian_rho50_prefix_only gaussian_rho25_prefix_only
    profiled_rho75_prefix_only profiled_rho50_prefix_only profiled_rho25_prefix_only
    oracle_rho75_prefix_only oracle_rho50_prefix_only oracle_rho25_prefix_only
  )
  if conditions_complete "${out}" 50 "${conditions[@]}"; then
    echo "fast_missing_logical=already_complete"
    return 0
  fi
  wait_for_h100_idle
  echo "fast_missing_logical=running"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/tmp/threshold-transformers-4.53.1:src \
    "${PY}" -m experiments.diffusion_attention_threshold_modeling eval-ruler-routing \
    --adapter fast_dllm_v2 --model-path "${FAST_MODEL}" --revision "${FAST_REV}" \
    --manifest-path "${ROOT}/manifests/fast_dllm_v2/manifest.json" --ruler-root "${RULER_ROOT}" \
    --output-dir "${out}" --num-samples 50 \
    --threshold-model "${ROOT}/profiled_threshold_model.json" \
    --conditions "${conditions[@]}" \
    --device cuda --precision bfloat16 --temperature 0 --quiet
}

run_physical() {
  local adapter=$1
  local out=$2
  local manifest=$3
  local py_path=$4
  local model=$5
  local revision=$6
  if conditions_complete "${out}" 2 dense profiled_rho50 profiled_rho25; then
    echo "physical_${adapter}=already_complete"
    return 0
  fi
  wait_for_h100_idle
  echo "physical_${adapter}=running"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="${py_path}:src" \
    "${PY}" -m experiments.diffusion_attention_threshold_modeling eval-ruler-routing \
    --adapter "${adapter}" --model-path "${model}" --revision "${revision}" \
    --manifest-path "${manifest}" --ruler-root "${RULER_ROOT}" \
    --output-dir "${out}" --num-samples 2 \
    --threshold-model "${ROOT}/profiled_threshold_model.json" \
    --conditions dense profiled_rho50 profiled_rho25 \
    --routing-execution physical --performance-warmups 1 --performance-repeats 3 \
    --device cuda --precision bfloat16 --temperature 0 --quiet
}

run_blocksize() {
  local adapter=$1
  local q=$2
  local out=$3
  local manifest=$4
  local py_path=$5
  local model=$6
  local revision=$7
  if conditions_complete "${out}" 2 dense gaussian_rho50; then
    echo "blocksize_${adapter}_q${q}=already_complete"
    return 0
  fi
  wait_for_h100_idle
  echo "blocksize_${adapter}_q${q}=running"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="${py_path}:src" \
    "${PY}" -m experiments.diffusion_attention_threshold_modeling eval-ruler-routing \
    --adapter "${adapter}" --model-path "${model}" --revision "${revision}" \
    --manifest-path "${manifest}" --ruler-root "${RULER_ROOT}" \
    --output-dir "${out}" --num-samples 2 \
    --conditions dense gaussian_rho50 --q-block-size "${q}" --kv-block-size 64 \
    --device cuda --precision bfloat16 --temperature 0 --quiet
}

ACTIVE_PID=${1:-}
wait_for_pid "${ACTIVE_PID}"
run_fast_missing_logical
run_physical diffusion_gemma "${ROOT}/physical_diffusion_gemma" "${ROOT}/manifests/smoke_diffusion_gemma/manifest.json" /tmp/threshold-transformers-5.11 "${DG_MODEL}" "${DG_REV}"
run_physical fast_dllm_v2 "${ROOT}/physical_fast_dllm_v2" "${ROOT}/manifests/smoke_fast_dllm_v2/manifest.json" /tmp/threshold-transformers-4.53.1 "${FAST_MODEL}" "${FAST_REV}"
for q in 32 64 128; do
  run_blocksize diffusion_gemma "${q}" "${ROOT}/blocksize_diffusion_gemma_q${q}" "${ROOT}/manifests/smoke_diffusion_gemma/manifest.json" /tmp/threshold-transformers-5.11 "${DG_MODEL}" "${DG_REV}"
  run_blocksize fast_dllm_v2 "${q}" "${ROOT}/blocksize_fast_dllm_v2_q${q}" "${ROOT}/manifests/smoke_fast_dllm_v2/manifest.json" /tmp/threshold-transformers-4.53.1 "${FAST_MODEL}" "${FAST_REV}"
done

PYTHONPATH=src "${PY}" -m experiments.diffusion_attention_threshold_modeling report-ruler --output-dir "${ROOT}" --ruler-root "${RULER_ROOT}"
PYTHONPATH=src "${PY}" -m experiments.diffusion_attention_threshold_modeling canonical-report --bundle-root "${ROOT}"
PYTHONPATH=src "${PY}" -m experiments.diffusion_attention_threshold_modeling audit-ruler16k --bundle-root "${ROOT}"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "${PY}" -m pytest -q
echo "finished=$(date -u --iso-8601=seconds)"
