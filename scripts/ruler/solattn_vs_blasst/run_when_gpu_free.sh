#!/usr/bin/env bash
set -euo pipefail

# Resumable host-side monitor/launcher for the canonical DiffusionGemma study.
# It deliberately requires a large free-memory margin before loading the
# ~49-GiB BF16 checkpoint, so it will not contend with another H100 job.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${ROOT}"
PYTHON=${PYTHON:-/home/exouser/miniconda3/envs/ljy_dlm/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b}
MODEL_REVISION=${MODEL_REVISION:-f7f5b7f5fa82ffc52addd066915886d497f5517b}
RULER_ROOT=${RULER_ROOT:-/home/exouser/dyh/RULER_c3f5e3b_clean}
RAW_ROOT=${RAW_ROOT:-${ROOT}/results/blasst/fast_dllm_v2/ruler/ruler_cache/official_raw/16384}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/results/diffusion_gemma_solattn_vs_blasst_ruler16k}
MIN_FREE_MIB=${MIN_FREE_MIB:-65000}
POLL_SECONDS=${POLL_SECONDS:-300}
SMOKE_MAX_NEW_TOKENS=${SMOKE_MAX_NEW_TOKENS:-32}
SMOKE_STEPS=${SMOKE_STEPS:-4}

LOG_DIR=${OUTPUT_DIR}/monitor
LOG_FILE=${LOG_DIR}/monitor.log
STATUS_FILE=${LOG_DIR}/status.json
mkdir -p "${LOG_DIR}"

trap 'write_status failed "${CURRENT_PHASE:-startup}" "${CURRENT_CONDITION:-}" "${LAST_FREE_MIB:-0}" "${LAST_USED_MIB:-0}" "${LAST_UTIL_GPU:-0}"' ERR

timestamp() { date -u +%FT%TZ; }

write_status() {
  local state=$1
  local phase=$2
  local condition=${3:-}
  local free=${4:-0}
  local used=${5:-0}
  local util=${6:-0}
  printf '{"timestamp":"%s","state":"%s","phase":"%s","condition":"%s","free_mib":%s,"used_mib":%s,"utilization_gpu":%s,"min_free_mib":%s}\n' \
    "$(timestamp)" "${state}" "${phase}" "${condition}" "${free}" "${used}" "${util}" "${MIN_FREE_MIB}" > "${STATUS_FILE}"
}

CURRENT_PHASE=waiting
CURRENT_CONDITION=
LAST_FREE_MIB=0
LAST_USED_MIB=0
LAST_UTIL_GPU=0

gpu_snapshot() {
  nvidia-smi --query-gpu=memory.free,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); print $1, $2, $3}'
}

echo "[$(timestamp)] monitor started; waiting for >=${MIN_FREE_MIB} MiB free" | tee -a "${LOG_FILE}"

while true; do
  snapshot=$(gpu_snapshot 2>/dev/null || true)
  if [[ -z "${snapshot}" ]]; then
    write_status unavailable waiting 0 0 0
    echo "[$(timestamp)] GPU query unavailable; retrying in ${POLL_SECONDS}s" | tee -a "${LOG_FILE}"
    sleep "${POLL_SECONDS}"
    continue
  fi
  read -r free_mib used_mib util_gpu <<<"${snapshot}"
  if [[ ! "${free_mib:-}" =~ ^[0-9]+$ || ! "${used_mib:-}" =~ ^[0-9]+$ || ! "${util_gpu:-}" =~ ^[0-9]+$ ]]; then
    write_status unavailable waiting 0 0 0
    echo "[$(timestamp)] GPU query returned nonnumeric telemetry; retrying in ${POLL_SECONDS}s" | tee -a "${LOG_FILE}"
    sleep "${POLL_SECONDS}"
    continue
  fi
  LAST_FREE_MIB=${free_mib}; LAST_USED_MIB=${used_mib}; LAST_UTIL_GPU=${util_gpu}
  write_status waiting occupied "" "${free_mib}" "${used_mib}" "${util_gpu}"
  if (( free_mib < MIN_FREE_MIB )); then
    echo "[$(timestamp)] H100 occupied: free=${free_mib} MiB used=${used_mib} MiB util=${util_gpu}%; retrying in ${POLL_SECONDS}s" | tee -a "${LOG_FILE}"
    sleep "${POLL_SECONDS}"
    continue
  fi
  echo "[$(timestamp)] H100 threshold met: free=${free_mib} MiB; starting study" | tee -a "${LOG_FILE}"
  break
done

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=${ROOT}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}

if [[ ! -f "${OUTPUT_DIR}/manifest.json" ]]; then
  CURRENT_PHASE=prepare
  write_status running prepare "" "${free_mib}" "${used_mib}" "${util_gpu}"
  "${PYTHON}" -m experiments.diffusion_gemma_solattn_vs_blasst_ruler16k prepare \
    --raw-root "${RAW_ROOT}" \
    --ruler-root "${RULER_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --revision "${MODEL_REVISION}" \
    --output-dir "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"
fi

if [[ ! -f "${OUTPUT_DIR}/smoke/status.json" || "$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", ""))' "${OUTPUT_DIR}/smoke/status.json" 2>/dev/null || true)" != "passed" ]]; then
  CURRENT_PHASE=smoke
  CURRENT_CONDITION=smoke_dense
  write_status running smoke "smoke_dense" "${free_mib}" "${used_mib}" "${util_gpu}"
  mkdir -p "${OUTPUT_DIR}/smoke"
  "${PYTHON}" - "${OUTPUT_DIR}" "${MODEL_PATH}" "${MODEL_REVISION}" "${RULER_ROOT}" <<'PY'
import json, pathlib, sys
from dllm.evaluation.ruler.io import read_jsonl, sha256_file, write_json, write_jsonl

root = pathlib.Path(sys.argv[1])
study = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
calibration = read_jsonl(root / "calibration.jsonl")[:2]
smoke_rows = root / "smoke" / "final.jsonl"
write_jsonl(smoke_rows, calibration)
smoke_manifest = {
    "schema_version": 1,
    "study": study["study"],
    "ruler": study["ruler"],
    "context_length": study["context_length"],
    "split_seed": study["split_seed"],
    "tasks": study["tasks"],
    "tokenizer": study.get("tokenizer", {}),
    "final": {"path": str(smoke_rows), "sha256": sha256_file(smoke_rows), "count": len(calibration)},
    "calibration": study["calibration"],
}
write_json(root / "smoke" / "manifest.json", smoke_manifest)
PY

  # First run is native dense. A second run uses the BLASST binding with
  # masking disabled, so exact generated-token parity exercises the real
  # registry integration before any sparse route is attempted.
  "${PYTHON}" -m experiments.diffusion_gemma_solattn_vs_blasst_ruler16k run \
    --study-manifest "${OUTPUT_DIR}/smoke/manifest.json" \
    --model-path "${MODEL_PATH}" --revision "${MODEL_REVISION}" \
    --ruler-root "${RULER_ROOT}" --output-dir "${OUTPUT_DIR}/smoke/dense_a" \
    --conditions dense --device cuda --precision bfloat16 2>&1 | tee -a "${LOG_FILE}"
  "${PYTHON}" - "${OUTPUT_DIR}" "${MODEL_PATH}" "${MODEL_REVISION}" "${RULER_ROOT}" <<'PY'
import json, pathlib, sys
from dllm.evaluation.ruler.runner import RulerRunConfig, run_evaluation

root = pathlib.Path(sys.argv[1]) / "smoke"
runner_manifest = root / "dense_a" / "dense" / "runner_manifest.json"
run_evaluation(RulerRunConfig(
    model_adapter="diffusion_gemma",
    model_path=sys.argv[2],
    revision=sys.argv[3],
    manifest_path=str(runner_manifest),
    ruler_root=sys.argv[4],
    output_dir=str(root / "blasst_noop"),
    num_samples=2,
    context_length=16384,
    attention_backend="blasst-reference",
    apply_blasst_mask=False,
    collect_attention_stats=False,
    device="cuda",
    precision="bfloat16",
))
PY

  "${PYTHON}" - "${OUTPUT_DIR}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]) / "smoke"
a = json.loads((root / "dense_a" / "dense" / "summary.json").read_text())
b = json.loads((root / "blasst_noop" / "summary.json").read_text())
assert a["actual_num_samples"] == b["actual_num_samples"] == 2
assert a["all_completions_nonempty"] and b["all_completions_nonempty"]
pa = {row["sample_id"]: row for row in json.loads("[" + ",".join((root / "dense_a" / "dense" / "predictions.jsonl").read_text().splitlines()) + "]")}
pb = {row["sample_id"]: row for row in json.loads("[" + ",".join((root / "blasst_noop" / "predictions.jsonl").read_text().splitlines()) + "]")}
assert pa.keys() == pb.keys()
for sample_id in pa:
    assert pa[sample_id]["prompt_hash"] == pb[sample_id]["prompt_hash"]
    assert pa[sample_id]["inference_seed"] == pb[sample_id]["inference_seed"]
    assert pa[sample_id]["completion_tokens"] == pb[sample_id]["completion_tokens"]
PY
fi

if [[ ! -f "${OUTPUT_DIR}/calibration/blasst_policy.json" ]]; then
  CURRENT_PHASE=calibrate
  write_status running calibrate "" "${free_mib}" "${used_mib}" "${util_gpu}"
  "${PYTHON}" -m experiments.diffusion_gemma_solattn_vs_blasst_ruler16k calibrate-blasst \
    --study-manifest "${OUTPUT_DIR}/manifest.json" \
    --model-path "${MODEL_PATH}" \
    --revision "${MODEL_REVISION}" \
    --ruler-root "${RULER_ROOT}" \
    --output-dir "${OUTPUT_DIR}/calibration" \
    --device cuda --precision bfloat16 2>&1 | tee -a "${LOG_FILE}"
fi

if [[ ! -f "${OUTPUT_DIR}/smoke/status.json" || "$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", ""))' "${OUTPUT_DIR}/smoke/status.json" 2>/dev/null || true)" != "passed" ]]; then
  CURRENT_PHASE=smoke
  CURRENT_CONDITION=smoke_blasst
  write_status running smoke "smoke_blasst" "${free_mib}" "${used_mib}" "${util_gpu}"
  "${PYTHON}" -m experiments.diffusion_gemma_solattn_vs_blasst_ruler16k run \
    --study-manifest "${OUTPUT_DIR}/smoke/manifest.json" \
    --threshold-policy "${OUTPUT_DIR}/calibration/blasst_policy.json" \
    --model-path "${MODEL_PATH}" --revision "${MODEL_REVISION}" \
    --ruler-root "${RULER_ROOT}" --output-dir "${OUTPUT_DIR}/smoke" \
    --conditions blasst_calibrated_s25 --device cuda --precision bfloat16 2>&1 | tee -a "${LOG_FILE}"
  "${PYTHON}" - "${OUTPUT_DIR}" <<'PY'
import csv, json, pathlib, sys
root = pathlib.Path(sys.argv[1]) / "smoke" / "blasst_calibrated_s25"
summary = json.loads((root / "summary.json").read_text())
assert summary["actual_num_samples"] == 2 and summary["all_completions_nonempty"]
step = list(csv.DictReader((root / "attention_stats" / "per_step.csv").open()))
layer = list(csv.DictReader((root / "attention_stats" / "per_layer.csv").open()))
head = list(csv.DictReader((root / "attention_stats" / "per_head.csv").open()))
types = {row.get("attention_type") for row in step}
assert {"local", "global"}.issubset(types)
assert {row.get("attention_type") for row in layer} >= {"local", "global"}
assert {row.get("attention_type") for row in head} >= {"local", "global"}
regions = set()
for row in step:
    value = row.get("region_counts", "")
    try:
        regions.update(json.loads(value).keys())
    except Exception:
        pass
assert {"prefix", "canvas"}.issubset(regions)
(pathlib.Path(sys.argv[1]) / "smoke" / "status.json").write_text(json.dumps({"state": "passed", "dense_parity": True, "blasst_examples": 2}) + "\n")
PY
fi

for condition in dense sol_gaussian_s25 sol_gaussian_s50 sol_gaussian_s75 sol_gaussian_s90 \
                 blasst_calibrated_s25 blasst_calibrated_s50 blasst_calibrated_s75 blasst_calibrated_s90; do
  if [[ -f "${OUTPUT_DIR}/${condition}/summary.json" && -f "${OUTPUT_DIR}/${condition}/predictions.jsonl" ]] \
      && [[ "$(wc -l < "${OUTPUT_DIR}/${condition}/predictions.jsonl")" -ge 50 ]]; then
    echo "[$(timestamp)] ${condition} already has a summary; resuming/skipping" | tee -a "${LOG_FILE}"
    continue
  fi
  CURRENT_PHASE=run
  CURRENT_CONDITION=${condition}
  write_status running run "${condition}" "${free_mib}" "${used_mib}" "${util_gpu}"
  "${PYTHON}" -m experiments.diffusion_gemma_solattn_vs_blasst_ruler16k run \
    --study-manifest "${OUTPUT_DIR}/manifest.json" \
    --threshold-policy "${OUTPUT_DIR}/calibration/blasst_policy.json" \
    --model-path "${MODEL_PATH}" \
    --revision "${MODEL_REVISION}" \
    --ruler-root "${RULER_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --conditions "${condition}" \
    --device cuda --precision bfloat16 2>&1 | tee -a "${LOG_FILE}"
done

write_status running report "" "${free_mib}" "${used_mib}" "${util_gpu}"
"${PYTHON}" -m experiments.diffusion_gemma_solattn_vs_blasst_ruler16k report \
  --output-dir "${OUTPUT_DIR}" --ruler-root "${RULER_ROOT}" 2>&1 | tee -a "${LOG_FILE}"
write_status complete complete "" "${free_mib}" "${used_mib}" "${util_gpu}"
echo "[$(timestamp)] study complete" | tee -a "${LOG_FILE}"
