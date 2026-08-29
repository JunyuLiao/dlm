#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
python_bin=${PYTHON_BIN:-/home/exouser/miniconda3/envs/ljy_dlm/bin/python}
: "${DIFFUSION_GEMMA_DEPENDENCY_PATH:=/tmp/diffusion-gemma-tf511-py312}"
: "${RULER_DEPENDENCY_PATH:=/tmp/ruler-smoke-deps}"
: "${RULER_ROOT:=/tmp/NVIDIA-RULER}"
: "${EXPERIMENT_ROOT:=${repo_root}/results/blasst/diffusion_gemma/ruler_8k_l0p95_g0p60}"
: "${MODEL_PATH:=google/diffusiongemma-26B-A4B-it}"
: "${MODEL_REVISION:=f7f5b7f5fa82ffc52addd066915886d497f5517b}"

export PATH="$(dirname -- "${python_bin}"):${PATH}"
export PYTHONPATH="${repo_root}/src:${DIFFUSION_GEMMA_DEPENDENCY_PATH}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

# Fixed evaluation size: 100 examples for each of the 13 official tasks.
tasks=niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2
manifest_dir=${EXPERIMENT_ROOT}/manifest
manifest=${manifest_dir}/manifest.json
dense_dir=${EXPERIMENT_ROOT}/dense_eager
sparse_dir=${EXPERIMENT_ROOT}/sparse_l0p95_g0p60
policy='{"local_blasst_lambda":0.95,"global_blasst_lambda":0.60}'

if [[ ! -s "${manifest}" || ! -s "${manifest_dir}/samples.jsonl" ]]; then
  "${python_bin}" -m dllm.cli.prepare_ruler \
    --ruler-root "${RULER_ROOT}" \
    --ruler-dependency-path "${RULER_DEPENDENCY_PATH}" \
    --tokenizer-path "${MODEL_PATH}" \
    --model-adapter diffusion_gemma \
    --revision "${MODEL_REVISION}" \
    --context-length 8192 \
    --length-mode total \
    --num-samples 1300 \
    --tasks "${tasks}" \
    --seed 42 \
    --output-dir "${manifest_dir}"
fi

"${python_bin}" - "${manifest}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
expected_tasks = {
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
}
if manifest.get("context_length") != 8192 or manifest.get("length_mode") != "total":
    raise SystemExit("manifest is not paper-style 8K total length")
if manifest.get("actual_num_samples") != 1300:
    raise SystemExit("manifest does not contain exactly 1,300 samples")
if set(manifest.get("tasks", ())) != expected_tasks:
    raise SystemExit("manifest does not contain all 13 official tasks")
if set(manifest.get("task_counts", {}).values()) != {100}:
    raise SystemExit("manifest does not contain exactly 100 samples/task")
PY

common=(
  --model-adapter diffusion_gemma
  --model-path "${MODEL_PATH}"
  --revision "${MODEL_REVISION}"
  --manifest "${manifest}"
  --ruler-root "${RULER_ROOT}"
  --context-length 8192
  --num-samples 1300
  --q-tile-size 128
  --kv-tile-size 64
  --block-size 256
  --precision bfloat16
  --temperature 0
)

echo "DiffusionGemma fixed-policy RULER evaluation"
echo "  context: 8192 total tokens"
echo "  tasks: 13; samples: 100/task, 1300/run"
echo "  dense: eager; sparse: local lambda=0.95, global lambda=0.60"
echo "  tiles: Q=128, KV=64; native canvas=256"
echo "  output: ${EXPERIMENT_ROOT}"

# A matching dense run is required: the earlier 50/task baseline cannot be used
# for the additional prompts. Both evaluations resume by sample ID.
if [[ ! -s "${dense_dir}/summary.json" ]]; then
  "${python_bin}" -m dllm.cli.eval_ruler "${common[@]}" \
    --attention-backend eager-dense \
    --output-dir "${dense_dir}"
else
  echo "resume: keeping completed 1,300-sample eager-dense baseline"
fi

if [[ ! -s "${sparse_dir}/summary.json" ]]; then
  "${python_bin}" -m dllm.cli.eval_ruler "${common[@]}" \
    --attention-backend blasst-reference \
    --blasst-policy-json "${policy}" \
    --blasst-lambda 0.003 \
    --collect-attention-stats \
    --stats-level layer \
    --output-dir "${sparse_dir}"
else
  echo "resume: keeping completed 1,300-sample sparse run"
fi

"${python_bin}" "${repo_root}/scripts/ruler/blasst/report_diffusion_gemma_blasst_fixed.py" \
  "${EXPERIMENT_ROOT}" \
  --ruler-root "${RULER_ROOT}"

echo "completed: ${EXPERIMENT_ROOT}/report/report.md"
