#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
script_dir=${repo_root}/scripts/ruler
export PYTHONPATH="${repo_root}/src:${PYTHONPATH:-}"

: "${RULER_ROOT:?set RULER_ROOT to the pinned NVIDIA RULER checkout}"
SMOKE_ROOT=${SMOKE_ROOT:-/tmp/dllm-ruler-smoke/diffusion_gemma}
export MODEL_ADAPTER=diffusion_gemma
export MODEL_PATH=${MODEL_PATH:-google/diffusiongemma-26B-A4B-it}
export TOKENIZER_PATH=${MODEL_PATH}
export CONTEXT_LENGTH=512
export NUM_SAMPLES=1
export TASKS=fwe
export BLOCK_SIZE=256
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
export STEPS=${STEPS:-4}
export PRECISION=bfloat16

native_args=(
  --model "${MODEL_PATH}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --output "${SMOKE_ROOT}/native.json"
)
if [[ -n "${MODEL_REVISION:-}" ]]; then
  native_args+=(--revision "${MODEL_REVISION}")
fi
python "${script_dir}/smoke_diffusion_gemma_native.py" "${native_args[@]}"

manifest_root=${SMOKE_ROOT}/manifest
export OUTPUT_DIR=${manifest_root}
"${script_dir}/prepare.sh"
export MANIFEST=${manifest_root}/manifest.json

export ATTENTION_BACKEND=dense
export OUTPUT_DIR=${SMOKE_ROOT}/dense
"${script_dir}/eval_one.sh"

export ATTENTION_BACKEND=blasst-reference
export VERIFY_DENSE_AFTER_BLASST=1
export OUTPUT_DIR=${SMOKE_ROOT}/blasst-reference
"${script_dir}/eval_one.sh"

python - "${SMOKE_ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
dense = json.loads((root / "dense" / "summary.json").read_text())
blasst = json.loads((root / "blasst-reference" / "summary.json").read_text())
for name, summary in (("dense", dense), ("blasst-reference", blasst)):
    assert summary["requested_num_samples"] == summary["actual_num_samples"] == 1, name
    assert summary["all_completions_nonempty"], name
    assert summary["full_checkpoint_on_cuda"], name
stats = blasst["attention_sparsity"]
assert stats["eligible_tiles"] > 0
assert stats["retained_tiles"] + stats["skipped_tiles"] == stats["eligible_tiles"]
assert blasst["post_blasst_dense_verified"]
print(json.dumps({"dense": dense, "blasst-reference": blasst}, indent=2, sort_keys=True))
PY
