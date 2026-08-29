#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${OUTPUT_DIR:?set OUTPUT_DIR to the sweep root}"
sweep_root=${OUTPUT_DIR}
lambdas=(0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.5)

export ATTENTION_BACKEND=blasst-reference
for value in "${lambdas[@]}"; do
  export BLASST_LAMBDA=${value}
  export OUTPUT_DIR="${sweep_root}/lambda_${value}"
  "${script_dir}/../evaluation/eval_one.sh"
done
