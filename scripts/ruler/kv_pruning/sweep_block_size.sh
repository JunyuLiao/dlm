#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${OUTPUT_DIR:?set OUTPUT_DIR to the sweep root}"
sweep_root=${OUTPUT_DIR}
blocks=(1 2 4 8 16 32 64)

for block in "${blocks[@]}"; do
  export BLOCK_SIZE=${block}
  export OUTPUT_DIR="${sweep_root}/block_${block}"
  "${script_dir}/../evaluation/eval_one.sh"
done
