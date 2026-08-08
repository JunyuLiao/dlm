#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${OUTPUT_DIR:?set OUTPUT_DIR to the comparison root}"
comparison_root=${OUTPUT_DIR}

for backend in dense blasst-reference; do
  export ATTENTION_BACKEND=${backend}
  export OUTPUT_DIR="${comparison_root}/${backend}"
  "${script_dir}/eval_one.sh"
done

