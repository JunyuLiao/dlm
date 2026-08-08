#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${MANIFEST_ROOT:?set MANIFEST_ROOT containing one subdirectory per context}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to the sweep root}"
sweep_root=${OUTPUT_DIR}
contexts=(512 1024 2048 4096 8192 16384)

for context in "${contexts[@]}"; do
  export CONTEXT_LENGTH=${context}
  export MANIFEST="${MANIFEST_ROOT}/${context}/manifest.json"
  export OUTPUT_DIR="${sweep_root}/context_${context}"
  "${script_dir}/eval_one.sh"
done

