#!/usr/bin/env bash
set -euo pipefail
python -m eval.eval_proxy_one_step --mode oracle
python -m eval.eval_proxy_one_step --mode realistic --output artifacts/proxy_one_step_realistic.json
python -m eval.eval_proxy_generation
python -m eval.estimate_proxy_cost \
  --pre-qk-sparsity "${PRE_QK_SPARSITY:-0.05}" \
  --downstream-sparsity "${DOWNSTREAM_SPARSITY:-0.25}"
