#!/usr/bin/env bash
set -euo pipefail
calibration="${1:-artifacts/proxy_calibration/policy_summary.json}"
contexts="${NUM_CONTEXTS:-4}"
for budget in 1e-2 1e-3 1e-4; do
  for transition in 0.9:0.5 0.5:0.15; do
    previous="${transition%%:*}"
    current="${transition##*:}"
    label="${previous}_to_${current}"
    python -m eval.eval_proxy_llada_one_step \
      --mode realistic --policy score --calibration "$calibration" \
      --previous-mask-ratio "$previous" --current-mask-ratio "$current" \
      --false-skip-budget "$budget" --num-contexts "$contexts" \
      --output "artifacts/proxy_score_${label}_realistic_${budget}.json"
    python -m eval.eval_proxy_llada_one_step \
      --mode realistic --policy max_score --calibration "$calibration" \
      --previous-mask-ratio "$previous" --current-mask-ratio "$current" \
      --false-skip-budget "$budget" --num-contexts "$contexts" \
      --output "artifacts/proxy_max_score_${label}_realistic_${budget}.json"
  done
done
