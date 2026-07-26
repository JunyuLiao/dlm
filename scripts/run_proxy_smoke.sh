#!/usr/bin/env bash
set -euo pipefail
output="${1:-artifacts/proxy_smoke}"
python scripts/smoke_proxy_traces.py --output-dir "$output/traces"
python -m tracing.validate_trace_integrity "$output/traces"
python -m analysis.analyze_cross_step "$output/traces" --output-dir "$output/analysis"
python -m analysis.analyze_cross_layer "$output/traces" --output-dir "$output/analysis"
python -m analysis.analyze_cross_head "$output/traces" --output-dir "$output/analysis"
python -m analysis.plot_proxy_relationships "$output/traces" --output-dir "$output/analysis"
python -m analysis.analyze_proxy_coverage "$output/traces" --output-dir "$output/analysis"
python -m proxy.calibrate_proxy_thresholds "$output/traces" --output-dir "$output/calibration"
python -m eval.eval_proxy_one_step --length 48 --q-block-size 16 --kv-block-size 8 \
  --lambda-value 1.0 --output "$output/one_step.json"
python -m eval.eval_proxy_generation --length 48 --lambda-value 1.0 --output "$output/generation.json"
python -m eval.estimate_proxy_cost --sequence-length 48 --layers 3 --heads 4 --head-dim 8 \
  --q-tile-rows 16 --kv-tile-cols 8 --pre-qk-sparsity 0.05 --output "$output/cost.json"
