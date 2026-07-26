#!/usr/bin/env bash
set -euo pipefail
trace_dir="${1:-artifacts/proxy_traces}"
python -m tracing.validate_trace_integrity "$trace_dir"
python -m analysis.analyze_cross_step "$trace_dir"
python -m analysis.analyze_cross_layer "$trace_dir"
python -m analysis.analyze_cross_head "$trace_dir"
python -m analysis.plot_proxy_relationships "$trace_dir"
python -m analysis.analyze_proxy_coverage "$trace_dir"
python -m proxy.calibrate_proxy_thresholds "$trace_dir"
