#!/usr/bin/env bash
set -euo pipefail
python scripts/collect_proxy_traces.py \
  --context-length 4096 --num-contexts 32 --batch-size 1 \
  --mask-ratios 0.9,0.5,0.15 --sample-fraction 0.05 \
  --bitpack-masks --output-dir artifacts/proxy_traces_v3 --resume
