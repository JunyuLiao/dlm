#!/usr/bin/env bash
# Record low-frequency, non-intrusive status for a resumable sweep.
set -euo pipefail
pid="$1"
root="$2"
log="$root/monitor.log"
while kill -0 "$pid" 2>/dev/null; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  counts="$(find "$root/conditions" -name predictions.jsonl -type f -exec sh -c 'printf "%s=%s " "$(basename "$(dirname "$1")")" "$(wc -l < "$1")"' _ {} \; 2>/dev/null || true)"
  gpu="$(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null || true)"
  printf '%s pid=%s %s gpu=%s\n' "$timestamp" "$pid" "$counts" "$gpu" >> "$log"
  sleep 900
done
timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s pid=%s exited\n' "$timestamp" "$pid" >> "$log"
