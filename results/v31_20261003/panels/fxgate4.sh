#!/bin/bash
# FX determinism block (after fxgate3.sh): lean 8192 again WITHOUT the fix (_ctl2, a repeat of _ctl) and again WITH
# it (_fxrep, a repeat of _fx), same adapter (ov_pa4fx) and S1 LongBench think cells. On 4-6 of 64 cells lean with vs
# without the fix diverged mid-output (dense: 64/64 identical), so: is the lean path itself run-to-run deterministic
# (ctl vs ctl2, fx vs fxrep)? Same race-free pause as fxgate3.sh (waits for $AFTER, then for the sc2 job on the GPU).
# env: W PY MODEL ROOT SHARD P CH [AFTER]
set -u
cd $W
if [ -n "${AFTER:-}" ]; then until grep -q '^resumed' $W/$AFTER 2>/dev/null; do sleep 10; done; fi
while true; do
  kill -0 $CH 2>/dev/null || { echo "refused: chain $CH gone $(date -u)" >> $W/status_fxgate4; exit 1; }
  H=$(ps -eo pid,ppid,args | awk -v ch=$CH '$2==ch && $0 ~ /v31_paired_host8.sh/ {print $1}')
  if [ -n "$H" ]; then
    T=$(ps -eo pid,ppid,args | awk -v h=$H '$2==h && $0 ~ /timeout/ {print $1}')
    if [ -n "$T" ]; then
      PYP=$(ps -eo pid,ppid | awk -v t=$T '$2==t {print $1}')
      if [ -n "$PYP" ] && nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -qx "$PYP"; then break; fi
    fi
  fi
  sleep 5
done
trap 'kill -CONT $CH 2>/dev/null; echo "resumed $(date -u)" >> $W/status_fxgate4' EXIT
kill -STOP $CH
echo "paused $CH (job in flight: python $PYP on the GPU) $(date -u)" >> $W/status_fxgate4
echo "done ready $(date -u)" > $W/status_fx4wait
JOBS=$W/jobs_fx4.txt WAITFILE=status_fx4wait CHAIN=status_fx4chain OV=ov_pa4fx bash $W/jobs_chain.sh
echo "fx4 rc=$? $(date -u)" >> $W/status_fxgate4
