#!/bin/bash
# FX control block, race-free version of fxgate2.sh (whose control job raced the sc2 job started at the same second
# on dlm2 and was refused by vLLM's startup memory check). Lean 8192 WITHOUT the fix, same adapter as the FX arms
# (ov_pa4fx = ov_pa4 adapter 7338e2769d45), same S1 LongBench think cells (seeds 1-2): tells the fix's effect (control
# vs lean_fx, expected identical) from the adapter-version effect (control vs the sc1 lean 8192 run, adapter
# 789917ef104b). Optionally waits until $AFTER has a "resumed" line (an earlier gate has resumed the chain); then waits
# until the sc2 chain's job in flight is ON THE GPU (its python running AND listed by nvidia-smi), so no GPU waiter of
# its own can still be pending; then pauses the chain (SIGSTOP; nothing killed), runs $W/jobs_fx2.txt, and resumes it
# (SIGCONT, also on any exit). env: W PY MODEL ROOT SHARD P CH [AFTER]
set -u
cd $W
if [ -n "${AFTER:-}" ]; then until grep -q '^resumed' $W/$AFTER 2>/dev/null; do sleep 10; done; fi
while true; do
  kill -0 $CH 2>/dev/null || { echo "refused: chain $CH gone $(date -u)" >> $W/status_fxgate3; exit 1; }
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
trap 'kill -CONT $CH 2>/dev/null; echo "resumed $(date -u)" >> $W/status_fxgate3' EXIT
kill -STOP $CH
echo "paused $CH (job in flight: python $PYP on the GPU) $(date -u)" >> $W/status_fxgate3
echo "done ready $(date -u)" > $W/status_fx3wait
JOBS=$W/jobs_fx2.txt WAITFILE=status_fx3wait CHAIN=status_fx3chain OV=ov_pa4fx bash $W/jobs_chain.sh
echo "fx3 rc=$? $(date -u)" >> $W/status_fxgate3
