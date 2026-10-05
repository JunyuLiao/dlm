#!/bin/bash
# FX control block: lean 8192 WITHOUT the fix, same adapter as the FX arms (ov_pa4fx = ov_pa4 adapter 7338e2769d45),
# same S1 LongBench think cells (seeds 1-2). Tells the fix's effect (control vs lean_fx: expected identical) from the
# adapter-version effect (control vs the sc1 lean 8192 run, adapter 789917ef104b; 3/32 cells differed on dlm2).
# Waits until fxgate.sh has resumed the sc2 chain and that chain has a job in flight (its python running), then pauses
# the chain (SIGSTOP; nothing killed), runs $W/jobs_fx2.txt and resumes it (SIGCONT, also on any exit).
# env: W PY MODEL ROOT SHARD P CH
set -u
cd $W
until grep -q '^resumed' $W/status_fxgate 2>/dev/null; do sleep 10; done
while true; do
  H=$(ps -eo pid,ppid,args | awk -v ch=$CH '$2==ch && $0 ~ /v31_paired_host8.sh/ {print $1}')
  [ -n "$H" ] && [ -n "$(ps -eo ppid,args | awk -v h=$H '$1==h && $0 ~ /timeout/')" ] && break
  kill -0 $CH 2>/dev/null || { echo "refused: chain $CH gone $(date -u)" >> $W/status_fxgate2; exit 1; }
  sleep 5
done
trap 'kill -CONT $CH 2>/dev/null; echo "resumed $(date -u)" >> $W/status_fxgate2' EXIT
kill -STOP $CH
echo "paused $CH (job in flight under $H) $(date -u)" >> $W/status_fxgate2
echo "done ready $(date -u)" > $W/status_fx2wait
JOBS=$W/jobs_fx2.txt WAITFILE=status_fx2wait CHAIN=status_fx2chain OV=ov_pa4fx bash $W/jobs_chain.sh
echo "fx2 rc=$? $(date -u)" >> $W/status_fxgate2
