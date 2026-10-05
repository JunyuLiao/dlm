#!/bin/bash
# FX block (FA4_LOCAL_FIX end to end): pauses the sc2 chain runner $CH (SIGSTOP; the job in flight finishes normally;
# refuses unless that job's python is running, so no second GPU waiter can race this block), runs jobs_chain.sh on
# $W/jobs_fx.txt with overlay ov_pa4fx (the bench with FA4_LOCAL_FIX; adapter identical to ov_pa4), then resumes the
# chain (SIGCONT, also on any exit). Same S1 cells, shard, tags and seeds as the arms it is compared with.
# env: W PY MODEL ROOT SHARD P CH
set -u
cd $W
H=$(ps -eo pid,ppid,args | awk -v ch=$CH '$2==ch && $0 ~ /v31_paired_host8.sh/ {print $1}')
if [ -z "$H" ] || [ -z "$(ps -eo ppid,args | awk -v h=$H '$1==h && $0 ~ /timeout/')" ]; then
  echo "refused: chain $CH has no job in flight $(date -u)" >> $W/status_fxgate; exit 1
fi
trap 'kill -CONT $CH 2>/dev/null; echo "resumed $(date -u)" >> $W/status_fxgate' EXIT
kill -STOP $CH
echo "paused $CH (job in flight under $H) $(date -u)" >> $W/status_fxgate
echo "done ready $(date -u)" > $W/status_fxwait
JOBS=$W/jobs_fx.txt WAITFILE=status_fxwait CHAIN=status_fxchain OV=ov_pa4fx bash $W/jobs_chain.sh
echo "fx rc=$? $(date -u)" >> $W/status_fxgate
