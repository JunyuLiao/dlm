#!/bin/bash
# Swap the job list of a running chain runner between jobs: only if the job in flight is the expected one (last line of
# its status file is "<LABEL> start") and its python is on the GPU; kills the runner (SIGKILL; the job in flight runs to
# completion under its orphaned host8) and starts jobs_chain.sh on $NEWJOBS (its first job waits for the GPU).
# env: W CH STATUS LABEL NEWJOBS CHAIN OV
set -u
cd $W
last=$(tail -1 $W/$STATUS)
case "$last" in "$LABEL start "*) ;; *) echo "refused: $STATUS last line: $last"; exit 1 ;; esac
H=$(ps -eo pid,ppid,args | awk -v ch=$CH '$2==ch && $0 ~ /v31_paired_host8.sh/ {print $1}')
T=$(ps -eo pid,ppid,args | awk -v h=$H '$2==h && $0 ~ /timeout/ {print $1}')
PYP=$(ps -eo pid,ppid | awk -v t=$T '$2==t {print $1}')
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -qx "$PYP" || { echo "refused: python $PYP not on GPU"; exit 1; }
E=$(tr '\0' '\n' < /proc/$CH/environ | grep -E '^(W|PY|MODEL|ROOT|SHARD|P)=' | tr '\n' ' ')
kill -KILL $CH
echo "done ready $(date -u)" > $W/status_swapwait
env $E JOBS=$NEWJOBS WAITFILE=status_swapwait CHAIN=$CHAIN OV=$OV nohup setsid bash $W/jobs_chain.sh > $W/${CHAIN}.log 2>&1 < /dev/null &
sleep 2
echo "swapped: runner $CH -> $(ps -eo pid,args | awk '$2=="bash" && $3 ~ /jobs_chain/ {print $1}'), in flight python $PYP ($LABEL), new list $NEWJOBS"
