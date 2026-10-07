#!/bin/bash
# Wait until the H100 has enough free memory for one vLLM engine, then run the given command.
# The GPU is shared: another tenant's job may hold most of the device. Never kill it; poll.
# usage: scripts/v32_wait_gpu.sh NEED_GIB CMD...
set -u
NEED_GIB=${1:?usage: v32_wait_gpu.sh NEED_GIB CMD...}
shift
POLL=${POLL_S:-60}
# nvidia-smi reports MiB; convert the requirement once so the comparison is in one unit.
NEED_MIB=$(( NEED_GIB * 1024 ))
while :; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  used=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')
  if [ "${free:-0}" -ge "$NEED_MIB" ]; then
    echo "$(date -u) free=$(( free / 1024 ))GiB >= ${NEED_GIB}GiB, starting: $*" >&2
    exec "$@"
  fi
  echo "$(date -u) free=$(( free / 1024 ))GiB < ${NEED_GIB}GiB, holding [$used] (other tenants are not touched)" >&2
  sleep "$POLL"
done