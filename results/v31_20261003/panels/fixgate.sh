#!/bin/bash
# mpk: GPU check of FA4_LOCAL_FIX between two sc2 jobs. Pauses the sc2 chain runner (SIGSTOP; no job is killed, the
# job in flight finishes normally) so its next job cannot take the GPU, waits for the GPU to empty, runs the LOCAL
# window micro-benchmark three ways -- a: the vLLM call unpatched (outputs saved), afix: patched (own compile caches;
# compared bitwise with a), b: static window (compared with a) -- then resumes the chain (SIGCONT, also on any exit).
# env: W PY CH (pid of the sc2 chain runner)
set -u
cd $W
trap 'kill -CONT $CH 2>/dev/null; echo "resumed $(date -u)" >> $W/status_fixgate' EXIT
kill -STOP $CH
echo "paused $CH $(date -u)" >> $W/status_fixgate
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 5; done
F=$W/fa4fix
run() {  # NAME CACHE ARGS...
  local n=$1 c=$2; shift 2
  mkdir -p $c/tmp
  ( cd $F && VLLM_CACHE_ROOT=$c/vllm XDG_CACHE_HOME=$c/xdg TMPDIR=$c/tmp TRITON_CACHE_DIR=$c/triton TVM_FFI_CACHE_DIR=$c/tvm \
    CUDA_CACHE_PATH=$c/cuda CUTE_DSL_CACHE_DIR=$c/cute FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$c/facute PYTHONNOUSERSITE=1 \
    timeout 900 $PY $F/v31_local_window_bench.py $F/lw_$n.jsonl "$@" ) > $F/lw_$n.log 2>&1
  echo "lw $n rc=$? $(date -u)" >> $W/status_fixgate
}
run a $W/cache --mode a --save $F/out_a.pt
run afix $W/cache_fix --mode a --fix --compare $F/out_a.pt
run b $W/cache --mode b --compare $F/out_a.pt
echo "done $(date -u)" >> $W/status_fixgate
