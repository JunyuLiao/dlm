#!/bin/bash
# dlm2 GPU-slot gate for the LLaDA2.2-mini headroom profile (user-approved download + ~1 GPU hour): after sc1x, run the
# prepared job (/home/exouser/dyh/llada22_20261004/llada22_profile_job.sh) only if its READY flag exists, else skip; sc2
# waits on this gate's "done" line, so the two never share the GPU. env: W PY
set -u
cd $W
until grep -q '^done ' $W/status_sc1xchain 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
# first: the LOCAL sliding-window decode-attention micro-benchmark (nsys found 25 LOCAL calls costing more than the 5
# GLOBAL ones at 128K; ~10 GPU minutes)
C=$W/cache
( cd $W && VLLM_CACHE_ROOT=$C/vllm XDG_CACHE_HOME=$C/xdg TMPDIR=$C/tmp TRITON_CACHE_DIR=$C/triton TVM_FFI_CACHE_DIR=$C/tvm   CUDA_CACHE_PATH=$C/cuda CUTE_DSL_CACHE_DIR=$C/cute FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$C/facute PYTHONNOUSERSITE=1   timeout 1200 $PY $W/v31_local_window_bench.py $W/local_window_bench.jsonl ) > $W/local_window_bench.log 2>&1
echo "localbench rc=$? $(date -u)" >> $W/status_llprof
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
J=/home/exouser/dyh/llada22_20261004
if [ -f $J/READY ] && [ -f $J/llada22_profile_job.sh ]; then
  echo "llprof start $(date -u)" >> $W/status_llprof
  timeout 6000 bash $J/llada22_profile_job.sh > $J/job_run.log 2>&1
  echo "llprof rc=$? $(date -u)" >> $W/status_llprof
else
  echo "llprof skipped (not READY) $(date -u)" >> $W/status_llprof
fi
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
echo "done $(date -u)" >> $W/status_llprof
