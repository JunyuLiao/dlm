#!/bin/bash
# mpk: LOCAL window micro-benchmark, one process per variant (FA4's compile cache does not key on dynamic_causal), run
# in the GPU gap after the sc2 job in flight; the rest of sc2 waits on this gate. env: W PY
set -u
cd $W
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
C=$W/cache
for m in a b c; do
  ( cd $W && VLLM_CACHE_ROOT=$C/vllm XDG_CACHE_HOME=$C/xdg TMPDIR=$C/tmp TRITON_CACHE_DIR=$C/triton TVM_FFI_CACHE_DIR=$C/tvm \
    CUDA_CACHE_PATH=$C/cuda CUTE_DSL_CACHE_DIR=$C/cute FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$C/facute PYTHONNOUSERSITE=1 \
    timeout 900 $PY $W/v31_local_window_bench.py $W/local_window_bench_$m.jsonl --mode $m ) > $W/local_window_bench_$m.log 2>&1
  echo "localbench $m rc=$? $(date -u)" >> $W/status_mpkgate2
done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
echo "done $(date -u)" >> $W/status_mpkgate2
