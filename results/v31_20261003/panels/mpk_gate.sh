#!/bin/bash
# mpk GPU-slot gate between sc1x and sc2: the LOCAL sliding-window decode-attention micro-benchmark (~10 GPU minutes;
# dlm2's first attempt failed on a call-signature mismatch, now verbose with the int32 dynamic-causal variant).
# sc2 waits on this gate's "done" line. env: W PY
set -u
cd $W
until grep -q '^done ' $W/status_sc1xchain 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
C=$W/cache
( cd $W && VLLM_CACHE_ROOT=$C/vllm XDG_CACHE_HOME=$C/xdg TMPDIR=$C/tmp TRITON_CACHE_DIR=$C/triton TVM_FFI_CACHE_DIR=$C/tvm \
  CUDA_CACHE_PATH=$C/cuda CUTE_DSL_CACHE_DIR=$C/cute FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$C/facute PYTHONNOUSERSITE=1 \
  timeout 1200 $PY $W/v31_local_window_bench.py $W/local_window_bench.jsonl ) > $W/local_window_bench.log 2>&1
echo "localbench rc=$? $(date -u)" >> $W/status_mpkgate
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
echo "done $(date -u)" >> $W/status_mpkgate
