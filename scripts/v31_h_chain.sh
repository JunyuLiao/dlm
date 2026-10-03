#!/bin/bash
# Panel h (efficiency switches, overlay6): after panel g, run the new GPU unit tests; if they all pass, run the panel.
# env from the launcher: W PY MODEL ROOT SHARD
set -u
C=$W/cache; O=$W/overlay6
until grep -q '^done' $W/status_g 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
cd $ROOT/deploy/v27_r17_6064109
PYTHONPATH=src:. PYTHONNOUSERSITE=1 V27_ADAPTER_DIR=$O TRITON_CACHE_DIR=$C/triton TVM_FFI_CACHE_DIR=$C/tvm \
  CUDA_CACHE_PATH=$C/cuda CUTE_DSL_CACHE_DIR=$C/cute FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$C/facute TMPDIR=$C/tmp \
  XDG_CACHE_HOME=$C/xdg timeout 1500 $PY $O/v31_run_tests.py $O/test_v31_copy_merge.py $O/test_v31_mage_fa4.py \
  $O/test_v31_dp_chunked.py $O/test_v31_fa4_observe.py $O/test_v31_logit_stats.py > $W/tests_h.out 2>&1
rc=$?
echo "tests rc=$rc $(date -u)" >> $W/status_h_tests
if [ $rc != 0 ]; then echo "done (tests failed, panel h not run) $(date -u)" >> $W/status_h; exit 1; fi
cd $W
W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD DATASETS=longbench_v2_32k,longbench_v2_64k MEM=0.90 TAG=h FIX_51994=1 \
  LABEL_SUFFIX=_fast LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4 \
  BENCH=$W/v31_vllm_paired_bench6.py OVERLAY=$O \
  ARMS='method:PIECEWISE method:PIECEWISE:main_cgate method:PIECEWISE:m2c method:PIECEWISE:r12 mage:PIECEWISE:1024 mage:PIECEWISE:4096' \
  bash $W/v31_paired_host6.sh > $W/host_h.log 2>&1
