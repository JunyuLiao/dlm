#!/bin/bash
# overlay7 chain (chunked DP build with a runtime chunk count): wait for $WAIT, run the GPU unit tests, then
# [panel h if RUN_H=1] -> panel h2 (warm, reported) -> panel i (96K). env: W PY MODEL ROOT SHARD WAIT RUN_H
set -u
O=$W/overlay7
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
ARMS6='method:PIECEWISE method:PIECEWISE:main_cgate method:PIECEWISE:m2c method:PIECEWISE:r12 mage:PIECEWISE:1024 mage:PIECEWISE:4096'
until grep -q '^done' $W/$WAIT 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
C=$W/cache
( cd $ROOT/deploy/v27_r17_6064109 && PYTHONPATH=src:. PYTHONNOUSERSITE=1 V27_ADAPTER_DIR=$O TRITON_CACHE_DIR=$C/triton \
  TVM_FFI_CACHE_DIR=$C/tvm CUDA_CACHE_PATH=$C/cuda CUTE_DSL_CACHE_DIR=$C/cute FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$C/facute \
  TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg timeout 1500 $PY $O/v31_run_tests.py $O/test_v31_dp_chunked.py \
  $O/test_v31_copy_merge.py $O/test_v31_mage_fa4.py $O/test_v31_fa4_observe.py $O/test_v31_logit_stats.py \
  > $W/tests_h7.out 2>&1 )
rc=$?
echo "tests7 rc=$rc $(date -u)" >> $W/status_h7_tests
[ $rc != 0 ] && exit 1
cd $W
run() {  # TAG SUFFIX DATASETS MEM ARMS [fast]
  env W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD DATASETS=$3 MEM=$4 TAG=$1 FIX_51994=1 LABEL_SUFFIX=$2 \
    BENCH=$W/v31_vllm_paired_bench6.py OVERLAY=$O ARMS="$5" $6 bash $W/v31_paired_host6.sh > $W/host_$1$2.log 2>&1
}
[ "$RUN_H" = 1 ] && run h _fast longbench_v2_32k,longbench_v2_64k 0.90 "$ARMS6" "$FAST"
run h2 _fast2 longbench_v2_32k,longbench_v2_64k 0.90 "$ARMS6" "$FAST"
run i "" longbench_v2_96k 0.92 'dense:default dense:PIECEWISE' ""
sed -i 's/^done/dense-done/' $W/status_i
run i _fast longbench_v2_96k 0.92 'method:PIECEWISE method:PIECEWISE:m2c method:PIECEWISE:r12 mage:PIECEWISE:1024' "$FAST"
