#!/bin/bash
# Panel ak (after pa): (1) the second dense baseline -- the all-kept arm, i.e. the SAME FA4 block-sparse path as the
# sparse arms with every tile kept -- on gl's LongBench-v2 64K + 96K cells and gl128's 128K cells (same cells / host as
# gl's dense FULL runs), so sparsity's own gain is separated from the kernel path's; (2) mpk only (nsys present; GPU
# hardware counters are admin-only, RmProfilingAdminOnly=1, so no ncu): an nsys CUDA kernel trace of the kernel audit
# (synthetic tensors) at 64K / 128K -- grid sizes and durations of the dense and block-sparse FA4 kernels, for a wave /
# SM-occupancy analysis. env: W PY MODEL ROOT LBSHARD NSYS (1 on mpk)
set -u
cd $W
until grep -q '^done ' $W/status_pachain 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
B="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$LBSHARD FIX_51994=1 MEM=0.90 OVERLAY=$W/ov_gs BENCH=$W/bench_ov_gs.py"
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
env $B $FAST TAG=gl DATASETS=longbench_v2_64k,longbench_v2_96k CELLS=$W/cells_confirm.json ARMS='allkept:PIECEWISE' \
  bash $W/v31_paired_host8.sh > $W/host_ak_conf.log 2>&1
sed -i "s/^done /done_allkept /" $W/status_gl
env $B $FAST TAG=gl128 DATASETS=longbench_v2_128k CELLS=$W/cells_lb128k.json MAN_DIR=$W/manifests_lb128k ARMS='allkept:PIECEWISE' \
  bash $W/v31_paired_host8.sh > $W/host_ak_128.log 2>&1
sed -i "s/^done /done_allkept /" $W/status_gl128
echo "allkept complete $(date -u)" >> $W/status_akchain
if [ "${NSYS:-0}" = 1 ]; then
  until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
  C=$W/cache; O=$W/ov_kern; mkdir -p $W/nsys
  ( cd $ROOT/deploy/v27_r17_6064109 && PYTHONPATH=src:. PYTHONNOUSERSITE=1 V27_ADAPTER_DIR=$O TRITON_CACHE_DIR=$C/triton \
    TVM_FFI_CACHE_DIR=$C/tvm CUDA_CACHE_PATH=$C/cuda CUTE_DSL_CACHE_DIR=$C/cute FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$C/facute \
    TMPDIR=$C/tmp XDG_CACHE_HOME=$C/xdg timeout 3600 /usr/local/bin/nsys profile --trace=cuda --sample=none --cpuctxsw=none \
    --force-overwrite=true -o $W/nsys/kernel_audit $PY $O/v31_sparse_kernel_audit.py $W/nsys/audit_rows.jsonl \
    --keys 65536,131072 --kept 0.0625,0.125,1 --splits 2,4 --patterns kvshared,indep,skew --reps 3 --warm 2 ) > $W/nsys/profile.log 2>&1
  echo "nsys rc=$? $(date -u)" >> $W/status_akchain
  /usr/local/bin/nsys stats --report cuda_gpu_trace --format csv --output $W/nsys/trace $W/nsys/kernel_audit.nsys-rep > $W/nsys/stats.log 2>&1
  echo "nsys stats rc=$? $(date -u)" >> $W/status_akchain
fi
echo "done $(date -u)" >> $W/status_akchain
