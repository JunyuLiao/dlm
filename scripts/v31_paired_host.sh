#!/bin/bash
# Run the seed-paired vLLM panel (scripts/v31_vllm_paired_bench.py) on one host: for each ARM:CG in $ARMS, one fresh
# engine over this host's shard of the cells. Every arm of a cell runs on the same host. Usage on the host:
#   W=<dyh run dir> PY=<vllm env python> MODEL=<model dir> ROOT=<m3_output_numerics_v21 root> SHARD=k/3 \
#   ARMS='dense:default dense:PIECEWISE method:PIECEWISE' DATASETS=longbench_v2_32k,longbench_v2_64k MEM=0.85 TAG=a \
#   nohup bash v31_paired_host.sh > host_a.log 2>&1 &
# W must contain v31_vllm_paired_bench.py, overlay/vllm_adapter.py and cells.json (private: has item ids).
set -u
C=$W/cache
D=$ROOT/deploy/v27_r17_6064109
CFG=$ROOT/v23/v27_r17_001/host.fragment_configs/ruler64k_r17/M3_R6_A64_fused_dp_async_m1ln2_c0_fa4.json
MAN=$ROOT/v23/v27_lb_e14_001/manifests
mkdir -p $C/tmp $W/private $W/public
export VLLM_CACHE_ROOT=$C/vllm XDG_CACHE_HOME=$C/xdg TMPDIR=$C/tmp TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton \
  HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 TVM_FFI_CACHE_DIR=$C/tvm CUDA_CACHE_PATH=$C/cuda CUTE_DSL_CACHE_DIR=$C/cute \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$C/facute PYTHONNOUSERSITE=1 PYTHONPATH=src:. V27_ADAPTER_DIR=$W/overlay \
  SHARD DATASETS MEM REPEATS=${REPEATS:-1}
cd $D
for spec in $ARMS; do
  arm=${spec%%:*}; cg=${spec#*:}
  until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 20; done
  echo "$arm $cg start $(date -u)" >> $W/status_$TAG
  timeout 21600 $PY $W/v31_vllm_paired_bench.py $MODEL $MAN $W/cells.json $W/public/${TAG}_${arm}_${cg}.jsonl \
    $W/private/${TAG}_${arm}_${cg}.private.jsonl $arm $cg $([ $arm = method ] && echo $CFG) \
    > $W/${TAG}_${arm}_${cg}.out 2> $W/${TAG}_${arm}_${cg}.err
  echo "$arm $cg rc=$? $(date -u)" >> $W/status_$TAG
done
echo "done $(date -u)" >> $W/status_$TAG
