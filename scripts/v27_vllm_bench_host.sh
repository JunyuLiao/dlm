#!/bin/bash
# Run scripts/v27_vllm_method_bench.py on dllm for a list of arms, one fresh engine per arm, each after the GPU is idle.
# Used for the 2026-10-02 vLLM smokes (results/.../vllm_port_probe_1002/smoke_*.jsonl). Copy to the host and run:
#   ARMS='dense: dense:PIECEWISE allkept:PIECEWISE method:PIECEWISE' TAG=5 DS=longbench_v2_64k MEM=0.90 LIMIT=2 \
#   V27_ADAPTER_PROFILE=1 nohup bash v27_vllm_bench_host.sh > bench5.log 2>&1 &
# Before running, copy the current files to the host:
#   experiments/numerical_qk_reuse/vllm_adapter.py -> $W/overlay/vllm_adapter.py
#   scripts/v27_vllm_method_bench.py -> $W/v27_vllm_method_bench.py
# The core is imported from a panel deploy ($D) so the frozen main-arm config ($CFG) validates byte-for-byte; the
# adapter comes from the overlay. Completions go only to $W/private (never commit them).
# ARMS entries are arm:cudagraph_mode ('' = vLLM default); arms are dense | native | allkept | method; the adapter arms
# need PIECEWISE.
W=/home/exouser/dyh/dlm_models_20261002/vllm_method; C=$W/cache
D=/home/exouser/dyh/m3_output_numerics_v21_20260927/deploy/v27_r17_6064109
CFG=/home/exouser/dyh/m3_output_numerics_v21_20260927/v23/v27_r17_001/host.fragment_configs/ruler64k_r17/M3_R6_A64_fused_dp_async_m1ln2_c0_fa4.json
MODEL=/home/exouser/dyh/junyu_frontier_v18_20260926/bridge/model_complete
MAN=${MAN:-/home/exouser/dyh/m3_output_numerics_v21_20260927/v23/v27_lb_e14_001/manifests}
CELLS=${CELLS:-/home/exouser/dyh/dlm_models_20261002/vllm_check_1002/cells.json}
PY=/home/exouser/dyh/dlm_models_20261002/envs/vllm/bin/python
TAG=${TAG:-1}
mkdir -p $C/tmp $W/private
export VLLM_CACHE_ROOT=$C/vllm XDG_CACHE_HOME=$C/xdg TMPDIR=$C/tmp TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton \
  HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 TVM_FFI_CACHE_DIR=$C/tvm CUDA_CACHE_PATH=$C/cuda CUTE_DSL_CACHE_DIR=$C/cute \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$C/facute PYTHONNOUSERSITE=1 PYTHONPATH=src:. V27_ADAPTER_DIR=$W/overlay \
  VLLM_BENCH_BLOCK=32 VLLM_BENCH_MEM=${MEM:-0.85} VLLM_BENCH_BATCHED=16384 VLLM_BENCH_LIMIT=${LIMIT:-2} \
  VLLM_BENCH_PRIVATE=$W/private/bench${TAG}_completions.jsonl
cd $D
for spec in ${ARMS}; do
  arm=${spec%%:*}; cg=${spec#*:}
  until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 20; done
  echo "$arm ${cg:-default} start $(date -u)" >> $W/bench${TAG}_status
  VLLM_BENCH_CG=$cg timeout 14400 $PY $W/v27_vllm_method_bench.py $MODEL $MAN $CELLS $W/smoke_${TAG}.jsonl $arm \
    $([ $arm = method ] && echo $CFG || echo -) ${DS:-longbench_v2_32k} \
    > $W/bench${TAG}_${arm}_${cg:-default}.out 2> $W/bench${TAG}_${arm}_${cg:-default}.err
  echo "$arm ${cg:-default} rc=$? $(date -u)" >> $W/bench${TAG}_status
done
echo "bench done $(date -u)" >> $W/bench${TAG}_status
