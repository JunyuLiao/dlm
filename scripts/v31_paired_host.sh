#!/bin/bash
# Run the seed-paired vLLM panel (scripts/v31_vllm_paired_bench.py) on one host: for each entry of $ARMS, one fresh
# engine over this host's shard of the cells. Every arm of a cell runs on the same host. Usage on the host:
#   W=<dyh run dir> PY=<vllm env python> MODEL=<model dir> ROOT=<m3_output_numerics_v21 root> SHARD=k/3 \
#   ARMS='dense:default dense:PIECEWISE method:PIECEWISE method:PIECEWISE:base_cgate' \
#   DATASETS=longbench_v2_32k,longbench_v2_64k MEM=0.90 TAG=a [FIX_51994=1] [WAIT_FOR=<status file>] \
#   nohup bash v31_paired_host.sh > host_a.log 2>&1 &
# ARMS entries: arm:cudagraph_mode[:variant]; a variant names configs/<variant>.json (v31_make_variant_config.py),
# default = the frozen R17 main config. FIX_51994=1 backports the upstream FULL-graph causal-buffer fix.
# MAN_DIR overrides the generation-manifest directory (e.g. the v31 RULER pool). CELLS overrides the cell file (default $W/cells.json; e.g. the confirmation set). LABEL_SUFFIX names an execution variant (e.g. _dpc for DP_BUILD=chunked); LOGIT_STATS / DP_BUILD / OBSERVE /
# MAGE_SELECT pass through the environment to the bench.
# W must contain v31_vllm_paired_bench.py, overlay/vllm_adapter.py and cells.json (private: has item ids).
set -u
C=$W/cache
BENCH=${BENCH:-$W/v31_vllm_paired_bench.py}; OVERLAY=${OVERLAY:-$W/overlay}
D=$ROOT/deploy/v27_r17_6064109
CFG=$ROOT/v23/v27_r17_001/host.fragment_configs/ruler64k_r17/M3_R6_A64_fused_dp_async_m1ln2_c0_fa4.json
MAN=${MAN_DIR:-$ROOT/v23/v27_lb_e14_001/manifests}
mkdir -p $C/tmp $W/private $W/public
export VLLM_CACHE_ROOT=$C/vllm XDG_CACHE_HOME=$C/xdg TMPDIR=$C/tmp TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton \
  HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 TVM_FFI_CACHE_DIR=$C/tvm CUDA_CACHE_PATH=$C/cuda CUTE_DSL_CACHE_DIR=$C/cute \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$C/facute PYTHONNOUSERSITE=1 PYTHONPATH=src:. V27_ADAPTER_DIR=$OVERLAY \
  SHARD DATASETS MEM REPEATS=${REPEATS:-1} FIX_51994=${FIX_51994:-0}
if [ -n "${WAIT_FOR:-}" ]; then until grep -q '^done' "$WAIT_FOR" 2>/dev/null; do sleep 30; done; fi
cd $D
for spec in $ARMS; do
  arm=${spec%%:*}; rest=${spec#*:}; cg=${rest%%:*}; name=main
  [ "$rest" != "$cg" ] && name=${rest#*:}
  cfg=$CFG; [ "$name" != main ] && cfg=$W/configs/$name.json
  label=${arm}_${cg}; [ "$arm" = method ] && label=${arm}_${name}_${cg}
  mk=1024; [ "$arm" = mage ] && [ "$name" != main ] && mk=$name && label=${arm}${mk}_${cg}
  [ "$FIX_51994" = 1 ] && label=${label}_fix
  label=${label}${LABEL_SUFFIX:-}
  until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 20; done
  echo "$label start $(date -u)" >> $W/status_$TAG
  MAGE_K=$mk timeout 21600 $PY $BENCH $MODEL $MAN ${CELLS:-$W/cells.json} $W/public/${TAG}_${label}.jsonl \
    $W/private/${TAG}_${label}.private.jsonl $arm $cg $([ $arm = method ] && echo $cfg) \
    > $W/${TAG}_${label}.out 2> $W/${TAG}_${label}.err
  echo "$label rc=$? $(date -u)" >> $W/status_$TAG
done
echo "done $(date -u)" >> $W/status_$TAG
