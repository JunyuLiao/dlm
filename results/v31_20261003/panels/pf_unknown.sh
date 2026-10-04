#!/bin/bash
# Profile of the context-dependent non-GLOBAL step cost (feasibility study, branch research/v31-scale-feasibility-20261004:
# the dense step grows 0.305 ms per 1K prompt tokens, the five GLOBAL calls explain 0.128). nsys CUDA trace with
# per-node CUDA-graph kernels of the official dense path (vLLM FULL + PR #51994 fix) on three RULER v34 items of one task
# (niah_single_1, 32K / 64K / 128K, seed 1), then the per-kernel GPU trace as CSV for the analysis. mpk only (nsys).
# env: W PY MODEL ROOT P
set -u
C=$W/cache; D=$ROOT/deploy/v27_r17_6064109; O=$W/ov_pa4; mkdir -p $W/nsys $C/tmp
python3 - <<PYEOF
import json
c = json.load(open('$W/sc1/cells_sc_ruler.json'))
sel = [x for x in c if x['seed'] == 1 and x['id'].endswith('niah_single_1_p0000')]
json.dump(sel, open('$W/nsys/pfu_cells.json', 'w'), indent=1)
print(len(sel))
PYEOF
export VLLM_CACHE_ROOT=$C/vllm XDG_CACHE_HOME=$C/xdg TMPDIR=$C/tmp TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton \
  HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 TVM_FFI_CACHE_DIR=$C/tvm CUDA_CACHE_PATH=$C/cuda CUTE_DSL_CACHE_DIR=$C/cute \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$C/facute PYTHONNOUSERSITE=1 PYTHONPATH=src:. V27_ADAPTER_DIR=$O \
  DATASETS=ruler32k_v34ofc,ruler64k_v34ofc,ruler128k_v34ofc MEM=0.90 REPEATS=1 FIX_51994=1 MAX_MODEL_LEN=136192
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
cd $D
timeout 3600 /usr/local/bin/nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --force-overwrite=true -o $W/nsys/pfu_dense $PY $W/bench_ov_pa4.py $MODEL $P/ruler_v34ofc $W/nsys/pfu_cells.json \
  $W/nsys/pfu_dense.jsonl $W/nsys/pfu_dense.private.jsonl dense default > $W/nsys/pfu_dense.out 2> $W/nsys/pfu_dense.err
echo "pfu nsys rc=$? $(date -u)" >> $W/status_sc1xchain
/usr/local/bin/nsys stats --report cuda_gpu_trace --format csv --output $W/nsys/pfu_trace $W/nsys/pfu_dense.nsys-rep > $W/nsys/pfu_stats.log 2>&1
echo "pfu stats rc=$? $(date -u)" >> $W/status_sc1xchain
