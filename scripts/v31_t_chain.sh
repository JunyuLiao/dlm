#!/bin/bash
# Panel t (overlay12): RULER accuracy at MATCHED kept fractions, to separate "selects better" from "keeps more":
# MAGE k=6144 (~19% / 9.4% of prefix tiles at 32K / 64K, matching m2c's realized 17% / 9.5%) and the method's
# fixed-budget k12 (risk and mass ranking; 12% per head and block, matching MAGE k=4096 at 32K). After panel s.
set -u
O=$W/overlay12
until grep -q '^done ' $W/status_s 2>/dev/null; do sleep 20; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
cd $W
env W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD DATASETS=ruler32k,ruler64k MEM=0.90 TAG=t FIX_51994=1 \
  CELLS=$W/cells_ruler31.json MAN_DIR=$W/manifests_ruler31 LABEL_SUFFIX=_fast \
  LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4 \
  BENCH=$W/v31_vllm_paired_bench12.py OVERLAY=$O \
  ARMS='mage:PIECEWISE:6144 method:PIECEWISE:m2c_k12 method:PIECEWISE:m2c_k12_mass' \
  bash $W/v31_paired_host8.sh > $W/host_t.log 2>&1
