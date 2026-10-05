#!/bin/bash
# Panel rc (mpk; dense FULL + fix only; official-protocol bench bench_ov_ofc.py = research/v31-official-protocol-20261004
# @ 3ce93b26f, records bound to pool / prompt / budget / pinned max_model_len), after fc:
#  (1) rc: GPU confirmation of the official RULER rendering (ruler_v33ofc: model turn, prefilled empty thought block,
#      RULER's answer prefix verbatim) -- one cell per task at 32K and 128K (26 cells, cells_rc_ofc.json, private);
#  (2) noop / noopref: opener ablation -- vt + the 8 niah tasks at 32K (135 cells) rendered WITHOUT the empty thought
#      block (ruler_v33noop) vs the same ids in the official pool;
#  (3) gwpilot: GraphWalks 90k bin, dense, 16384-token budget: the cap rate before the budget is frozen;
#  (4) dp: determinism probe for the MAGE step-1 selection path (panel gs control flipped 1/174 cells): the same
#      control twice more on the mpk shard.
# env: W PY MODEL ROOT
set -u
cd $W
until grep -q '^done ' $W/status_fc 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
P=/media/volume/dllm-1/dyh/pools_v31_official
D="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT FIX_51994=1 MEM=0.90 BENCH=$W/bench_ov_ofc.py OVERLAY=$W/ov_gs ARMS=dense:default"
env $D TAG=rc CELLS=$W/cells_rc_ofc.json MAN_DIR=$P/ruler_v33ofc DATASETS=ruler32k_v33ofc,ruler128k_v33ofc \
  MAX_MODEL_LEN=136192 bash $W/v31_paired_host8.sh > $W/host_rc.log 2>&1
env $D TAG=noop CELLS=$P/ruler_v33noop/cells_ruler_v33noop.json MAN_DIR=$P/ruler_v33noop DATASETS=ruler32k_v33noop \
  MAX_MODEL_LEN=136192 bash $W/v31_paired_host8.sh > $W/host_noop.log 2>&1
env $D TAG=noopref CELLS=$P/ruler_v33ofc/cells_ruler32k_v33ofc_vt_niah.json MAN_DIR=$P/ruler_v33ofc DATASETS=ruler32k_v33ofc \
  MAX_MODEL_LEN=136192 bash $W/v31_paired_host8.sh > $W/host_noopref.log 2>&1
env $D TAG=gwpilot CELLS=$P/graphwalks_b16k/cells_graphwalks_90k_b16k_pilot.json MAN_DIR=$P/graphwalks_b16k \
  DATASETS=graphwalks_90k_b16k MAX_MODEL_LEN=110592 bash $W/v31_paired_host8.sh > $W/host_gwpilot.log 2>&1
for rep in a b; do
  env W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=1/3 DATASETS=ruler32k,ruler64k MEM=0.90 TAG=dp FIX_51994=1 \
    CELLS=$W/cells_ruler31.json MAN_DIR=$W/manifests_ruler31 BENCH=$W/bench_ov_gs.py OVERLAY=$W/ov_gs \
    LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4 \
    MAGE_FRAC=0.12 MAGE_GRAN=qblock_max MAGE_STEP=1 LABEL_SUFFIX=_frac12_qblock_max_step1_$rep ARMS='mage:PIECEWISE:0' \
    bash $W/v31_paired_host8.sh > $W/host_dp_$rep.log 2>&1
  sed -i "s/^done /done_$rep /" $W/status_dp
done
echo "done $(date -u)" >> $W/status_rc
