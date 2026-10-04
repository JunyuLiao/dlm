#!/bin/bash
# Screening round S1, short-prompt part S1s (EXPLORATORY; overlay ov_pa3), after sc1: the same seven arms on AIME26 (all
# 30 problems) and HumanEval (40 of 164; the other 124 held out), thinking on, 8192-token budget, sampler seeds 1 and 2
# (v31_screen_suite.py --short, sc_suite_short.json; pool manifests_ae, pinned max_model_len 13312). Prompts are short:
# the sparse path acts on the model's own reasoning once the context passes the token budget, so this part checks that
# the method does not hurt short-prompt tasks (accuracy only; no speed headroom at <= 8.4K tokens).
# env: W PY MODEL ROOT SHARD
set -u
cd $W
until grep -q '^done ' $W/status_sc1chain 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
O=$W/ov_pa3; S=$W/sc1
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
LEAN="MAGE_GRAN=qblock_max MAGE_STEP=1 MAGE_CARRY=1"
arm() {  # NAME BUDGET(dense|tokens) EXTRA...: one arm over the four parts
  local L=$1 K=$2; shift 2
  for part in aime he; do
    case $part in
      aime) D=aime26 ;;
      he)   D=humaneval ;;
    esac
    M=$W/manifests_ae; X=13312
    B="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$SHARD FIX_51994=1 MEM=0.90 OVERLAY=$O BENCH=$W/bench_ov_pa3.py
      TAG=sc1$part DATASETS=$D CELLS=$S/cells_sc_$part.json MAN_DIR=$M MAX_MODEL_LEN=$X"
    if [ $K = dense ]; then
      env $B ARMS='dense:default' bash $W/v31_paired_host8.sh > $W/host_sc1${part}_$K.log 2>&1
    else
      env $B $FAST $LEAN "$@" LABEL_SUFFIX=_lean$L ARMS="mage:PIECEWISE:$K" bash $W/v31_paired_host8.sh \
        > $W/host_sc1${part}_$K$L.log 2>&1
    fi
    sed -i "s/^done /done_$K$L /" $W/status_sc1$part
  done
  echo "$K$L complete $(date -u)" >> $W/status_sc1schain
}
arm '' dense
arm '' 4096
arm _t50_cgate 4096 MAGE_RESELECT_TRIGGER=0.5 MAGE_ROWW=cgate
arm _t50_cgate_k2048 4096 MAGE_RESELECT_TRIGGER=0.5 MAGE_ROWW=cgate MAGE_RESELECT_K=2048
arm _t50_cgate_k2048 8192 MAGE_RESELECT_TRIGGER=0.5 MAGE_ROWW=cgate MAGE_RESELECT_K=2048
arm _t50_margin 4096 MAGE_RESELECT_TRIGGER=0.5 MAGE_ROWW=margin
arm '' 8192
echo "done $(date -u)" >> $W/status_sc1schain
