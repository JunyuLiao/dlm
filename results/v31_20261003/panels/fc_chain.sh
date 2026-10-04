#!/bin/bash
# Panel fc (branch research/v31-forced-canvas-20261004; bench bench_ov_fc.py on overlay ov_gs; EXPLORATORY), after gs2:
# step inflation and fidelity on the SAME canvases. LongBench-v2 64K + 96K confirmation cells (pf's shards).
#  (1) reference: dense PIECEWISE with per-canvas reseeding, its committed tokens recorded (private);
#  (2) self-check: the same configuration forced with its own record (must reproduce every step count, agree 1.0);
#  (3) calibration: dense FULL (default) forced with the PIECEWISE record -- the execution-mode noise floor;
#  (4) arms forced with the same record: MAGE k=4096, m2c, m2c k12 mass;
#  (5) m2c + the step-20 dense rescue (DENSE_WHEN=step:20; Junyu's dense-rescue idea, collaboration candidate) and
#      the MAGE-port per-head unit (qblock_max, 12%, selection at step 1) without and with MAGE_CARRY: RULER answers are
#      one canvas long, so panel gs could not see the carry; here every canvas after the first uses it.
# Timing fields of these records are not valid (the mode synchronizes). Every arm waits for an empty GPU.
# env: W PY MODEL ROOT LBSHARD
set -u
cd $W
until grep -q '^done ' $W/status_gb 2>/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
REF=$W/private/fc_dense_PIECEWISE_ref.tokens.jsonl
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
LB="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$LBSHARD FIX_51994=1 MEM=0.90 CELLS=$W/cells_confirm.json TAG=fc
  DATASETS=longbench_v2_64k,longbench_v2_96k BENCH=$W/bench_ov_fc.py OVERLAY=$W/ov_gs"
if [ ! -s $REF ]; then
  env $LB FORCE_RECORD=$REF LABEL_SUFFIX=_ref ARMS='dense:PIECEWISE' bash $W/v31_paired_host8.sh > $W/host_fc_ref.log 2>&1
  sed -i "s/^done /done_ref /" $W/status_fc
fi
if [ ! -s $REF ]; then echo "ABORT: empty reference record $REF $(date -u)" >> $W/status_fc; echo "done $(date -u)" >> $W/status_fc; exit 1; fi
env $LB FORCE_REF=$REF LABEL_SUFFIX=_self ARMS='dense:PIECEWISE' bash $W/v31_paired_host8.sh > $W/host_fc_self.log 2>&1
sed -i "s/^done /done_self /" $W/status_fc
env $LB FORCE_REF=$REF LABEL_SUFFIX=_forced ARMS='dense:default' bash $W/v31_paired_host8.sh > $W/host_fc_full.log 2>&1
sed -i "s/^done /done_full /" $W/status_fc
env $LB $FAST FORCE_REF=$REF LABEL_SUFFIX=_forced \
  ARMS='mage:PIECEWISE:4096 method:PIECEWISE:m2c method:PIECEWISE:m2c_k12_mass' \
  bash $W/v31_paired_host8.sh > $W/host_fc_arms.log 2>&1
sed -i "s/^done /done_arms /" $W/status_fc
env $LB $FAST FORCE_REF=$REF DENSE_WHEN=step:20 LABEL_SUFFIX=_forced_s20 ARMS='method:PIECEWISE:m2c'   bash $W/v31_paired_host8.sh > $W/host_fc_s20.log 2>&1
sed -i "s/^done /done_s20 /" $W/status_fc
for c in 0 1; do
  env $LB $FAST FORCE_REF=$REF MAGE_FRAC=0.12 MAGE_GRAN=qblock_max MAGE_STEP=1 MAGE_CARRY=$c     LABEL_SUFFIX=_forced_qbm_step1_carry$c ARMS='mage:PIECEWISE:0' bash $W/v31_paired_host8.sh > $W/host_fc_carry$c.log 2>&1
  sed -i "s/^done /done_carry$c /" $W/status_fc
done
echo "done $(date -u)" >> $W/status_fc
