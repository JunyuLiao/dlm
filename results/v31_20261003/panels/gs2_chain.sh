#!/bin/bash
# Panel gs2 (overlay ov_gs; EXPLORATORY), after panel gs: the group-shared decision inside the method itself.
#  RISK_GROUP=kv on m2c k12 mass (fixed 12% per unit, mass ranking, re-decision every 6 steps, carry_first): one top-k
#  per (KV head, 128-row block) on the max of the group's worst-row values -- same tiles per head, identical lists.
#  REVISED 10:25 UTC after panel gs: on the MAGE port the group-shared unit lost 4.3 cwe points (0.34 official) to
#  the per-head unit at the same work and saved nothing in time (held call 0.253 vs 0.257 ms at equal counts), so the
#  RISK_GROUP=kv arms are dropped. What remains: the per-head fixed-fraction method m2c k12 mass (equal counts per
#  (head, block), i.e. balanced CTAs) --
#  (1) per-call GPU cost (PROFILE_MODE=events, 3 LongBench-v2 64K cells);
#  (2) end to end on the LongBench-v2 64K + 96K confirmation cells (dense reference from gs).
# Every arm waits for an empty GPU (v31_paired_host8.sh). env: W PY MODEL ROOT RSHARD LBSHARD
set -u
cd $W
until grep -q '^done ' $W/status_gs 2>/dev/null; do sleep 30; done
if grep -q 'tests FAILED' $W/status_gs; then echo "skipped: gs tests failed $(date -u)" >> $W/status_gs2; echo "done $(date -u)" >> $W/status_gs2; exit 1; fi
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 10; done
O=$W/ov_gs
FAST="LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4"
LB="W=$W PY=$PY MODEL=$MODEL ROOT=$ROOT SHARD=$LBSHARD FIX_51994=1 MEM=0.90 CELLS=$W/cells_confirm.json OVERLAY=$O"
env $LB $FAST TAG=gs2p DATASETS=longbench_v2_64k LIMIT=3 BENCH=$O/v31_step_profile.py PROFILE_MODE=events   PROFILE_OUT=$W/public/gs2p_profile_method_m2c_k12_mass_ctl.jsonl LABEL_SUFFIX=_ctl ARMS='method:PIECEWISE:m2c_k12_mass'   bash $W/v31_paired_host8.sh > $W/host_gs2p_ctl.log 2>&1
sed -i "s/^done /done_ctl /" $W/status_gs2p
env $LB $FAST TAG=gs2l DATASETS=longbench_v2_64k,longbench_v2_96k BENCH=$W/bench_ov_gs.py LABEL_SUFFIX=_ctl   ARMS='method:PIECEWISE:m2c_k12_mass' bash $W/v31_paired_host8.sh > $W/host_gs2l_ctl.log 2>&1
sed -i "s/^done /done_ctl /" $W/status_gs2l
echo "done $(date -u)" >> $W/status_gs2
