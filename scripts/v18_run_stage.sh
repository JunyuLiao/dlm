#!/usr/bin/env bash
set -euo pipefail
HOST=$(hostname)
if [ "$HOST" = mpk ]; then
 P=/media/volume/dllm-1/dyh/junyu_frontier_v18_20260926
 PY=/home/exouser/miniconda3/envs/ljy_dlm/bin/python
 MODEL=/home/exouser/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b
 LIB=/home/exouser/dyh/numerical_qk_reuse_native_20260924/build_cp1/value_direction_db080045f7a5fbce.so
 TORCH_LIB=/home/exouser/dyh/numerical_qk_reuse_native_20260924/build_cp1/torch_4c65c048754f9fb7/value_direction_torch_4c65c048754f9fb7.so
 INVENTORY=$P/bridge/tensor_inventory_complete.json
 export PYTHONPATH=src:.
 export TRITON_CACHE_DIR=/media/volume/dllm-1/dyh/numerical_qk_overnight_runtime_20260925/tc_warmup
elif [ "$HOST" = dllm ]; then
 P=/home/exouser/dyh/junyu_frontier_v18_20260926
 R=$P/bridge/runtime
 PY=$R/miniconda3/envs/ljy_dlm/bin/python
 MODEL=$P/bridge/model_complete
 LIB=$P/bridge/binaries/value_direction_db080045f7a5fbce.so
 TORCH_LIB=$P/bridge/binaries/value_direction_torch_4c65c048754f9fb7.so
 INVENTORY=$P/bridge/tensor_inventory_complete.json
 export PYTHONPATH=$R/.local/lib/python3.10/site-packages:src:.
 export PYTHONNOUSERSITE=1
 export LD_LIBRARY_PATH=$P/bridge/binaries:$R/.local/lib/python3.10/site-packages/torch/lib:$R/miniconda3/envs/ljy_dlm/lib
 export TRITON_CACHE_DIR=$P/triton_cache
else exit 4; fi
DRIVER_DEPLOY=$1
STAGE=$2
case "$STAGE" in initial|aime|remainder) ;; *) exit 4;; esac
SEGMENT=${3:-$STAGE}
if [ "$SEGMENT" != "$STAGE" ]; then
 case "$SEGMENT" in aime_c[0-9][0-9][0-9]) [ "$STAGE" = aime ] || exit 4 ;; *) exit 4 ;; esac
fi
BUDGET_NAME=${4:-campaign_budget.json}
case "$BUDGET_NAME" in campaign_budget.json|campaign_budget_extension_20260926.json) ;; *) exit 4 ;; esac
cd "$P/deploy/bridge_json"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=4
mkdir -p "$P/evaluation/ledgers" "$P/evaluation/status" "$P/logs" "$P/private_eval"
for H in mpk dllm; do
 for D in ruler aime; do touch "$P/evaluation/ledgers/${H}_${D}.jsonl"; done
done
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then echo 'GPU process present: stage not launched'; exit 4; fi
if [ "$STAGE" = aime ]; then DATASET=aime; else DATASET=ruler; fi
STATUS="$P/evaluation/status/${HOST}_${SEGMENT}"
if [ -e "$STATUS.started.json" ]; then echo 'Stage marker already exists: inspect before explicit recovery'; exit 4; fi
start=$(date +%s)
printf '{"host":"%s","stage":"%s","pid":%s,"pgid":%s,"start":%s}\n' "$HOST" "$STAGE" "$$" "$(ps -o pgid= -p $$ | tr -d ' ')" "$start" > "$STATUS.started.json"
set +e
"$PY" "$P/deploy/$DRIVER_DEPLOY/scripts/v18_stage_driver.py" --stage "$STAGE" \
 --ruler-protocol "$P/evaluation/ruler4k_primary_protocol.json" \
 --aime-protocol "$P/evaluation/aime26_primary_protocol.json" \
 --budget "$P/deploy/$DRIVER_DEPLOY/results/junyu_frontier_v18_20260926/$BUDGET_NAME" \
 --private "$P/private_eval/$DATASET" --ledger "$P/evaluation/ledgers/${HOST}_${DATASET}.jsonl" \
 --lock "$P/gpu.lock" \
 --ruler-ledger "$P/evaluation/ledgers/mpk_ruler.jsonl" "$P/evaluation/ledgers/dllm_ruler.jsonl" \
 --aime-ledger "$P/evaluation/ledgers/mpk_aime.jsonl" "$P/evaluation/ledgers/dllm_aime.jsonl" \
 --timeout 900 --execute
rc=$?
set -e
end=$(date +%s)
printf '{"host":"%s","stage":"%s","start":%s,"end":%s,"gpu_process_seconds":%s,"rc":%s}\n' "$HOST" "$STAGE" "$start" "$end" "$((end-start))" "$rc" > "$STATUS.done.json"
exit "$rc"
