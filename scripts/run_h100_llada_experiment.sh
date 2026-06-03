#!/usr/bin/env bash
set -euo pipefail

# H100 真实模型统一入口：
# 1) 用 LLaDA 这类 masked diffusion LM 跑多 block probe。
# 2) 每个 block 的 step 来自模型 confidence，并且 block k 以前面已生成 block 为上下文。
# 3) 再复用 dlm_block_sampling_benchmark.py 生成 summary / prompt_block_steps / plots。

PYTHON=${PYTHON:-python3}
RUN_ID=${RUN_ID:-$(date -u +"%Y%m%d_%H%M%S")}
MODEL=${MODEL:-GSAI-ML/LLaDA-8B-Instruct}
PROMPTS=${PROMPTS:-data/prompts_heterogeneous.jsonl}
OUTDIR=${OUTDIR:-outputs/h100_llada/$RUN_ID}
BATCH_SIZES=${BATCH_SIZES:-${BATCH_SIZE:-"2 4 8 16"}}
BLOCK_SIZES=${BLOCK_SIZES:-${BLOCK_SIZE:-"16 32 64"}}
BATCH_SIZES=$(echo "$BATCH_SIZES" | tr "," " ")
BLOCK_SIZES=$(echo "$BLOCK_SIZES" | tr "," " ")
TRIALS=${TRIALS:-5}
NUM_BLOCKS=${NUM_BLOCKS:-4}
MAX_STEPS=${MAX_STEPS:-${MAX_STEPS_PER_BLOCK:-64}}
CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.90}
ACCEPTANCE_POLICY=${ACCEPTANCE_POLICY:-topk}
DTYPE=${DTYPE:-bf16}
DEVICE_MAP=${DEVICE_MAP:-none}
MASK_TOKEN_ID=${MASK_TOKEN_ID:-126336}

mkdir -p "$OUTDIR"

echo "[h100 runner] real LLaDA H100 data"
echo "[h100 runner] RUN_ID=$RUN_ID OUTDIR=$OUTDIR"
echo "[h100 runner] BATCH_SIZES=$BATCH_SIZES BLOCK_SIZES=$BLOCK_SIZES NUM_BLOCKS=$NUM_BLOCKS MAX_STEPS_PER_BLOCK=$MAX_STEPS CONFIDENCE_THRESHOLD=$CONFIDENCE_THRESHOLD ACCEPTANCE_POLICY=$ACCEPTANCE_POLICY"

$PYTHON scripts/llada_block_step_probe.py \
  --model "$MODEL" \
  --prompts "$PROMPTS" \
  --run-id "$RUN_ID" \
  --batch-sizes $BATCH_SIZES \
  --block-sizes $BLOCK_SIZES \
  --trials "$TRIALS" \
  --num-blocks "$NUM_BLOCKS" \
  --max-steps "$MAX_STEPS" \
  --confidence-threshold "$CONFIDENCE_THRESHOLD" \
  --acceptance-policy "$ACCEPTANCE_POLICY" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  --mask-token-id "$MASK_TOKEN_ID" \
  --out "$OUTDIR/per_request_rows.csv" \
  --block-steps-out "$OUTDIR/block_steps.csv" \
  --batch-block-out "$OUTDIR/batch_block_latency.csv"

echo "[h100 runner] plotting from real LLaDA CSV: $OUTDIR/per_request_rows.csv"
$PYTHON scripts/dlm_block_sampling_benchmark.py \
  --mode plot-csv \
  --input-csv "$OUTDIR/per_request_rows.csv" \
  --outdir "$OUTDIR/plots" \
  --data-label "real LLaDA H100 data"

printf '\nDone. Real-model H100 outputs:\n'
printf '  %s/per_request_rows.csv\n' "$OUTDIR"
printf '  %s/block_steps.csv\n' "$OUTDIR"
printf '  %s/batch_block_latency.csv\n' "$OUTDIR"
printf '  %s/plots/summary.csv\n' "$OUTDIR"
printf '  %s/plots/difficulty_summary.csv\n' "$OUTDIR"
printf '  %s/plots/prompt_block_steps.csv\n' "$OUTDIR"
printf '  %s/plots/exp1_batch_speed_gap.png\n' "$OUTDIR"
printf '  %s/plots/exp2_steps_vs_perf.png\n' "$OUTDIR"
printf '  %s/plots/exp3_block_size_latency.png\n' "$OUTDIR"
printf '  %s/plots/exp4_prompt_difficulty_steps.png\n' "$OUTDIR"
