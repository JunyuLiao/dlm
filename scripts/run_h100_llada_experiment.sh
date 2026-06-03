#!/usr/bin/env bash
set -euo pipefail

# H100 真实模型统一入口：
# 1) 用 LLaDA 这类 masked diffusion LM 跑多 block probe。
# 2) 每个 block 的 step 来自模型 confidence，并且 block k 以前面已生成 block 为上下文。
# 3) 再复用 dlm_block_sampling_benchmark.py 生成 summary / prompt_block_steps / plots。

PYTHON=${PYTHON:-python3}
MODEL=${MODEL:-GSAI-ML/LLaDA-8B-Instruct}
PROMPTS=${PROMPTS:-data/prompts_heterogeneous.jsonl}
OUTDIR=${OUTDIR:-outputs/h100_llada}
BATCH_SIZES=${BATCH_SIZES:-"2 4 8 16"}
BLOCK_SIZES=${BLOCK_SIZES:-"16 32 64"}
TRIALS=${TRIALS:-5}
NUM_BLOCKS=${NUM_BLOCKS:-4}
MAX_STEPS=${MAX_STEPS:-64}
CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.90}
DTYPE=${DTYPE:-bf16}
DEVICE_MAP=${DEVICE_MAP:-none}
MASK_TOKEN_ID=${MASK_TOKEN_ID:-126336}

mkdir -p "$OUTDIR"

$PYTHON scripts/llada_block_step_probe.py \
  --model "$MODEL" \
  --prompts "$PROMPTS" \
  --batch-sizes $BATCH_SIZES \
  --block-sizes $BLOCK_SIZES \
  --trials "$TRIALS" \
  --num-blocks "$NUM_BLOCKS" \
  --max-steps "$MAX_STEPS" \
  --confidence-threshold "$CONFIDENCE_THRESHOLD" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  --mask-token-id "$MASK_TOKEN_ID" \
  --out "$OUTDIR/per_request_rows.csv" \
  --block-steps-out "$OUTDIR/block_steps.csv"

$PYTHON scripts/dlm_block_sampling_benchmark.py \
  --mode plot-csv \
  --input-csv "$OUTDIR/per_request_rows.csv" \
  --outdir "$OUTDIR/plots"

printf '\nDone. Real-model H100 outputs:\n'
printf '  %s/per_request_rows.csv\n' "$OUTDIR"
printf '  %s/block_steps.csv\n' "$OUTDIR"
printf '  %s/plots/summary.csv\n' "$OUTDIR"
printf '  %s/plots/difficulty_summary.csv\n' "$OUTDIR"
printf '  %s/plots/prompt_block_steps.csv\n' "$OUTDIR"
printf '  %s/plots/exp1_batch_speed_gap.png\n' "$OUTDIR"
printf '  %s/plots/exp2_steps_vs_perf.png\n' "$OUTDIR"
printf '  %s/plots/exp3_block_size_latency.png\n' "$OUTDIR"
printf '  %s/plots/exp4_prompt_difficulty_steps.png\n' "$OUTDIR"
