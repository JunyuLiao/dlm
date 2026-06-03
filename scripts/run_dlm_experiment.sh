#!/usr/bin/env bash
set -euo pipefail

# 一键跑完整模拟实验：
# 1) 用带 difficulty 的 prompt 文件生成 prompt-aware token_steps。
# 2) 输出 per_request_rows.csv / summary.csv / difficulty_summary.csv。
# 3) 如果装了 matplotlib，会输出 exp1~exp4 四张图。

OUTDIR=${OUTDIR:-outputs/prompt_difficulty_demo}
PROMPTS=${PROMPTS:-data/prompts_heterogeneous.jsonl}
PYTHON=${PYTHON:-python3}

$PYTHON scripts/dlm_block_sampling_benchmark.py \
  --mode simulate \
  --prompt-file "$PROMPTS" \
  --batch-sizes 2 4 8 16 \
  --block-sizes 16 32 64 \
  --schedulers sync dynamic \
  --trials 50 \
  --outdir "$OUTDIR"

printf '\nDone. Key outputs:\n'
printf '  %s/per_request_rows.csv\n' "$OUTDIR"
printf '  %s/summary.csv\n' "$OUTDIR"
printf '  %s/difficulty_summary.csv\n' "$OUTDIR"
printf '  %s/prompt_block_steps.csv\n' "$OUTDIR"
printf '  %s/exp1_batch_speed_gap.png\n' "$OUTDIR"
printf '  %s/exp2_steps_vs_perf.png\n' "$OUTDIR"
printf '  %s/exp3_block_size_latency.png\n' "$OUTDIR"
printf '  %s/exp4_prompt_difficulty_steps.png\n' "$OUTDIR"
