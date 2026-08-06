#!/usr/bin/env bash
set -euo pipefail

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1

MODEL_PATH=${MODEL_PATH:-inclusionAI/LLaDA2.1-mini}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/llada2_1_mini}
GPUS=${GPUS:-0;1;2;3}
TP_SIZE=${TP_SIZE:-4}
GEN_LENGTH=${GEN_LENGTH:-2048}
BLOCK_LENGTH=${BLOCK_LENGTH:-32}
THRESHOLD=${THRESHOLD:-0.5}
EDITING_THRESHOLD=${EDITING_THRESHOLD:-0.0}
MAX_POST_STEPS=${MAX_POST_STEPS:-16}

for task in gsm8k_llada_mini mbpp_sanitized_llada_mini; do
  output_path=${OUTPUT_DIR}/${task}
  python eval_dinfer_sglang.py \
    --tasks "${task}" \
    --confirm_run_unsafe_code \
    --model dInfer_eval \
    --model_args "model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},editing_threshold=${EDITING_THRESHOLD},max_post_steps=${MAX_POST_STEPS},show_speed=True,save_dir=${output_path},parallel_decoding=threshold,cache=prefix,warmup_times=0,use_compile=True,tp_size=${TP_SIZE},parallel=tp,cont_weight=0,use_credit=False,prefix_look=0,after_look=0,gpus=${GPUS},model_type=llada2.1-mini,use_bd=True,save_samples=False" \
    --output_path "${output_path}" \
    --include_path ./tasks \
    --apply_chat_template
done
