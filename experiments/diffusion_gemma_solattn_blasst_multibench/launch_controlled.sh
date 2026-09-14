#!/usr/bin/env bash
# Run from the repository root. Resume skips every completed immutable shard.
set -uo pipefail
export PYTHONPATH=src:.
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
python_bin=/home/exouser/miniconda3/envs/ljy_dlm/bin/python
experiment_module=experiments.diffusion_gemma_solattn_blasst_multibench
"$python_bin" -m "$experiment_module" prepare || exit 1
"$python_bin" -m "$experiment_module" smoke || exit 1
"$python_bin" -m "$experiment_module" run
"$python_bin" -m "$experiment_module" grade-livecodebench
"$python_bin" -m "$experiment_module" report
