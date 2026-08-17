# DiffusionGemma BLASST MATH500 evaluation

This benchmark follows the reasoning-task protocol in BLASST (arXiv
2512.12087): NeMo-Skills prompt construction and symbolic math grading,
temperature 0.6, top-p 0.95, ten independent generations for each of the 500
problems, and majority/self-consistency scoring. It applies local/global BLASST
thresholds 0.9/0.6 to DiffusionGemma decoder attention and reports physical
tile sparsity separately for native sliding and full-attention layers.

DiffusionGemma's optional thinking control is disabled and generation is capped
at 2,048 tokens (eight native 256-token canvases), with normal EOS early
stopping. This is a model-specific formatting choice: in validation, thinking
mode ignored NeMo-Skills' boxed-answer contract and produced no extractable
answer, whereas standard mode produced a complete reasoned solution followed
by the correct `\boxed{}` response. NeMo-Skills' zero-shot math prompt and
strict symbolic scorer are unchanged.

The NeMo-Skills source and prepared dataset are intentionally external and
pinned in `run_config.json`. Prepare them first:

```bash
git clone https://github.com/NVIDIA-NeMo/Skills.git /home/exouser/ljy/Skills
cd /home/exouser/ljy/Skills
python nemo_skills/dataset/math-500/prepare.py
```

Run the full, resumable experiment:

```bash
PYTHONPATH=/home/exouser/ljy/dlm/src \
HF_HUB_OFFLINE=1 \
/home/exouser/miniconda3/envs/ljy_dlm/bin/python \
  benchmarks/diffusion_gemma_math500/run_experiment.py \
  --nemo-skills-root /home/exouser/ljy/Skills \
  --output-dir results/diffusion_gemma_math500_blasst_l0p9_g0p6
```

Run the matched dense baseline by adding `--dense-baseline` and using a fresh
output directory. This bypasses BLASST installation entirely while preserving
the model revision, prompts, seeds, sampling configuration, and aggregation:

```bash
PYTHONPATH=/home/exouser/ljy/dlm/src \
HF_HUB_OFFLINE=1 \
/home/exouser/miniconda3/envs/ljy_dlm/bin/python \
  benchmarks/diffusion_gemma_math500/run_experiment.py \
  --nemo-skills-root /home/exouser/ljy/Skills \
  --output-dir results/diffusion_gemma_math500_dense \
  --dense-baseline
```

`predictions.jsonl` is fsync'd after each generation. Re-running the identical
command resumes by problem and sample index. `summary.json`,
`self_consistency.jsonl`, and `report.md` are written after all 5,000
generations finish.

The attention backend is a correctness/reference implementation: it applies
the BLASST mask and counts physically skippable QxKV tiles, but materializes
dense QK scores. Its sparsity is meaningful; its wall time is not an optimized
sparse-kernel speedup measurement.

## Fixed bottom-k block-maximum study

`run_blockmax_quantiles.py` implements the supervisor-proposed policy: for
every layer/head/query row it computes dense QK, ranks 64-token KV tiles by
block maximum, and drops the lowest k% for k = 0, 25, 50, 75, 90, and 95. It
uses a deterministic 50-problem sample from NeMo-Gym MATH-500 and ten paired
temperature-0.6/top-p-0.95 rollouts per problem:

```bash
PYTHONPATH=/home/exouser/ljy/dlm/src:/home/exouser/ljy/dlm \
HF_HUB_OFFLINE=1 \
/home/exouser/miniconda3/envs/ljy_dlm/bin/python \
  benchmarks/diffusion_gemma_math500/run_blockmax_quantiles.py \
  --nemo-gym-root /tmp/nemo-gym-math500 \
  --output-dir results/diffusion_gemma_math500_blockmax_k50
```

The run is resumable after every generation. The primary reported score is
NeMo-Gym majority@10 symbolic accuracy; pass@1 (average over ten rollouts) and
oracle pass@10 are secondary diagnostics.
