# DiffusionGemma Sol-Attn vs. BLASST on RULER16K

This package is an isolated, resumable reference-mask experiment.  It uses
the five default pinned NVIDIA RULER tasks, a disjoint 2-example-per-task
calibration set, and a 10-example-per-task final set.

The public sparsity value is the target fraction of skipped 64×64 tiles.  The
Sol-Attn-like route uses one analytic Gaussian beta per target:

| target skipped | beta |
|---:|---:|
| 25% | -0.674490 |
| 50% | 0 |
| 75% | 0.674490 |
| 90% | 1.281552 |

BLASST calibration records dense block-max margins once, evaluates the
declared lambda grid offline, fits `lambda * L = alpha * exp(gamma * s)`, and
exports separate local/global lambdas.  The final set is never used to choose
thresholds.

Typical flow:

```bash
PYTHONPATH=src python -m experiments.diffusion_gemma_solattn_vs_blasst_ruler16k prepare \
  --raw-root results/blasst/fast_dllm_v2/ruler/ruler_cache/official_raw/16384 \
  --ruler-root /path/to/RULER --model-path /path/to/diffusiongemma \
  --revision PINNED_REVISION

PYTHONPATH=src python -m experiments.diffusion_gemma_solattn_vs_blasst_ruler16k calibrate-blasst \
  --study-manifest results/diffusion_gemma_solattn_vs_blasst_ruler16k/manifest.json \
  --model-path /path/to/diffusiongemma --ruler-root /path/to/RULER \
  --output-dir results/diffusion_gemma_solattn_vs_blasst_ruler16k/calibration

PYTHONPATH=src python -m experiments.diffusion_gemma_solattn_vs_blasst_ruler16k run \
  --study-manifest results/diffusion_gemma_solattn_vs_blasst_ruler16k/manifest.json \
  --threshold-policy results/diffusion_gemma_solattn_vs_blasst_ruler16k/calibration/blasst_policy.json \
  --model-path /path/to/diffusiongemma --ruler-root /path/to/RULER

PYTHONPATH=src python -m experiments.diffusion_gemma_solattn_vs_blasst_ruler16k report
```

The reference implementation reports full-tile sparsity separately from
BLASST valid-QK-element sparsity.  It makes no latency, throughput, or
speedup claim.
