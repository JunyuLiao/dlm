# DiffusionGemma Sol-Attn vs. BLASST on RULER16K

## Whole-tile execution update

New BLASST runs use `blasst_mask_semantics: physical_tile_v1`. A tile is
skipped only when every valid active query row votes to skip. Otherwise all
structurally valid positions are retained. Row votes are diagnostics, not
an extra execution mask. Valid-QK-element sparsity now counts elements in
wholly skipped tiles, including partial/structurally masked tile sizes.

Use a **fresh output directory** for corrected experiments, e.g.
`results/diffusion_gemma_solattn_vs_blasst_ruler16k_physical_tile_v1/`, instead
of the historical paths in the original command examples below. Old prediction
shards, statistics, and policy files are protected from silent overwrite/resume.
Reports label the recorded semantics and reject mixed old/new BLASST conditions.
Historical results must not be interpreted as corrected-method accuracy.

Dense margin traces remain reusable as read-only inputs: offline evaluation
now derives physical element masks from the row votes. Legacy policies fitted
to **physical tile sparsity** remain compatible with that unchanged routing
decision; policies fitted to old row-element sparsity require recalibration.
Write regenerated policies to new paths.

Inactive queries do not vote or enter statistics. They nevertheless obey the
physical execution mask. If an inactive query loses its last valid key, its
attention output is zero—not a softmax of all negative infinities. Such empty
rows must be checked using the **post-routing** validity mask.

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
