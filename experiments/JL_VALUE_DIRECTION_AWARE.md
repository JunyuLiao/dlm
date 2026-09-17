# Value-direction-aware sparse attention

This experiment series adds attention-weighted value-direction information to
DiffusionGemma physical-tile selection. The numerical implementations and
versioned evaluation recipes are preserved unchanged from their audited runs.

## Implementation

The primary router forms FP32 token-value sketches with fixed, hashed random
projections. It computes each candidate block's attention-weighted projected
mean, then estimates the centered change from adding that block to the output
of previously retained blocks. The change is normalized by the existing valid-KV
RMS scale. A physical 128-query by 64-KV tile is skipped only when every valid
query's score is strictly below its attention-type-specific threshold.

First-support blocks and threshold ties are retained. Skipping leaves the
retained running state unchanged. Structural masks, GQA, prefix/canvas
eligibility and ordinary attention renormalization are preserved. Actual
attention uses original values, not sketches. Full-dimensional block PV is not
used to construct projected routing scores; it is reserved for the explicit
full-dimensional control and diagnostics.

- [`diffusion_gemma_jl_output_aware/reference.py`](diffusion_gemma_jl_output_aware/reference.py): trusted online routing reference.
- [`diffusion_gemma_jl_output_aware/projections.py`](diffusion_gemma_jl_output_aware/projections.py): deterministic projection matrices, hashes and value-checked sketch reuse.
- [`diffusion_gemma_jl_output_aware/kernels.py`](diffusion_gemma_jl_output_aware/kernels.py): GPU block statistics and physical-tile routing.
- [`diffusion_gemma_jl_output_aware/routing.py`](diffusion_gemma_jl_output_aware/routing.py): native attention integration and numerical boundary checks.
- [`diffusion_gemma_jl_output_aware/shared_analysis.py`](diffusion_gemma_jl_output_aware/shared_analysis.py): shared-QKV output error, risk underestimation and overlapping block-type diagnostics.

The package also preserves uncentered projected-contribution, exact-mass and
type-aware controls from earlier experiment versions. Their presence does not
mean every control completed every benchmark. Consult each version's audit.
Cached `mass` baselines use the prior max-based mass bound, not `mass_exact`.

## Completed studies

| Version | Scope | Report |
| --- | --- | --- |
| v6 | Directional 50% comparisons, AIME30 and first LongBench50 | [report](../results/diffusion_gemma_jl_directional50_v6/report.md) |
| v7 | Other 50 cached LongBench questions, frozen policies | [report](../results/diffusion_gemma_jl_longbench_remaining50_v7/report.md) |
| v8 | AIME Gaussian dimensions 8/16/24 versus 32/full | [report](../results/diffusion_gemma_jl_aime_gaussian_dimensions_v8/report.md) |
| v9 | AIME Gaussian dimensions 2/4 added | [report](../results/diffusion_gemma_jl_aime_gaussian_tiny_dimensions_v9/report.md) |
| v11 | AIME dimension1 and LongBench100 dimensions1/2/8, with cached comparisons | [report](../results/diffusion_gemma_jl_lowrank_aime30_longbench100_v11/report.md) |
| v12 | AIME seed43: dense/aggressive BLASST/mass/full/Gaussian8 | [report](../results/diffusion_gemma_jl_aime_seed43_v12/report.md) |

The v11 bundle contains the consolidated seed42 sweep: 1,260 audited outputs.
The v12 bundle contains 150 fresh outputs with unchanged thresholds and
projection seed1729; only generation seed changes from42 to43. All sparse
conditions in these two bundles request50% physical sparsity. Achieved sparsity
is measured, not assumed. Version10 is a preserved failed CPU preflight; use
the corrected version11 driver for the completed low-rank study.

## Measurement and scope

Physical sparsity is summed skipped eligible tiles divided by summed eligible
tiles, separately whole/global/local. Retained attention mass and execution-local
output error use dense attention on each run's corresponding QKV. Shared-state
diagnostics are separate from generated-sequence divergence. Positional token
agreement continues after divergence; missing/extra positions disagree.

The current cohort is all30 AIME26 problems and100 previously sampled LongBench
v2 questions. AIME uses the existing prompt/scorer and2048 output tokens;
LongBench uses pinned NeMo prompt/MCQ evaluation,4096 output tokens and the
existing32K input cap. Six historical calibration members per benchmark are
included in headline scores; noncalibration24/94 are reported separately. These
are previously examined examples, not fresh held-out confirmation. Native
temperature0 is a sentinel for the0.4–0.8 schedule, not greedy decoding.

QK, block softmax and projected PV remain computed. The retained original-value
output uses a masked dense-shaped matmul. Physical skipping is not a claim that
all tile arithmetic is avoided. No hardware speedup over FlashAttention is
claimed.

## Published artifacts and reproduction

Git includes compact reports, aggregate JSON/CSV tables, thresholds, plots,
source/execution contracts, test evidence and audit/verification records for the
completed versions above. Their bytes are unchanged from the local result
bundles. Audit files intentionally reference additional local evidence.

Downloaded reference repositories, model weights, full prompts/token manifests,
per-example raw generations, compressed per-layer routing records, shared-QKV
snapshots and archived validation source copies are **not** included in this
compact Git publication. They remain preserved locally. A clone therefore is
not a self-contained raw-evidence archive: full audit regeneration and the
historical orchestration tests require those original caches, with matching
hashes and paths. The versioned drivers retain their original local model and
predecessor paths; they are not a turnkey fresh-download evaluation pipeline.

With the original compatible caches and environment restored, run from the
repository root:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m experiments.diffusion_gemma_jl_lowrank_multibench_v2 report
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m experiments.diffusion_gemma_jl_lowrank_multibench_v2 verify
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m experiments.diffusion_gemma_jl_aime_seed43 report
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m experiments.diffusion_gemma_jl_aime_seed43 verify
```

Use a disposable pytest output directory rather than overwriting frozen
`results/.../tests.xml` files. Existing execution fingerprints include those
original validation artifacts.
