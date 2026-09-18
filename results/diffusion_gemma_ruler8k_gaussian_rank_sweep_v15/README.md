# RULER8K Gaussian-rank sweep: published results

Start with the [interpretation](interpretation.md) and [complete report](report.md).
The completed experiment has 2,470 audited final outputs across 19 conditions:
130 matched questions (10 each from all 13 official RULER tasks), dense, and
BLASST/mass-only/full-dimensional centered/Gaussian ranks 1/2/4/8/16/32 at
50% and 75% target physical sparsity. There are 26 disjoint calibration
questions and 13 development questions. These previously examined examples
are not fresh held-out confirmation.

Version 15 adds 1,300 final evaluations and reuses all 1,170 compatible
[version 14 outputs](../diffusion_gemma_ruler8k_jl130_v14/report.md). Version 13's
failed smoke bookkeeping evidence is preserved separately. Neither numerical
kernels nor previous results were modified. The original 41-test CPU/CUDA gate,
two-input real-model smoke, full final audit, and independent raw-only report
regeneration all passed. No final inference failures were recorded.

## Main finding

At approximately 76% actual sparsity, Gaussian32 scores 47.38%, close to the
full-dimensional reference's 49.78%, versus Gaussian2's 2.50%. This recovers
about 95% of the observed Gaussian2-to-full score gap. Identical-support
projection diagnostics support information loss as a major contributor, not
the sole cause. Accuracy is not monotonic in rank: Gaussian16 underperforms
Gaussian8. At 50%, small projections already perform well. The unchanged
30-token VT budget depresses dense accuracy; the disclosed non-VT sensitivity
analysis preserves the principal 75% finding.

## Setup and measurement

- Pinned DiffusionGemma revision `f7f5b7f5fa82ffc52addd066915886d497f5517b`, BF16,
  128-query x 64-KV physical tiles, prefix plus canvas eligible, native GQA and
  structural masks. Projection and routing arithmetic use FP32, with TF32 off.
- Generation seed 42; projection seed 1729; fixed Gaussian matrices per
  layer/native KV head. Matrices are not nested across ranks. Seeds 2718 and
  31415 are additional diagnostic checks, not generation replications.
- Original official task-specific output budgets, native 256-token canvas,
  48-step maximum, thinking disabled. Requested temperature zero is the native
  schedule sentinel, not greedy decoding. The pinned RULER scorer is used.
- Sparsity is summed skipped eligible physical tiles divided by summed
  eligible tiles, reported overall and separately for local/global layers.
- Local/global thresholds are calibrated independently on calibration data
  and frozen before final evaluation. BLASST permits lambda above one only
  when the joint local=global=1 whole-model ceiling cannot reach the target;
  its 75% condition needs an aggressive local threshold.
- Shared-state diagnostics cover 390 calibration snapshots, one question per
  task and all 30 layers at step zero. The common-support probe holds the
  full-dimensional retained history fixed across ranks and seeds.
- No hardware speedup is measured or claimed. QK, block softmax and projected
  PV remain computed; physical tile deletion does not eliminate all tile work.

## Published files versus local evidence

This is a compact Git publication, following earlier studies in this
repository. It includes the unchanged experiment/test source, report and
interpretation, aggregate CSV/JSON tables, per-task tables, shared-state
diagnostic aggregates, projection hashes, frozen policies/configurations,
figures, original test evidence, and audit/regeneration proofs. The completed
v14 baseline's compact artifacts and the v13 failure metadata are also kept.

Multi-gigabyte raw generation/routing shards, shared QKV tensors, proposal
caches, full prompts/token manifests, large work-accounting dumps, downloaded
reference repositories and model weights are **not included in Git**. They
remain unchanged locally. Audit JSON intentionally records hashes and paths
to that additional evidence. A fresh clone is therefore **not** a
self-contained raw-evidence archive, and cannot run historical `report`,
`verify`, or cache-dependent tests without restoring those original caches.
The code preserves its audited local/predecessor paths rather than silently
rewriting provenance for publication.

The frozen canonical audit remains unchanged. `interpretation_proof.json`
separately records provenance for the post-audit interpretation.

## Source and reproduction

- [Version 15 driver](../../experiments/diffusion_gemma_ruler8k_gaussian_sweep.py)
- [Projection diagnostics](../../experiments/diffusion_gemma_ruler8k_projection_diagnostics.py)
- [Report extension](../../experiments/diffusion_gemma_ruler8k_gaussian_report.py)
- [Post-audit interpretation](../../experiments/diffusion_gemma_ruler8k_gaussian_interpretation.py)
- [Version 14 baseline driver](../../experiments/diffusion_gemma_ruler8k_jl_v14.py)
- [Rank-sweep tests](../../tests/test_ruler8k_gaussian_sweep.py)

With the original compatible caches and pinned environment restored, from the
repository root:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m experiments.diffusion_gemma_ruler8k_gaussian_sweep report
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m experiments.diffusion_gemma_ruler8k_gaussian_sweep verify
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m experiments.diffusion_gemma_ruler8k_gaussian_interpretation
```

Use temporary pytest output directories. Never overwrite the original
`tests.xml` files: frozen execution fingerprints include their exact bytes.
