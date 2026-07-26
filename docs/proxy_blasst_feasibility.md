# Pre-QK proxy-BLASST feasibility study

## Status and recommendation

The tracing, held-out analysis, calibration, counterfactual execution, safety,
and analytical cost paths are implemented behind opt-in scripts. The accepted
fused Triton kernel is unchanged. The repository schedule remains the source
of truth: high/mid/low-noise lambdas are
`0.04858582466840744`, `0.4334796965122223`, and `1.0`.

**Current recommendation: no-go on a CUDA proxy kernel yet.** The CPU smoke
run validates mechanics, but is synthetic and cannot establish LLaDA
predictability or generation quality. Its held-out results also demonstrate
why high aggregate agreement is insufficient: no tested low-overhead rule met
99.9% PRE_SKIP precision with nonzero coverage. A kernel prototype should be
considered only after the model-scale commands below pass the stated criteria.

The first realistic 4096-token binary-reuse run skipped QK for 6.57% of
physical tiles, but only 86.55% of those decisions were correct. Exact BLASST
skipped approximately 27.27% of tiles; binary reuse recalled only 20.87% of
those opportunities. This gap is expected across the high-to-mid transition:
lambda changes from 0.0486 to 0.4335, so a previous high-noise binary mask
cannot represent the many additional mid-noise skips. Continuous previous
scores are the primary route to higher coverage. The binary result remains a
no-go because it made 18,539 false skips, including 7,524 new-maximum events.

## Robust continuous-score continuation

Trace schema v2 fixes two issues found in the first attempted model run:

- Short repository corpora are deterministically repeated and rotated into
  heterogeneous native-length contexts instead of aborting below 4096 tokens.
- Sampling is keyed by sample and spatial tile, excluding step/layer/head, so
  every selected coordinate retains all legal source-target relations. Under
  the old independent 5% sampler, cross-source joins retained only about
  0.25%. GPU tensors are now copied in bounded batches and unselected tiles do
  not compute trace quantiles.

Calibration is source-noise to target-noise specific, sequence-length specific,
and hierarchical (global, layer, then layer/head). A threshold is deployable
only when both calibration and prompt-disjoint validation satisfy one-sided
Wilson upper bounds for total false skips and a ten-times stricter new-maximum
budget. Familywise confidence is Bonferroni-adjusted across policies, budgets,
and groups. Low-support thresholds fall back to a safer parent threshold or
EXACT. Runtime policies can use either one previous-step score or the monotonic
maximum of previous-step and previous-layer scores. Missing/cascaded scores
always force EXACT.

`coverage_headroom.json` deliberately separates an optimistic, label-tuned
risk/coverage frontier from deployable certification. This shows whether more
than binary reuse's 6.6% is statistically plausible without presenting an
oracle-tuned threshold as an optimization result.

## What is implemented

- Exact physical-tile traces with request/sample/seed, diffusion step and
  block, noise, layer/head/GQA group, positions, valid rows, exact `R` and
  `log(R)`, exact skip, new-maximum events, row quantiles, and region flags.
- Deterministic fractional sampling, bounded incremental NPZ shards, schema
  validation, optional packed masks, and optional 4/8-bit `log(R)` copies.
  Full QK matrices are never stored. Output-contribution norm is recorded as
  `NaN` because obtaining it would add a duplicate PV operation in this path.
- Coordinate-only joins for previous step, previous layer, all legally earlier
  heads, non-adjacent steps/layers, prompt-disjoint splits, global and
  layer/head/noise/GQA/region metrics, conditional score bins, heatmaps, and
  head clusters.
- Binary reuse, per-layer/head/noise score thresholds, leader and clustered
  heads, binary intersections, monotonic max-score rules, and a two-feature
  logistic ablation at false-skip budgets `1e-2` through `1e-5`.
- A source-only predictor API with only `PRE_SKIP` and `EXACT`. There is no
  predicted-keep shortcut. Oracle targets are held in a separate store and
  are consulted only after the decision for scoring.
- Counterfactual execution in which PRE_SKIP does not compute target QK, does
  not update the running maximum, and performs neither softmax nor PV. Separate
  metadata stores support oracle-source and realistic cascading execution.
- Warmup, local/diagonal/sink anchors, periodic refresh, and noise-specific
  thresholds.
- A traffic/operation model for packed bits, 4/8-bit scores, and FP16 scores.
  It reports algorithmic work avoidance and never treats Python timing as a
  kernel speedup.

## Smoke evidence

The committed smoke artifact contains six sample-disjoint synthetic requests,
three noise levels, three layers, four heads, 48 tokens, and 3,888 physical
tile records. It is a framework test, not a model result.

| Relation | Binary PRE_SKIP precision | Continuous log-score correlation | Finding |
|---|---:|---:|---|
| Previous denoising step | no source binary skips | Pearson 0.9903 | scores are stable, but calibrated nonzero coverage failed held-out safety |
| Previous layer | 77.27% | Pearson 0.9902 | unsafe despite 99.58% aggregate mask agreement |
| Earlier head | 6.12% | Pearson 0.2715 | clearly unsafe in this smoke distribution |

The worst previous-layer head had 75% binary precision and two false skips
that were new-maximum events. Earlier-head precision ranged from 0% to 11.11%
by target head; the worst head had 100% false skips. Per-layer/head threshold
calibration that had zero calibration errors generalized as poorly as 57.14%
or 71.43% precision, showing that these tiny splits cannot support `1e-5`
claims. Conservative max-score and logistic policies correctly selected zero
coverage at the strict budgets.

The safe one-step reference produced exact-BLASST-equivalent output because it
selected zero PRE_SKIP tiles in this small run. This is correctness evidence,
not evidence of useful sparsity. See `artifacts/proxy_smoke/` for all CSV/JSON
summaries and cross-step/layer/head plots.

## Cost model and kernel mapping

For one 4096-token LLaDA request with 32 layers, 32 heads, 128x64 tiles, and a
hypothetical 5% pre-QK sparsity, the model counts 2,097,152 physical tiles and
104,858 avoided QK tile operations. It estimates 1.718 GB of logical K-tile
traffic avoided before cache effects. Previous-step storage is 256 KiB for a
packed bit mask, 1 MiB for 4-bit scores, 2 MiB for 8-bit scores, or 4 MiB for
FP16 scores. One-source K-traffic break-even sparsity is approximately
0.00076%, 0.00305%, 0.00610%, and 0.01221%, respectively. These are analytical
logical-load estimates; L2 reuse, coalescing, synchronization, occupancy, and
metadata placement determine real performance.

- Previous step is the cleanest mapping: cache per-request metadata and read it
  without same-step synchronization.
- Previous layer is already available but adds a write/read per layer; whether
  it remains in L2 must be measured.
- Leader heads require ordered execution or CTA/warpgroup co-location. An
  offline correlation is not evidence of speedup if followers serialize.
- Packed masks have the lowest overhead. Continuous scores are justified only
  if their improved safe coverage is substantial on held-out LLaDA prompts.

## Commands

CPU smoke and tests:

```bash
bash scripts/run_proxy_smoke.sh
python -m unittest discover -s tests
```

Exact CUDA/LLaDA trace collection and held-out analysis:

```bash
bash scripts/collect_proxy_traces.sh
bash scripts/run_proxy_analysis.sh artifacts/proxy_traces
bash scripts/run_proxy_score_eval.sh artifacts/proxy_calibration/policy_summary.json
```

Reference counterfactual and cost evaluation:

```bash
bash scripts/run_proxy_eval.sh
```

True LLaDA masked-token evaluation (CUDA; deliberately slow PyTorch reference):

```bash
python -m eval.eval_proxy_llada_one_step --mode oracle
python -m eval.eval_proxy_llada_one_step --mode realistic \
  --output artifacts/proxy_llada_one_step_realistic.json
python -m eval.eval_proxy_llada_generation --mode realistic
```

Before a CUDA prototype, collect multiple prompt classes/seeds and require at
least 99.9% held-out PRE_SKIP precision, nontrivial correct physical pre-QK
sparsity, close masked-token agreement to exact BLASST, no high-noise collapse,
positive modeled savings, and stability in the worst layer/head group.
