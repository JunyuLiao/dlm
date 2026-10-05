# Why BLASST loses to value-aware routing in the current DiffusionGemma runs

This note records the investigation prompted by the AIME26 and RULER results. The
working hypothesis is not that the methods are benchmark dependent. They use the
same model and the same iterative denoising process, and the largest gap appears
on the semantic tasks that need stable distributed attention. The evidence points
to a mismatch between BLASST's routing state and the output state, followed by a
QK-only selectivity error. The value signal and query sensitivity then improve the
allocation further.

## 1. The result is reproducible across the two relevant evaluations

### AIME26, uniform one-pair calibration, target 50%

From [`query_adaptive_aime_uniform_baselines_v1/report.md`](../../results/query_adaptive_aime_uniform_baselines_v1/report.md):

| Method | Overall / global / local physical sparsity | Accuracy | Mean steps | Cap canvases |
|---|---:|---:|---:|---:|
| dense reference | 0 / 0 / 0% | 46.67% | 14.97 | 0 |
| Gaussian32 | 49.16 / 52.75 / 48.27% | 47.78% | 19.65 | 3 |
| aggressive BLASST | 49.63 / 46.81 / 50.31% | 32.22% | 27.03 | 73 |

The overall physical budgets are effectively matched. The global/local allocation
difference is a possible secondary factor, but it cannot explain the same pattern
in the RULER run where all three sparsity strata are matched closely.

### RULER4K130, target 70%

The archived value-direction run has the same 128x64 tile geometry and matched
physical budgets for Gaussian32 and BLASST. Gaussian32 scores 85.28% and BLASST
82.51%; BLASST also takes 3619 decoder calls versus 2123 for Gaussian32 and has
token agreement with native dense of 36.18% versus 63.99%. See
[`value_direction_hopper_v1/final_ruler4k130_v1/report.md`](../../results/value_direction_hopper_v1/final_ruler4k130_v1/report.md)
and [`systems_report_v3/report.md`](../../results/value_direction_hopper_v1/systems_report_v3/report.md).

The query-adaptive value router improves the same operating point further:

| Method | Overall / global / local sparsity | Accuracy | Mean calls |
|---|---:|---:|---:|
| archived BLASST | 69.68 / 69.86 / 69.56% | 82.51% | 27.84 |
| C_s70 | 69.7 / 70.3 / 69.3% | 89.7% | 8.65 |
| T_s70 | 69.8 / 70.9 / 69.1% | 89.7% | 6.78 |

The per-task pattern is diagnostic: BLASST falls to 46% on code-word extraction,
70%/60% on the two QA tasks, while Gaussian32 is 77%/80%/70%. The NIAH tasks are
mostly 100% for both. This is a failure on tasks requiring distributed semantic
support, rather than a generic benchmark label effect.

## 2. The first causal distinction: routing state versus retained output state

For a query row (i) and KV tile (j), the aggressive BLASST decision is

\[
r^B_{ij}=b_{ij}-\bar b_{i,j-1},\qquad
\bar b_{ij}=\max(\bar b_{i,j-1},b_{ij}),
\]

where (b_{ij}) is the tile's maximum QK logit. In the current implementation
\(\bar b\) is updated even when tile (j) is skipped. The retained softmax
normalizer and projected output are updated only when the tile is retained.
Therefore a skipped tile can raise the future routing reference without entering the
output state. Later tiles are judged against an omitted maximum. This can create a
cascade of skips and makes the routing state inconsistent with the attention state.

The repository makes this explicit:

* The legacy streaming emulator selects `seen_m` for BLASST, updates it on every
  tile, and updates retained `m` only for included tiles
  ([`operators.py:205-227`](../../experiments/diffusion_gemma_value_aware/operators.py:205)).
* The Hopper kernel carries the same convention: “compare against maxima of PRIOR
  seen blocks, then update seen even if this tile drops”
  ([`value_direction.cu:520-523`](../../experiments/value_direction_hopper/csrc/value_direction.cu:520)).

The value router does not use this all-history max as its output-risk reference. It
computes a candidate change to the retained projected output and commits the
projected state only for retained tiles. Query sensitivity is applied to that
candidate before the physical all-row vote
([`value_direction.cu:505-522`](../../experiments/value_direction_hopper/csrc/value_direction.cu:505)).

This is more than an implementation detail. It is a causal difference in the
denoising trajectory. The AIME quick50 controls isolate it:

| AIME quick-review condition, target 50% | Physical sparsity | Retained mass | Relative operator error | Correct / 10 |
|---|---:|---:|---:|---:|
| aggressive BLASST (all-history max) | 50.77% | 69.53% | 0.5095 | 4 |
| no-value control with retained-state gap | 49.12% | 78.24% | 0.3200 | 6 |
| value-aware score | 49.63% | 77.95% | 0.3080 | 6 |

The no-value control nearly recovers the value-aware result while removing the V
factor. This makes the retained-state convention the first mechanism to test; it
would be incorrect to attribute the whole improvement to the value statistic.
The quick50 set is exploratory (10 AIME questions), so this is a mechanism signal,
not a final accuracy estimate. Source: [`quick50_review/report.md`](../../results/diffusion_gemma_value_aware_followup/quick50_review/report.md).

## 3. Why a QK maximum is a poor output-contribution proxy

Even with a corrected retained-state reference, a tile maximum is only one scalar.
For a tile of width (W=64), its unnormalized attention mass is

\[
S_j=\sum_{k\in j}e^{a_{ik}},\qquad S_j\le W e^{b_{ij}}.
\]

A negative max gap does not imply small $S_j$. A tile can contain many logits just
below the previous maximum. With a gap of -2.2, 64 such entries have an upper mass
of $64e^{-2.2}\approx7.1$ relative to one previous-max entry. The length
normalization in the BLASST threshold does not recover the missing within-tile
log-sum-exp or the value direction.

The output change is instead approximately

\[
\Delta O_{ij}=\alpha_{ij}(\mu_{ij}-O_{i,j-1}),
\]

where $\alpha$ is the block probability mass and $\mu$ is the block's weighted value
mean. Two tiles with the same QK maximum can have very different $\alpha$ and very
different $\mu$. Cancellation can make a large-mass tile harmless; a moderate-mass
tile aligned with a semantic direction can be decisive. BLASST cannot distinguish
these cases because it does not form either quantity.

The shared-state AIME calibration diagnostics compare masks at exactly the same
50% physical tile budget:

| Score | Retained mass | Relative output error |
|---|---:|---:|
| QK max / gap-only | 77.87% | 0.3316 |
| exact contribution diagnostic | 83.76% | 0.2504 |
| exact mass diagnostic | 84.64% | 0.2391 |

The contribution diagnostic lowers error 24.5% relative to gap-only and retains
5.89 percentage points more mass. On the independent LongBench calibration states,
the corresponding error reduction is 61.6% and the mass increase is 13.81 points.
These are shared dense-state diagnostics, not downstream generation results, but
they directly measure the selectivity failure. See
[`diagnostics/report.md`](../../results/diffusion_gemma_value_aware_followup/diagnostics/report.md).

The current AIME seed-43 shared-state aggregate points in the same direction. It is
not an exact matched-budget comparison, so it should be treated as directional:

| Score | Physical sparsity | Retained mass | Relative output error |
|---|---:|---:|---:|
| aggressive BLASST | 68.23% | 58.83% | 0.8878 |
| mass at 50% | 50.00% | 77.16% | 0.5304 |
| full centered value risk | 38.02% | 83.04% | 0.2088 |

## 4. Why this becomes a denoising problem rather than a one-call error

The sparse operator changes the canvas and the accepted/renoised positions. A bad
tile decision therefore changes the next Q/K state, which changes the next mask.
The state mismatch above makes BLASST especially vulnerable to this feedback:

1. A tile with a large but non-recording maximum is skipped.
2. Its maximum still raises `seen`.
3. Future tiles fail the record-gap test even when they carry useful mass or value
   direction.
4. The retained denominator/output does not contain the skipped tile.
5. The next denoising canvas receives a larger attention error, causing more
   prediction changes and slower native stopping.

The observed cap/step statistics are consistent with this mechanism: at AIME 50%,
BLASST has 27.03 mean steps and 73 cap canvases versus Gaussian32's 19.65 and 3;
on RULER it has 3619 versus 2123 calls. This is not proof of mediation, but it is
the expected direction if routing errors are amplified by the denoising loop.

The physical vote also matters. A 128x64 tile is skipped only when all valid query
rows vote to skip. BLASST's row score is query-indifferent, so it cannot preferentially
protect an unstable or answer-relevant row. C/T sensitivity changes the row scores
before the same physical vote, preserving tiles for unstable queries and allowing
stable rows to become cheap. This explains why C/T improve the semantic RULER tasks
without changing the physical tile budget.

## 5. Alternative explanations checked, and what remains open

* **Sparsity mismatch:** small AIME global/local differences remain, but RULER is
  matched within roughly one point in all strata. A reallocation control is still
  required for the AIME claim.
* **Aggressive lambda extension:** the archived BLASST final uses the repository's
  lambda>1 inverse-valid-length extension. A lambda<=1/capped run is a necessary
  control, but the capped historical point is not a matched 50%/70% comparison.
* **Backend arithmetic:** Gaussian32 and BLASST use the same Hopper backend in the
  archived RULER comparison, and the AIME uniform run uses the same 128x64 kernel
  family. The no-value control therefore points to routing semantics rather than
  a different decoder implementation.
* **Projection rank or V norms:** scalar V norms add almost no signal because the
  model's value normalization makes norms nearly constant. The useful information
  is value direction/cancellation, not a token norm multiplier. See
  [`value_signal_differences.csv`](../../results/diffusion_gemma_value_aware_followup/diagnostics/value_signal_differences.csv).
* **Calibration:** calibration changes which tiles are selected but does not create
  the retained-mass/error gap on the same dense state. Thresholds should be frozen
  only after the routing semantics are fixed.

## 6. Decisive next experiments

### A. Same-state matched-support replay (highest priority)

On saved AIME and RULER Q/K/V states, force exactly the same physical tile count
and all-row closure for:

1. BLASST with all-history `seen` (current implementation);
2. BLASST with retained-only running max;
3. retained-state gap-only (no V term);
4. projected contribution/value risk;
5. query-weighted gap-only and query-weighted value risk.

Report retained mass, true dense output L2, fraction of skipped update energy,
rank/Spearman and AUROC against exact output contribution, with local/global and
prefix/canvas splits. This separates state semantics from score information without
denoising feedback.

### B. Minimal Hopper ablation

Add a diagnostic `blasst_retained` mode. In the BLASST router, update `seen[r]`
only when `decision.eligible && !decision.skip`; retain the current threshold and
run a fresh calibration because the sparsity distribution changes. Compare it
against current BLASST and the no-value value-mode control on a small AIME/RULER
subset before a full sweep. This is the cheapest test of the strongest hypothesis.

### C. Hybrid score after the state test

If retained-only BLASST recovers most of the gap, keep the simpler state fix and add
only a small output-aware term, for example

\[
r=\text{gap}_{\rm retained}+\lambda_v\log(\|\alpha\mu\|/R),
\]

or use the projected change ΔO already computed by the Hopper value router. If
retained-only BLASST remains weak, the QK selectivity failure is causal; prioritize
the projected value-direction score and query weighting. A direct query-conditioned
control is `gap + log(s_i)` with the same BLASST gap and no V computation.

### D. Trajectory replay

Teacher-force the same denoising canvas through each policy for several calls, then
release the trajectory. Measure one-call output error, next-call flip/acceptance
changes, cap probability and final answer. This tests whether the operator gap is
amplified by feedback rather than merely correlated with it.

## Decision rule for the next iteration

Do not choose a new sensitivity formula yet. First run A/B on retained-state
semantics. If `blasst_retained` substantially increases retained mass and reduces
steps/caps, the all-history reference is the primary defect. If it does not, and
matched-support replay still favors contribution/value risk, then the primary defect
is QK-max selectivity. Only after that split should we tune the C/T query weighting
or add a lightweight projected-value hybrid.

This explains the current AIME and RULER results with a common causal mechanism,
while preserving the distinction between a strong diagnostic and a proven
end-to-end accuracy claim.
