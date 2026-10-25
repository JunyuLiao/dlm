# Completed 25-question end-to-end review — 2026-10-25 UTC

The requested screen is complete: 10 AIME26, 10 HumanEval, and 5 original,
context-eligible long-band LongBench-v2 questions, seeds 42/43, nine arms.
All 450 scored cells have exact full audit/clean output-hash, N, C, and T
agreement; all clean timings have zero new CUDA capture/JIT events. The 19
frozen closeout checks and supplemental source-binding/runtime checks pass.
The H100 is idle. This is an exposed small-cohort screen, not noninferiority
evidence or a production-method promotion.

The result does **not** establish accuracy-preserving trajectory acceleration.
The query-direction response variant has a descriptive AIME quality lead, but
loses one HumanEval case, fails to improve LongBench quality, and is slower than
dense on all three tasks. LongBench also favors the inherited V31 baseline and
the sham over this response variant. Do not optimize these selectors as a
production path on the strength of this screen.

## Official task outcomes and synchronized request cost

Counts are item/seed cells: 20 per short task, 10 for LongBench. There are only
10, 10, and 5 independent question clusters. W is mean clean request seconds.

| Arm | AIME correct/20 | HumanEval correct/20 | LongBench correct/10 | AIME W | HE W | LB W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_full_fix51994 | 12 | 20 | 2 | 4.991 | 0.739 | 63.764 |
| dense_piecewise | 12 | 20 | 2 | 5.053 | 0.750 | 63.904 |
| allkept_layer29 | 11 | 19 | 2 | 5.651 | 0.821 | 77.044 |
| v31_optimized | 6 | 19 | 4 | 7.242 | 0.949 | 28.857 |
| current_error_fixed4 | 9 | 19 | 3 | 20.577 | 2.128 | 235.355 |
| response_guard_qdirection | 15 | 19 | 2 | 16.682 | 2.070 | 287.100 |
| margin_guard_weighted_bias | 10 | 18 | 3 | 19.744 | 2.248 | 220.607 |
| validity_reuse_mass_certificate | 12 | 19 | 2 | 55.044 | 5.548 | 536.777 |
| sham_fixed4 | 10 | 19 | 4 | 17.459 | 2.083 | 216.756 |

W begins before engine.add_request on the frozen token inputs and ends after
the final device synchronization. It includes prefill, selection, retained
attention, and the native sampler; it excludes model startup, preparation of
tokenized inputs, audit instrumentation, and task scoring. S excludes prefill.
Startup and all explicit warm/audit/clean requests are counted in allocation.
FULL and PIECEWISE output/N/C/T match in all 50 cells; their actual decode modes
are independently audited. All-kept is a systematic consumer treatment, not
noise to subtract: it already changes AIME and HumanEval task outcomes.

## Native trajectory

The full tables in ../summary.md and per_cell_metrics.csv include N, C, N/C,
T, S/N, S, W, caps and physical work for every arm. Selected totals:

| Task | Arm | Total N | Total C | Aggregate N/C | Total T |
| --- | --- | ---: | ---: | ---: | ---: |
| AIME | dense FULL | 5426 | 425 | 12.767 | 107609 |
| AIME | response | 4778 | 385 | 12.410 | 96709 |
| AIME | sham | 5037 | 397 | 12.688 | 99830 |
| AIME | V31 | 7031 | 492 | 14.291 | 125180 |
| HumanEval | dense FULL | 741 | 89 | 8.326 | 19335 |
| HumanEval | response | 696 | 87 | 8.000 | 19688 |
| HumanEval | sham | 708 | 85 | 8.329 | 19880 |
| HumanEval | V31 | 842 | 91 | 9.253 | 20626 |
| LongBench | dense FULL | 12128 | 298 | 40.698 | 75608 |
| LongBench | response | 16031 | 404 | 39.681 | 102904 |
| LongBench | sham | 11781 | 298 | 39.534 | 75383 |
| LongBench | V31 | 6833 | 196 | 34.862 | 49114 |

Aggregate N/C weights the different realized canvases. Paired question-level
ratios are a different estimand and can move in the opposite direction when
interventions change output length and canvas count. Response versus FULL:

| Task | Paired geometric N ratio | Paired N/C ratio [95% interval] | Paired W ratio [95% interval] |
| --- | ---: | --- | --- |
| AIME | 0.857 | 0.970 [0.923, 1.009] | 3.152 [2.779, 3.537] |
| HumanEval | 0.899 | 0.978 [0.837, 1.136] | 2.379 [1.893, 2.890] |
| LongBench | 2.070 | 1.138 [0.973, 1.330] | 4.216 [2.434, 7.305] |

These are descriptive paired question-cluster bootstrap intervals (10000
draws, both seeds kept in each cluster); they are not multiplicity-adjusted
confirmatory tests. N/C intervals come from paired_trajectory001; W intervals
come from analysis001. Nonsignificance does not establish noninferiority.

On AIME the response quality difference is +15 pp [0, +35] against FULL,
with 0 dense-correct→response-wrong and 3 reverse transitions. But output T
also falls (paired ratio 0.874), and N/C versus sham is 0.998 [0.946, 1.040].
The AIME result does not establish response-specific adaptive acceleration.
On HumanEval the response loses HumanEval/71 seed43 to a final-code syntax
failure: one dense-correct→candidate-wrong, no reverse. LongBench response
and dense have the same two correct cells, while response loses two cells
against V31 and two against sham, with no reverse improvements in either
comparison. LongBench valid parsed choices are 4/10 response, 6/10 dense,
8/10 V31 and 6/10 sham. Output caps are 5/10 response, 4/10 dense, 2/10 V31
and 3/10 sham. A capped/wrong reference is not quality preservation evidence.

First-canvas mean calls, FULL/response/sham: AIME 10.20/10.35/10.40;
HumanEval 10.05/9.40/9.35; LongBench 23.90/26.10/21.20. The LongBench
confidence-only bottleneck persists: 7689/12128 dense calls versus
9888/16031 response calls. Native 48-call canvas caps increase from 145/298
dense canvases to 206/404 response canvases. Output caps and canvas caps are
different events. All traces retain the native uncommitted tail (one call per
request) and separately report committed-canvas counts; no forced continuation
or canvas2 substitution was used.

## Diagnostic interpretation and work

At the common first selection boundary, all 200 score/current-error bank
comparisons agree exactly on candidate current-error, mass and feasibility.
Across 50 cells × 32 routing units, only 352/1600 units have at least four
feasible alternatives: 0/640 AIME, 32/640 HumanEval, 320/320 LongBench.
Response chooses different supports in 46/50 cells (467 unit choices), margin
in 25/50 (232), and sham in 46/50 (465); certified reuse initially matches
current-error. Response has 194 distinct current-error/mass-matched unit
choices, margin 168, sham 281. **None** has at least 25% full-response
separation under the predeclared matching rule. These units and masks are not
independent task samples. AIME's first bank fails the ranking-identification
floor; do not retrospectively call its quality lead an oracle gate success.

This agrees with the diagnostic warning that current output fidelity alone
cannot certify native future states or quality. It supplies direct end-to-end
negative evidence for these concrete repeated-intervention methods, but the
non-identifying response bank prevents a general rejection of all possible
response mechanisms. The experiment does not separate SA propagation from AS
direct intervention or held SS interaction; the prior AA/SA/AS/SS stage remains
partial and is not relabeled complete.

New methods prune only wholly-prefix GLOBAL layer29 rectangles (all 16 query
heads, queryblocks0/1, Q128×KV64), retaining current-canvas/boundary support,
other layers, LOCAL, prefill and commit. Every unit deletes
max(1, B−ceil(0.9B)) eligible prefix blocks, or falls back to all-kept: small B
uses one deletion; growing prefixes use approximately 90% retention. Frozen
guards and fixed4 refresh are unchanged.
Actual overall GLOBAL consumer sparsity is about 1.08–1.94%; V31 uses five
GLOBAL layers with K128/K8192 and reaches 45.47–90.83%. V31 is a practical
baseline with different work, not a matched-work selector attribution control.
Later trajectories also change physical work and feasibility among new arms.

Current-output bias and query derivatives are exact for the current attention
operator. The response selector minimizes J_S u and records (J_S−J_A)u
separately, but predicts neither future K/V motion nor full-model contraction.
The margin selector is a previous-margin-weighted attention surrogate, not the
proposed exact downstream MarginGuard oracle. The certificate recomputes the
current-canvas partition and resets every canvas; it is a retained-mass bound,
not an accuracy certificate. See ../THEORY_AND_LIMITS.md.

Selector cost overwhelms removed consumer work. On LongBench response removes
165880192 layer29 consumer rectangles but performs 1718295168 discovery QK
and 1887136320 discovery PV rectangles. Its refresh support-management CUDA
event intervals total 1990.788 audit seconds; discovery is nested inside that
interval and must not be added to it. These are audit event intervals, not
active GPU utilization or a clean per-kernel speed claim. Clean request W is
the performance endpoint. Certified reuse refreshes 9819 times with zero held
calls on LongBench; across the full screen it refreshes 15550 times with only
5 held calls and 11371 failed certificate checks. Preserve this negative.

## Go/no-go and next bounded experiment

- Response query-direction variant: no-go for production, speed or general
  accuracy-preserving acceleration. Preserve the AIME quality lead as
  descriptive; first fix response-bank identification before a new causal study.
- Margin-weighted surrogate: no-go for promotion (10/20 AIME, 18/20 HE,
  3/10 LB; slower than dense). No evidence here justifies an exhaustive
  downstream MarginGuard influence oracle or a cheaper proxy.
- Current-error fixed4: retain as fidelity control, not a quality certificate
  or accelerated method (9/20 AIME, 19/20 HE, 3/10 LB).
- Current-canvas-aware certified reuse: park this concrete conservative bound;
  frequent refresh and high W defeat practical reuse. The older norm-only
  track remains separately parked.
- V31: retain Yuhan's inherited reuse baseline and its distinct work scope;
  its stronger LongBench screen result does not erase the AIME regression or
  establish general dense noninferiority. Gaussian32/value routing, C-gate,
  V31/V2 and prior negatives retain separate provenance.

The most informative next step is a separately frozen operator/first-canvas
bank-identification leaf, using within-routing-unit equal-work, similar-error
and similar-mass candidates with actual response separation. Use full Q/K/V
motion components or predeclared finite-difference diagnostic directions to
test whether useful separation exists; do not search future task outcomes to
manufacture a bank. Only if that leaf identifies alternatives should a fresh
from-prompt AA/SA/AS/SS study test pulse versus hold and score prespecified
continuations. Fresh from-prompt branches avoid assuming the failed restored
output-assembly certificate is fixed. Do not expand into kernels, a production
selector, another benchmark panel, or a later-canvas campaign automatically.

## Identity, resources and review order

Branch denoising_trajectory_research; work began at 5f95bab0. Protocol
88e46d4088a094968ebbeaed05b0ce4b9cf6e824a643675eba257ac9920d35dd,
generation source4ac2b53f, V31 repair binding1429aaf3 (source21a3a507),
non-V31 long binding1636acaa (source018ab5f9). Completion/readout sources
were checkpointed before their execution; final commit is recorded by Git.
All 559 frozen source hashes match. No new native RNG draws, extra model
forwards, stale K/V, cached attention outputs, sampler/stopping/cap changes,
or original prompt changes are admitted. No sparse restored-output claim.

Complete successful panel: 49117.370 allocated-worker seconds (13.644 h),
405458 counted engine/startup/profile forwards, one H100 worker at a time.
Earlier preserved qualification/preflight/failure allocation is at least
1536.107 seconds and 17658 known forwards, plus an unavailable failed-V31
counter. Exact process-exit allocation for the superseded all-kept preflight
was not persisted. Do not claim an exact all-attempt total. See resources.json.
No campaign GPU ceiling remains; no watchdog fired. H100 check at
2026-10-25 02:18:47 UTC: no compute process, 0MiB, 0% utilization.

32 focused CPU tests passed before generation; the final 36-test CPU suite
also passes. Earlier replay output-assembly qualification, mixed-mode V31
accounting, interrupted controllers and CPU-reader failures remain preserved.
The accepted V31 data use separately bound constant-mode audit/clean engines;
the algorithm and budgets are unchanged. No paused campaign resumed, no other
GPU workload was killed, no push requested. Nine original over-limit LongBench
items remain blocked without truncation. The five eligible long questions
cover only two domains. Previous exposure, small clusters, unadjusted screening
intervals and absent response-identification pairs limit inference.

Inspect this review, ../summary.md, ../final_complete.json,
../closeout001/verification.json, ../binding_review001/verification.json,
../scoring001/official_scores.json, ../paired_trajectory001/summary.json,
../support_analysis001/summary.json, ../trajectory_analysis001/summary.json,
per_cell_metrics.csv, selector_accounting.json and COMMANDS.md first.
