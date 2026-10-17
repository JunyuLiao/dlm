# 2026-10-17: complete results and statistical references

Read [method definitions and setup](01_METHODS_AND_SETUP.md) before comparing
numbers. The completed native study has 358 input/seed cells and 5,370 officially
scored generation trials. Every structurally valid result is included. No
candidate passes the predeclared all-workload accuracy/call screen. Clean timing
remains paused. This is exploratory development, not held-out confirmation.

## Official native accuracy and trajectories

Each entry is **official score / mean denoising calls N per trial**. RULER is
its official task-average 0–100 score; the other scores are accuracy percent.

| Method | AIME26 | LongBench-v2 | MATH100 | RULER128K |
|---|---:|---:|---:|---:|
| V31 control | 51.111 / 295.300 | 45.703 / 251.039 | 95.000 / 61.468 | 86.250 / 5.240 |
| Full-value swaps | 52.222 / 301.672 | 46.875 / 192.023 | 95.000 / 60.590 | 87.308 / 5.231 |
| Direction-guarded residual | 51.111 / 296.911 | 43.750 / 249.516 | 95.500 / 60.970 | 86.058 / 5.317 |
| Raw residual | 50.556 / 296.861 | 43.750 / 327.094 | 94.500 / 61.145 | 86.346 / 5.375 |
| Swaps + direction guard | 51.111 / 298.611 | 47.656 / 299.438 | 95.000 / 60.725 | 85.962 / 5.481 |
| Dense FULL fix51994 | 55.556 / 298.422 | 51.562 / 234.625 | 95.000 / 61.370 | 86.538 / 5.750 |
| Dense PIECEWISE | 58.889 / 281.800 | 46.875 / 258.781 | 95.000 / 63.020 | 86.538 / 5.654 |
| All-kept FA4 | 58.889 / 284.222 | 51.562 / 305.953 | 95.000 / 58.560 | 84.856 / 5.558 |

Candidate rows average two trials/cell; V31 averages four bookends; references
one trial/cell. This is not answer ensembling. Primary incremental contrasts
use shared-engine V31 controls. Fine-grained differences against separate-engine
dense references are limited by observed runtime variation.

The [complete human-readable statistics](../../results/research_update_20261017/attempt001/native_statistics.md)
include every arm's median/p95 N, N/C, C, T, cap rate, audit-only S/W/S-per-N,
and both interval analyses for score, N, C and T. The
[source-bound synthesis and figure](../../results/v31_value_decision_reuse_20261015/synthesis_20261017/attempt001/summary.md)
provide an additional presentation of the same frozen reports.

Audit S and W in the generated Markdown are totals across each arm's own
unequal trial count. Divide by that count for per-trial audit costs. Audit S/N
is an amortized instrumented span, not a clean per-forward price or speedup.

## Interpretation supported by the data

* **AIME:** all four candidates increase mean N. Full swaps add 1.111 score
  points (problem CI [-2.778,5.000]) and 6.372 calls (+2.16%, CI [-5.228,18.058]).
  Score gain changes sign across engine repeats. No reliable improvement.
* **LongBench:** full swaps reduce N by 59.016 (-23.51%), with problem interval
  [-143.250,-3.988] and engine-block interval [-119.172,-14.945]. Score increases
  1.172 points, but interval [-6.250,8.594] includes zero. Calls fall on 41/64
  cells, rise on 22 and tie on 1. Five cells explain 89.69% of the net saving.
  One always-wrong capped cell accounts for 58.56% (2549→337 calls,
  16384→3026 tokens). The 26 always-correct cells also save 38.231 calls on
  average. These conditioned/tail-trimmed summaries are descriptive and biased
  by conditioning; they do not replace the unconditional result.
* **MATH:** combined swaps/guard reduces N by 0.7425 (-1.21%, problem interval
  [-1.5775,-0.0575]) with unchanged aggregate score. Guard's +0.5 points is one
  additional correct trial/200. Only 46/200 candidate trials reach a sparse
  anchor; 45 execute a correction path. Most trials do not test sparse reuse.
* **RULER:** full swaps add 1.058 score points (task interval [0,3.077]); N
  changes by only -0.0096 (-0.18%, interval [-0.0577,0.0481]). All candidate
  score/hash/N-C-T vectors match across repeats. Three changed cells drive
  the gain: multikey3 0→100, cwe 60→90 and cwe 80→60. Eleven task scores are
  unchanged. Repeating identical trajectories does not double independent
  evidence. All correction arms increase mean N.

Full swaps are the strongest candidate. Residual correction's much larger
same-state output-error gains do not produce a consistent native tradeoff.
Raw residual adds 30.30% LongBench calls; adding the guard to swaps changes
LongBench N from 192.023 to 299.438 and loses the RULER swap score gain.
This is evidence against using mean attention-output error alone to rank
methods, not proof of one unique cause. The next unsolved target is admission
and reuse lifetime of bounded swaps, including the AIME call regression.

## Same-state operator results: every promoted and rejected family

The initial screen evaluated all ten recorded operator arms. The whole-layer
screen then included all 16 heads and both Q128 tiles, five GLOBAL layers and
570 usable frames from 620 saved frames. The following rows are sparse held
frames; request003 is all-kept and contributes no sparse benefit evidence.

| Method | LongBench req001: ratio / >2% max-row harms (220 frames) | LongBench req002 (100) | AIME req004 (30) |
|---|---:|---:|---:|
| Mass anchor | 1.0000 / 0 | 1.0000 / 0 | 1.0000 / 0 |
| Full swaps | 0.9847 / 1 | 0.9733 / 8 | 0.9776 / 10 |
| Online hybrid swaps | 0.9861 / 0 | 0.9762 / 8 | 0.9796 / 5 |
| Raw residual | 0.6355 / 26 | 0.7399 / 25 | 0.8327 / 11 |
| Omitted moments | 0.5398 / 126 | 0.7694 / 68 | 0.9059 / 20 |
| Direction-guarded residual | 0.6444 / 0 | 0.7753 / 3 | 0.8291 / 2 |
| Norm-guarded residual | 0.8281 / 0 | 0.9701 / 0 | 0.9818 / 0 |

Ratios are mean post-O error relative to the mass anchor, from their respective
frozen replay summaries. The guard uses a whole-layer signed residual. Causal
query extrapolation worsens age-one prediction on 100/100 screened sparse units
and was rejected. The initial cap-2/lifetime/tangent results are retained in
[operator_screen.csv](../../results/research_update_20261017/attempt001/operator_screen.csv).
These are four previously examined trajectories, not four independent
benchmark panels. AIME K27 replay is counterfactual on original K128 states.

Decision diagnostics cover 57 paired inputs; 39 supply 265 sparse held frames.
The initial panel and purposive coverage extension remain separate. Counts
below are top-1 / acceptance disagreement rows against dense-reference suffixes.

| Diagnostic arm | LongBench initial (99 frames) | AIME initial (35) | RULER initial (14) | MATH coverage (75) | RULER coverage (42) |
|---|---:|---:|---:|---:|---:|
| Mass anchor | 1068 / 134 | 26 / 16 | 24 / 4 | 107 / 20 | 27 / 52 |
| Full swaps | 1035 / 132 | 21 / 14 | 23 / 2 | 101 / 19 | 25 / 51 |
| Online hybrid swaps | 1054 / 127 | 21 / 14 | 25 / 2 | 101 / 19 | 24 / 51 |
| Fixed-K C_gate swaps | 1034 / 133 | 23 / 15 | 23 / 2 | 98 / 14 | 24 / 53 |
| Shuffled C_gate swaps | 1023 / 137 | 22 / 17 | 22 / 2 | 100 / 13 | 25 / 53 |
| Direction guard | 792 / 125 | 24 / 12 | 12 / 8 | 98 / 21 | 20 / 34 |
| Raw residual | 711 / 132 | 19 / 7 | 10 / 6 | 97 / 19 | 19 / 39 |
| Omitted moments | 573 / 105 | 28 / 13 | 9 / 4 | 97 / 31 | 17 / 42 |
| Norm guard | 1055 / 136 | 26 / 16 | 24 / 5 | 107 / 20 | 27 / 52 |
| Swaps + direction guard | 803 / 130 | 20 / 11 | 15 / 6 | 104 / 17 | 20 / 37 |

The initial MATH panel has zero sparse frames; the coverage extension supplies
its sparse evidence. Correlated row/frame counts are not task accuracy trials.
The [decision CSV](../../results/research_update_20261017/attempt001/same_state_decisions.csv)
also exposes KL, previously-unsettled-row metrics, common-noise disagreement,
stop flips, item-level improvements/regressions and request-cluster intervals.
Eager same-state parity does not imply compiled native repeatability.

## Physical work, latency boundaries and available statistics

For each method, report totals and normalize by its own trial count. A fixed
K is a balanced support budget, not matched whole-request work after native
trajectories diverge. Full swaps change GLOBAL kept tiles by +2.66% AIME,
-18.87% LongBench, -2.53% MATH and -0.022% RULER. Their extra full-QK and full-PV
discovery are each 75,482 / 923,246 / 2,891 / 108,390 rectangles per trial.
RULER has only 0.94 held calls/anchor versus LongBench 7.81; amortization differs.

Overall denominator: decode GLOBAL full-support plus LOCAL window-intersecting
head × Q128 × KV64 rectangles, including initial/refresh/carry phases. LOCAL
skips are zero. Requested prefix budget, GLOBAL sparsity and overall sparsity
are distinct. Extra discovery, candidate evaluation, projection and correction
work are reported separately from FA4 consumer work.

| Published file under results/research_update_20261017/attempt001 | Statistics preserved |
|---|---|
| `native_results.csv` | Every numeric native-summary field: score/trials; N/C/T totals, means and medians; p95/caps; GLOBAL/LOCAL/overall work; selector work; extra refinement counters; audit S, W and S/N; cumulative prefix accounting. Nested lists remain JSON cells. |
| `native_comparisons.csv` | Score/N/C/T deltas, both cluster intervals, per-repeat deltas and local screen outcome. |
| `native_repeats.csv` | Each candidate replicate, each V31 bookend and each reference score/mean N. |
| `native_phases.csv` | Dense, initial, carried, refresh and held GLOBAL eligible/kept/skipped counts, fractions and per-trial normalization. Phase calls are layer invocations, not N. |
| `native_control_noise.csv` | Within-engine and cross-engine score/N/C/T variation and intervals; output-hash disagreements. |
| `native_trajectory_exposure.csv` | Anchor/swap/correction exposure, held calls per anchor, outcome-conditioned C/N/T, exact N product decomposition, tail sensitivity. Correction-path counts do not imply nonzero alpha on every row. |
| `operator_screen.csv`, `whole_layer.csv` | All initial operator arms and all whole-layer aggregate error, swap and harm statistics. |
| `reuse_guard.csv`, `reuse_guard_by_layer.csv` | Error ratios, maximum-row harms, mean coefficient and per-layer results. |
| `same_state_decisions.csv` | All initial/coverage decision aggregates and their request-cluster intervals. |
| `final_complete.json` | Exporter hash and public input/output hashes; precisely scoped public validation. |

Per-item/frame distributions, mask IoU, retained mass, reuse age, prediction
residual, exact budgets, source pins and receipts remain in the original numeric
records linked by each campaign. The tables above do not invent unavailable
confidence intervals, causal explanations or clean timings.
Frame-level decision records additionally retain maximum KL/logit error,
entropy and entropy changes, margin crossings, stop direction and previously
unsettled counts. These are diagnostics, not final-answer accuracy statistics.

## Variability, failed attempts and verification

V31 bookend output hashes differ 13/180 AIME, 6/128 LongBench, 1/200 MATH,
0/208 RULER; official correctness differs only once on LongBench. Across engines,
first-control hashes differ 73/90, 4/64, 100/100 and 0/104 respectively. All
structurally valid outcomes remain in the score. The completed control-only
history and ABBA warmup factor show that value intervention is not necessary
for drift; 4/18 control-warmup and 2/18 method-warmup groups drifted. This small
factor neither identifies the numerical cause nor proves a warmup benefit.

Older independent-process accuracy remains interpretation-held; the exact-parity
replacement failed on a long AIME control bookend. They are not pooled into
the completed replicated study. Failed suffix/no-op attempts are retained with
their source/status, never presented as efficacy evidence. Their protocols and
receipts are published alongside the successful stages.

71 H100 tests passed with zero skips/failures; native method/test hashes remain
unchanged. Final native audit verifies 1,099 source files, 225 attempts, 1,710
bound files and all 5,370 records: zero measured graph captures, LOCAL skips or
reported fixed-budget errors. `final_integrity_audit_attempt002.json` supersedes
the first independent LOCAL check's field-name mistake; generation and scores
did not change. GPU reservation was 74,655.1 seconds native plus 4,830.6 seconds
qualification, including startup/warmup, excluding earlier diagnostics. These
are reservation costs, not clean latency. No full503 or timing expansion ran.

Original [conclusions](../../results/v31_value_decision_reuse_20261015/conclusions_20261017.md),
[native completion](../../results/v31_value_decision_reuse_20261015/native_replicated_20261016_attempt001/final_complete.json),
and [study completion](../../results/v31_value_decision_reuse_20261015/final_complete.json)
remain immutable; this publication adds navigation and reproducible exports.
