# Completed three-direction study: findings and limits

All three directions were designed, implemented and screened on identical
states before native accuracy evaluation. The resulting four-arm native study
is complete. Bounded full-value swaps are the strongest candidate, but none
passes the predeclared requirement to improve the accuracy/call tradeoff
across every workload. No full503 expansion is promoted. Timing remains paused.

The inherited cross-step-reuse machinery is Yuhan's V31 baseline. The new
selection, trust and reconstruction variants belong to this study; they are
not relabelings of historical V2 or the earlier Gaussian32/C_gate methods.

## What was implemented and what passed each gate

1. **Mass-qualified, fixed-budget value swaps.** Start from the actual V31 map;
   form a pool by mass rank, not KV position. Evaluate full-dimensional signed
   output error through the real O projection, with bounded proposals and
   combined-support recomputation. Admit at most eight swaps, protecting row
   mass and worst-row error. K never changes. Full-information ranking and an
   online full-V proposal variant were both evaluated. The online variant uses
   full-information admission and is explicitly a hybrid, not historical V2.
   Full ranking was stronger in the decision screen and advanced to native runs.
   C_gate did not outperform neutral/shuffled weights consistently and stopped
   at diagnostics.
2. **Reuse-lifetime prediction and directional trust.** Extrapolating the
   previous query movement failed the age-one prediction screen on all 100
   examined sparse units. It was rejected. The refined online guard instead
   uses the current selected output and cached anchor residual. With post-O
   movement m and residual r, alpha=clip(1-<m,r>/||r||²,0,1). This minimizes a
   historical-anchor surrogate; it is not a bound on current dense error or
   task loss. A fused held-call kernel avoids O projection and omitted-key
   scans on held calls. It advanced alongside its raw-residual ablation.
3. **Reconstruction of omitted output.** Raw anchor residual and omitted
   numerator/denominator moments were evaluated separately. Moment correction
   improved some average metrics but produced harmful row/acceptance tails;
   it was not promoted. Raw residual was retained as a mechanistic baseline.
   Guarded residual and swaps-plus-guard test whether trust or selection makes
   reconstruction useful. These are changed output consumers, not selector-only
   variants.

Native deployment is limited to GLOBAL layer 29, the layer qualified by the
same-state suffix intervention. Earlier GLOBAL layers retain V31 behavior.
Every head/Q128 tile keeps K eligible prefix KV64 blocks (27 for AIME/MATH,
128 for LongBench/RULER), plus the common mandatory/carry blocks. LOCAL is
dense. Proposal, acceptance, stopping, re-noising, self-conditioning and output
limits remain native. No diagnostic suffix forward is counted as native work.

## Evidence before native evaluation

The whole-layer operator replay used all 16 heads and both query tiles across
five GLOBAL layers, with 570 usable frames from four previously examined
trajectories. Full swaps reduced mean post-O error approximately 1.5–2.7% on
the two sparse LongBench trajectories and 2.2% on the sparse AIME trajectory.
The guarded residual reduced it 22.5–35.6% on LongBench and 17.1% on AIME,
while substantially reducing the raw residual's harmful maximum-row tails.

Same-state model diagnostics covered 57 paired inputs, of which 39 supplied
265 sparse held frames. Observation-only outputs, N/C/T, RNG and exact eager
suffix parity were checked. In the initial LongBench screen, dense-reference
top-1/acceptance disagreements were 1068/134 for V31, 1035/132 for full swaps,
and 792/125 for the guard. These gains justified a development test, not an
accuracy prediction. Frames are correlated, coverage extensions are purposive,
and eager diagnostic states differ from compiled native trajectories.

The unchanged method source passed 71 H100 tests with no failures/skips; its
12 recorded source/test hashes still match. Coordinator/analysis checks and
all four workload qualifications also passed. Bounded batched pair evaluation,
an optimized full-V online Triton traversal and a fused correction kernel were
implemented, but no clean timing benchmark was run.

Sources: `whole_layer/report.md`, `reuse_guard/report.md`,
`native_inclusion.md`, and `receipts/h100_qualification/attempt002/receipt.json`.

## Official native results

Each entry below is official score / mean denoising calls per trial. RULER is
the official task-average score on a 0–100 scale; the others are accuracy %.

| Method | AIME26 | LongBench-v2 | MATH100 | RULER128K |
|---|---:|---:|---:|---:|
| V31 control | 51.111 / 295.300 | 45.703 / 251.039 | 95.000 / 61.468 | 86.250 / 5.240 |
| Full-value swaps | 52.222 / 301.672 | 46.875 / 192.023 | 95.000 / 60.590 | 87.308 / 5.231 |
| Direction-guarded residual | 51.111 / 296.911 | 43.750 / 249.516 | 95.500 / 60.970 | 86.058 / 5.317 |
| Raw residual | 50.556 / 296.861 | 43.750 / 327.094 | 94.500 / 61.145 | 86.346 / 5.375 |
| Swaps + direction guard | 51.111 / 298.611 | 47.656 / 299.438 | 95.000 / 60.725 | 85.962 / 5.481 |

Full results, all dense/all-kept references, N/C, canvases, output lengths,
tails, cap rates, intervals and physical work are in
`synthesis_20261017/attempt001/summary.md`, `results.csv` and `comparisons.csv`.
The accompanying PNG/SVG/PDF show paired accuracy/call changes and uncertainty.

**AIME blocks promotion.** Every candidate has higher mean N. Full swaps gain
1.111 score points but add 6.372 calls (+2.16%). The score interval is
[-2.778,5.000] points; its sign reverses between engine repetitions. This is
not reliable accuracy improvement.

**LongBench supports bounded swaps, mainly through tail reduction.** Full
swaps save 59.016 calls (-23.51%); the problem-cluster interval is
[-143.250,-3.988], and the separate engine-block interval is
[-119.172,-14.945]. The +1.172-point accuracy interval [-6.250,8.594] includes
zero. Calls fall on 41/64 cells, but five cells explain 89.69% of net savings.
One always-wrong length-capped cell alone explains 58.56%: 2549→337 calls,
16384→3026 tokens. Twenty-six always-correct cells also save 38.231 calls on
average. Outcome conditioning and trimming are descriptive, selection-biased
diagnostics, not replacement estimands. Algebraically, -25.376 calls come from
the canvas-count term and -33.640 from the calls-per-canvas term; this is not
a causal attribution to improved reasoning or stopping.

**MATH provides a small, low-exposure signal.** Combined swaps/guard saves
0.7425 calls (-1.21%) with unchanged score. Its problem interval is
[-1.5775,-0.0575]. The guard's +0.5-point score difference corresponds to one
additional correct candidate trial among 200. Only 46/200 candidate trials
reach a sparse anchor, and 45 execute a correction path. This largely all-kept
workload cannot establish broad sparse-routing efficacy.

**RULER's swap gain is repeatable but narrow.** Full swaps add 1.058 score
points, with task-cluster interval [0,3.077], while N changes by just -0.0096
(-0.18%, interval [-0.0577,0.0481]). All candidate score, output-hash and N/C/T
vectors match exactly across the two repetitions. The gain comes from one
`niah_multikey_3` cell changing 0→100 and two `cwe` cells changing 60→90 and
80→60. Eleven task scores are unchanged; identical repeats do not double the
independent evidence. All three correction arms increase mean N. See
`reports/ruler128k/task_diagnostic.json` under the replicated native study.

## Why the operator gains did not deliver the intended native improvement

The data establish a mismatch between the optimized surrogate and the native
objective. They do not identify one unique causal mechanism.

* Smaller mean attention-output error is insufficient. Guarded reconstruction
  has much larger operator gains than swapping, yet loses 1.953 LongBench
  accuracy points and barely changes N. Raw reconstruction increases N by
  30.30%. This directly rejects ranking these methods by mean PV error alone.
* The historical dense anchor can become stale. The guard limits correction
  against that anchor, not the unknown current dense output. Its operator
  protection is useful, but it does not certify acceptance or stopping over
  a changed native trajectory. The failed query extrapolation also argues
  against assuming smooth, predictable next-query movement.
* Selection and correction are not empirically additive. Adding the guard to
  full swaps changes LongBench N from 192.023 to 299.438 and loses the RULER
  swap score gain. This combined variant does not improve the tradeoff.
* Cross-step reuse has little amortization opportunity on some workloads.
  RULER has one canvas per request and about five calls. Swaps average only
  0.94 held calls per anchor, versus 7.81 on LongBench. Extra full-QK/PV
  discovery is therefore especially difficult to justify from call savings
  on RULER. This is work accounting, not a measured latency conclusion.

The useful next research target is the **admission and lifetime of a bounded
swap**, keeping V31 as the fixed-budget anchor. Evidence favors that over
unconditional reconstruction or treating attention error as the final objective.
A future design must show on same states which swaps preserve native decision
margins over held calls, then eliminate the AIME call regression in a fresh
frozen panel. This study does not yet provide that predictor. C_gate's shuffled
ablation and failed temporal extrapolation prevent claiming either as a solved
mechanism. No additional variant or full accuracy campaign was silently launched.

## Fairness, variability, work and completion

Unchanged V31 showed both cross-engine variation and occasional same-engine
drift, including in control-only experiments. Neither warmup policy eliminated
it. The original independent-process and exact-parity campaigns retain their
held/failed status. A separately frozen replicated design retained every
structurally valid outcome: two engines per cell, reversed candidate order,
and both V31 bookends averaged. No best-of-two scoring or parity-based deletion
was used. Separate-engine dense references remain contextual comparisons.

The final campaign covers 358 input/seed cells and 5,370 officially scored
trials. It is exploratory, using previously examined prompts. Separate
problem/task and engine-block bootstrap intervals are not a simultaneous
confidence region and do not establish noninferiority. The all-workload
promotion screen has zero survivors.

The final integrity audit verifies 1,099 frozen execution files, 225 completed
attempts and 1,710 bound files. All 5,370 measured records have zero CUDA graph
captures and zero LOCAL skips; receipts report zero fixed-budget errors.
Use `final_integrity_audit_attempt002.json`: the first independent audit used
the wrong LOCAL field name and was superseded by a required-field check on
every record. No generation or reported metric changed.

Fixed K does not imply equal total work after trajectories diverge. Full-swap
GLOBAL kept work changes +2.66% on AIME, -18.87% on LongBench, -2.53% on MATH
and -0.022% on RULER. Extra full-QK and full-PV discovery, each per trial, are
75,482 / 923,246 / 2,891 / 108,390 rectangles respectively. These costs are
separate from the sparse consumer. Overall sparsity includes dense phases and
LOCAL window-intersecting work; it must not be confused with the prefix budget.

The replicated native campaign reserved the GPU for 74,655.1 seconds, plus
4,830.6 seconds for its qualifications. These elapsed reservations include
startup and warmup and exclude earlier diagnostic campaigns; they are not
kernel time or clean generation latency. The coordinator exited successfully,
the H100 has no compute process, all partial/failed attempts remain intact,
and clean timing remains paused.
