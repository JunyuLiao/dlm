# Native development inclusion decision

Frozen after completed same-state diagnostics; this qualifies a development test, not task accuracy or a deployment claim. Changes remain restricted to GLOBAL layer29. Clean timing remains paused.

Each entry below is dense-reference top1 / acceptance disagreement row comparisons. Frames within a request are correlated; the coverage extension is purposive and is kept separate.

| Panel | Sparse items / frames | V31 mask | Full swaps | Direction guard | Raw residual | Swaps + guard |
|---|---:|---:|---:|---:|---:|---:|
| LongBench initial | 8 / 99 | 1068 / 134 | 1035 / 132 | 792 / 125 | 711 / 132 | 803 / 130 |
| AIME initial | 5 / 35 | 26 / 16 | 21 / 14 | 24 / 12 | 19 / 7 | 20 / 11 |
| MATH long-trajectory coverage | 7 / 75 | 107 / 20 | 101 / 19 | 98 / 21 | 97 / 19 | 104 / 17 |
| RULER first-per-task | 5 / 14 | 24 / 4 | 23 / 2 | 12 / 8 | 10 / 6 | 15 / 6 |
| RULER additional vt/cwe | 14 / 42 | 27 / 52 | 25 / 51 | 20 / 34 | 19 / 39 | 20 / 37 |

## Included native modes

* `full_swap8`: direction1; preserves fixed K and mass/worst-row admission. Full ranking is selected over the online hybrid because initial LongBench and RULER top1 behavior is stronger; the online method remains a completed diagnostic ablation.
* `residual_direction_guard`: refined direction2 with direction3 reconstruction. It uses only current selected output and cached anchor data on held calls; it performs no omitted-key scan there.
* `residual`: direction3 mechanistic baseline. Raw residual caching is a known idea, not the novelty claim. Its inclusion is necessary to determine whether the direction guard improves native accuracy or trajectories beyond caching alone; it is not being certified as tail-safe.
* `full_swap8_direction_guard`: tests whether bounded support changes and online residual trust are complementary on native trajectories. Operator and decision evidence does not establish that the combination will beat either component.

## Excluded from this native panel

* Previous-query tangent: failed age1 prediction on all100 screened sparse units.
* C_gate: no reproducible advantage over neutral or shuffled weights. In MATH, C_gate has14 acceptance disagreements and shuffled weights13; both change the row objective without changing K.
* Cached omitted moments: stronger average KL/top1 gains, but MATH acceptance disagreements31 versus20 for mass. The initial RULER sample also contains one delayed-stop discrepancy. This does not support it as the next fewer-call candidate.
* Norm-only trust: almost a no-op on short tasks; direction-specific trust retains more useful correction.

## Limits and required native qualification

Initial data comprise35 paired items; the extension adds22. Every uninstrumented/active pair preserves output hashes,N,C,T and exact eager suffix parity. Only39 requests provide sparse held-frame evidence. Earlier GLOBAL layers have no suffix qualification and remain unchanged. Eager decision diagnostics cannot predict official correctness or a native call-count gain. AIME changes are small relative to the measured consumer numerical floor.

Guarded correction still has one more MATH acceptance disagreement (21 versus20), while its MATH top1 count improves107→98. RULER first-per-task acceptance worsens4→8, but additional vt/cwe acceptance improves52→34. These tradeoffs remain explicit; the development panel resolves them with official scores and native stopping.

Raw residual introduces one delayed dense-reference stop on initial LongBench; the guard avoids it. In additional RULER, the single premature dense-reference stop is shared by the mass baseline and every included mode; moment reconstruction removes it. None of these events alone establishes an accuracy effect.

Native development uses64 LongBench,90 AIME,100 MATH and104 RULER cells. Four fresh controls remain: V31,FULL fix51994,PIECEWISE dense and all-kept FA4. Sparse activation, fixed K, actual compiled dispatch and official scorer binding must pass smokes before the panels. Full503 promotion still requires no observed score/N loss per workload, at least one score gain, no material p95/cap regression, and complete accounting. No clean timing or speedup claim is authorized by this decision.
