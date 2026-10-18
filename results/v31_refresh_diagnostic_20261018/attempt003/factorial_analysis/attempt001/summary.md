# Bounded refresh diagnostic

bounded first-event causal diagnostic; instrumented timings are not deployable latency

Coverage: 48/48 complete paired groups. Full-state prefix pairing: **passed**.

Current output uses dense attention or the old common sparse support; future support is freshly selected or held. This experiment does not evaluate a cheap selector.

## longbench_v2

32 paired groups; 8 question clusters.

| Contrast | Score delta | Calls delta | Calls 95% cluster interval | Request delta, diagnostic s |
|---|---:|---:|---|---:|
| sparse_minus_dense | 0.03125 | 7.45312 | [-15.3125, 29.5004] | 0.176035 |
| fresh_minus_hold | 0.03125 | -16.7344 | [-36.9688, 2.85938] | -0.405257 |
| interaction | 0.1875 | -0.59375 | [-49.1875, 47.0312] | -0.0427978 |
| control_after_minus_before | 0 | 0.15625 | [-0.375, 0.84375] | -0.000127188 |

Control bookends disagree in output hash in 2/32 groups and in official score in 0/32 groups.

JSON includes every simple contrast, seed/engine strata, exposure groups, missing coverage, question contributions, leave-one-question-out sensitivity, and A/A variation. All adverse and capped outcomes remain included.

## ruler128k

16 paired groups; 4 question clusters.

| Contrast | Score delta | Calls delta | Calls 95% cluster interval | Request delta, diagnostic s |
|---|---:|---:|---|---:|
| sparse_minus_dense | 0.00625 | 0.375 | [-0.1875, 1.3125] | 0.00745344 |
| fresh_minus_hold | -0.00625 | -0.25 | [-0.5625, 0] | -0.00480281 |
| interaction | -0.0125 | 0 | [-1.125, 1.125] | -0.00439812 |
| control_after_minus_before | 0 | 0 | [0, 0] | -0.00672687 |

Control bookends disagree in output hash in 0/16 groups and in official score in 0/16 groups.

JSON includes every simple contrast, seed/engine strata, exposure groups, missing coverage, question contributions, leave-one-question-out sensitivity, and A/A variation. All adverse and capped outcomes remain included.

## Separate clean runtime headroom

selected batch-one headroom diagnostic; clean timing and audited work remain separate

| Arm | Requests | Mean score | Mean calls | Mean request s |
|---|---:|---:|---:|---:|
| allkept_fa4 | 4 | 0 | 338.5 | 13.2431 |
| dense_full_fix51994 | 4 | 0.5 | 241.5 | 10.2313 |
| dense_piecewise | 4 | 0.5 | 289.5 | 11.7634 |
| dynamic | 4 | 0.5 | 281.5 | 9.75161 |
| fixed16 | 4 | 0 | 212.5 | 8.01623 |

Execution sources: 6aab86325990edf6e5ab6bb72bcd909781234e97, db755c43be763a8af9edfe8a147ec333d8b8542b. Referenced clean leaves retain their original protocol and source; successor profiling is a separate execution source. Compare native trajectories before cost transfer.

JSON reports paired dynamic/fixed16 ratios and substrate-control differences. Audit spans, physical rectangles, and profiler activity are separate. The removable request-critical-path bound remains **unresolved**: inclusive spans and disjoint activity labels do not establish overlap or latency savings.

Score deltas retain each suite's official units. Intervals are descriptive; nonsignificance does not establish equivalence. Instrumented time is not a deployable latency estimate, and fewer calls do not establish speedup.

- Small diagnostic cohorts do not establish benchmark superiority or noninferiority.
- No automatic promotion, full503 resume, or deployable speedup follows from this report.
