# V28 held-order regroup component ablation

Source `8927280a1`; H100, one accepted historical prefix state per nominal length bin,
synthetic QKV with native strides and random pages. All three pass IEEE-FP32 masked
references. This tests the favorable held-order case, excluding offline search,
selector and initial list/split construction. Full gather, alias2 and scatter are timed.

| Nominal bin | Held total / natural Q64 | Consumer saving (ms) | Standalone gather/scatter (ms) |
|---|---:|---:|---:|
| 32768 | 1.04498 | 0.01987 | 0.05624 |
| 65536 | 1.00462 | 0.02357 | 0.06163 |
| 98304 | 1.03343 | 0.01701 | 0.06210 |

Geometric mean total ratio: **1.02753** (about 2.8% slower).
Standalone permutation times are not additive with consumer timing; the actual
held-total measurement is primary. Even with search completely amortized away,
this unfused implementation does not win in these three selected states. Do not
advance it to a request panel. This does not rule out every grouping heuristic
or a genuinely fused consumer; the broader CPU screen showed only ~1.06% overall
gated load-proxy saving before costs, so further engineering is low priority.

32 rotated-order timing repeats, held-list rebuild count zero. No request-level
speed/accuracy inference or confidence interval from these three component states.
Reserved GPU time including CPU search/imports/qualification: 45.8499 s.
