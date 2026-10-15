# Causal value-residual trust replay

Independent refinement after unconditional residual correction showed error-tail regressions. The guard uses current selected output and cached anchor selected/dense outputs, measured after the real O projection. It uses neither future queries nor current omitted-key output. All 16 heads and both query tiles are included. This is an exploratory operator diagnostic on the same previously examined trajectories.

For cached residual r and current selected movement m, the direction guard applies alpha=clip(1 - dot(m,r)/||r||², 0,1). This minimizes error against the old dense anchor, which is a causal surrogate, not a guaranteed bound on current dense error. A norm-only guard is an ablation.

| Request | Method | Sparse held frames | Mean error ratio | Mean-error harms | >2% max-row harms | Mean alpha |
|---|---|---:|---:|---:|---:|---:|
| request001 | mass_anchor | 220 | 1.0000 | 0 | 0 | 0.000 |
| request001 | residual | 220 | 0.6355 | 6 | 26 | 1.000 |
| request001 | residual_direction_guard | 220 | 0.6444 | 0 | 0 | 0.486 |
| request001 | residual_norm_guard | 220 | 0.8281 | 0 | 0 | 0.076 |
| request002 | mass_anchor | 100 | 1.0000 | 0 | 0 | 0.000 |
| request002 | residual | 100 | 0.7399 | 15 | 25 | 1.000 |
| request002 | residual_direction_guard | 100 | 0.7753 | 3 | 3 | 0.411 |
| request002 | residual_norm_guard | 100 | 0.9701 | 0 | 0 | 0.014 |
| request004 | mass_anchor | 30 | 1.0000 | 0 | 0 | 0.000 |
| request004 | residual | 30 | 0.8327 | 9 | 11 | 1.000 |
| request004 | residual_direction_guard | 30 | 0.8291 | 1 | 2 | 0.419 |
| request004 | residual_norm_guard | 30 | 0.9818 | 0 | 0 | 0.017 |

Request001/002 are LongBench; request004 is one late sparse AIME trajectory. All-kept states and anchor calls are excluded. The direction guard preserves much of the mean improvement while reducing observed harmful tails. The norm guard is safer here but often suppresses nearly all correction. Neither establishes task accuracy, native-call savings, or out-of-sample safety. Add both guards to same-state final-layer decision diagnostics before native promotion. Timing remains paused.
