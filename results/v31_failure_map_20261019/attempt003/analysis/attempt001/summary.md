# Bounded V31 failure-map diagnostic

Status: complete_no_screen_candidates. GPU reservation 5355.8/14400 seconds.

Execution source `5dcade2c27bbc8092bdbc94b231f9e324b0bcaee`; protocol `0ca7a002319049af7f0a7b379647ddca576af676909aa702ea78c0d9c31cb7e4`.

Development LongBench questions; native sampler and manifest output budgets preserved. Audit and clean outcomes are separate. N excludes per-canvas commit calls and C is derived by the inherited recorder from output length; N/C does not align causal states. No result here establishes held-out generalization or noninferiority.

## qualification

| Arm | K | Requests | Accuracy | N | N/C | T | W (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| control_before | 2048 | 2 | 0.00% | 131.000 | 15.412 | 2026.0 | 6.2918 |
| control_after | 2048 | 2 | 0.00% | 115.000 | 15.333 | 1742.5 | 5.9079 |
| control_before | 4096 | 2 | 50.00% | 115.000 | 13.529 | 1976.0 | 6.0350 |
| control_after | 4096 | 2 | 50.00% | 110.000 | 12.941 | 1952.5 | 5.9111 |
| control_before | 8192 | 2 | 0.00% | 98.500 | 12.312 | 1950.0 | 5.7211 |
| fixed16 | 8192 | 2 | 0.00% | 153.500 | 15.350 | 2399.5 | 6.7817 |
| control_after | 8192 | 2 | 0.00% | 98.500 | 12.312 | 1950.0 | 5.7301 |
| allkept_fa4 | dense | 2 | 50.00% | 130.000 | 14.444 | 2180.0 | 6.9678 |
| dense_full_fix51994 | dense | 2 | 50.00% | 154.000 | 15.400 | 2506.5 | 7.3023 |
| dense_piecewise | dense | 2 | 0.00% | 184.500 | 16.043 | 2810.5 | 8.2146 |

## screen

| Arm | K | Requests | Accuracy | N | N/C | T | W (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| allkept_fa4 | dense | 32 | 59.38% | 203.312 | 19.363 | 2559.0 | 8.9487 |
| dense_full_fix51994 | dense | 32 | 50.00% | 332.656 | 23.091 | 3556.6 | 11.8551 |
| dense_piecewise | dense | 32 | 56.25% | 288.375 | 22.842 | 3107.8 | 10.8529 |
| control_before | 2048 | 32 | 56.25% | 221.531 | 21.417 | 2522.8 | 7.8086 |
| control_after | 2048 | 32 | 56.25% | 254.625 | 22.202 | 2812.0 | 8.4060 |
| control_before | 4096 | 32 | 59.38% | 182.125 | 20.236 | 2173.7 | 7.0745 |
| control_after | 4096 | 32 | 62.50% | 184.188 | 20.254 | 2201.6 | 7.1177 |
| control_before | 8192 | 32 | 62.50% | 200.719 | 20.786 | 2338.8 | 7.5589 |
| fixed16 | 8192 | 32 | 59.38% | 285.781 | 25.193 | 2770.9 | 8.7852 |
| control_after | 8192 | 32 | 62.50% | 202.094 | 20.794 | 2356.7 | 7.6084 |

## Gates

Screen candidates: 0. Confirmed failures: not assessed.
Qualified repair groups: 0/0. Replicated mechanistic leads: 0.

See analysis.json for all paired intervals, bookend drift, selected-case reversals, and repair qualification errors. Repair and qualification W are instrumented and are not speed measurements. Older campaigns remain paused.
