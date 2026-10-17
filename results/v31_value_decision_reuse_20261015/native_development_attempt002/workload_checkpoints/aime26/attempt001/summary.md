# Complete aime26 native checkpoint

complete paired workload only; remaining workloads and full503 promotion remain pending; no adaptive arm changes.

| Arm | Official score | N/problem-trial | N/canvas | Canvases/trial | Output tokens/trial | Cap rate |
|---|---:|---:|---:|---:|---:|---:|
| current_v31_control | 53.333 | 290.111 | 13.403 | 21.644 | 5473.3 | 0.400 |
| dense_full_fix51994 | 58.889 | 272.744 | 13.148 | 20.744 | 5227.9 | 0.344 |
| dense_piecewise | 51.111 | 294.522 | 13.476 | 21.856 | 5517.6 | 0.433 |
| allkept_fa4 | 54.444 | 283.733 | 13.511 | 21.000 | 5293.4 | 0.389 |
| full_swap8 | 57.778 | 293.489 | 13.580 | 21.611 | 5461.8 | 0.400 |
| residual_direction_guard | 56.667 | 280.744 | 13.051 | 21.511 | 5425.1 | 0.378 |
| residual | 55.556 | 284.700 | 13.276 | 21.444 | 5412.1 | 0.389 |
| full_swap8_direction_guard | 53.333 | 297.078 | 13.732 | 21.633 | 5444.4 | 0.400 |

Paired differences versus V31; 95% cluster-bootstrap intervals are exploratory.

| Arm | Score difference (points) | 95% interval | Calls/trial difference | 95% interval |
|---|---:|---|---:|---|
| dense_full_fix51994 | +5.556 | [-3.333, +14.444] | -17.367 | [-34.778, -1.967] |
| dense_piecewise | -2.222 | [-12.222, +7.778] | +4.411 | [-11.723, +22.333] |
| allkept_fa4 | +1.111 | [-7.778, +11.111] | -6.378 | [-25.601, +12.367] |
| full_swap8 | +4.444 | [-3.333, +12.222] | +3.378 | [-12.489, +19.233] |
| residual_direction_guard | +3.333 | [-5.556, +11.111] | -9.367 | [-29.445, +10.922] |
| residual | +2.222 | [-7.778, +13.333] | -5.411 | [-27.445, +19.344] |
| full_swap8_direction_guard | -0.000 | [-6.667, +5.556] | +6.967 | [-10.567, +27.589] |
