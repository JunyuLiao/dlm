# Detailed evaluated-method results

All figures use native adaptive denoising. `N/problem` is mean denoising forwards per item; `N/canvas` is total forwards divided by canvas count; `T` is mean output tokens; `W` is recorded end-to-end wall time from instrumented audit runs and is not a clean timing claim. GLOBAL and overall sparsity are physical audited tile/rectangle sparsities.

## 503-item official LongBench audit

| Method | Accuracy | N/problem | N/canvas | T | W (s) | GLOBAL sparsity | Overall sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_full_fix51994 | 247/503 (49.1%) | 324.74 | 21.42 | 3754.3 | 14.02 | 0.00% | 0.00% |
| dense_piecewise | 247/503 (49.1%) | 324.74 | 21.42 | 3754.3 | 14.27 | 0.00% | 0.00% |
| allkept_fa4 | 252/503 (50.1%) | 292.76 | 20.64 | 3501.8 | 13.29 | 0.00% | 0.00% |
| current_v31_control | 268/503 (53.3%) | 279.71 | 20.57 | 3352.5 | 10.50 | 83.16% | 78.08% |
| value_v1_online_discard_mass | 247/503 (49.1%) | 239.17 | 18.83 | 3118.9 | 11.34 | 80.04% | 75.07% |
| value_v2_online_preserve_mass | 266/503 (52.9%) | 250.64 | 19.68 | 3132.8 | 11.53 | 84.11% | 78.91% |
| value_v3a_singleton_delete | 253/503 (50.3%) | 272.79 | 20.30 | 3309.8 | 11.78 | 83.08% | 78.05% |
| value_v3b_approx_batch8 | 242/503 (48.1%) | 341.26 | 24.78 | 3404.3 | 22.31 | 84.97% | 79.94% |
| value_v2_online_jcgate | 256/503 (50.9%) | 242.59 | 18.91 | 3153.6 | 11.58 | 78.07% | 73.18% |
| value_mass_gated_online_swap8 | 264/503 (52.5%) | 249.86 | 19.65 | 3123.8 | 11.63 | 82.50% | 77.40% |

The full audit paired result for swap8 versus control is -4 items / -0.795 pp, bootstrap 95% CI [-4.97,+3.38] pp, exact McNemar p=0.741.

## 15-item development panel

| Method | Accuracy | N/problem | N/canvas | T | W (s) | GLOBAL sparsity | Overall sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_v31_control | 7/15 (46.7%) | 146.87 | 15.74 | 2230.9 | 6.81 | 76.45% | 69.68% |
| value_mass_gated_online_swap8 | 9/15 (60.0%) | 137.80 | 18.79 | 1767.3 | 10.15 | 79.16% | 72.85% |
| value_mass_gated_online_swap16 | 8/15 (53.3%) | 147.00 | 16.83 | 2093.7 | 7.85 | 78.37% | 72.00% |
| value_mass_gated_online_phase_swap16 | 8/15 (53.3%) | 183.93 | 19.43 | 2291.4 | 8.63 | 81.44% | 75.53% |
| value_mass_gated_online_cgate_swap16 | 6/15 (40.0%) | 132.73 | 18.61 | 1716.6 | 7.33 | 80.38% | 74.63% |
| value_mass_gated_full_swap16 | 7/15 (46.7%) | 323.60 | 26.52 | 3029.2 | 11.25 | 85.97% | 80.66% |
| value_intersection_online | 7/15 (46.7%) | 232.20 | 18.83 | 3004.7 | 11.10 | 86.71% | 80.99% |
| value_intersection_full | 7/15 (46.7%) | 232.20 | 18.83 | 3004.7 | 10.75 | 86.71% | 80.99% |

Development interpretation: swap8 was 9/15, but its improvement did not survive the 503-item audit. The full-information, phase, C_gate, and intersection variants were not promoted.

## Coverage and audit status

- All full-audit arms shown above have 503/503 official scorer coverage.
- The new swap8 audit used four resumable shards with zero timed CUDA graph captures and zero validation errors.
- Timing remains paused; audit `W` values are descriptive instrumented records only.

