# Stable cheap refresh development panel

Previously examined development data; fresh controls, fixed native settings, no clean timing.

## Operator diagnostics

| Selector | True retained mass | Row error mean / p95 / max | Next-call change p95 / max | Swaps | Max cumulative |
|---|---:|---:|---:|---:|---:|
| v31_progress_hold | 0.754724 | 2.8991 / 8.3010 / 19.8184 | 0.0000 / 0.0000 | 0 | 0 |
| raw_s2 | 0.757182 | 2.8438 / 8.1084 / 19.3122 | 0.3668 / 15.9814 | 166215 | 561 |
| stable_l0_g2_s2 | 0.755321 | 2.8871 / 8.2598 / 19.8232 | 0.0470 / 14.3844 | 6073 | 8 |
| hold_no_progress | 0.761065 | 2.8273 / 8.3347 / 19.8742 | 0.0000 / 0.0000 | 0 | 0 |
| stable_probe4_s2 | 0.755265 | 2.8874 / 8.2595 / 19.8138 | 0.0430 / 12.4776 | 5368 | 8 |

## Native accuracy and denoising trajectory

| Arm | Workload | Correct / cells | Accuracy | Mean N | p95 N | Mean N/C | N ratio vs progress V31 | Paired wins/losses |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dense_full_fix51994 | longbench_v2_0shot_think | 25/64 | 39.06% | 338.12 | 972.8 | 18.76 | 1.4073 | 3/12 |
| dense_full_fix51994 | aime26 | 51/90 | 56.67% | 275.17 | 538.5 | 12.06 | 0.9988 | 13/11 |
| dense_piecewise | longbench_v2_0shot_think | 25/64 | 39.06% | 338.12 | 972.8 | 18.76 | 1.4073 | 3/12 |
| dense_piecewise | aime26 | 51/90 | 56.67% | 275.17 | 538.5 | 12.06 | 0.9988 | 13/11 |
| allkept_fa4 | longbench_v2_0shot_think | 29/64 | 45.31% | 323.56 | 1438.5 | 18.65 | 1.3467 | 3/8 |
| allkept_fa4 | aime26 | 51/90 | 56.67% | 278.57 | 562.8 | 12.22 | 1.0111 | 13/11 |
| v31_progress_dense_control | longbench_v2_0shot_think | 34/64 | 53.12% | 240.27 | 490.2 | 17.75 | 1.0000 | 0/0 |
| v31_progress_dense_control | aime26 | 49/90 | 54.44% | 275.50 | 552.1 | 12.10 | 1.0000 | 0/0 |
| raw_global_progress_dense_s2 | longbench_v2_0shot_think | 33/64 | 51.56% | 238.89 | 544.9 | 18.22 | 0.9943 | 4/5 |
| raw_global_progress_dense_s2 | aime26 | 50/90 | 55.56% | 272.01 | 552.1 | 12.02 | 0.9873 | 1/0 |
| stable_global_progress_dense_s2 | longbench_v2_0shot_think | 31/64 | 48.44% | 271.86 | 612.2 | 18.69 | 1.1315 | 4/7 |
| stable_global_progress_dense_s2 | aime26 | 49/90 | 54.44% | 274.34 | 552.1 | 12.06 | 0.9958 | 0/0 |
| v31_no_progress_dense_hold | longbench_v2_0shot_think | 33/64 | 51.56% | 262.69 | 646.9 | 19.79 | 1.0933 | 7/8 |
| v31_no_progress_dense_hold | aime26 | 55/90 | 61.11% | 283.29 | 554.6 | 12.28 | 1.0283 | 11/5 |
| raw_global_no_progress_dense_s1 | longbench_v2_0shot_think | 28/64 | 43.75% | 304.38 | 691.0 | 18.83 | 1.2668 | 3/9 |
| raw_global_no_progress_dense_s1 | aime26 | 55/90 | 61.11% | 283.30 | 554.6 | 12.28 | 1.0283 | 11/5 |
| raw_global_no_progress_dense_s4 | longbench_v2_0shot_think | 32/64 | 50.00% | 249.17 | 530.2 | 18.53 | 1.0371 | 5/7 |
| raw_global_no_progress_dense_s4 | aime26 | 55/90 | 61.11% | 283.28 | 554.6 | 12.28 | 1.0282 | 11/5 |
| raw_global_no_progress_dense_s8 | longbench_v2_0shot_think | 27/64 | 42.19% | 324.14 | 898.5 | 19.38 | 1.3491 | 3/10 |
| raw_global_no_progress_dense_s8 | aime26 | 55/90 | 61.11% | 283.14 | 554.6 | 12.28 | 1.0277 | 11/5 |
| stable_global_no_progress_dense_s1 | longbench_v2_0shot_think | 32/64 | 50.00% | 218.64 | 456.8 | 18.40 | 0.9100 | 4/6 |
| stable_global_no_progress_dense_s1 | aime26 | 55/90 | 61.11% | 283.70 | 554.6 | 12.29 | 1.0298 | 11/5 |
| stable_global_no_progress_dense_s4 | longbench_v2_0shot_think | 32/64 | 50.00% | 273.38 | 864.0 | 19.16 | 1.1378 | 3/5 |
| stable_global_no_progress_dense_s4 | aime26 | 55/90 | 61.11% | 282.70 | 554.6 | 12.26 | 1.0261 | 11/5 |
| stable_global_no_progress_dense_s8 | longbench_v2_0shot_think | 35/64 | 54.69% | 293.83 | 637.0 | 19.26 | 1.2229 | 4/3 |
| stable_global_no_progress_dense_s8 | aime26 | 55/90 | 61.11% | 283.87 | 554.6 | 12.28 | 1.0304 | 11/5 |

## Selector overhead and work

Audit measurements include observation and ranking instrumentation. QK is duplicated for selected tiles and mandatory tail; no selector PV or KV-head union is used. These measurements are not clean latency.

| Arm | Workload | Observations / updates | QK rectangles | Probe rectangles | Audit ms/observation | Max final state MiB | GLOBAL sparsity |
|---|---|---:|---:|---:|---:|---:|---:|
| dense_full_fix51994 | longbench_v2_0shot_think | 0 / 0 | 0 | 0 | 0.000 | 0.00 | 0.00% |
| dense_full_fix51994 | aime26 | 0 / 0 | 0 | 0 | 0.000 | 0.00 | 0.00% |
| dense_piecewise | longbench_v2_0shot_think | 0 / 0 | 0 | 0 | 0.000 | 0.00 | 0.00% |
| dense_piecewise | aime26 | 0 / 0 | 0 | 0 | 0.000 | 0.00 | 0.00% |
| allkept_fa4 | longbench_v2_0shot_think | 0 / 0 | 0 | 0 | 0.000 | 0.00 | 0.00% |
| allkept_fa4 | aime26 | 0 / 0 | 0 | 0 | 0.000 | 0.00 | 0.00% |
| v31_progress_dense_control | longbench_v2_0shot_think | 0 / 0 | 0 | 0 | 0.000 | 171.69 | 82.94% |
| v31_progress_dense_control | aime26 | 0 / 0 | 0 | 0 | 0.000 | 10.41 | 0.00% |
| raw_global_progress_dense_s2 | longbench_v2_0shot_think | 31260 / 31260 | 133042560 | 0 | 12.097 | 348.25 | 83.91% |
| raw_global_progress_dense_s2 | aime26 | 60 / 60 | 255360 | 0 | 11.452 | 30.88 | 0.00% |
| stable_global_progress_dense_s2 | longbench_v2_0shot_think | 36050 / 20102 | 153428800 | 0 | 6.766 | 348.25 | 84.74% |
| stable_global_progress_dense_s2 | aime26 | 80 / 46 | 340480 | 0 | 3.609 | 30.88 | 0.00% |
| v31_no_progress_dense_hold | longbench_v2_0shot_think | 0 / 0 | 0 | 0 | 0.000 | 171.69 | 87.67% |
| v31_no_progress_dense_hold | aime26 | 0 / 0 | 0 | 0 | 0.000 | 10.41 | 0.00% |
| raw_global_no_progress_dense_s1 | longbench_v2_0shot_think | 89090 / 89090 | 379167040 | 0 | 6.353 | 348.25 | 88.22% |
| raw_global_no_progress_dense_s1 | aime26 | 215 / 215 | 915040 | 0 | 2.591 | 30.88 | 0.00% |
| raw_global_no_progress_dense_s4 | longbench_v2_0shot_think | 16240 / 16240 | 69117440 | 0 | 8.014 | 348.25 | 86.28% |
| raw_global_no_progress_dense_s4 | aime26 | 45 / 45 | 191520 | 0 | 2.597 | 30.88 | 0.00% |
| raw_global_no_progress_dense_s8 | longbench_v2_0shot_think | 9835 / 9835 | 41857760 | 0 | 9.628 | 348.25 | 88.08% |
| raw_global_no_progress_dense_s8 | aime26 | 25 / 25 | 106400 | 0 | 2.726 | 30.88 | 0.00% |
| stable_global_no_progress_dense_s1 | longbench_v2_0shot_think | 63015 / 21141 | 268191840 | 0 | 6.222 | 348.25 | 86.91% |
| stable_global_no_progress_dense_s1 | aime26 | 215 / 182 | 915040 | 0 | 3.563 | 30.88 | 0.00% |
| stable_global_no_progress_dense_s4 | longbench_v2_0shot_think | 18525 / 12885 | 78842400 | 0 | 6.732 | 348.25 | 87.87% |
| stable_global_no_progress_dense_s4 | aime26 | 50 / 31 | 212800 | 0 | 3.580 | 30.88 | 0.00% |
| stable_global_no_progress_dense_s8 | longbench_v2_0shot_think | 8740 / 5135 | 37197440 | 0 | 7.079 | 348.25 | 87.78% |
| stable_global_no_progress_dense_s8 | aime26 | 20 / 5 | 85120 | 0 | 3.808 | 30.88 | 0.00% |

Problem-clustered confidence intervals, receipt counters, and physical accounting are in native_workload_results.json. Lower N and higher sparsity are not end-to-end speed claims. The 503-cell expansion is a later gate.

## Development recommendation

- `raw_global_progress_dense_s2`: retain as diagnostic; no expansion yet; trajectory gate passed=False.
- `stable_global_progress_dense_s2`: retain as diagnostic; no expansion yet; trajectory gate passed=False.
- `v31_no_progress_dense_hold`: retain as diagnostic; no expansion yet; trajectory gate passed=False.
- `raw_global_no_progress_dense_s1`: retain as diagnostic; no expansion yet; trajectory gate passed=False.
- `raw_global_no_progress_dense_s4`: retain as diagnostic; no expansion yet; trajectory gate passed=False.
- `raw_global_no_progress_dense_s8`: retain as diagnostic; no expansion yet; trajectory gate passed=False.
- `stable_global_no_progress_dense_s1`: retain as diagnostic; no expansion yet; trajectory gate passed=False.
- `stable_global_no_progress_dense_s4`: retain as diagnostic; no expansion yet; trajectory gate passed=False.
- `stable_global_no_progress_dense_s8`: retain as diagnostic; no expansion yet; trajectory gate passed=False.
