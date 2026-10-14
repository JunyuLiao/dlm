# Completed controls and short-workload threshold results

The authorized LongBench controls are complete and verified. The full threshold campaign is paused by user request.
No full LongBench V2 threshold run or finer threshold run was launched. No shared threshold can yet be selected.
Accuracy tuning on these evaluated panels is exploratory. Clean timing remains paused.


## LongBench-v2: 503 problems, seed 1

| Method | Correct | Accuracy | Calls/problem | Median calls | Canvases/problem | Calls/canvas | Tokens/problem | GLOBAL sparsity | Overall sparsity | Kept GLOBAL tiles | Work/V31 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_v31_control | 268/503 | 53.28% | 280.19 | 172.0 | 13.58 | 20.63 | 3348.3 | 83.20% | 78.12% | 5,833,185,760 | 1.0000 |
| dense_full_fix51994 | 247/503 | 49.11% | 324.74 | 167.0 | 15.16 | 21.42 | 3754.3 | 0.00% | 0.00% | 41,562,284,640 | 7.1251 |
| dense_piecewise | 247/503 | 49.11% | 324.74 | 167.0 | 15.16 | 21.42 | 3754.3 | 0.00% | 0.00% | 41,562,284,640 | 7.1251 |
| allkept_fa4 | 252/503 | 50.10% | 292.76 | 162.0 | 14.18 | 20.64 | 3501.8 | 0.00% | 0.00% | 36,334,866,240 | 6.2290 |

## AIME26: 30 problems x seeds 42/43/44

| Method | Correct | Accuracy | Calls/problem | Median calls | Canvases/problem | Calls/canvas | Tokens/problem | GLOBAL sparsity | Overall sparsity | Kept GLOBAL tiles | Work/V31 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_v31_control | 53/90 | 58.89% | 287.51 | 270.5 | 21.24 | 13.53 | 5356.5 | 46.08% | 18.02% | 135,081,440 | 1.0000 |
| dense_full_fix51994 | 52/90 | 57.78% | 281.86 | 212.0 | 21.37 | 13.19 | 5379.2 | 0.00% | 0.00% | 244,106,880 | 1.8071 |
| dense_piecewise | 51/90 | 56.67% | 276.78 | 223.0 | 21.19 | 13.06 | 5343.0 | 0.00% | 0.00% | 237,475,360 | 1.7580 |
| allkept_fa4 | 49/90 | 54.44% | 288.22 | 262.5 | 21.57 | 13.36 | 5439.1 | 0.00% | 0.00% | 251,986,240 | 1.8654 |
| V2 tau=0.018 | 45/90 | 50.00% | 284.38 | 263.0 | 21.52 | 13.21 | 5439.1 | 16.09% | 6.31% | 208,920,250 | 1.5466 |
| V2 tau=0.0195 | 52/90 | 57.78% | 284.69 | 240.5 | 21.51 | 13.23 | 5414.7 | 16.56% | 6.45% | 205,343,431 | 1.5201 |
| V2 tau=0.021 | 60/90 | 66.67% | 278.68 | 262.0 | 21.46 | 12.99 | 5398.7 | 17.11% | 6.53% | 192,949,091 | 1.4284 |
| V2 tau=0.0225 | 55/90 | 61.11% | 284.14 | 236.5 | 21.57 | 13.18 | 5434.1 | 19.05% | 7.40% | 197,470,094 | 1.4619 |
| V2 tau=0.024 | 50/90 | 55.56% | 287.77 | 244.5 | 21.76 | 13.23 | 5490.3 | 20.62% | 8.09% | 200,003,157 | 1.4806 |

## MATH500: fixed sampled 100 problems, seed 1

| Method | Correct | Accuracy | Calls/problem | Median calls | Canvases/problem | Calls/canvas | Tokens/problem | GLOBAL sparsity | Overall sparsity | Kept GLOBAL tiles | Work/V31 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_v31_control | 95/100 | 95.00% | 62.02 | 49.0 | 6.38 | 9.72 | 1509.1 | 10.13% | 2.23% | 18,899,040 | 1.0000 |
| dense_full_fix51994 | 93/100 | 93.00% | 63.65 | 45.0 | 6.43 | 9.90 | 1528.5 | 0.00% | 0.00% | 22,954,560 | 1.2146 |
| dense_piecewise | 95/100 | 95.00% | 59.98 | 45.0 | 6.19 | 9.69 | 1464.5 | 0.00% | 0.00% | 20,227,360 | 1.0703 |
| allkept_fa4 | 95/100 | 95.00% | 59.05 | 47.0 | 6.15 | 9.60 | 1459.9 | 0.00% | 0.00% | 19,532,640 | 1.0335 |
| V2 tau=0.018 | 92/100 | 92.00% | 60.64 | 47.0 | 6.24 | 9.72 | 1480.3 | 5.26% | 1.16% | 19,593,063 | 1.0367 |
| V2 tau=0.0195 | 94/100 | 94.00% | 64.21 | 46.0 | 6.44 | 9.97 | 1531.9 | 6.02% | 1.35% | 21,294,493 | 1.1267 |
| V2 tau=0.021 | 95/100 | 95.00% | 61.80 | 47.5 | 6.25 | 9.89 | 1479.5 | 5.81% | 1.28% | 19,552,374 | 1.0346 |
| V2 tau=0.0225 | 95/100 | 95.00% | 59.79 | 45.5 | 6.22 | 9.61 | 1474.8 | 6.07% | 1.31% | 18,526,039 | 0.9803 |
| V2 tau=0.024 | 92/100 | 92.00% | 59.77 | 43.0 | 6.13 | 9.75 | 1454.2 | 7.10% | 1.55% | 18,532,236 | 0.9806 |

## LongBench difficulty and length strata

| Method | Easy | Hard | Short | Medium | Long |
|---|---:|---:|---:|---:|---:|
| current_v31_control | 57.8% | 50.5% | 67.2% | 46.5% | 43.5% |
| dense_full_fix51994 | 49.0% | 49.2% | 61.1% | 42.8% | 41.7% |
| dense_piecewise | 49.0% | 49.2% | 61.1% | 42.8% | 41.7% |
| allkept_fa4 | 51.6% | 49.2% | 65.6% | 41.4% | 41.7% |

## AIME26 per-seed accuracy

| Method | Seed 42 | Seed 43 | Seed 44 | Total |
|---|---:|---:|---:|---:|
| allkept_fa4 | 15/30 | 17/30 | 17/30 | 49/90 |
| current_v31_control | 17/30 | 18/30 | 18/30 | 53/90 |
| dense_full_fix51994 | 19/30 | 18/30 | 15/30 | 52/90 |
| dense_piecewise | 17/30 | 18/30 | 16/30 | 51/90 |
| v2_tau_0p018 | 16/30 | 15/30 | 14/30 | 45/90 |
| v2_tau_0p0195 | 16/30 | 20/30 | 16/30 | 52/90 |
| v2_tau_0p021 | 21/30 | 19/30 | 20/30 | 60/90 |
| v2_tau_0p0225 | 19/30 | 20/30 | 16/30 | 55/90 |
| v2_tau_0p024 | 19/30 | 14/30 | 17/30 | 50/90 |

## Parsing and coverage

| Workload | Method | Scored | Capped | Unparsed | No final response | Input-truncated items |
|---|---|---:|---:|---:|---:|---:|
| aime26 | allkept_fa4 | 90 | 35 | 32 | 32 | n/a |
| aime26 | current_v31_control | 90 | 32 | 28 | 27 | n/a |
| aime26 | dense_full_fix51994 | 90 | 30 | 30 | 29 | n/a |
| aime26 | dense_piecewise | 90 | 31 | 30 | 30 | n/a |
| aime26 | v2_tau_0p018 | 90 | 35 | 33 | 32 | n/a |
| aime26 | v2_tau_0p0195 | 90 | 33 | 27 | 27 | n/a |
| aime26 | v2_tau_0p021 | 90 | 24 | 23 | 22 | n/a |
| aime26 | v2_tau_0p0225 | 90 | 30 | 25 | 25 | n/a |
| aime26 | v2_tau_0p024 | 90 | 34 | 31 | 31 | n/a |
| longbench_v2 | allkept_fa4 | 503 | 26 | 51 | 44 | 237 |
| longbench_v2 | current_v31_control | 503 | 18 | 30 | 28 | 237 |
| longbench_v2 | dense_full_fix51994 | 503 | 30 | 56 | 48 | 237 |
| longbench_v2 | dense_piecewise | 503 | 30 | 56 | 48 | 237 |
| math500 | allkept_fa4 | 100 | 4 | 4 | 4 | n/a |
| math500 | current_v31_control | 100 | 3 | 3 | 3 | n/a |
| math500 | dense_full_fix51994 | 100 | 7 | 6 | 6 | n/a |
| math500 | dense_piecewise | 100 | 4 | 4 | 4 | n/a |
| math500 | v2_tau_0p018 | 100 | 6 | 6 | 6 | n/a |
| math500 | v2_tau_0p0195 | 100 | 5 | 6 | 6 | n/a |
| math500 | v2_tau_0p021 | 100 | 5 | 3 | 3 | n/a |
| math500 | v2_tau_0p0225 | 100 | 2 | 2 | 2 | n/a |
| math500 | v2_tau_0p024 | 100 | 4 | 6 | 6 | n/a |

## Paired accuracy versus V31

| Workload | Method | Delta (pp) | Problem-bootstrap 95% interval (pp) | V31-only / candidate-only correct | Exploratory McNemar p |
|---|---|---:|---|---:|---:|
| aime26 | allkept_fa4 | -4.44 | [-14.44, +4.44] | 12 / 8 | 0.5034 |
| aime26 | dense_full_fix51994 | -1.11 | [-12.22, +10.00] | 11 / 10 | 1.0000 |
| aime26 | dense_piecewise | -2.22 | [-12.22, +5.56] | 8 / 6 | 0.7905 |
| aime26 | v2_tau_0p018 | -8.89 | [-16.67, -1.11] | 13 / 5 | 0.0963 |
| aime26 | v2_tau_0p0195 | -1.11 | [-8.89, +6.67] | 7 / 6 | 1.0000 |
| aime26 | v2_tau_0p021 | +7.78 | [-2.22, +17.78] | 6 / 13 | 0.1671 |
| aime26 | v2_tau_0p0225 | +2.22 | [-7.78, +13.33] | 9 / 11 | 0.8238 |
| aime26 | v2_tau_0p024 | -3.33 | [-10.00, +3.33] | 9 / 6 | 0.6072 |
| longbench_v2 | allkept_fa4 | -3.18 | [-6.56, +0.20] | 47 / 31 | 0.0888 |
| longbench_v2 | dense_full_fix51994 | -4.17 | [-7.95, -0.40] | 57 / 36 | 0.0375 |
| longbench_v2 | dense_piecewise | -4.17 | [-7.95, -0.40] | 57 / 36 | 0.0375 |
| math500 | allkept_fa4 | +0.00 | [-4.00, +4.00] | 2 / 2 | 1.0000 |
| math500 | dense_full_fix51994 | -2.00 | [-6.00, +2.00] | 3 / 1 | 0.6250 |
| math500 | dense_piecewise | +0.00 | [-4.00, +4.00] | 2 / 2 | 1.0000 |
| math500 | v2_tau_0p018 | -3.00 | [-8.00, +1.00] | 4 / 1 | 0.3750 |
| math500 | v2_tau_0p0195 | -1.00 | [-6.00, +3.00] | 3 / 2 | 1.0000 |
| math500 | v2_tau_0p021 | +0.00 | [-3.00, +3.00] | 1 / 1 | 1.0000 |
| math500 | v2_tau_0p0225 | +0.00 | [-3.00, +3.00] | 1 / 1 | 1.0000 |
| math500 | v2_tau_0p024 | -3.00 | [-8.00, +1.00] | 4 / 1 | 0.3750 |

Interpretation: AIME uses 30 problem clusters, keeping its three seeds together; the cell-level McNemar value is exploratory.
The V2 0.021 AIME accuracy difference is +7.78 pp, with a paired problem-bootstrap interval spanning zero.
V2 0.021 retains more physical GLOBAL work than V31 on the short workloads; fewer denoising calls alone do not establish speedup.
GLOBAL work counts audited Q128 x KV64 rectangles over native denoising calls. LOCAL is dense; prefill and commit work are separate.
Canvas C is the reported visible output canvas count; receipt canvas IDs are ranked separately for progress-axis work.
LongBench truncated counts describe frozen input truncation, while capped counts describe the generation output limit.
MATH500 uses bound math-verify symbolic equivalence with the official final-response parser; the fixed subset effective sampling seed is 20261111.

Pending, resumable: the five initial LongBench V2 thresholds and two finer thresholds on all three workloads.
Existing frozen protocols, completed attempts, and private generations are retained. Resume only after user instruction.
