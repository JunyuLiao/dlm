# V2 tau=0.021: complete LongBench-v2 comparison

503 paired seed-1 problems; unchanged native sampler and selector-only FA4 execution. Previously examined benchmark; no held-out claim.

| Method | Correct | Accuracy | N/problem | Median N | C/problem | N/C | Tokens/problem | Kept GLOBAL tiles | GLOBAL sparsity | Overall sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_v31_control | 268/503 | 53.28% | 280.19 | 172.0 | 13.58 | 20.63 | 3348.3 | 5,833,185,760 | 83.20% | 78.12% |
| dense_full_fix51994 | 247/503 | 49.11% | 324.74 | 167.0 | 15.16 | 21.42 | 3754.3 | 41,562,284,640 | 0.00% | 0.00% |
| dense_piecewise | 247/503 | 49.11% | 324.74 | 167.0 | 15.16 | 21.42 | 3754.3 | 41,562,284,640 | 0.00% | 0.00% |
| allkept_fa4 | 252/503 | 50.10% | 292.76 | 162.0 | 14.18 | 20.64 | 3501.8 | 36,334,866,240 | 0.00% | 0.00% |
| v2_tau_0p021 | 254/503 | 50.50% | 247.59 | 166.0 | 12.93 | 19.15 | 3177.7 | 5,243,209,214 | 82.54% | 77.41% |

| V2 versus | Accuracy delta (pp) | Paired bootstrap 95% CI (pp) | McNemar p | N ratio | Kept GLOBAL ratio |
|---|---:|---|---:|---:|---:|
| current_v31_control | -2.783 | [-6.362, +0.795] | 0.1654 | 0.88365 | 0.89886 |
| dense_full_fix51994 | +1.392 | [-2.584, +5.368] | 0.5584 | 0.76242 | 0.12615 |
| dense_piecewise | +1.392 | [-2.584, +5.368] | 0.5584 | 0.76242 | 0.12615 |
| allkept_fa4 | +0.398 | [-3.579, +4.374] | 0.9212 | 0.84570 | 0.14430 |

## Official difficulty and length strata

| Method | Easy | Hard | Short | Medium | Long | Capped | Unparsed | No final response | Input-truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_v31_control | 57.8% | 50.5% | 67.2% | 46.5% | 43.5% | 18 | 30 | 28 | 237 |
| dense_full_fix51994 | 49.0% | 49.2% | 61.1% | 42.8% | 41.7% | 30 | 56 | 48 | 237 |
| dense_piecewise | 49.0% | 49.2% | 61.1% | 42.8% | 41.7% | 30 | 56 | 48 | 237 |
| allkept_fa4 | 51.6% | 49.2% | 65.6% | 41.4% | 41.7% | 26 | 51 | 44 | 237 |
| v2_tau_0p021 | 53.6% | 48.6% | 63.9% | 44.2% | 40.7% | 13 | 34 | 29 | 237 |

GLOBAL work is receipt-measured physical FA4 consumer work. Selector discovery and selection are additional costs.
Equal sparsity, equal retained work, equal native trajectory and equal throughput are distinct claims.
Native stopping stays enabled. Audit wall time is instrumented; clean timing and every other threshold remain paused.
