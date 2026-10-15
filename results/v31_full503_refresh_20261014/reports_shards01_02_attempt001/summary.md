# Paused after shard02: LongBench-v2 partial results

All 13 arms completed the same 252 seed-1 problems: 126 in each of shards 01 and 02. The full 503-problem experiment is paused; 251 problems per arm remain. These are partial, previously examined benchmark results, not a completed 503-problem audit. No shard03 worker launched.

| Arm | Shard01 /126 | Shard02 /126 | Combined /252 | Accuracy | Mean N | N vs V31 | N/C | Output tokens | GLOBAL skipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense FULL (fixed) | 71 | 56 | 127 | 50.40% | 310.55 | +6.48% | 18.51 | 3827.9 | 0.00% |
| Dense PIECEWISE | 60 | 57 | 117 | 46.43% | 360.62 | +23.64% | 18.68 | 4063.1 | 0.00% |
| All-kept FA4 | 64 | 58 | 122 | 48.41% | 319.56 | +9.56% | 18.74 | 3728.4 | 0.00% |
| V31 progress control | 72 | 58 | 130 | 51.59% | 291.66 | +0.00% | 18.60 | 3501.7 | 83.30% |
| No-progress hold | 66 | 63 | 129 | 51.19% | 328.33 | +12.57% | 19.35 | 3463.5 | 87.80% |
| Fresh every step | 72 | 62 | 134 | 53.17% | 269.02 | -7.76% | 18.02 | 3350.8 | 91.17% |
| Raw / progress / 2 | 67 | 59 | 126 | 50.00% | 343.20 | +17.67% | 19.01 | 3786.3 | 84.38% |
| Stable / progress / 2 | 70 | 50 | 120 | 47.62% | 289.14 | -0.87% | 18.66 | 3405.6 | 83.17% |
| Raw / no progress / 2 | 65 | 60 | 125 | 49.60% | 267.54 | -8.27% | 18.50 | 3283.7 | 86.77% |
| Stable / no progress / 2 | 71 | 56 | 127 | 50.40% | 279.56 | -4.15% | 18.80 | 3215.9 | 87.03% |
| Stable / no progress / 1 | 65 | 58 | 123 | 48.81% | 284.93 | -2.31% | 18.73 | 3320.1 | 87.24% |
| Stable / no progress / 4 | 66 | 55 | 121 | 48.02% | 305.04 | +4.59% | 19.13 | 3340.3 | 87.54% |
| Stable / no progress / 8 | 70 | 56 | 126 | 50.00% | 295.51 | +1.32% | 18.99 | 3301.1 | 87.38% |

## Paired factor comparisons

| Candidate minus reference | Correct delta | Wins / losses | Accuracy delta pp [95% CI] | N ratio [95% CI] | N/C ratio | Output-length ratio |
|---|---:|---:|---|---|---:|---:|
| Stable / progress / 2 minus Raw / progress / 2 | -6 | 19 / 25 | -2.38 [-7.54, 2.78] | 0.842 [0.710, 0.994] | 0.981 | 0.899 |
| Stable / no progress / 2 minus Raw / no progress / 2 | +2 | 19 / 17 | +0.79 [-3.97, 5.56] | 1.045 [0.877, 1.243] | 1.016 | 0.979 |
| Raw / progress / 2 minus Raw / no progress / 2 | +1 | 24 / 23 | +0.40 [-5.16, 5.56] | 1.283 [1.086, 1.515] | 1.027 | 1.153 |
| Stable / progress / 2 minus Stable / no progress / 2 | -7 | 17 / 24 | -2.78 [-7.94, 1.98] | 1.034 [0.857, 1.252] | 0.992 | 1.059 |
| Stable / no progress / 2 minus Stable / no progress / 1 | +4 | 23 / 19 | +1.59 [-3.57, 6.75] | 0.981 [0.817, 1.184] | 1.004 | 0.969 |
| Stable / no progress / 4 minus Stable / no progress / 1 | -2 | 21 / 23 | -0.79 [-5.95, 4.37] | 1.071 [0.912, 1.259] | 1.022 | 1.006 |
| Stable / no progress / 8 minus Stable / no progress / 1 | +3 | 26 / 23 | +1.19 [-4.37, 6.75] | 1.037 [0.843, 1.283] | 1.014 | 0.994 |
| V31 progress control minus No-progress hold | +1 | 24 / 23 | +0.40 [-4.76, 5.95] | 0.888 [0.712, 1.106] | 0.961 | 1.011 |
| Fresh every step minus V31 progress control | +4 | 22 / 18 | +1.59 [-3.57, 6.75] | 0.922 [0.749, 1.146] | 0.969 | 0.957 |

Intervals use 10,000 paired problem-cluster bootstrap replicates, seed 51994; they are descriptive and unadjusted for multiple comparisons. Nonsignificance does not establish noninferiority. The four interval-2 arms isolate the stable bundle and progress conditionally; intervals 1/2/4/8 are compared only for stable no-progress.

## Operator diagnostics and execution checks

The pre-generation CPU/CUDA/FA4 gates, complete independent factor replay, and new every-step real-Q/K/V qualification passed; see the parent study qualification receipts. The first-round cross-arm audit and this 3,276-cell source/sampler/runtime/hash audit pass. All evaluated requests have zero new CUDA captures and all LOCAL calls remain dense. Warmup graph captures are outside evaluated requests. Native GLOBAL physical counts include actual dense-refresh/carry phases; K=128 is the eligible-prefix budget, with existing protected tail/structural tiles preserved. Short prefixes keep all available tiles.

No-progress hold retains the existing per-canvas initialization/carry and disables mid-canvas progress refresh. Fresh every step selects from current full-support V31 mass without a previous-map bonus or cache, discards the dense observer output, then returns native sparse FA4 output on that same call; its extra observer QK/PV and tail-QK work are separate in native_results.json. No extra transformer/denoising forward is inserted.

## Selector overhead and clean timing

native_results.json reports selector CUDA-event audit totals, state memory, physical work, and instrumented GPU reservation seconds. These are audit measurements, not clean latency. Clean timing remains intentionally absent and paused. N, N/C and output length describe the native trajectory, not speedup. The full study has no final_complete.json. Resume requires an explicit user instruction.
