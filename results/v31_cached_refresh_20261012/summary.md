# V31 cached-refresh study summary

The 154-cell native panel is complete for the 16 runnable arms: 64 LongBench-v2 subset cells plus 90 AIME26 cells (30 problems × 3 seeds). Official scoring and receipts cover every completed arm. The frozen spec also names four variants whose native map wiring was not available in this study; they remain operator diagnostics and are listed separately below.

## Native accuracy and trajectory

| Arm | LongBench-v2 (64) | AIME avg@k (90) | Mean denoising forwards | Mean ΔN vs V31 | Cells with N > V31 | GLOBAL physical sparsity | Mean map IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dense_full_fix51994` | 46.9% | 50.0% | 282.97 | 43.99 | 86/154 | 0.0% | — |
| `dense_piecewise` | 39.1% | 56.7% | 301.33 | 62.35 | 77/154 | 0.0% | — |
| `allkept_fa4` | 45.3% | 56.7% | 297.27 | 58.29 | 81/154 | 0.0% | — |
| `current_v31_control` | 53.1% | 54.4% | 238.98 | 0.00 | 0/154 | 73.6% | — |
| `cached_refresh_global_topk_m160` | 51.6% | 55.6% | 238.19 | -0.79 | 37/154 | 74.9% | 0.7344 |
| `cached_refresh_global_topk_m256` | 51.6% | 56.7% | 238.31 | -0.67 | 36/154 | 75.2% | 0.7318 |
| `cached_refresh_global_topk_m512` | 51.6% | 54.4% | 238.51 | -0.47 | 35/154 | 74.9% | 0.7347 |
| `cached_refresh_bounded_m160` | 46.9% | 53.3% | 375.33 | 136.35 | 117/154 | 75.7% | 0.9465 |
| `cached_refresh_bounded_m256` | 46.9% | 54.4% | 374.01 | 135.03 | 117/154 | 75.5% | 0.9468 |
| `cached_refresh_bounded_m512` | 46.9% | 53.3% | 374.36 | 135.38 | 117/154 | 75.5% | 0.9468 |
| `cached_refresh_bounded_m256_decay` | 50.0% | 55.6% | 327.27 | 88.29 | 116/154 | 71.8% | 0.9088 |
| `cached_refresh_bounded_m256_phase_protected` | 45.3% | 53.3% | 348.88 | 109.90 | 108/154 | 78.2% | 0.9545 |
| `cached_refresh_bounded_m256_cgate` | 48.4% | 53.3% | 366.08 | 127.10 | 120/154 | 76.0% | 0.9483 |
| `cached_refresh_bounded_m256_every_1_step` | 50.0% | 54.4% | 370.95 | 131.97 | 118/154 | 72.4% | 0.9528 |
| `cached_refresh_bounded_m256_every_4_steps` | 46.9% | 54.4% | 332.86 | 93.88 | 103/154 | 74.1% | 0.9421 |
| `cached_refresh_bounded_m256_progress_triggered` | 46.9% | 53.3% | 374.83 | 135.85 | 117/154 | 75.4% | 0.9471 |

The current V31 control is 53.1% on the LongBench subset and 54.44% AIME avg@k. No cached arm exceeds that LongBench control in this panel. The global-top-K arms have near-control mean denoising counts, but each still has some cells with more forwards than V31. Bounded-swap arms show substantially more cells above the V31 trajectory gate; they are not promotion candidates under the frozen fixed-trajectory rule.

## Selector and fairness audit

| Arm | Policy/config | Dense refreshes | Cheap refreshes | Updates / no-op | Swaps | Forced dense fallbacks | Selector audit (s) | LOCAL skipped | Graph captures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cached_refresh_global_topk_m160` | global_topk, M=160, B=0, period=2 | 770 | 66080 | 23610 / 42470 | 60839801 | 0 | 500.9 | 0 | 0 |
| `cached_refresh_global_topk_m256` | global_topk, M=256, B=0, period=2 | 770 | 66155 | 23895 / 42260 | 61487001 | 0 | 510.8 | 0 | 0 |
| `cached_refresh_global_topk_m512` | global_topk, M=512, B=0, period=2 | 770 | 66140 | 23605 / 42535 | 60834152 | 0 | 510.6 | 0 | 0 |
| `cached_refresh_bounded_m160` | bounded_swaps, M=160, B=8, period=2 | 770 | 118210 | 91373 / 26837 | 9105889 | 0 | 1004.2 | 0 | 0 |
| `cached_refresh_bounded_m256` | bounded_swaps, M=256, B=8, period=2 | 770 | 117805 | 91015 / 26790 | 8975180 | 0 | 1009.7 | 0 | 0 |
| `cached_refresh_bounded_m512` | bounded_swaps, M=512, B=8, period=2 | 770 | 117855 | 91003 / 26852 | 8976451 | 0 | 1025.3 | 0 | 0 |
| `cached_refresh_bounded_m256_decay` | bounded_swaps, M=256, B=8, period=2, decay | 770 | 100912 | 99709 / 1203 | 7565930 | 2 | 861.2 | 0 | 0 |
| `cached_refresh_bounded_m256_phase_protected` | bounded_swaps, M=256, B=8, period=2, phase | 770 | 108061 | 65933 / 42128 | 7570098 | 2 | 932.3 | 0 | 0 |
| `cached_refresh_bounded_m256_cgate` | bounded_swaps, M=256, B=8, period=2, C_gate tie | 770 | 115345 | 88354 / 26991 | 8315913 | 0 | 1140.8 | 0 | 0 |
| `cached_refresh_bounded_m256_every_1_step` | bounded_swaps, M=256, B=8, period=1 | 770 | 245516 | 166853 / 78663 | 15109775 | 2 | 2030.9 | 0 | 0 |
| `cached_refresh_bounded_m256_every_4_steps` | bounded_swaps, M=256, B=8, period=4 | 770 | 44571 | 37210 / 7361 | 3449134 | 4 | 379.9 | 0 | 0 |
| `cached_refresh_bounded_m256_progress_triggered` | bounded_swaps, M=256, B=8, period=2 | 770 | 118090 | 90982 / 27108 | 8933116 | 0 | 1028.2 | 0 | 0 |

All 16 arms have 154/154 records, complete receipts, zero receipt validation errors, zero order errors, zero non-finite attention calls, zero unexpected LOCAL skips, and zero timed CUDA graph captures. The bounded-swap synthetic replay verified exactly `K=128` and the `B=8` cap; the native receipts report the same fixed-budget configuration. Selector audit seconds are instrumentation totals and are not clean latency.

## Scope and promotion gate

- The runnable native scope is complete under `attempt009` (primary bounded M=256) and `attempt010` (the 15-arm matrix).
- Native production gaps: separate online-value-only tie-break, full-information/full-V tie-break, strict intersection, and filled intersection. These are present in operator replay only because their native map wiring was not available.
- The 154-cell panel is not the requested 503-cell LongBench-v2 audit. No 503-cell promotion claim is made.
- Several arms violate the per-problem denoising-call gate relative to `current_v31_control`; promotion is therefore blocked even where subset accuracy is competitive.
- Clean timing remains intentionally absent and paused.

Generated artifacts: [`final_report.json`](/home/exouser/ljy/dlm/results/v31_cached_refresh_20261012/final_report.json), [`native_panel.csv`](/home/exouser/ljy/dlm/results/v31_cached_refresh_20261012/tables/native_panel.csv), [`fairness_audit.json`](/home/exouser/ljy/dlm/results/v31_cached_refresh_20261012/tables/fairness_audit.json), [`operator replay`](/home/exouser/ljy/dlm/results/v31_cached_refresh_20261012/operator_replay/summary.json).
