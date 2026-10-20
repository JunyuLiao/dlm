# HumanEval: corrected K256 and K128 results

Updated 2026-10-20 (Hong Kong). All 164 tasks, primary seed 42, thinking on and off, one H100 80 GB. Both panels are complete and independently verified. This report reorganizes the immutable combined publication; it adds no generation or timing runs.

## Main findings

- **K256:** V2 gains three correct answers with thinking and one without, but increases mean calls and request latency. Full-value swaps and guarded residual do not achieve the joint accuracy/latency objective.
- **K128:** V2 has the strongest observed joint lead over V31: 158 vs 157 correct with thinking and 160 vs 156 without; mean request speedups are 1.060x and 1.085x. Both speedup confidence intervals include 1. The non-thinking V2 calibration is explicitly unmatched.
- Native dense FULL is faster in mean request time than every tested sparse/value method in all four cells. All-kept FA4 is retained as a separate attention-consumer control.
- Audit and clean outputs/trajectories match for every method. These are reproducible technical results under the pinned runtime, with primary accuracy evidence from one seed.

See the [reproducibility problem and implemented fixes](../../../docs/HUMANEVAL_REPRODUCIBILITY_FIXES_20261020.md).

## Frozen setup and method scope

| Item | Setting |
|---|---|
| Model / runtime | DiffusionGemma-26B-A4B, BF16, batch 1; vLLM 0.30.0, Torch 2.13.0 |
| Scientific deploy | `435511e43cf186a836761667f36fbf4f0f6061ec` |
| K256 protocol | `55370024da6427395312efeac1f700953c2d64aa5f1e65b2e2cfdac565e90c86` |
| K128 protocol | `187719fc92ac1c6facfab7527257875773b58b80a0d4667e0738cfdd6183b10e` |
| Geometry / scope | Per-query-head Q128 x KV64 tiles; GLOBAL layers 5/11/17/23/29; all LOCAL attention dense |
| Budget | K256 = four selectable prefix KV64 blocks; K128 = two; mandatory canvas/boundary tiles retained separately |
| Native decoding | Canvas 256; maximum 48 calls/canvas; temperatures 0.8 to 0.4; entropy bound 0.1; confidence threshold 0.005; stability threshold 1; frozen output limits |
| Numerical policy | Explicit deterministic model/helper compilation, horizontal combo fusion/benchmarking disabled; overlap disabled only for mass-observing FA4 |
| Calibration | V2 thresholds selected using first-canvas work, with no accuracy used; examined prompts overlap evaluation |

V31 inherits Yuhan's cross-step-reuse machinery. The value-aware selectors and refinements are separately named study variants.

| Display name | Recorded method | Selection/value scope |
|---|---|---|
| Dense FULL | `dense_full_fix51994` | Native FULL decode with the mask fix corresponding to PR #51994 |
| Dense PIECEWISE | `dense_piecewise` | Matched native PIECEWISE dense control |
| All-kept FA4 | `allkept_fa4` | Same sparse-consumer family with every tile retained |
| V31 | `control_before` | Attention-mass selection/reuse, fixed per-tile prefix K, all five GLOBAL layers |
| V2 | `value_v2_online_preserve_mass` | Online running-softmax-corrected value selection; Gaussian32; all five GLOBAL layers; variable threshold, not exact K |
| V3a | `value_v3a_singleton_delete` | Full-row information with individual-block value scoring/top-K; Gaussian32; all five GLOBAL layers |
| Full-value swaps | `full_swap8` | Full-dimensional V (512) bounded swaps at layer 29; V31 in the other GLOBAL layers |
| Direction-guarded residual | `residual_direction_guard` | Full-dimensional value correction at layer 29; V31 in the other GLOBAL layers |

The extra `control_after` label is a V31 technical bookend, not a ninth scientific method. Dense FULL, PIECEWISE and all-kept runs are budget-independent and shared between K256 and K128; repeated table entries are the same trials. Every selector arm runs freshly at each budget.

## Metrics and denominators

- Accuracy is the official HumanEval pass@1 result, shown as percent and correct/164. Audit/clean rescoring agrees.
- `N`: mean denoising forwards per problem, excluding commit calls. `C`: mean canvases. `N/C`: total forwards divided by total canvases, not the mean of per-item ratios.
- `T`: mean generated output tokens, including thinking where enabled. `S`: generation span excluding prefill. `W`: request time including prefill and all method work. Times below are clean means.
- `S/N`: total generation time divided by total forwards, in milliseconds. It is an amortized trajectory cost, not an isolated forward benchmark.
- GLOBAL sparsity = `(GLOBAL eligible - GLOBAL kept) / GLOBAL eligible`. Overall sparsity divides the same skipped count by GLOBAL eligible plus native-window LOCAL eligible rectangles. LOCAL sparsity is zero for every arm.
- Physical counts are Q128/KV64 attention-consumer rectangles over the audit denoising trajectory. They exclude prefill, commit attention and additional selector/QK/PV work; they are not CUDA CTA counts. Clean W includes complete method cost.

## Accuracy, denoising and clean end-to-end performance

### K256, thinking on

| Method | Accuracy (correct/164) | N/problem | C/problem | N/C | T | S ms | W ms | S/N ms | GLOBAL sparsity | Overall sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense FULL (fix #51994) | 98.17% (161/164) | 48.079 | 5.195 | 9.255 | 1202.51 | 929.363 | 952.487 | 19.330 | 0.000% | 0.000% |
| Dense PIECEWISE | 98.17% (161/164) | 48.079 | 5.195 | 9.255 | 1202.51 | 940.269 | 959.672 | 19.557 | 0.000% | 0.000% |
| All-kept FA4 | 100.00% (164/164) | 51.945 | 5.482 | 9.476 | 1266.24 | 1091.848 | 1111.270 | 21.019 | 0.000% | 0.000% |
| V31 | 96.95% (159/164) | 49.256 | 5.122 | 9.617 | 1189.19 | 1104.579 | 1123.988 | 22.425 | 48.325% | 10.922% |
| V2 | 98.78% (162/164) | 50.043 | 5.268 | 9.499 | 1223.60 | 1128.215 | 1147.622 | 22.545 | 25.618% | 5.824% |
| V3a | 96.95% (159/164) | 51.854 | 5.323 | 9.741 | 1233.70 | 1186.225 | 1205.628 | 22.876 | 50.587% | 11.857% |
| Full-value swaps | 94.51% (155/164) | 49.085 | 5.183 | 9.471 | 1198.24 | 1805.949 | 1825.361 | 36.792 | 48.880% | 11.202% |
| Direction-guarded residual | 96.34% (158/164) | 52.543 | 5.293 | 9.927 | 1228.07 | 1301.915 | 1321.328 | 24.778 | 50.979% | 11.921% |

### K256, thinking off

| Method | Accuracy (correct/164) | N/problem | C/problem | N/C | T | S ms | W ms | S/N ms | GLOBAL sparsity | Overall sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense FULL (fix #51994) | 95.12% (156/164) | 9.470 | 1.341 | 7.059 | 228.49 | 183.129 | 205.666 | 19.339 | 0.000% | 0.000% |
| Dense PIECEWISE | 95.12% (156/164) | 9.470 | 1.341 | 7.059 | 228.49 | 184.888 | 203.842 | 19.525 | 0.000% | 0.000% |
| All-kept FA4 | 95.12% (156/164) | 9.957 | 1.341 | 7.423 | 236.62 | 209.045 | 228.095 | 20.994 | 0.000% | 0.000% |
| V31 | 95.73% (157/164) | 10.774 | 1.372 | 7.853 | 244.56 | 242.932 | 262.178 | 22.547 | 20.031% | 3.579% |
| V2 | 96.34% (158/164) | 10.854 | 1.409 | 7.706 | 247.77 | 244.821 | 264.080 | 22.557 | 5.168% | 0.889% |
| V3a | 95.73% (157/164) | 10.305 | 1.335 | 7.717 | 234.32 | 240.810 | 260.063 | 23.369 | 9.504% | 1.587% |
| Full-value swaps | 95.12% (156/164) | 10.445 | 1.341 | 7.786 | 235.07 | 294.975 | 314.227 | 28.240 | 14.663% | 2.515% |
| Direction-guarded residual | 95.12% (156/164) | 10.280 | 1.341 | 7.664 | 235.20 | 245.081 | 264.348 | 23.839 | 10.074% | 1.685% |

### K128, thinking on

| Method | Accuracy (correct/164) | N/problem | C/problem | N/C | T | S ms | W ms | S/N ms | GLOBAL sparsity | Overall sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense FULL (fix #51994) | 98.17% (161/164) | 48.079 | 5.195 | 9.255 | 1202.51 | 929.363 | 952.487 | 19.330 | 0.000% | 0.000% |
| Dense PIECEWISE | 98.17% (161/164) | 48.079 | 5.195 | 9.255 | 1202.51 | 940.269 | 959.672 | 19.557 | 0.000% | 0.000% |
| All-kept FA4 | 100.00% (164/164) | 51.945 | 5.482 | 9.476 | 1266.24 | 1091.848 | 1111.270 | 21.019 | 0.000% | 0.000% |
| V31 | 95.73% (157/164) | 57.415 | 5.360 | 10.712 | 1243.13 | 1278.469 | 1297.955 | 22.267 | 59.870% | 14.630% |
| V2 | 96.34% (158/164) | 53.323 | 5.476 | 9.738 | 1277.45 | 1204.595 | 1224.068 | 22.590 | 56.641% | 14.744% |
| V3a | 93.90% (154/164) | 58.445 | 5.335 | 10.954 | 1245.00 | 1324.915 | 1344.402 | 22.669 | 61.818% | 15.724% |
| Full-value swaps | 94.51% (155/164) | 55.982 | 5.341 | 10.481 | 1233.15 | 1999.532 | 2019.004 | 35.718 | 59.735% | 14.802% |
| Direction-guarded residual | 94.51% (155/164) | 56.518 | 5.341 | 10.581 | 1237.51 | 1402.708 | 1422.193 | 24.819 | 60.891% | 15.400% |

### K128, thinking off

| Method | Accuracy (correct/164) | N/problem | C/problem | N/C | T | S ms | W ms | S/N ms | GLOBAL sparsity | Overall sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense FULL (fix #51994) | 95.12% (156/164) | 9.470 | 1.341 | 7.059 | 228.49 | 183.129 | 205.666 | 19.339 | 0.000% | 0.000% |
| Dense PIECEWISE | 95.12% (156/164) | 9.470 | 1.341 | 7.059 | 228.49 | 184.888 | 203.842 | 19.525 | 0.000% | 0.000% |
| All-kept FA4 | 95.12% (156/164) | 9.957 | 1.341 | 7.423 | 236.62 | 209.045 | 228.095 | 20.994 | 0.000% | 0.000% |
| V31 | 95.12% (156/164) | 11.488 | 1.384 | 8.300 | 243.12 | 255.146 | 274.204 | 22.210 | 29.656% | 5.477% |
| V2 | 97.56% (160/164) | 10.439 | 1.354 | 7.712 | 233.58 | 233.637 | 252.697 | 22.381 | 18.355% | 3.096% |
| V3a | 96.34% (158/164) | 10.165 | 1.329 | 7.647 | 231.34 | 234.295 | 253.350 | 23.050 | 14.152% | 2.362% |
| Full-value swaps | 95.73% (157/164) | 10.280 | 1.317 | 7.806 | 229.80 | 320.709 | 339.769 | 31.196 | 15.792% | 2.650% |
| Direction-guarded residual | 96.34% (158/164) | 11.829 | 1.402 | 8.435 | 249.40 | 283.234 | 302.297 | 23.943 | 32.730% | 6.100% |

## Paired differences versus V31

Accuracy differences are percentage points; call differences are candidate minus V31. Speedup is mean V31 request time divided by mean candidate request time, so values above 1 favor the candidate. Brackets are paired question-bootstrap 95% intervals (20,000 resamples), conditional on the observed engines. They do not cover new-seed or independent full-run variation. Candidate-only/control-only counts expose answer changes hidden by aggregate accuracy.

| K / thinking | Method | Accuracy delta pp [95% CI] | N delta [95% CI] | Request speedup [95% CI] | Candidate-only / V31-only correct |
|---|---|---:|---:|---:|---:|
| 256 / on | V2 | +1.829 [-0.610, +4.878] | +0.787 [-2.439, +4.354] | 0.979 [0.916, 1.042] | 4 / 1 |
| 256 / on | V3a | +0.000 [-3.049, +3.049] | +2.598 [-0.335, +5.707] | 0.932 [0.883, 0.985] | 3 / 3 |
| 256 / on | Full-value swaps | -2.439 [-5.488, +0.000] | -0.171 [-2.756, +2.250] | 0.616 [0.587, 0.649] | 1 / 5 |
| 256 / on | Direction-guarded residual | -0.610 [-3.049, +1.829] | +3.287 [+0.000, +7.762] | 0.851 [0.787, 0.903] | 2 / 3 |
| 256 / off | V2 | +0.610 [+0.000, +1.829] | +0.079 [-0.982, +1.140] | 0.993 [0.913, 1.074] | 1 / 0 |
| 256 / off | V3a | +0.000 [+0.000, +0.000] | -0.470 [-2.476, +0.750] | 1.008 [0.913, 1.178] | 0 / 0 |
| 256 / off | Full-value swaps | -0.610 [-1.829, +0.000] | -0.329 [-2.421, +1.238] | 0.834 [0.726, 0.979] | 0 / 1 |
| 256 / off | Direction-guarded residual | -0.610 [-1.829, +0.000] | -0.494 [-2.457, +0.707] | 0.992 [0.897, 1.157] | 0 / 1 |
| 128 / on | V2 | +0.610 [-3.049, +4.268] | -4.091 [-13.262, +5.762] | 1.060 [0.897, 1.239] | 5 / 4 |
| 128 / on | V3a | -1.829 [-6.098, +1.829] | +1.030 [-5.707, +9.701] | 0.965 [0.843, 1.076] | 4 / 7 |
| 128 / on | Full-value swaps | -1.220 [-4.878, +1.829] | -1.433 [-7.732, +5.354] | 0.643 [0.575, 0.712] | 3 / 5 |
| 128 / on | Direction-guarded residual | -1.220 [-4.878, +1.829] | -0.896 [-8.061, +6.049] | 0.913 [0.819, 1.025] | 3 / 5 |
| 128 / off | V2 | +2.439 [+0.000, +5.488] | -1.049 [-3.835, +1.195] | 1.085 [0.905, 1.321] | 5 / 1 |
| 128 / off | V3a | +1.220 [-1.220, +3.659] | -1.323 [-3.659, +0.079] | 1.082 [0.970, 1.264] | 3 / 1 |
| 128 / off | Full-value swaps | +0.610 [+0.000, +1.829] | -1.207 [-3.128, -0.012] | 0.807 [0.725, 0.915] | 1 / 0 |
| 128 / off | Direction-guarded residual | +1.220 [+0.000, +3.049] | +0.341 [-0.653, +1.970] | 0.907 [0.801, 0.979] | 2 / 0 |

## V2 work calibration and whole-trajectory work

The threshold is frozen from work only before primary accuracy generation. The first-canvas target is a kept-fraction ratio within 1% and a sparsity difference within 0.25 percentage points. Absolute kept counts are shown separately because changed trajectories also change eligible work. The predeclared bounded search retains the nearest work-only threshold if the target cannot be reached.

| K / thinking | V2 threshold | First-canvas absolute kept ratio, V2/V31 | Kept-fraction ratio | Sparsity delta pp | Match |
|---|---:|---:|---:|---:|---|
| 256 / on | 0.09448519267394 | 0.993973 | 1.001726 | -0.170895 | Pass |
| 256 / off | 0.09812993965151 | 0.981305 | 1.000967 | -0.096011 | Pass |
| 128 / on | 0.5600716206066 | 1.002772 | 1.000254 | -0.023822 | Pass |
| 128 / off | 0.7157410137532 | 0.936804 | 0.994373 | +0.531160 | **Unmatched** |

**K128 thinking-off V2 is unmatched:** its first-canvas sparsity is 0.531160 pp higher than V31, beyond the 0.25 pp tolerance, and it retains 6.32% fewer absolute GLOBAL tiles on that canvas. The threshold was retained by the predeclared work-only rule. Accuracy was not used to select it.

First-canvas matching never establishes equal total attention work. Whole-trajectory GLOBAL kept work is:

| K / thinking | V31 GLOBAL kept | V2 GLOBAL kept | V2/V31 |
|---|---:|---:|---:|
| 256 / on | 14,679,520 | 21,695,069 | 1.477914 |
| 256 / off | 2,460,480 | 2,812,186 | 1.142942 |
| 128 / on | 15,302,560 | 16,445,696 | 1.074702 |
| 128 / off | 2,424,000 | 2,156,222 | 0.889531 |

Thus K128 thinking-on V2 has fewer model calls but about 7.47% more kept GLOBAL consumer work; its lower mean latency is not evidence of equal work or cheaper individual attention operations. At K128 thinking-off it has both fewer calls and fewer total kept GLOBAL tiles, with the disclosed first-canvas mismatch. GLOBAL and overall sparsity can move differently because the LOCAL/GLOBAL denominators change with the trajectory.

## Tail behavior and output limits

These statistics retain long trajectories and failed/capped requests. An output cap is a request hitting its generation limit; it is distinct from a canvas reaching 48 denoising calls. No tail removal is used in headline means.

| K / thinking | Method | N p95 | N maximum | Output-capped requests | Canvases reaching 48 calls |
|---|---|---:|---:|---:|---:|
| 256 / on | Dense FULL (fix #51994) | 103.85 | 516 | 0 | 1 |
| 256 / on | Dense PIECEWISE | 103.85 | 516 | 0 | 1 |
| 256 / on | All-kept FA4 | 120.00 | 422 | 0 | 2 |
| 256 / on | V31 | 119.85 | 488 | 1 | 0 |
| 256 / on | V2 | 100.85 | 554 | 1 | 0 |
| 256 / on | V3a | 113.00 | 446 | 1 | 0 |
| 256 / on | Full-value swaps | 105.00 | 505 | 1 | 0 |
| 256 / on | Direction-guarded residual | 138.65 | 504 | 1 | 0 |
| 256 / off | Dense FULL (fix #51994) | 19.85 | 59 | 0 | 0 |
| 256 / off | Dense PIECEWISE | 19.85 | 59 | 0 | 0 |
| 256 / off | All-kept FA4 | 22.00 | 65 | 0 | 0 |
| 256 / off | V31 | 24.85 | 163 | 0 | 0 |
| 256 / off | V2 | 24.70 | 108 | 0 | 1 |
| 256 / off | V3a | 25.00 | 72 | 0 | 0 |
| 256 / off | Full-value swaps | 22.85 | 129 | 0 | 0 |
| 256 / off | Direction-guarded residual | 22.85 | 76 | 0 | 0 |
| 128 / on | Dense FULL (fix #51994) | 103.85 | 516 | 0 | 1 |
| 128 / on | Dense PIECEWISE | 103.85 | 516 | 0 | 1 |
| 128 / on | All-kept FA4 | 120.00 | 422 | 0 | 2 |
| 128 / on | V31 | 126.70 | 506 | 1 | 0 |
| 128 / on | V2 | 133.85 | 522 | 3 | 0 |
| 128 / on | V3a | 145.20 | 555 | 2 | 0 |
| 128 / on | Full-value swaps | 140.70 | 492 | 1 | 0 |
| 128 / on | Direction-guarded residual | 140.65 | 648 | 1 | 0 |
| 128 / off | Dense FULL (fix #51994) | 19.85 | 59 | 0 | 0 |
| 128 / off | Dense PIECEWISE | 19.85 | 59 | 0 | 0 |
| 128 / off | All-kept FA4 | 22.00 | 65 | 0 | 0 |
| 128 / off | V31 | 23.85 | 214 | 0 | 1 |
| 128 / off | V2 | 20.85 | 133 | 0 | 0 |
| 128 / off | V3a | 24.70 | 56 | 0 | 0 |
| 128 / off | Full-value swaps | 24.00 | 81 | 0 | 0 |
| 128 / off | Direction-guarded residual | 24.70 | 203 | 0 | 0 |

K128 thinking-on V2 has three output-capped requests versus one for V31. Its lower mean calls do not establish uniformly shorter trajectories: its p95 is 133.85 calls versus V31's 126.70. Preserve this tail when interpreting the mean lead.

## Complete physical attention-consumer accounting

All LOCAL skipped counts are zero. Eligible/kept counts below are summed across all 164 audit requests; they do not count additional selector/PV computations.

| K / thinking | Method | GLOBAL eligible | GLOBAL kept | GLOBAL skipped | LOCAL eligible |
|---|---|---:|---:|---:|---:|
| 256 / on | Dense FULL (fix #51994) | 26,309,120 | 26,309,120 | 0 | 93,804,800 |
| 256 / on | Dense PIECEWISE | 26,309,120 | 26,309,120 | 0 | 93,804,800 |
| 256 / on | All-kept FA4 | 31,819,040 | 31,819,040 | 0 | 104,620,400 |
| 256 / on | V31 | 28,407,360 | 14,679,520 | 13,727,840 | 97,286,000 |
| 256 / on | V2 | 29,167,200 | 21,695,069 | 7,472,131 | 99,129,600 |
| 256 / on | V3a | 31,853,920 | 15,740,000 | 16,113,920 | 104,044,800 |
| 256 / on | Full-value swaps | 28,906,080 | 14,776,800 | 14,129,280 | 97,222,400 |
| 256 / on | Direction-guarded residual | 32,311,840 | 15,839,680 | 16,472,160 | 105,863,600 |
| 256 / off | Dense FULL (fix #51994) | 2,241,600 | 2,241,600 | 0 | 11,203,200 |
| 256 / off | Dense PIECEWISE | 2,241,600 | 2,241,600 | 0 | 11,203,200 |
| 256 / off | All-kept FA4 | 2,477,760 | 2,477,760 | 0 | 12,352,400 |
| 256 / off | V31 | 3,076,800 | 2,460,480 | 616,320 | 14,145,600 |
| 256 / off | V2 | 2,965,440 | 2,812,186 | 153,254 | 14,274,400 |
| 256 / off | V3a | 2,533,600 | 2,292,800 | 240,800 | 12,637,600 |
| 256 / off | Full-value swaps | 2,734,560 | 2,333,600 | 400,960 | 13,207,200 |
| 256 / off | Direction-guarded residual | 2,546,080 | 2,289,600 | 256,480 | 12,675,600 |
| 128 / on | Dense FULL (fix #51994) | 26,309,120 | 26,309,120 | 0 | 93,804,800 |
| 128 / on | Dense PIECEWISE | 26,309,120 | 26,309,120 | 0 | 93,804,800 |
| 128 / on | All-kept FA4 | 31,819,040 | 31,819,040 | 0 | 104,620,400 |
| 128 / on | V31 | 38,132,640 | 15,302,560 | 22,830,080 | 117,916,400 |
| 128 / on | V2 | 37,929,120 | 16,445,696 | 21,483,424 | 107,785,200 |
| 128 / on | V3a | 40,978,880 | 15,646,560 | 25,332,320 | 120,130,400 |
| 128 / on | Full-value swaps | 37,634,720 | 15,153,760 | 22,480,960 | 114,239,600 |
| 128 / on | Direction-guarded residual | 39,029,760 | 15,264,320 | 23,765,440 | 115,290,000 |
| 128 / off | Dense FULL (fix #51994) | 2,241,600 | 2,241,600 | 0 | 11,203,200 |
| 128 / off | Dense PIECEWISE | 2,241,600 | 2,241,600 | 0 | 11,203,200 |
| 128 / off | All-kept FA4 | 2,477,760 | 2,477,760 | 0 | 12,352,400 |
| 128 / off | V31 | 3,445,920 | 2,424,000 | 1,021,920 | 15,212,000 |
| 128 / off | V2 | 2,640,960 | 2,156,222 | 484,738 | 13,016,000 |
| 128 / off | V3a | 2,453,440 | 2,106,240 | 347,200 | 12,247,200 |
| 128 / off | Full-value swaps | 2,513,600 | 2,116,640 | 396,960 | 12,467,200 |
| 128 / off | Direction-guarded residual | 3,750,880 | 2,523,200 | 1,227,680 | 16,374,400 |

## Overhead and interpretation

Full-value swaps and guarded residual retain expensive extra moment construction/selection or correction work at layer 29. Moving proposal evaluation to the device does not eliminate that complete work. Reported audit anchor CUDA-event totals are available in the source summary JSON under `audit_selector_times_ms`; they are instrumented module spans, not separate clean kernel benchmarks.

- With K256 thinking on, swaps reduce calls only from 49.256 to 49.085 while raising W from 1123.988 to 1825.361 ms; accuracy falls from 159 to 155 correct. Guarded residual raises both calls and W and scores 158.
- With K128 thinking on, swaps and residual save about 1.43 and 0.90 calls/problem but take 2019.004 and 1422.193 ms versus V31's 1297.955 ms; each scores 155 versus 157.
- K128 non-thinking V3a also has a favorable mean point (158 correct, 10.165 calls, 253.350 ms), but its speedup interval includes 1. No across-setting superiority follows.
- The tested fixed-budget/value methods have no mean request-speed advantage over native dense FULL on HumanEval. Short contexts and dense LOCAL/prefill/commit work limit the removable attention cost.

Per-forward S/N values above include trajectory-dependent costs. No new isolated kernel or attention-module clean timing was run for this documentation, so the tables support measured request-level outcomes rather than a claim about an optimally fused selector.

## Reproducibility, technical controls and evidence limits

| K | Full audit/clean comparisons | Main records independently rescored | Qualification records rescored | Score mismatches / timeouts |
|---|---:|---:|---:|---:|
| 256 | 22/22 pass | 5,904 | 432 | 0 / 0 |
| 128 | 22/22 pass | 5,904 | 432 | 0 / 0 |

Each full audit/clean comparison checks the same method/question/seed, including private completion equality, output hash, N/C/T, sampler counts and emission intervals. Independent engine-prefix replication also passes. The 5,904 main rows per budget are nine labels x 164 problems x two thinking modes x two passes, not 5,904 independent problems. Shared dense references appear logically in both budgets.

| K / thinking | V31 after/before mean W | Output / trajectory changes |
|---|---:|---|
| 256 / on | 1.000044 | 0 |
| 256 / off | 1.000752 | 0 |
| 128 / on | 0.999988 | 0 |
| 128 / off | 0.997788 | 0 |

The full164 primary seed is 42. Seed43 repeats six examined questions per method/mode and remains descriptive. Calibration reuses examined prompts and is not held-out confirmation. The question-bootstrap intervals condition on these engines; they do not establish seed-robust superiority or noninferiority.

The reproducibility corrections are verified on these HumanEval runs. Historical LongBench-v2 drift has not been causally attributed to them by a corrected-runtime replay. Older paused campaigns and frozen results remain intact.

## Publication and reproduction

- [Combined compact tables](../../humaneval_corrected_k256_k128_20261020/attempt001/summary.md), [full machine-readable methods CSV](../../humaneval_corrected_k256_k128_20261020/attempt001/methods.csv), and [paired intervals](../../humaneval_corrected_k256_k128_20261020/attempt001/paired_vs_v31.json).
- [Frozen source/protocol bindings](../../humaneval_corrected_k256_k128_20261020/attempt001/config.json), [completion and file hashes](../../humaneval_corrected_k256_k128_20261020/attempt001/final_complete.json), and [resource accounting](../../humaneval_corrected_k256_k128_20261020/attempt001/resources.json).
- [K256 full summary](../../humaneval_corrected_k256_k128_20261020/attempt001/k256/summary.json) and [K128 full summary](../../humaneval_corrected_k256_k128_20261020/attempt001/k128/summary.json) contain first-canvas counts, physical totals, audit overhead, per-item limits and all technical controls.
- [K256 audit/clean certificate](../../humaneval_corrected_k256_k128_20261020/attempt001/k256/full164_audit_clean_equivalence.json) and [K128 certificate](../../humaneval_corrected_k256_k128_20261020/attempt001/k128/full164_audit_clean_equivalence.json).
- [K256 independent engine-prefix proof](../../humaneval_corrected_k256_k128_20261020/attempt001/k256/independent_prefix_replication.json) and [K128 proof](../../humaneval_corrected_k256_k128_20261020/attempt001/k128/independent_prefix_replication.json).
- [Independent combined readback/source receipt](../../humaneval_corrected_validation_20261020/attempt001/receipt.json).

Public per-item exports contain only sanitized metrics and hashes; prompts, gold answers, raw completions and token IDs remain private. The checked-in CSV/JSON supports CPU-only table reproduction. Native generation and official rescoring require the authorized frozen dataset/model/runtime; do not treat a public metric reaggregation as independent regeneration.

For a quick CPU-only check from the repository root:

```bash
python - <<'PY_CHECK'
import hashlib, json
from pathlib import Path
root = Path('results/humaneval_corrected_k256_k128_20261020/attempt001')
marker = json.loads((root / 'final_complete.json').read_text())
assert marker['status'] == 'complete'
for name, digest in marker['files_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
for budget in (256, 128):
    proof = json.loads((root / f'k{budget}/full164_audit_clean_equivalence.json').read_text())
    assert len(proof['comparisons']) == 22
    assert all(c['passed'] for c in proof['comparisons'])
print('Publication hashes and both full audit/clean certificates pass')
PY_CHECK
```
