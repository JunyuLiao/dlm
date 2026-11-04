# TraDo-8B-Instruct RULER32K: completed baseline comparison

Completed four new arms /520requests on the same130-problem cohort. The existing current-tile V2 result is reused and officially re-scored on CPU; zero V2 generation requests. Prior reports and experiments remain unchanged.

| Method | Official retrieval score | Full-credit /130 | Mean output tokens | N total | C total | N/request | C/request | N/C | Tile sparsity | EOS / caps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native dense | 64.64% | 70 | 31.4 | 2836 | 1124 | 21.82 | 8.65 | 2.523 | 0.00% | 128 / 2 |
| All-kept FA4 | 64.85% | 71 | 31.3 | 2804 | 1122 | 21.57 | 8.63 | 2.499 | 0.00% | 127 / 3 |
| V31 | 63.49% | 68 | 29.9 | 2669 | 1076 | 20.53 | 8.28 | 2.480 | 49.97% | 127 / 3 |
| V3a | 64.96% | 71 | 30.0 | 2643 | 1079 | 20.33 | 8.30 | 2.449 | 49.97% | 129 / 1 |
| V2 current-tile init (existing) | 64.72% | 71 | 31.0 | 2767 | 1113 | 21.28 | 8.56 | 2.486 | 49.45% | 126 / 4 |

Official RULER score is fractional target-retrieval credit, averaged over the balanced13tasks; full-credit count is the number of completely correct samples. N is denoising forwards,C native canvases; N/C is pooled. Caps count non-EOS outputs at128tokens.

Setup:130problems,10/task across13official tasks,seed42. Same frozen RULER32K prompts as prior V2; prompt lengths29798–32472,128output cap,32768context limit,no truncation. Original TraDo revision2d37bd3e and8c0d14b0 source; BF16,batch1,oneH100. Native4-token block,4denoising steps,temperature1,top-p1,top-k0,dynamic confidence0.9,native EOS. All36denoising layers eligible; dense prefill and commits.

V31/V3a keep max(round(J/2),mandatory_count) physical KV64tiles per query head/Q128block, including mandatory current tiles; ties-to-even rounding follows effective context length at each selection. V31 ranks attention mass; V3a uses Gaussian32 full-information singleton risk(seed1729+layer). Existing FA4 current-full-V consumer,refresh every fourth call,no carry. No threshold calibration is required for these fixed budgets.

Sparsity is1−sum(kept)/sum(eligible), measured on actual denoising consumer trajectories including mandatory tiles and selection/reuse calls, excluding selector discovery,prefill,commits. It is geometric tile sparsity,not whole-model FLOPs. Dense native SDPA and all-kept FA4 remain separate controls. Instrumented worker times are accounting,not clean latency.

Paired differences (candidate minus reference,percentage points;10000problem-bootstrap resamples):
- V31 vs Native dense: -1.15pp,problem95%CI[-3.15,+0.46],13-task-cluster95%CI[-2.88,+0.00].
- V3a vs V31: +1.47pp,problem95%CI[-0.32,+3.78],13-task-cluster95%CI[-0.71,+4.42].
- V2 current-tile init (existing) vs V31: +1.23pp,problem95%CI[-1.77,+4.42],13-task-cluster95%CI[-2.23,+5.58].
- V2 current-tile init (existing) vs V3a: -0.24pp,problem95%CI[-2.76,+2.24],13-task-cluster95%CI[-2.50,+2.06].
- V2 current-tile init (existing) vs Native dense: +0.08pp,problem95%CI[-3.19,+3.38],13-task-cluster95%CI[-3.69,+4.62].

Per-task retrieval scores:

| Task | Native dense | All-kept FA4 | V31 | V3a | Existing current-tile V2 |
|---|---:|---:|---:|---:|---:|
| cwe | 12.00% | 12.00% | 12.00% | 12.00% | 12.00% |
| fwe | 93.33% | 90.00% | 93.33% | 90.00% | 93.33% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% | 100.00% | 90.00% |
| niah_multikey_2 | 40.00% | 40.00% | 40.00% | 40.00% | 40.00% |
| niah_multikey_3 | 20.00% | 20.00% | 10.00% | 10.00% | 10.00% |
| niah_multiquery | 82.50% | 80.00% | 80.00% | 77.50% | 77.50% |
| niah_multivalue | 72.50% | 75.00% | 70.00% | 85.00% | 82.50% |
| niah_single_1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 90.00% | 90.00% | 90.00% | 90.00% | 90.00% |
| qa_1 | 20.00% | 30.00% | 20.00% | 30.00% | 40.00% |
| qa_2 | 30.00% | 30.00% | 30.00% | 30.00% | 30.00% |
| vt | 80.00% | 76.00% | 80.00% | 80.00% | 76.00% |

Validation:five focused CPU tests,independent CPU/CUDA selector masks and FP32 masked-attention checks,exact historical all-kept output/count replay,16complete new shards,520unique source-bound new cells,per-selection exact budget/mandatory/eligibility assertions. Existing130V2scores independently re-score exactly. Same prompt/seed/cap/runtime/model identities are verified. No V2 worker or previous Gemma campaign is resumed.

The cohort was previously examined and all methods are exploratory. Intervals describe observed paired sample/task uncertainty,not seed variability or noninferiority. Raw prompts,gold,tokens and generations remain private.
