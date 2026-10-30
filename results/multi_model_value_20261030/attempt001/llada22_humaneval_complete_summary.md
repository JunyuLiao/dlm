# LLaDA2.2 HumanEval complete panel

164 problems, seed42, native 32-token blocks; consumer sparsity targets about50%.

| Arm | Correct/164 | N | C | N/C pooled | Output tokens | Consumer sparsity |
|---|---:|---:|---:|---:|---:|---:|
| dense_native | 141/164 | 4717 | 1405 | 3.3573 | 39862 | 0.000% |
| allkept_fa4 | 138/164 | 5153 | 1439 | 3.5810 | 40870 | 0.000% |
| mass_topk_reuse_r4 | 83/164 | 6029 | 1285 | 4.6918 | 35900 | 49.978% |
| value_v2_gaussian32_reuse_r4 | 10/164 | 2331 | 503 | 4.6342 | 10317 | 49.216% |
| value_v3a_singleton_reuse_r4 | 103/164 | 6941 | 1422 | 4.8812 | 40227 | 49.990% |

V3a minus mass accuracy: 12.20 pp; question-clustered95% interval [3.05, 21.34] pp. Calls/request increase 5.561; paired geometric N/C ratio 0.99754. This is an accuracy/call tradeoff, not a denoising convergence gain.

V2 has severe quality loss and substantially shorter output; lower N is not useful convergence evidence. All-kept consumer effects remain separate. This is one completed model/benchmark, not campaign completion. Instrumented timing is not clean speedup evidence. A redundant41-cell mass repeat is excluded from primary coverage and matches all output/call/canvas/tile identities.
