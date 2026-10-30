# Three-model value evaluation

Instrumented execution; no clean latency claim.

| Model / task / arm | Score | N | C | N/C | T | Consumer sparsity |
|---|---:|---:|---:|---:|---:|---:|
| llada22/aime26/allkept_fa4 | 32.22% (90) | 154593 | 20767 | 7.444 | 662063 | 0.00% |
| llada22/aime26/dense_native | 40.00% (90) | 138512 | 19813 | 6.991 | 631678 | 0.00% |
| llada22/aime26/mass_topk_reuse_r4 | 23.33% (90) | 119064 | 13271 | 8.972 | 422252 | 50.00% |
| llada22/aime26/value_v2_gaussian32_reuse_r4 | 10.00% (90) | 38202 | 3950 | 9.671 | 123666 | 75.21% |
| llada22/aime26/value_v3a_singleton_reuse_r4 | 27.78% (90) | 125912 | 14869 | 8.468 | 473315 | 50.00% |
| llada22/humaneval/allkept_fa4 | 84.15% (164) | 5153 | 1439 | 3.581 | 40870 | 0.00% |
| llada22/humaneval/dense_native | 85.98% (164) | 4717 | 1405 | 3.357 | 39862 | 0.00% |
| llada22/humaneval/mass_topk_reuse_r4 | 50.61% (164) | 6029 | 1285 | 4.692 | 35900 | 49.98% |
| llada22/humaneval/value_v2_gaussian32_reuse_r4 | 6.10% (164) | 2331 | 503 | 4.634 | 10317 | 49.22% |
| llada22/humaneval/value_v3a_singleton_reuse_r4 | 62.80% (164) | 6941 | 1422 | 4.881 | 40227 | 49.99% |
