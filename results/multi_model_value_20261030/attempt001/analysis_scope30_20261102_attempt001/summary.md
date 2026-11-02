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
| llada22/ruler128k/allkept_fa4 | 84.82% (130) | 1044 | 237 | 4.405 | 3664 | 0.00% |
| llada22/ruler128k/dense_native | 85.44% (130) | 1015 | 237 | 4.283 | 3663 | 0.00% |
| llada22/ruler128k/mass_topk_reuse_r4 | 86.01% (130) | 1017 | 237 | 4.291 | 3689 | 50.00% |
| llada22/ruler128k/value_v2_gaussian32_reuse_r4 | 86.23% (130) | 1024 | 236 | 4.339 | 3656 | 49.19% |
| llada22/ruler128k/value_v3a_singleton_reuse_r4 | 85.03% (130) | 1040 | 237 | 4.388 | 3662 | 50.00% |
| trado/aime26/allkept_fa4 | 4.44% (90) | 47348 | 22921 | 2.066 | 91413 | 0.00% |
| trado/aime26/dense_native | 8.89% (90) | 48516 | 24203 | 2.005 | 96555 | 0.00% |
| trado/aime26/mass_topk_reuse_r4 | 3.33% (90) | 39895 | 17583 | 2.269 | 70081 | 50.00% |
| trado/aime26/value_v2_gaussian32_reuse_r4 | 6.67% (90) | 47505 | 22888 | 2.076 | 91284 | 60.45% |
| trado/aime26/value_v3a_singleton_reuse_r4 | 2.22% (90) | 38130 | 16180 | 2.357 | 64451 | 49.97% |
| trado/humaneval/allkept_fa4 | 78.05% (164) | 9425 | 5076 | 1.857 | 19796 | 0.00% |
| trado/humaneval/dense_native | 79.27% (164) | 9612 | 5225 | 1.840 | 20383 | 0.00% |
| trado/humaneval/mass_topk_reuse_r4 | 42.68% (164) | 11616 | 5105 | 2.275 | 19934 | 49.55% |
| trado/humaneval/value_v2_gaussian32_reuse_r4 | 76.22% (164) | 10059 | 5362 | 1.876 | 20933 | 48.84% |
| trado/humaneval/value_v3a_singleton_reuse_r4 | 47.56% (164) | 11233 | 5600 | 2.006 | 21901 | 49.67% |
| ultrallada/humaneval/allkept_fa4 | 1.22% (164) | 167936 | 2624 | 64.000 | 167936 | 0.00% |
| ultrallada/humaneval/dense_native | 1.22% (164) | 167936 | 2624 | 64.000 | 167936 | 0.00% |
| ultrallada/humaneval/mass_topk_reuse_r4 | 0.00% (164) | 167936 | 2624 | 64.000 | 167936 | 48.99% |
| ultrallada/humaneval/value_v2_gaussian32_reuse_r4 | 0.00% (164) | 167936 | 2624 | 64.000 | 167936 | 49.45% |
| ultrallada/humaneval/value_v3a_singleton_reuse_r4 | 0.00% (164) | 167936 | 2624 | 64.000 | 167936 | 48.99% |
