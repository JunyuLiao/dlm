# Retained multi-model evaluation

Sparse arms target 48–52% consumer sparsity. Per the user's revised stopping rule, final AIME V2 target drift is disclosed without recalibration or another benchmark rerun. AIME V2 is transductively count-calibrated; other arms are frozen primary panels. Ultra AIME/RULER were cancelled; TraDo RULER is unsupported. No clean latency claim.

Execution deviation: AIME V2 reruns used newer shared runtime/attention modules. Generation AST and kernel dependencies match for these two models; a subsequent synthetic H100 audit confirms bitwise output/map and exact work-count parity. Source hashes differ from the original controls; retain this qualification and the after-execution timing of the audit.

| Model/task | Arm | Accuracy | N | C | N/C | T | Sparsity |
|---|---|---:|---:|---:|---:|---:|---:|
| llada22/aime26 | All-kept | 32.22% | 154593 | 20767 | 7.444 | 662063 | 0.00% |
| llada22/aime26 | Dense | 40.00% | 138512 | 19813 | 6.991 | 631678 | 0.00% |
| llada22/aime26 | V31 | 23.33% | 119064 | 13271 | 8.972 | 422252 | 50.00% |
| llada22/aime26 | V2 | 31.11% | 122995 | 13290 | 9.255 | 422632 | 51.72% |
| llada22/aime26 | V3a | 27.78% | 125912 | 14869 | 8.468 | 473315 | 50.00% |
| llada22/humaneval | All-kept | 84.15% | 5153 | 1439 | 3.581 | 40870 | 0.00% |
| llada22/humaneval | Dense | 85.98% | 4717 | 1405 | 3.357 | 39862 | 0.00% |
| llada22/humaneval | V31 | 50.61% | 6029 | 1285 | 4.692 | 35900 | 49.98% |
| llada22/humaneval | V2 | 6.10% | 2331 | 503 | 4.634 | 10317 | 49.22% |
| llada22/humaneval | V3a | 62.80% | 6941 | 1422 | 4.881 | 40227 | 49.99% |
| llada22/ruler128k | All-kept | 84.82% | 1044 | 237 | 4.405 | 3664 | 0.00% |
| llada22/ruler128k | Dense | 85.44% | 1015 | 237 | 4.283 | 3663 | 0.00% |
| llada22/ruler128k | V31 | 86.01% | 1017 | 237 | 4.291 | 3689 | 50.00% |
| llada22/ruler128k | V2 | 86.23% | 1024 | 236 | 4.339 | 3656 | 49.19% |
| llada22/ruler128k | V3a | 85.03% | 1040 | 237 | 4.388 | 3662 | 50.00% |
| trado/aime26 | All-kept | 4.44% | 47348 | 22921 | 2.066 | 91413 | 0.00% |
| trado/aime26 | Dense | 8.89% | 48516 | 24203 | 2.005 | 96555 | 0.00% |
| trado/aime26 | V31 | 3.33% | 39895 | 17583 | 2.269 | 70081 | 50.00% |
| trado/aime26 | V2 | 6.67% | 48514 | 23887 | 2.031 | 95292 | 49.78% |
| trado/aime26 | V3a | 2.22% | 38130 | 16180 | 2.357 | 64451 | 49.97% |
| trado/humaneval | All-kept | 78.05% | 9425 | 5076 | 1.857 | 19796 | 0.00% |
| trado/humaneval | Dense | 79.27% | 9612 | 5225 | 1.840 | 20383 | 0.00% |
| trado/humaneval | V31 | 42.68% | 11616 | 5105 | 2.275 | 19934 | 49.55% |
| trado/humaneval | V2 | 76.22% | 10059 | 5362 | 1.876 | 20933 | 48.84% |
| trado/humaneval | V3a | 47.56% | 11233 | 5600 | 2.006 | 21901 | 49.67% |
| ultrallada/humaneval | All-kept | 1.22% | 167936 | 2624 | 64.000 | 167936 | 0.00% |
| ultrallada/humaneval | Dense | 1.22% | 167936 | 2624 | 64.000 | 167936 | 0.00% |
| ultrallada/humaneval | V31 | 0.00% | 167936 | 2624 | 64.000 | 167936 | 48.99% |
| ultrallada/humaneval | V2 | 0.00% | 167936 | 2624 | 64.000 | 167936 | 49.45% |
| ultrallada/humaneval | V3a | 0.00% | 167936 | 2624 | 64.000 | 167936 | 48.99% |
