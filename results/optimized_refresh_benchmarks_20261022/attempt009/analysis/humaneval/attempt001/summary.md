# humaneval: optimized refresh comparison

V2 threshold 0.560071620607, fixed by explicit user instruction without new calibration. Provenance: results/humaneval_value_observe_no_overlap_k128_20261019/attempt001/calibration/thinking_on/full164_threshold.json. Current physical work is measured below; prior matching does not establish current work equivalence or 40-50% held sparsity. Timings below are the separate clean pass.

| Method | Correct | Accuracy | Mean W (s) | N/C | S/N (ms) | GLOBAL sparsity | Held sparsity | Refresh (audit ms) | Held (audit ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2_online_optimized | 159/164 | 96.95% | 1.1755 | 9.462 | 22.113 | 66.25% | 68.95% | 0.151 | 0.038 |
| v31_optimized | 160/164 | 97.56% | 1.2589 | 10.358 | 21.756 | 59.18% | 73.51% | 0.116 | 0.030 |
| dense_full_fix51994 | 161/164 | 98.17% | 0.9531 | 9.255 | 19.332 | 0.00% | n/a | n/a | n/a |
| dense_piecewise | 161/164 | 98.17% | 0.9590 | 9.255 | 19.543 | 0.00% | n/a | n/a | n/a |
| allkept_fa4 | 161/164 | 98.17% | 1.0502 | 9.226 | 21.038 | 0.00% | n/a | n/a | n/a |

V2 physical GLOBAL work by phase: carried: 1807110/3963360 kept (54.40% sparse); dense: 195200/195200 kept (0.00% sparse); held: 7513170/24193920 kept (68.95% sparse); initial: 1116536/3383520 kept (67.00% sparse); refresh: 1152670/3178080 kept (63.73% sparse). First-canvas work is 1771491/1925440 kept (8.00% sparse); whole-request work is 11784686/34914080 kept (66.25% sparse).

LOCAL and prefill stay dense. Inherited native adaptive stopping and the resident direct-paged held consumer are preserved. Refresh and held event timing are audit-only; request/generation timing is clean. S/N is an amortized per-step cost, not a measured single-forward price. Paired problem-clustered confidence intervals are in paired_vs_v31.json. This workload uses a previously examined/fixed threshold and is not fresh calibration or held-out confirmation.
