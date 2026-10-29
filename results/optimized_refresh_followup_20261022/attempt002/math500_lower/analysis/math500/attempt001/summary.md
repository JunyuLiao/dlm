# math500: optimized refresh comparison

V2 threshold 0.2; held calibration sparsity 46.58% from 5606720 eligible and 2995331 kept physical GLOBAL tiles across 32 calibration cells. Target range was 40-50%; in range=True. Calibration uses computation only. Timings below are the separate clean pass.

| Method | Correct | Accuracy | Mean W (s) | N/C | S/N (ms) | GLOBAL sparsity | Held sparsity | Refresh (audit ms) | Held (audit ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2_online_optimized | 438/500 | 87.60% | 1.5721 | 10.434 | 21.232 | 47.60% | 49.36% | 0.150 | 0.053 |
| v31_optimized | 433/500 | 86.60% | 1.6069 | 11.058 | 20.975 | 59.37% | 73.05% | 0.121 | 0.029 |
| dense_full_fix51994 | 449/500 | 89.80% | 1.3322 | 10.241 | 18.508 | 0.00% | n/a | n/a | n/a |
| dense_piecewise | 449/500 | 89.80% | 1.3440 | 10.241 | 18.749 | 0.00% | n/a | n/a | n/a |
| allkept_fa4 | 450/500 | 90.00% | 1.4394 | 10.327 | 20.217 | 0.00% | n/a | n/a | n/a |

V2 physical GLOBAL work by phase: carried: 9558697/15317760 kept (37.60% sparse); dense: 494080/494080 kept (0.00% sparse); held: 51646982/101993600 kept (49.36% sparse); initial: 6381409/13072640 kept (51.19% sparse); refresh: 7214299/12825920 kept (43.75% sparse). First-canvas work is 5083763/5169920 kept (1.67% sparse); whole-request work is 75295467/143704000 kept (47.60% sparse).

LOCAL and prefill stay dense. Inherited native adaptive stopping and the resident direct-paged held consumer are preserved. Refresh and held event timing are audit-only; request/generation timing is clean. S/N is an amortized per-step cost, not a measured single-forward price. Paired problem-clustered confidence intervals are in paired_vs_v31.json. The full mandated coverage includes the four transductive calibration questions; their prior exposure is disclosed and they are not held-out evidence.
