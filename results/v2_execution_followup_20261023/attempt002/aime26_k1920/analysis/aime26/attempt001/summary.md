# aime26: optimized refresh comparison

V2 threshold 0.0594603557501, fixed by explicit user instruction without new calibration. Provenance: user fixed LongBench0.021 or preserved short-benchmark V2. Current physical work is measured below; prior matching does not establish current work equivalence or 40-50% held sparsity. Timings below are the separate clean pass.

| Method | Correct | Accuracy | Mean W (s) | N/C | S/N (ms) | GLOBAL sparsity | Held sparsity | Refresh (audit ms) | Held (audit ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2_online_optimized | 50/90 | 55.56% | 6.4140 | 13.541 | 20.876 | 44.94% | 45.58% | 0.288 | 0.101 |
| v31_optimized | 54/90 | 60.00% | 5.9013 | 13.229 | 20.756 | 41.54% | 48.97% | 0.215 | 0.080 |
| dense_full_fix51994 | 51/90 | 56.67% | 5.1751 | 13.497 | 18.283 | 0.00% | n/a | n/a | n/a |
| dense_piecewise | 51/90 | 56.67% | 5.2394 | 13.497 | 18.527 | 0.00% | n/a | n/a | n/a |
| allkept_fa4 | 49/90 | 54.44% | 5.9087 | 13.254 | 19.968 | 0.00% | n/a | n/a | n/a |

V2 physical GLOBAL work by phase: carried: 12765691/20454080 kept (37.59% sparse); dense: 101760/101760 kept (0.00% sparse); held: 116207366/213536000 kept (45.58% sparse); initial: 9449664/19151040 kept (50.66% sparse); refresh: 11392990/19013920 kept (40.08% sparse). First-canvas work is 1127892/1133120 kept (0.46% sparse); whole-request work is 149917471/272256800 kept (44.94% sparse).

LOCAL and prefill stay dense. Inherited native adaptive stopping and the resident direct-paged held consumer are preserved. Refresh and held event timing are audit-only; request/generation timing is clean. S/N is an amortized per-step cost, not a measured single-forward price. Paired problem-clustered confidence intervals are in paired_vs_v31.json. This workload uses a previously examined/fixed threshold and is not fresh calibration or held-out confirmation.
