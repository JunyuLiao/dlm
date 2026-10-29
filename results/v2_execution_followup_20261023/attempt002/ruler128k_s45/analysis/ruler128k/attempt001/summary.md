# ruler128k: optimized refresh comparison

V2 threshold 0.00185615530061; held calibration sparsity 52.08% from 15379200 eligible and 7369332 kept physical GLOBAL tiles across 13 calibration cells. Target range was 40-50%; in range=True. Calibration uses computation only. Timings below are the separate clean pass.

| Method | Correct | Accuracy | Mean W (s) | N/C | S/N (ms) | GLOBAL sparsity | Held sparsity | Refresh (audit ms) | Held (audit ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2_online_optimized | 80/104 | 87.45% | 6.6401 | 6.000 | 58.534 | 51.86% | 61.99% | 6.293 | 1.633 |
| v31_optimized | 79/104 | 86.78% | 6.5946 | 5.548 | 53.002 | 47.33% | 93.49% | 3.432 | 0.222 |
| dense_full_fix51994 | 81/104 | 86.68% | 6.6089 | 5.875 | 54.441 | 0.00% | n/a | n/a | n/a |
| dense_piecewise | 81/104 | 86.68% | 6.6016 | 5.875 | 52.975 | 0.00% | n/a | n/a | n/a |
| allkept_fa4 | 81/104 | 87.12% | 6.6066 | 5.913 | 52.564 | 0.00% | n/a | n/a | n/a |

V2 physical GLOBAL work by phase: carried: 13099346/33973600 kept (61.44% sparse); dense: 33907040/33907040 kept (0.00% sparse); held: 28829012/75853440 kept (61.99% sparse); initial: 11641646/33907040 kept (65.67% sparse); refresh: 10580548/26052800 kept (59.39% sparse). First-canvas work is 84958246/169720320 kept (49.94% sparse); whole-request work is 98057592/203693920 kept (51.86% sparse).

LOCAL and prefill stay dense. Inherited native adaptive stopping and the resident direct-paged held consumer are preserved. Refresh and held event timing are audit-only; request/generation timing is clean. S/N is an amortized per-step cost, not a measured single-forward price. Paired problem-clustered confidence intervals are in paired_vs_v31.json. The full mandated coverage includes the four transductive calibration questions; their prior exposure is disclosed and they are not held-out evidence.
