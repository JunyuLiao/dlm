# math500: optimized refresh comparison

V2 threshold 3.2; held calibration sparsity 38.81% from 115840 eligible and 70880 kept physical GLOBAL tiles across 4 calibration cells. Target range was 40-50%; in range=False. Calibration uses computation only. The calibration target was missed: retained held tiles equal the mandatory tile floor on these calibration trajectories. The protocol accepts this measured ceiling and reports full-panel work separately. Timings below are the separate clean pass.

| Method | Correct | Accuracy | Mean W (s) | N/C | S/N (ms) | GLOBAL sparsity | Held sparsity | Refresh (audit ms) | Held (audit ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2_online_optimized | 392/500 | 78.40% | 1.8613 | 12.465 | 20.784 | 77.58% | 79.43% | 0.160 | 0.027 |
| v31_optimized | 433/500 | 86.60% | 1.6069 | 11.058 | 20.975 | 59.37% | 73.05% | 0.121 | 0.029 |
| dense_full_fix51994 | 449/500 | 89.80% | 1.3322 | 10.241 | 18.508 | 0.00% | n/a | n/a | n/a |
| dense_piecewise | 449/500 | 89.80% | 1.3440 | 10.241 | 18.749 | 0.00% | n/a | n/a | n/a |
| allkept_fa4 | 450/500 | 90.00% | 1.4394 | 10.327 | 20.217 | 0.00% | n/a | n/a | n/a |

V2 physical GLOBAL work by phase: carried: 5668160/16351040 kept (65.33% sparse); dense: 494080/494080 kept (0.00% sparse); held: 31668000/153918720 kept (79.43% sparse); initial: 3391040/14073920 kept (75.91% sparse); refresh: 3351840/13930240 kept (75.94% sparse). First-canvas work is 5043200/5436320 kept (7.23% sparse); whole-request work is 44573120/198768000 kept (77.58% sparse).

LOCAL and prefill stay dense. Inherited native adaptive stopping and the resident direct-paged held consumer are preserved. Refresh and held event timing are audit-only; request/generation timing is clean. S/N is an amortized per-step cost, not a measured single-forward price. Paired problem-clustered confidence intervals are in paired_vs_v31.json. The full mandated coverage includes the four transductive calibration questions; their prior exposure is disclosed and they are not held-out evidence.
