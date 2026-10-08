# Synthetic selector qualification

These are synchronized synthetic discovery/selection measurements at native GLOBAL geometry, not model accuracy or end-to-end measurements. The projection is Gaussian32 seed 1729. Online threshold 0.01 is diagnostic only; the fixed budget is 8192 tokens.

| prefix | statistics ms | V1 ms | V2 ms | V3a ms | exact V3b ms | batch8 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 0.511 | 0.330 | 0.335 | 1.118 | 0.745 | 0.726 |
| 32768 | 2.356 | 3.015 | 2.829 | 1.861 | 53.211 | 19.235 |
| 65536 | 4.340 | 5.493 | 5.076 | 2.644 | 210.787 | 44.138 |
| 131072 | 8.402 | 10.922 | 10.114 | 4.767 | 844.271 | 139.745 |

Exact greedy and batch8 trade selection cost for projected objective quality. At 128K their maximum relative sketch errors are approximately 0.01571 and 0.01771; their mask Jaccard is 0.0405 on this synthetic case. Similar errors do not mean identical decisions or proven accuracy parity. These timings exclude projection and the model attention output call.

The failed diagnostic serializer attempt is retained under attempt001. Mathematical and consumer qualification: 32 independent tests plus 2 residual controls passed on H100. Four historical route-only test failures reproduce on untouched cdc12221; the residual test passes in an isolated process.
