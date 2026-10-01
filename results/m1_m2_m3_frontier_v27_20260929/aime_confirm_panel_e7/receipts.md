# E7 receipts (AIME26, 30 problems x seeds 404-909 = 180 cells per arm)

Integrity: 1,260/1,260 runs ok; 180 cells, each cell on one host (dllm, mpk, dlm2: 420 runs each);
3 runs with timed new graphs (excluded in the Wc column).

GLOBAL-layer routing counters, summed over the 180 requests of each arm:

| arm | routed calls | gated to native dense (key < 2K) | first-call carry | bootstrap dense | held decisions | decoder calls |
|---|---:|---:|---:|---:|---:|---:|
| D_fa4_allkept | 0 | 0 | 0 | 0 | 0 | 53,263 |
| M1_R1_A8_fa4 | 228,730 | 0 | 0 | 20,350 | 0 | 53,886 |
| M2c_R1_A8_fa4 | 229,660 | 0 | 0 | 20,445 | 0 | 54,110 |
| M3_R3_A8_fa4 | 233,615 | 0 | 0 | 20,470 | 155,040 | 54,911 |
| M3_R6_A64_fused_dp_async_m1ln2_fa4 | 232,310 | 0 | 0 | 20,570 | 202,175 | 54,690 |
| M3_R6_..._m1ln2_c0_gate2k_fa4 | 163,070 | 74,845 | 13,115 | 875 | 141,590 | 53,179 |
| B_A64_fused_rp_async_m1ln2_c0_gate2k_fa4 | 170,860 | 74,845 | 13,470 | 875 | 170,860 | 54,879 |

The gate and the first-call carry both ran as intended: about 31% of GLOBAL calls of the gated arms had fewer
than 2K keys and ran the native dense kernel, and the c0 arms bootstrap densely only on canvas 0.
