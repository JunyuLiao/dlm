# E9 receipts, accuracy tests and direct arm-to-arm ratios (LongBench-v2 64K, 24 items x seeds 404-707 = 96 cells per arm)

Integrity: 576/576 runs ok; 96 cells, each on one host (dllm, mpk, dlm2: 192 runs each); no timed new graphs.
Dense and the -ln2 + carry_first arm are token-identical to the same cells in E5 (96/96 each).

Effective-method fields (all 96 runs of each arm agree):

| arm | V term | risk_topk | proj_rank | risk_value | mu_mode | carry_first | carried-first calls | bootstrap dense |
|---|---|---|---:|---|---|---|---:|---:|
| M3_R6_A64_fused_dp_async_topk88_c0_fa4 | projected V, rank 32 | k12 | 32 | – | exact | true | 5,225 | 480 |
| ..._topk88_r8_c0_fa4 | projected V, rank 8 | k12 | 8 | – | exact | true | 5,315 | 480 |
| M2c_R6_A64_fused_dp_async_topk88_c0_fa4 | M2 tile-mean V | k12 | – | – | pooled_compact (20,225 builds) | true | 5,225 | 480 |
| ..._topk88_mass_c0_fa4 | none (attention mass) | k12 | – | mass | exact | true | 5,115 | 480 |
| M3_R6_A64_fused_dp_async_m1ln2_c0_fa4 | projected V, rank 32 (threshold −ln2) | – | 32 | – | exact | true | 5,190 | 480 |

Bootstrap dense 480 = 96 requests x 5 GLOBAL layers (canvas 0 only).

Accuracy, exact McNemar:

| arm | correct (dense 51) | vs dense (p) | vs rank 32 at k12 (p) |
|---|---:|---|---|
| rank 32, k12 | 53 | +8/−6 (0.79) | – |
| rank 8, k12 | 50 | +8/−9 (1.00) | +6/−9 (0.61) |
| M2 tile mean, k12 | 54 | +9/−6 (0.61) | +10/−9 (1.00) |
| mass only, k12 | 54 | +7/−4 (0.55) | +6/−5 (1.00) |
| −ln2 threshold | 49 | +6/−8 (0.79) | +5/−9 (0.42) |

Direct paired request-wall ratios (geometric mean over 96 cells, item-clustered bootstrap 95% CI, seed 7):

| ratio | W [CI] |
|---|---|
| mass / rank 32 | 0.960 [0.905, 1.018] |
| rank 8 / rank 32 | 0.971 [0.915, 1.039] |
| M2 tile mean / rank 32 | 1.010 [0.945, 1.085] |
| mass / −ln2 threshold | 0.999 [0.961, 1.037] |
| −ln2 threshold / rank 32 | 0.961 [0.902, 1.026] |
