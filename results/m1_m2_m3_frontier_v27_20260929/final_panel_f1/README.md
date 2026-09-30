# v27 final-configuration panels (piecewise_v3, 8 arms)

Frozen before generation; two H100 hosts with every arm of a cell on one host. Arms: dense D_fa4_allkept; Fan plain
M1/M2c/M3 (A8); B and M3 R6 DP -ln2; each of the latter two also with a 32768-key length gate (dense below 32K keys).

- LongBench panel `v27_final_lb_f1_79c81baf9ac23611` (`../specs/v27_final_lb_f1.json`): LongBench-v2 pool (12 items,
  10-19K tokens) and the 32K/64K panel items (12 + 12), seeds 101/202/303, 36 cells per arm per dataset.
  Summary `summary_lb.md` / `summary_lb.csv`, cells `cells_lb.csv`. All 864 executions succeeded.
- AIME panel `v27_final_aime_f1_6d5d56ee2ab4dc8d` (`../specs/v27_final_aime_f1.json`): all 30 AIME26 problems x
  seeds 101/202/303/404 (120 cells per arm). All 960 executions succeeded. Summary `summary_aime.md` /
  `summary_aime.csv`, cells `cells_aime.csv`.

## LongBench (paired against D_fa4_allkept)

| arm | 64K W [95% CI] | 64K decode S | 64K correct (dense 18) | 32K W [95% CI] | 32K correct (dense 28) | pool W | pool correct (dense 20) |
|---|---|---:|---:|---|---:|---:|---:|
| Fan M1 R1 A8 | 1.267 [1.200, 1.336] | 1.413 | 20 | 1.329 [1.144, 1.544] | 25 | 1.303 | 22 |
| Fan M2c R1 A8 | 1.491 [1.365, 1.623] | 1.788 | 16 | 1.326 [1.139, 1.553] | 27 | 1.298 | 20 |
| Fan M3 R3 A8 | 1.120 [0.989, 1.248] | 1.181 | 21 | 1.154 [0.993, 1.324] | 29 | 1.173 | 22 |
| B | 0.864 [0.825, 0.905] | 0.788 | 20 | 0.935 [0.862, 1.016] | 31 | 1.044 | 18 |
| B, 32K gate | 0.858 [0.815, 0.903] | 0.791 | 20 | 0.982 [0.917, 1.052] | 29 | 1.005 | 20 |
| M3 R6 DP -ln2 | 0.869 [0.805, 0.933] | 0.783 | 23 | 0.902 [0.780, 1.015] | 31 | 1.076 | 19 |
| M3 R6 DP -ln2, 32K gate | 0.871 [0.809, 0.935] | 0.783 | 23 | 0.909 [0.777, 1.030] | 31 | 1.006 | 20 |

- 64K: B and M3 R6 DP -ln2 reproduce the sweep and the variant panel (0.868 / 0.872 / 0.869 for M3 R6 DP -ln2,
  identical correct counts: piecewise_v3 generates the same tokens as piecewise_v2).
- The 32K gate makes the 10-19K pool exactly dense (same tokens, +0/-0, 0.5% gate overhead) and keeps the 64K gain.
- Fan's plain methods are 1.12-1.49x dense at every length, with unchanged step counts.

## AIME26 (120 cells per arm; dense 65 correct)

| arm | correct | +/- vs dense | sign test p | request W [95% CI] | steps/canvas | sec/task | steps/task |
|---|---:|---|---:|---|---:|---:|---:|
| dense D_fa4_allkept | 65 | | | 1 | 1 | 14.34 | 301.1 |
| Fan M1 R1 A8 | 66 | +13/-12 | 1.00 | 1.056 [1.016, 1.101] | 0.998 | 15.32 | 301.6 |
| Fan M2c R1 A8 | 67 | +12/-10 | 0.83 | 1.013 [0.944, 1.081] | 0.987 | 14.76 | 294.4 |
| Fan M3 R3 A8 | 63 | +8/-10 | 0.82 | 0.990 [0.918, 1.055] | 1.001 | 14.38 | 303.5 |
| B | 58 | +11/-18 | 0.27 | 1.020 [0.944, 1.089] | 1.038 | 14.91 | 326.7 |
| B, 32K gate | 65 | +0/-0 | 1.00 | 0.987 [0.928, 1.040] | 1.000 | 14.18 | 301.1 |
| M3 R6 DP -ln2 | 65 | +9/-9 | 1.00 | 0.971 [0.898, 1.044] | 1.004 | 14.24 | 309.5 |
| M3 R6 DP -ln2, 32K gate | 65 | +0/-0 | 1.00 | 0.969 [0.917, 1.003] | 1.000 | 13.90 | 301.1 |

- M3 R6 DP -ln2 is accuracy-neutral on AIME without any gate (65 vs 65 of 120, +9/-9); Fan's plain M1/M2c/M3 are
  within noise (63-67). B is the only arm trending down (58, p = 0.27; 26 vs 35 of 60 in the earlier panel).
- No arm is significantly faster on AIME: prefixes stay below about 9K keys, where GLOBAL attention is 4-8% of a
  forward. The gated arms compute exactly what dense computes, so their 0.97-0.99 request ratios measure timing
  noise. Fan M1 is significantly slower (per-step selector).
