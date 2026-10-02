# E10 receipts: M2 with the same optimizations as M3 (LongBench-v2 32K/64K, 24 items x seeds 404-707 = 96 cells per arm per bin)

- 576/576 runs ok; one host per cell (dllm, mpk, dlm2); no timed new graphs.
- Dense and M3 R6 DP -ln2 + carry_first are token-identical to E5 on these cells (192/192 each).
- Effective method (all 192 runs per arm agree): M2c arm mu_mode=pooled_compact, M3 arm mu_mode=exact; both
  carry_first=true, threshold_shift=minus_ln2, decision_interval=6. Carried-first calls: M2c 11,720, M3 11,995.

Direct M2c / M3 paired request-wall ratio (item-clustered bootstrap, seed 7) and accuracy (exact McNemar):

| bin | M2c / M3 W [CI] | correct M2c-only / M3-only (p) |
|---|---|---|
| 32K | 0.987 [0.914, 1.063] | 11 / 8 (0.65) |
| 64K | 0.983 [0.940, 1.024] | 6 / 6 (1.00) |
