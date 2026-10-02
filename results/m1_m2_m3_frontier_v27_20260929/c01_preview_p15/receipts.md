# P15 receipts: observe_carried (c01) accuracy-first preview

- Protocol `v27_c01_preview_p15_3e50c12a37ee0486`, deploy `v27_p15_5bf19f1`. The 237 v27 tests pass on dllm.
- 180/180 runs ok on mpk and dlm2, one host per cell. Timed new graphs: 1 dense and 1 q64c pair on AIME (excluded in Wc).
- Path receipts: in the c01 arm every observation call after a request's first canvas took its output from the carried
  map (AIME 1,530 of 1,630; 32K 1,245 of 1,325; 64K 980 of 1,060; 96K 325 of 365). The rest are first-canvas
  observations, which have no carried map and stay dense fused.

c01 / q64c (direct, paired; small preview, not a result):

| bin | cells | accuracy (+/-) | per-step S/N [CI] | W [CI] | steps per canvas |
|---|---:|---|---|---|---:|
| AIME26 | 20 | 15 / 14 (+1/-0) | 0.910 [0.750, 1.007] | 0.943 [0.740, 1.136] | 0.995 |
| 32K | 16 | 10 / 12 (+1/-3) | 0.983 [0.950, 1.002] | 1.058 [0.913, 1.238] | 1.038 |
| 64K | 16 | 8 / 8 | 0.949 [0.913, 0.983] | 0.977 [0.864, 1.078] | 1.107 |
| 96K | 8 | 5 / 5 | 0.994 [0.979, 1.010] | 1.012 [0.866, 1.213] | 1.019 |

Readings:
- No accuracy-drop signal against dense: c01 / dense is 15/14 (AIME), 10/11 (32K), 8/8 (64K), 5/4 (96K).
- The per-step saving at 64K and on AIME is larger than the kernel-level call-1 estimate.
- 64K shows more steps per canvas (1.107). This is checked in E15.
