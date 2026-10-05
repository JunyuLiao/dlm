# E15 receipts: observe_carried (c01) panel with six fresh seeds (1616–2121)

- Protocol `v27_c01_e15_e83919d6411343b2`, deploy `v27_e15_f63a1f2`, run dir `v27_lb_e15_001`.
- 1,602/1,602 runs ok (534 per arm) on mpk and dlm2, one host per cell. Timed new graphs: 1 dense pair and 1 main
  pair (excluded in Wc).
- Stages 32K and AIME26 were launched by the host-side chain `host_chain_e15.py`.
- Path receipt: in the c01 arm, 41,505 of 44,175 observation calls took their output from the carried map. The rest
  are first canvases, which stay on the dense fused path.

## c01 / M3 + c0 (direct, paired)

| bin | cells | accuracy (+/-, p) | per-step S/N [CI] | W [CI] | N [CI] |
|---|---:|---|---|---|---|
| AIME26 | 180 | 97 / 100 (+12/-15, 0.70) | **0.983 [0.961, 0.996]** | 1.018 [0.974, 1.065] | 1.042 [0.996, 1.090] |
| 32K | 144 | 95 / 90 (+17/-12, 0.46) | 1.004 [1.000, 1.009] | 0.948 [0.912, 0.985] | 0.930 [0.881, 0.981] |
| 64K | 144 | 70 / 78 (+5/-13, 0.096) | 0.995 [0.993, 0.999] | 1.008 [0.971, 1.044] | 1.014 [0.949, 1.079] |
| 96K | 66 | 30 / 31 (+6/-7, 1.0) | 1.002 [0.976, 1.034] | 0.977 [0.898, 1.056] | 0.965 [0.830, 1.110] |

Against dense, c01 at 64K is 70 vs 83 (+4/-17, exact McNemar p about 0.007).

**Verdict: c01 is not adopted.**
- The per-step saving is 0–1.7% (the kernel-level call-1 saving spread over 14–18 steps per canvas). P15's −5% at 64K
  was a small-sample effect.
- c01 adds an accuracy risk at 64K.
- The W differences come from step counts, not from c01's per-step cost.

## Main result pooled over E13 + E14 + E15 (18 seeds; `direct_c0_vs_dense_18seeds.md`)

| bin | cells | W [CI] | S | per-step | N | accuracy main / dense (p) |
|---|---:|---|---:|---:|---:|---|
| 32K | 432 | **0.950 [0.920, 0.980]** | 0.937 | 0.921 | 1.018 | 267 / 274 (0.55) |
| 64K | 432 | **0.867 [0.841, 0.892]** | 0.787 | 0.814 | 0.967 | 228 / 238 (0.24) |
| 96K | 198 | **0.818 [0.731, 0.898]** | 0.728 | 0.745 | 0.977 | 93 / 76 (0.012, sparse higher) |
| AIME26 (E15 only) | 180 | 0.951 [0.894, 1.001] | 0.950 | 1.024 | 0.928 | 100 / 94 (0.41) |

- At 32K, E15's main arm took more steps (N 1.10). That moves the pooled 32K W from 0.925 (12 seeds) to 0.950.
- On AIME, W 0.951 comes from 7% fewer steps; per step the main arm is 2.4% slower than dense.
