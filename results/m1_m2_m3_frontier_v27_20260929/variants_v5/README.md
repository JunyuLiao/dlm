# v27 variant panel v5 (piecewise_v3): layer and tail variants, Fan M1/M2c/M3 on the final substrate

Protocol `v27_long_lb_variants_v5_e9b1c8a1b6de7422` (`../specs/v27_long_lb_variants_v5.json`), frozen before
generation: the same 12 + 12 LongBench-v2 32K/64K panel items and seeds 101/202/303 as the threshold sweep
(36 cells per arm per bin), substrate `piecewise_v3` (tokens identical to piecewise_v2, per-call KV concat removed),
two H100 hosts with every arm of a cell on one host. Summary: `summary.md`, `summary.csv`; cells: `cells.csv`.

**Correctness checks from the receipts.** Every arm ran on piecewise_v3 with no new graphs in timed requests and one
host per cell. no_first made exactly 1.0 native GLOBAL call per decoder call and late3 exactly 2.0; cap24 entered
dense only at step 24; -ln2 arms ran with the global log threshold -3.874 (= -3.180 - ln2). B and M3 R6 DP -ln2
reproduce the threshold sweep's correct counts exactly (piecewise_v3 generates the same tokens as piecewise_v2).

## 64K (dense 18 of 36 correct)

| arm | correct | +/- | request W [95% CI] | decode S [95% CI] | steps/canvas | S/N |
|---|---:|---|---|---|---:|---:|
| Fan M1 R1 A8 | 20 | +3/-1 | 1.267 [1.199, 1.336] | 1.416 | 0.977 | 1.474 |
| Fan M2c R1 A8 | 16 | +0/-2 | 1.497 [1.372, 1.630] | 1.792 | 0.997 | 1.708 |
| Fan M3 R3 A8 | 21 | +4/-1 | 1.117 [0.986, 1.240] | 1.182 | 0.999 | 1.182 |
| **B** | 20 | +5/-3 | **0.861 [0.821, 0.903]** | **0.788 [0.747, 0.833]** | 1.044 | 0.795 |
| B, first layer dense | 18 | +3/-3 | 0.913 [0.848, 0.982] | 0.854 | 1.020 | 0.838 |
| M3 R6 DP, first layer dense | 21 | +5/-2 | 0.914 [0.788, 1.027] | 0.876 | 1.052 | 0.839 |
| **M3 R6 DP, -ln2** | 23 | +5/-0 | **0.872 [0.812, 0.936]** | **0.783 [0.714, 0.867]** | 0.952 | 0.833 |
| M3 R6 DP, -ln2, first layer dense | 18 | +1/-1 | 0.922 [0.817, 1.046] | 0.881 | 0.999 | 0.862 |
| M3 R6 DP, -ln2, layers 5 and 11 dense | 17 | +1/-2 | 0.901 [0.796, 0.996] | 0.863 | 0.981 | 0.898 |
| M3 R6 DP, -ln2, dense after step 24 | 21 | +5/-2 | 0.909 [0.842, 0.991] | 0.841 | 0.965 | 0.860 |

## 32K (dense 28 of 36 correct)

| arm | correct | +/- | request W [95% CI] | steps/canvas |
|---|---:|---|---|---:|
| Fan M1 R1 A8 | 25 | +4/-7 | 1.329 [1.160, 1.529] | 1.027 |
| Fan M2c R1 A8 | 27 | +3/-4 | 1.326 [1.139, 1.552] | 1.007 |
| Fan M3 R3 A8 | 29 | +6/-5 | 1.154 [0.993, 1.326] | 0.991 |
| B | 31 | +5/-2 | 0.929 [0.857, 1.007] | 1.015 |
| B, first layer dense | 22 | +3/-9 | 1.068 [0.954, 1.209] | 1.047 |
| M3 R6 DP, first layer dense | 24 | +3/-7 | 1.016 [0.884, 1.163] | 1.052 |
| M3 R6 DP, -ln2 | 31 | +6/-3 | 0.898 [0.772, 1.016] | 0.996 |
| M3 R6 DP, -ln2, first layer dense | 28 | +6/-6 | 0.906 [0.788, 1.022] | 0.974 |
| M3 R6 DP, -ln2, layers 5 and 11 dense | 24 | +4/-8 | 1.091 [0.992, 1.211] | 1.049 |
| M3 R6 DP, -ln2, dense after step 24 | 30 | +6/-4 | 0.935 [0.798, 1.072] | 0.984 |

## Findings

- None of the layer or tail variants beats plain M3 R6 DP at -ln2 or B at 64K. Keeping the first GLOBAL layers dense
  raises the kept attention mass but costs a dense layer (about 2.7 ms per call at 64K), more than it saves in steps.
- The best settings reproduce across substrates and panels: B 0.861 (sweep: 0.866), M3 R6 DP -ln2 0.872 (sweep:
  0.868) at 64K.
- Fan's plain M1/M2c/M3 are 1.12-1.50x dense in request time at 64K and 1.15-1.33x at 32K on the final substrate,
  with unchanged step counts: their per-step selector and re-observation cost more than the attention they skip
  (`../kernel_bench/README.md`).
- Accuracy stays within the paired noise at this size; the swings (e.g. B first layer dense 22 vs 28 at 32K,
  +3/-9, p = 0.15) are not significant.
