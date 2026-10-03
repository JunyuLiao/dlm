# Results ledger (verified results only)

Rules for this file:
- An entry needs a scored, frozen panel or a documented benchmark whose summary is in the repo.
- Historical HF entries use `D_fa4_allkept` unless stated. Current headline speed evidence must use native vLLM dense (FA4 dynamic-causal with effective split-KV).
- Ratios are method / dense: paired geometric mean with a question-clustered bootstrap 95% CI. < 1 means faster.
- W includes prefill. Historical HF W times the generation call, excluding method setup/cleanup;
  native V18b W includes adapter begin/end, but excludes HTTP/network time. S excludes prefill.
  S/N = amortized per step. NC = steps per canvas. T = output tokens.
- Accuracy is strict-correct cells, method vs dense.
- Paths are relative to `results/m1_m2_m3_frontier_v27_20260929/`.
- Protocol ids are the frozen protocol identities; frozen files are private (`E:/dlm/v23_private/`).
- Arm shorthand: **M3** = `M3_R6_A64_fused_dp_async_m1ln2_fa4` (M3 R6 DP −ln2). **Fan plain** = `M1_R1_A8_fa4` /
  `M2c_R1_A8_fa4` / `M3_R3_A8_fa4`.

## Historical HF results (piecewise_v5)

Baseline correction (2026-10-02): these speed ratios use FA4 num_splits=1 and are not
headline evidence against the fastest official serving. vLLM dynamic-causal split-KV is
faster. Keep these results unchanged as history; the new vLLM panel is not scored yet.

Audit correction (2026-10-02): historical claims of zero capture refer only to Dynamo
`unique_graphs` unless a direct CUDA counter is documented. Repeated-seed cell McNemar p
values are exploratory; counts do not prove noninferiority. Mass-only is prefix-ranking-only,
with V-dependent tail/observation retained. The appended audit entry and
`docs/VARIANT_FAIRNESS_AUDIT_20261002.md` supersede stronger historical wording.

### L1. E4 large-seed confirmation: LongBench-v2 32K and 64K, 24 items × 6 new seeds (404–909) = 144 cells per arm per bin

| arm | 64K W [CI] | 64K S | 64K acc (dense 75) | 32K W [CI] | 32K S | 32K acc (dense 86) |
|---|---|---:|---:|---|---:|---:|
| M3 | **0.879 [0.820, 0.926]** | 0.807 | 77 | **0.940 [0.897, 0.980]** | 0.921 | 86 |
| B | 0.876 [0.812, 0.935] | 0.805 | 81 | 0.941 [0.882, 1.003] | 0.918 | 88 |
| M3, observation at call 2 (obs2) | 0.884 [0.845, 0.921] | 0.820 | 77 | 0.925 [0.860, 0.994] | 0.899 | 92 |
| M3, R3 | 0.889 [0.816, 0.950] | 0.823 | 74 | 0.937 [0.886, 0.988] | 0.917 | 88 |
| Fan plain M1 / M2c / M3 | 1.258 / 1.341 / 1.093 | — | 74 / 85 / 80 | 1.251 / 1.293 / 1.060 | — | 94 / 90 / 94 |

- **Execution and integrity.** 2,304/2,304 runs ok. No timed run captured a new graph. Every arm of a cell ran on one
  host (dllm, mpk, dlm2). Receipts confirm the intended paths:
  - obs2 has 2× bootstrap-dense calls per observation;
  - R3 has effective decision interval 3;
  - the −ln2 arms use global threshold −3.874.
- **Steps did not inflate:** NC 0.97–1.03 and T 0.95–1.01 for M3/B/obs2/R3. Prefill P is 0.99–1.01 for every arm
  (prefill is not optimized by any method).
- **Caveats.**
  - The items are the same 24 per bin as earlier panels. The first 12 (formal) were used to choose −ln2.
  - Held-out-only split (ratio of summed walls, our script, not the panel summary's estimator):
    - 64K M3 0.85 [0.73, 0.96];
    - 32K M3 0.96 [0.91, 1.01] (n.s.); 32K obs2 0.95 [0.91, 0.98].
  - Summary copied from the coordinator's scoring output; not yet committed at the time of writing.
- **Source.** `lb_confirm_panel_e4/summary.md`, spec `specs/v27_lb_confirm_e4.json`, protocol
  `v27_lb_confirm_e4_d3a7edb87a2b800a`, deploy commit 9e47ddb9d.

### L1b. E5 overhead / tail variants on the E4 cells (same 288 cells, hosts and seeds; piecewise_v5)

| arm | 64K W [CI] | 64K S | 64K acc (dense 75) | 32K W [CI] | 32K S | 32K acc (dense 86) |
|---|---|---:|---:|---|---:|---:|
| M3 (reference; tokens identical to E4) | 0.879 [0.818, 0.927] | 0.807 | 77 | 0.942 [0.900, 0.984] | 0.920 | 86 |
| **M3 + `carry_first`** | **0.853 [0.802, 0.895]** | **0.775** | 76 | **0.907 [0.858, 0.952]** | **0.878** | 92 |
| B + `carry_first` | 0.858 [0.800, 0.912] | 0.774 | 80 | 0.931 [0.883, 0.980] | 0.907 | 86 |
| M3 + `stable1` gate | 0.891 [0.832, 0.937] | 0.820 | 75 | 0.953 [0.908, 0.998] | 0.939 | 81 |
| B + `stable1` gate | 0.889 [0.831, 0.939] | 0.826 | 79 | 0.938 [0.875, 1.005] | 0.920 | 88 |

- **carry_first vs its own reference** (direct pairs, geometric mean, question-clustered bootstrap):
  - M3: 64K W 0.971 [0.939, 1.001], 32K 0.963 [0.914, 1.011];
  - accuracy discordance balanced (64K 7/8, 32K 15/9);
  - B: 64K 0.981, 32K 0.995.
- **stable1** is slower and costs accuracy at 32K (M3 81 vs 86). Rejected.
- **Integrity.**
  - 2,016/2,016 runs ok, no timed new graphs.
  - Dense, M3 and B tokens are identical to E4 (288/288 each).
  - Receipts: c0 bootstrap-dense = 5 calls (canvas 0 only) per request; stable1 fired in 225/288 M3 runs.
- **Source.** `lb_overhead_panel_e5/summary.md`, spec `specs/v27_lb_overhead_e5.json`, protocol
  `v27_lb_overhead_e5_1c93b97271a3a266`, deploy commit 758723513.

### L1c. E6 96K: the 6 items whose prompts fit one H100 (≤ about 95K tokens) × seeds 404–909 = 36 cells per arm (piecewise_v5)

| arm | W [CI] | S | per-step S/N [CI] | N [CI] | T | acc (dense 12) |
|---|---|---:|---|---|---:|---:|
| **M3 + `carry_first`** | **0.744 [0.597, 0.903]** | 0.619 | 0.762 [0.747, 0.776] | 0.812 [0.628, 1.041] | 0.858 | 16 |
| M3 | 0.819 [0.690, 0.968] | 0.720 | 0.762 [0.731, 0.785] | 0.946 | 0.934 | 14 |
| B | 0.786 [0.641, 0.946] | 0.668 | 0.743 [0.722, 0.760] | 0.900 | 0.906 | 16 |
| Fan plain M1 / M2c / M3 | 1.158 / 1.385 / 1.045 | — | 1.62 / 1.83 / 1.25 | — | — | 14 / 17 / 15 |

- The per-step decode cost (−24 to −26%) is robust: tight CIs.
- Part of the request gain comes from fewer steps or shorter outputs (N 0.81, T 0.86 for M3 + c0). With only 6
  items these are noisy: N's CI crosses 1, and per-item W ranges 0.28–1.30.
- With N = 1 the expected request ratio is about 0.85.
- **Integrity.** 252/252 runs ok, no timed new graphs, one host per cell (dllm, mpk). c0 receipts: 5 bootstrap-dense
  calls per request.
- **Source.** `lb96k_confirm_panel_e6/summary.md`, spec `specs/v27_lb96k_confirm_e6.json`, protocol
  `v27_lb96k_confirm_e6_8ab3aa35d1874ca1`.

### L1c2. 96K pooled: E6 + E6b, 11 fitting items × seeds 404–909 = 66 cells per arm (piecewise_v5)

E6b adds the 5 other fitting items of the frozen 96K pool (pool positions 13–24, prompts ≤ 95,074 tokens), same arms,
seeds and substrate. Pooled ratios are paired per cell with the bootstrap over 11 items.

| arm | W [CI] | S [CI] | per-step S/N [CI] | N [CI] | T | acc (dense 25) |
|---|---|---|---|---|---:|---:|
| **M3 + `carry_first`** | **0.822 [0.703, 0.944]** | **0.727 [0.589, 0.887]** | 0.754 [0.738, 0.768] | 0.965 [0.772, 1.190] | 0.960 | 30 |
| M3 | 0.858 [0.761, 0.956] | 0.783 [0.664, 0.918] | 0.761 [0.740, 0.781] | 1.028 | 1.004 | 31 |
| B | 0.865 [0.752, 0.969] | 0.789 [0.655, 0.928] | 0.738 [0.725, 0.751] | 1.069 | 1.031 | 31 |
| Fan plain M1 / M2c / M3 | 1.257 / 1.448 / 1.121 | 1.48 / 1.80 / 1.22 | 1.59 / 1.83 / 1.22 | — | — | 32 / 32 / 33 |

- **96K: M3 + c0 is 18% faster end to end and 27% faster in generation, with no accuracy loss** (30 vs 25). The
  per-step saving (−25%) is tight; the step count is now neutral (N 0.96, CI spans 1).
- E6b alone (5 items, 30 cells): per-step S/N 0.745 for M3 + c0, but N 1.19 [0.87, 1.48], so W 0.927 [0.782, 1.052]
  (n.s.); B's N 1.31 [1.19, 1.44]. E6 alone had N 0.81. **At 96K the step count varies by item set in both
  directions; only the pooled panel supports a request-level claim.**
- **Integrity.** E6b 210/210 runs ok, no timed new graphs, one host per cell (dllm, mpk, dlm2), c0 receipts 5
  bootstrap-dense calls per request. E6b's deploy is one code commit after E6's (c36b1f933 adds opt-in V-term options,
  off in these arms).
- **Source.** `lb96k_pooled_e6_e6b/summary.md` and `README.md`; E6b alone `lb96k_extend_panel_e6b/summary.md`, spec
  `specs/v27_lb96k_extend_e6b.json`, protocol `v27_lb96k_extend_e6b_d1b1c0c74a1dc69d`.

### L1d. E7 AIME large-seed confirmation: 30 problems × seeds 404–909 = 180 cells per arm (piecewise_v5)

| arm | acc (dense 99) | +/− | W [CI] | Wc [CI] | S | per-step S/N [CI] | N |
|---|---:|---|---|---|---:|---|---:|
| **M3 + `carry_first` + 2K gate** | **99** | +11/−11 | **1.005 [0.967, 1.042]** | 1.016 [0.985, 1.047] | 1.005 | 1.005 [0.982, 1.017] | 1.000 |
| M3 | 96 | +16/−19 | 1.033 [0.977, 1.089] | 1.044 [0.995, 1.095] | 1.032 | 1.010 [0.988, 1.023] | 1.022 |
| B (−ln2) + `carry_first` + 2K gate | 93 | +9/−15 | 1.057 [1.023, 1.096] | 1.065 [1.029, 1.105] | 1.059 | 1.032 [1.015, 1.055] | 1.026 |
| Fan plain M1 | 97 | +14/−16 | 1.089 [1.050, 1.126] | 1.094 | 1.090 | 1.077 [1.070, 1.084] | 1.012 |
| Fan plain M2c | 92 | +15/−22 | 1.115 [1.046, 1.183] | 1.115 | 1.113 | 1.097 [1.088, 1.105] | 1.015 |
| Fan plain M3 | 99 | +19/−19 | 1.069 [1.013, 1.129] | 1.081 | 1.068 | 1.032 [1.007, 1.047] | 1.035 |

- **AIME has no speed gain; the best low-overhead variant is neutral.** M3 + c0 + 2K gate is W 1.005 with accuracy
  99 vs 99. No arm's accuracy difference is significant (exact McNemar p ≥ 0.31).
- Fan plain M1/M2c/M3 are 7–12% slower than dense; B with the same gate is 6% slower.
- **Receipts.** About 31% of GLOBAL calls in the gated arms had fewer than 2K keys and ran native dense
  (74,845 calls). `carried_first_calls` = 13,115 (M3) and 13,470 (B); bootstrap dense is canvas 0 only (875 calls).
- **Integrity.** 1,260/1,260 runs ok, one host per cell (dllm, mpk, dlm2 × 420), 3 runs with timed new graphs
  (excluded in Wc).
- **Source.** `aime_confirm_panel_e7/summary.md` and `receipts.md`, spec `specs/v27_aime_confirm_e7.json`,
  protocol `v27_aime_confirm_e7_0515fa3125588ff0`, deploy `v27_e7_ceefaf2`.

### L1e. E8 V-term ablation on AIME at a fixed 70% target sparsity: 30 problems × seeds 404–909 = 180 cells per arm (piecewise_v5)

All method arms use the M3 R6 DP selector with risk top-k keeping 30% of eligible prefix tiles; only the V term of
the risk changes. SparseD (port) at keep 0.3 is the external reference.

| arm (V term) | acc (dense 99) | vs dense (p) | vs rank 32 (p) | W [CI] | N [CI] | T | per-step S/N |
|---|---:|---|---|---|---|---:|---:|
| projected V, rank 32 (current) | 87 | +14/−26 (0.081) | – | 1.138 [1.078, 1.202] | 1.127 [1.065, 1.193] | 1.089 | 1.011 |
| projected V, rank 16 | 86 | +10/−23 (0.035) | +15/−16 (1.00) | 1.119 [1.059, 1.190] | 1.105 | 1.065 | 1.013 |
| projected V, rank 8 | 87 | +8/−20 (0.036) | +16/−16 (1.00) | 1.113 [1.029, 1.198] | 1.112 | 1.080 | 1.003 |
| projected V, rank 4 | 94 | +18/−23 (0.53) | +20/−13 (0.30) | 1.131 [1.058, 1.210] | 1.128 | 1.084 | 1.002 |
| M2 tile-mean V | 92 | +13/−20 (0.30) | +19/−14 (0.49) | 1.172 [1.071, 1.268] | 1.121 | 1.083 | 1.047 |
| no V (attention mass only) | 91 | +15/−23 (0.26) | +18/−14 (0.60) | 1.071 [1.013, 1.123] | 1.065 [1.019, 1.113] | 1.020 | 1.005 |
| SparseD 70% (port) | 94 | +13/−18 (0.47) | +26/−19 (0.37) | 1.020 [0.971, 1.069] | 1.031 | 1.022 | 0.991 |

- **No evidence that looking at V selects better at matched sparsity.** No V term differs significantly from
  rank 32 (all p ≥ 0.30). The projected-V arms at rank 32/16/8 are the lowest (86–87). Rank 16 and rank 8 lose to dense
  significantly (p ≈ 0.035); mass-only, M2 tile-mean V, rank 4 and SparseD do not.
- There is no monotone rank trend (rank 4 scores 94, ranks 8–32 score 86–87).
- At 70% sparsity every arm of our selector takes more steps and writes longer outputs (N 1.07–1.13, T 1.02–1.09), so
  W is 7–17% slower than dense. SparseD stays near dense (N 1.03).
- This replicates L8 (V-aware selection worse at high AIME sparsity) with 3× the cells. It does not reproduce the
  group-internal RULER8K finding that V helps (that selector uses current QK; ours uses historical QK).
- **Receipts.** Every run's `effective_method` shows the intended `risk_topk`, `proj_rank`, `risk_value` and `mu_mode`
  (M2: 54,105 compact pool builds). 1,440/1,440 runs ok, one host per cell, 3 runs with timed new graphs. Dense tokens
  are identical to E7 (180/180).
- **Source.** `aime_vterm_panel_e8/summary.md` and `receipts.md`, spec `specs/v27_aime_vterm_e8.json`, protocol
  `v27_aime_vterm_e8_bd293196ff730851`, deploy `v27_e8_021431b`.

### L1f. E9 V-term ablation at 64K under M3 + `carry_first`, fixed 88% sparsity: 24 items × seeds 404–707 = 96 cells per arm (piecewise_v5)

All method arms are M3 R6 DP + `carry_first`. At `risk_topk='k12'` (keep 12% of eligible prefix tiles) only the V term
changes; the −ln2 threshold arm (E5 best) is the operating-point reference.

| arm | W [CI] | S [CI] | per-step S/N | N [CI] | acc (dense 51) |
|---|---|---|---:|---|---:|
| projected V, rank 32, k12 | 0.880 [0.825, 0.932] | 0.818 [0.745, 0.892] | 0.781 | 1.047 [0.954, 1.141] | 53 |
| projected V, rank 8, k12 | 0.854 [0.791, 0.924] | 0.777 | 0.785 | 0.990 | 50 |
| M2 tile-mean V, k12 | 0.888 [0.828, 0.948] | 0.826 | 0.780 | 1.059 | 54 |
| **no V (attention mass only), k12** | **0.845 [0.789, 0.894]** | 0.772 [0.700, 0.841] | 0.784 | 0.986 | **54** |
| −ln2 threshold (projected V, rank 32) | 0.846 [0.781, 0.894] | 0.768 [0.686, 0.835] | 0.816 | 0.942 | 49 |

- **At 64K, looking at V does not select better than attention mass alone.** Accuracy is 50–54 vs dense 51 for every
  V term (all p ≥ 0.42 vs rank 32 and vs dense). Direct speed ratios between V terms are all n.s.: mass / rank 32
  0.960 [0.905, 1.018]; mass / −ln2 0.999 [0.961, 1.037].
- The per-step cost hardly depends on the V term (S/N 0.780–0.785 at k12), so the projected-V computation is cheap.
  The k12 arms are cheaper per step than −ln2 (0.78 vs 0.82); −ln2 therefore keeps more than 12% of the tiles here.
- Together with E8 (AIME), the V term is not a supported contribution: the selector can rank by attention mass with
  no loss. This moves our selector closer to score-only selectors such as SparseD.
- **Receipts.** Every run shows the intended `risk_topk`, `proj_rank`, `risk_value`, `mu_mode` and `carry_first`;
  carried-first calls 5,115–5,315 per arm; bootstrap dense 480 (canvas 0 only). 576/576 runs ok, one host per cell,
  no timed new graphs. Dense and the −ln2 arm are token-identical to E5 on these cells (96/96).
- **Source.** `lb64_vterm_panel_e9/summary.md` and `receipts.md`, spec `specs/v27_lb64_vterm_e9.json`, protocol
  `v27_lb64_vterm_e9_5b125c63817898f6`, deploy `v27_e9_efea330` (code `efea33024`).

### L1g. E10 M2 with the same optimizations as M3, at the operating point: LongBench-v2 32K/64K, 24 items × seeds 404–707 = 96 cells per arm per bin (piecewise_v5)

M2c (tile-mean projected V) on exactly the M3 R6 DP −ln2 + `carry_first` pipeline; only `mu_mode` differs.

| bin | arm | W [CI] | S | per-step S/N | N | acc (dense) |
|---|---|---|---:|---:|---:|---:|
| 32K | M3 + c0 | 0.909 [0.829, 0.978] | 0.880 | 0.926 | 0.95 | 56 (59) |
| 32K | M2c + c0 | 0.897 [0.826, 0.968] | 0.874 | 0.905 | 0.96 | 59 (59) |
| 64K | M3 + c0 | 0.845 [0.782, 0.894] | 0.768 | 0.816 | 0.94 | 49 (51) |
| 64K | M2c + c0 | 0.830 [0.762, 0.896] | 0.734 | 0.810 | 0.91 | 49 (51) |

- **M2 and M3 tie under the same optimizations.** Direct M2c / M3: 32K 0.987 [0.914, 1.063], 64K 0.983 [0.940, 1.024];
  accuracy M2c-only / M3-only 11/8 (p 0.65) and 6/6 (p 1.00). Per-step costs are equal.
- **Integrity.** 576/576 ok, one host per cell, no timed new graphs; dense and M3 + c0 token-identical to E5 (192/192);
  receipts show `mu_mode` pooled_compact vs exact, `carry_first` true for both (carried-first calls 11,720 / 11,995).
- **Source.** `lb_m2opt_panel_e10/summary.md`, `receipts.md`; spec `specs/v27_lb_m2opt_e10.json`, protocol
  `v27_lb_m2opt_e10_c965792c1cb6542d`, deploy `v27_e10_4252356`.

### L1h. E11 V-term preview at 95% sparsity: LongBench-v2 64K, 24 items × seeds 404/505 = 48 cells per arm (piecewise_v5; preview)

M3 R6 DP + `carry_first` at `risk_topk='k5'` (keep 5% of eligible prefix tiles; new named keep, code `9d8ae5e`).

| arm | W [CI] | S | per-step S/N | N [CI] | acc (dense 23) |
|---|---|---:|---:|---|---:|
| projected V, rank 32 | 0.887 [0.810, 0.960] | 0.807 | 0.764 | 1.06 [0.91, 1.21] | 24 |
| M2c tile mean | 0.905 [0.831, 0.985] | 0.839 | 0.761 | 1.10 | 26 |
| attention mass only | 0.891 [0.818, 0.964] | 0.833 | 0.760 | 1.10 [0.97, 1.24] | 26 |

- **Even at 95% sparsity there is no V advantage and no accuracy loss** (48 cells): mass / rank 32 W 1.004
  [0.916, 1.099], accuracy +5/−3 (p 0.73); every arm vs dense p ≥ 0.51.
- Per-step cost improves to 0.76 (0.78 at 88%), but steps rise slightly (n.s.), so request W ≈ 0.89, not better than
  the 88% operating point (0.85).
- **Integrity.** 192/192 ok, one host per cell, no timed new graphs; receipts show `k5`, `carry_first`, and the intended
  `mu_mode` / `risk_value`. k5 unit test passes on dlm2.
- **Source.** `lb64_vterm_hi_panel_e11/summary.md`, `receipts.md`; spec `specs/v27_lb64_vterm_hi_e11.json`, protocol
  `v27_lb64_vterm_hi_e11_18d29fa43d5d11a9`, deploy `v27_e11_9d8ae5e`.

### Time breakdown on piecewise_v5 (2026-10-01): one real dense decoder call per workload

`scripts/v27_time_breakdown.py` on the frozen E7 (AIME, request 0, call 55) and E5 (32K/64K, request 0, call 5)
configurations. Compiled-forward kernels (CUDA-graph replay) grouped by kernel; step = forward + sampler/stop test.

| part (ms, % of step) | AIME (1,979 keys) | 32K (35,135) | 64K (71,772) |
|---|---|---|---|
| step | 27.0 | 37.0 | 44.7 |
| GLOBAL attention (5 layers, FA4) | 0.5 (2%) | 7.2 (19%) | 14.5 (32%) |
| LOCAL attention (25 layers, cuDNN SDPA) | 0.8 (3%) | 0.8 (2%) | 0.8 (2%) |
| MoE expert GEMM | 10.5 (39%) | 12.9 (35%) | 12.2 (27%) |
| MoE routing sort/top-k | 1.3 (5%) | 1.3 (3%) | 1.3 (3%) |
| other GEMMs (nvjet) | 3.5 (13%) | 3.5 (9%) | 3.5 (8%) |
| rest of forward (norm, elementwise, copies) | 6.1 (22%) | 7.0 (19%) | 8.1 (18%) |
| sampler + stop test | 4.3 (16%) | 4.3 (12%) | 4.3 (10%) |
| decoder-forward median in one request: dense → best variant | 23.5 → 23.6 (1.00) | 31.1 → 26.4 (0.85) | 38.2 → 27.7 (0.72) |

Data: `time_breakdown_v5/*_dense_first.jsonl`. One request per workload; a diagnostic, not a panel.

### L1i. E12 HumanEval: 164 tasks × seeds 404–909 = 984 cells per arm (piecewise_v5; mpk + dlm2)

Thinking ON, budget 8192, pass@1 by the official tests in the unprivileged sandbox on mpk. Prompts 80–460 tokens.

| arm | pass@1 (dense 947) | vs dense | W [CI] | S | per-step S/N | N (total steps) | steps per block |
|---|---:|---|---|---:|---:|---:|---:|
| M3 + c0 (−ln2) | 957 | +31/−21 | 1.023 [1.007, 1.039] | 1.025 | 1.028 | 0.997 | 9.14 (dense 9.22) |
| M3 + c0 + 2K gate | 948 | +4/−3 | 1.009 [1.004, 1.015] | 1.011 | 1.007 | 1.003 | 9.32 |
| 70% fixed, projected V rank 32 | 956 | +33/−24 | 1.094 [1.068, 1.124] | 1.101 | 1.024 | 1.075 | 9.79 |
| 70% fixed, rank 8 | 963 | +31/−15 | 1.095 [1.068, 1.122] | 1.103 | 1.024 | 1.077 | 9.61 |
| 70% fixed, rank 4 | 957 | +32/−22 | 1.101 [1.075, 1.129] | 1.108 | 1.025 | 1.081 | 9.87 |
| 70% fixed, M2c tile mean | 948 | +26/−25 | 1.100 [1.074, 1.128] | 1.109 | 1.026 | 1.081 | 9.84 |
| 70% fixed, mass only | 954 | +31/−24 | 1.100 [1.074, 1.127] | 1.108 | 1.026 | 1.080 | 9.89 |

- **No accuracy loss in any arm**, even at a forced 70% sparsity. Rank 8 is +31/−15 against dense (p 0.026), which is
  not significant after correcting for seven comparisons. No V term differs from rank 32 (all p ≥ 0.29).
- **No speed gain**: HumanEval is short-context (dense 48.8 total steps per request, 5.3 blocks, 9.2 steps per block).
  The main configuration is 2.3% slower; the 2K gate sends 85% of GLOBAL calls to dense and is 0.9% slower; forcing 70%
  adds 6–8% steps per block and is 9–10% slower.
- Answers the classmate hypothesis for HumanEval: the selector's V term is not needed here (all V terms tie).
- **Integrity.** 7,872/7,872 ok, one host per cell, 2 timed new graphs; per-host ratios agree (main 1.024 mpk /
  1.022 dlm2). Source: `results/humaneval_ruler_v27_20261002/humaneval_e12/` (`summary.md`, `steps.md`, `receipts.md`);
  protocol `v27_humaneval_e12_d844fb61c76dde6a`, deploy `v27_e12_5d5c3b9`.

### L1j. E13 q64 (64-row FA4 keep maps), 32K/64K/96K × 6 seeds (piecewise_v5)

| bin | arm | W [CI] | S | per-step S/N | N | acc (dense) |
|---|---|---|---:|---:|---:|---:|
| 32K | M3 + c0 | 0.908 [0.857, 0.956] | 0.877 | 0.925 | 0.949 | 92 (86) |
| 32K | M3 + c0 + q64 | 0.904 [0.845, 0.963] | 0.869 | 0.923 | 0.941 | 87 (86) |
| 64K | M3 + c0 | 0.854 [0.803, 0.896] | 0.776 | 0.815 | 0.951 | 76 (75) |
| 64K | M3 + c0 + q64 | 0.864 [0.801, 0.924] | 0.785 | 0.810 | 0.969 | 77 (75) |
| 96K | M3 + c0 | 0.849 [0.733, 0.969] | 0.771 | 0.746 | 1.034 | 31 (27) |
| 96K | M3 + c0 + q64 | 0.870 [0.794, 0.938] | 0.779 | 0.740 | 1.052 | 29 (27) |

- **q64 is non-negative but small.** Direct q64 / M3 + c0: per-step 0.998 (32K), **0.993 [0.990, 0.996] (64K, consistent on
  all three hosts)**, 0.993 (96K); request W 0.995 / 1.011 / 1.025, all n.s. (step-count noise). Accuracy unchanged.
- The offline estimate (22% less kept prefix work → about 1.5% per step at 64K) overstated the gain: carried call-0 maps
  stay at 128 rows (23% of list builds) and the refinement itself costs time.
- Regrouping within 64-row groups would add only ~5% of the remaining kept work (about 0.15% per step); not pursued.
- Source: `lb_q64_panel_e13/` (`summary.md`, `steps.md`, `receipts.md`); protocol `v27_lb_q64_e13_f038777e0ab0d142`.

### L1k. E14: q64 and q64c with six fresh seeds; pooled 12-seed main result (2026-10-02)

| comparison | 32K | 64K | 96K |
|---|---|---|---|
| **M3 + c0 / dense, W (12 seeds)** | **0.925 [0.892, 0.958]** | **0.860 [0.826, 0.891]** | **0.801 [0.705, 0.902]** |
| M3 + c0 / dense, generation-only S | 0.903 | 0.777 | 0.708 |
| M3 + c0 / dense, per step | 0.924 | 0.814 | 0.751 |
| M3 + c0 / dense, accuracy | 177 / 179 (p 0.91) | 150 / 155 (p 0.53) | 62 / 53 (p 0.15) |
| q64 / M3 + c0, per step (12 seeds) | 0.996 [0.993, 1.000] | **0.993 [0.991, 0.996]** | 0.995 [0.988, 1.003] |
| q64 / M3 + c0, W (12 seeds) | 1.002 | 1.004 | 1.012 |
| q64c / q64, per step (E14) | 1.002 | 0.998 [0.994, 1.002] | 0.989 [0.972, 1.003] |
| q64c / M3 + c0, N (E14) | 1.014 | 1.049 [0.989, 1.115] | 1.146 [1.004, 1.323] |

**Main result, robust over 12 seeds.** M3 + c0 is 7.5% / 14% / 20% faster end to end than the FA4 dense control at
32K / 64K / 96K, with no significant accuracy difference.

**q64.**
- It cuts per-step cost by 0.4–0.7%; this is significant at 64K and consistent on every host.
- It is invisible end to end, where step-count noise is about ±5%.
- Accuracy is unchanged.

**q64c.**
- The 64-row carried call-0 map adds no measurable per-step gain over q64 at 32K or 64K.
- It shows more steps at 96K (N 1.146 vs M3 + c0, CI just above 1; 66 vs 37 blocks at the step cap). This may be a
  real effect of the more aggressive carried call-0 map, or noise.
- It is not adopted.

Source: `lb_q64c_panel_e14/` (`summary.md`, `steps.md`, `receipts.md`, `direct_*.md`).

### L1l. R16: RULER 32K/64K accuracy check of the current pipeline (2026-10-02)

All 13 RULER tasks at 32K/64K × seeds 404/505/606, dllm.
- Correct answers per arm: dense 34 / 33; M3 + c0 34 / 33; + q64c 34 / 33.
- Fixed 88% (k12): rank-32 V 33 / 33, mass-only 33 / 33.
- No accuracy loss in any arm, and no V-term difference on long-context RULER.
- Not a speed target (about 5 calls per request).
- Source: `ruler_long_panel_r16/` (`summary.md`, `vterm.md`, `steps.md`, `receipts.md`).

### L1m. E15: observe_carried (c01) and the 18-seed main result (2026-10-02)

- **Main, M3 + c0 / dense, pooled E13 + E14 + E15 (18 seeds):**
  - W 0.950 [0.920, 0.980] (32K), 0.867 [0.841, 0.892] (64K), 0.818 [0.731, 0.898] (96K);
  - generation S 0.937 / 0.787 / 0.728; per step 0.921 / 0.814 / 0.745;
  - accuracy 267/274, 228/238, 93/76 (p 0.55, 0.24, 0.012 in favour of sparse).
- **AIME (E15, 180 cells):** W 0.951 [0.894, 1.001], from 7% fewer steps (per step 1.024); accuracy 100 vs 94.
- **c01 / main:**
  - per step 0.983 on AIME (significant), 1.004 (32K), 0.995 (64K), 1.002 (96K);
  - accuracy at 64K 70 vs 78 (p 0.096), and vs dense 70 vs 83 (p about 0.007).
  - **Not adopted.**
- Source: `c01_panel_e15/` (`receipts.md`, `summary.md`, `steps.md`, `direct_*.md`).

### L1n. P16: held-out check of the 64K accuracy gap (2026-10-02)

- Design: 6 LongBench-v2 64K items frozen from E13–E15 (the 3 most negative and 3 most positive main-minus-dense items)
  × 12 never-used seeds = 72 cells per arm.
- **Overall:** main 45 vs dense 44, W 0.857, per step 0.808.
- **The 3 worst items:** −20 pp in discovery (17 vs 28 of 54) → −8 pp on fresh seeds (14 vs 17 of 36; +5/−8, p 0.58).
  The gap is mostly selection noise; there is no evidence of a systematic 64K loss.
- Output protection is 38 / 72 (+8/−14) with N 1.035, so it is not adopted. −2ln2 gives 44 / 72 at per step 0.852,
  also not adopted.
- Source: `lb64_item_check_p16/receipts.md`.

### L1o. P17: C-gate preview (group member's query sensitivity; collaboration; 2026-10-02)

- Data: AIME 10, 64K 8, 96K 4 items × 2 seeds (44 cells per arm). **Preview.**
- **C gate at −ln2 vs main T, per step:** 1.071 [1.063, 1.079] (64K) and 1.103 [1.078, 1.125] (96K).
- **C gate at the base threshold vs main:** 1.027 at 64K, 1.017 at 96K.
- **Accuracy** (of 44): dense 27, main 30, C −ln2 27, C base 29; nothing significant.
- **Verdict:** not adopted. The C gate protects most of the canvas, so it keeps more tiles, and it shows no accuracy
  gain.
- Source: `cgate_preview_p17/receipts.md`.

### L1p. R17: RULER 32K / 64K / 92K with V-dimension variants (2026-10-02)

- Design: 13 tasks × 10 samples = 130 rows per length, seed 404, 8 arms, 3,120 runs; accuracy only.
  - 96K became 92K because 96K RULER rows (97.5K–98.2K tokens) exceed the one-H100 fit of our substrate.
- **Main vs dense:** 110 / 110, 98 / 97, 91 / 91; no difference anywhere.
- **Fixed sparsity:** losses only at 32K, mostly on `cwe`:
  - 95% at rank 32: 101 (p 0.004);
  - 88% at rank 8 / rank 4: 102 / 103 (p 0.021 / 0.039);
  - 88% at rank 32: 105 (p 0.13).

  At 64K / 92K every fixed arm equals dense.
- **V term:**
  - mass-only ≥ rank 32 everywhere;
  - at 95% sparsity, 107 vs 101 at 32K (p 0.031, mass-only better);
  - lower ranks are no better.

  The V term is now negative on AIME, LongBench 64K (88% / 95%), HumanEval and RULER 32K–92K.
- Source: `ruler_long_panel_r17/receipts.md`.

### Official-serving check and vLLM port probe (2026-10-02)

- **vLLM 0.30.0 native DiffusionGemma vs our dense control** on the same 18 E14 cells and host, batch 1, bf16:

  | | 32K | 64K | 96K |
  |---|---|---|---|
  | per step, vLLM / ours | 29.1 / 38.5 ms (0.76×) | 40.5 / 45.0 ms (0.90×) | 48.0 / 49.2 ms (0.98×) |
  | prefill, vLLM / ours | 0.89 / 2.03 s (0.44×) | 2.48 / 5.05 s (0.49×) | 4.05 / 7.20 s (0.57×) |

  Source: `vllm_dense_check_1002/README.md`.
- **FA4 block-sparse over vLLM's paged KV is not correct in this build.** The error is about 22 vs the reference;
  dense paged and contiguous sparse are exact. Source: `vllm_port_probe_1002/README.md`.

### Regroup offline and call-1 split (2026-10-02)

**Offline regrouping** (`regroup_offline_1002/`): real need matrices of 144 decisions (M3 + c0 + q64c, 32K/64K/96K
items 0–3). Kernel work relative to natural 64-row tiles (q64 = 1):

| grouping | 32K | 64K | 96K |
|---|---:|---:|---:|
| q128 (executed M3 map) | | 1.32 | 1.34 |
| q64r: sort by need count within the 128-row block | | 0.963 | 0.952 |
| sort by count over the head's 256 rows | 0.987 | 0.983 | 0.950 |
| chw/value_aware set key (lexicographic) | 1.065 | 1.055 | 1.035 |
| greedy clustering per head | 0.994 | 1.014 | 1.010 |
| cross-head (GQA) 8 positions × 8 heads | 1.052 | 1.065 | 1.060 |
| cross-head sort / set key / greedy | 1.04–1.10 | 1.05–1.09 | 1.04–1.08 |
| per-row bound | 0.25 | 0.22 | 0.21 |

Natural 64-row tiles are already the best structure tried. Positional neighbours share needs; heads and set-key
neighbours do not. No grouping removes more than about 5% of q64's tiles, which is less than the permutation costs,
so the regrouping line is closed. The router check matched in 48/48 decisions per bin.

**Call-1 split** (`observe_split_1002/obs_split.json`), per GLOBAL layer:

| keys | fused observation (current) | observation-only + FA4 sparse (c01) | FA4 dense |
|---|---:|---:|---:|
| 33K | 2.22 | 1.76 (0.80×) | 1.34 |
| 65K | 4.22 | 3.15 (0.75×) | 2.59 |
| 97K | 6.28 | 4.59 (0.73×) | 3.86 |

- With 14–18 steps per canvas at 32K/64K, c01's expected per-step gain is about 0.5–1%. This corrects the earlier
  3–4% estimate, which assumed about 9 steps per canvas.
- Our Triton observation-only kernel (QK plus summaries) is slower than FA4 full dense attention. A faster observation
  kernel is the remaining lever on this call.

### q64 / q64r / split-KV kernel bench on identical real states (2026-10-02)

Kernel level only (no end-to-end claim): 32K/64K/96K items 0–3, seed 404, 240 decisions per bin, each timed under
every map on the same Q/K/V. Median ratio per sparse GLOBAL call:

| bin | q64 / 128 | q64r / 128 (incl. permutation) | num_splits 2 / 1 | kept prefix 128 → q64 → q64r |
|---|---:|---:|---:|---|
| 32K | 0.914 | 1.021 | 1.05 | 21.8% → 17.3% → 16.4% |
| 64K | 0.887 | 1.006 | 1.04 | 11.9% → 9.0% → 8.6% |
| 96K | 0.899 | 1.014 | 1.04 | 7.9% → 5.9% → 5.4% |

- q64 is the useful part of the regrouping lead (−9 to −11% per sparse call).
- Within-block row regrouping (q64r) and FA4 split-KV are slower and are dropped.
- Next: E14 adds the 64-row carried call-0 map (q64c).
- Source: `q64_bench_1002/README.md`, `bench.jsonl`.

### Map drift between re-decisions (2026-10-02)

E5 configuration, 64K/32K items 0–3: every re-decision (calls 8, 14, …) changes the held map; relative to the first
decision of the canvas, 44–71% of its kept tiles change (kept-set Jaccard 0.59–0.70). The cause is the per-position
query sensitivity T = clamp(1 + 3·EMA(argmax flips), 1, 4), active in every run; the risk table, reference scale and
threshold do not change. Details: `map_drift_1002/README.md`.

### Query-row regrouping diagnostic (2026-10-02; idea from chw/value_aware)

64K/32K, 4 requests each, E5 configuration. Regrouping the 256 canvas rows within 128-row FA4 query tiles keeps
4.2% / 4.4% fewer prefix tiles (64K 14.0% → 13.4%, 32K 22.8% → 21.8%); at 64- and 32-row tiles regrouping again adds
only 3–5%. Finer tiles alone cut kept work by 22% (64 rows) and 39% (32 rows); the per-row bound is 2.1% / 4.2% kept.
Not integrated (end-to-end effect well below 1%). Details: `regroup_diag_1002/README.md`.

### Absolute step counts (2026-10-02): total steps per request and steps per block

`scripts/v27_step_stats.py` over successful first outputs. A block is one 256-token canvas; steps are decoder calls;
the cap is 48 steps per block. Ratios are paired within cells (geometric mean, item-clustered 95% CI). Every panel
directory now has `steps.md` / `steps.csv` with all arms and per-host means.

| workload | arm | requests | total steps mean / median | blocks per request | steps per block mean / median | blocks at 48 cap | N ratio [CI] | steps/block ratio [CI] |
|---|---|---|---|---|---|---|---|---|
| 32K (E5) | dense | 144 | 223.75 / 192.5 | 15.88 | 14.09 / 13.0 | 23/2286 | 1.0 [1.000,1.000] | 1.0 [1.000,1.000] |
| 32K (E5) | M3 + c0 | 144 | 214.78 / 186.0 | 15.08 | 14.24 / 12.0 | 18/2172 | 0.9485 [0.880,1.011] | 1.0042 [0.977,1.031] |
| 32K (E5) | M3 | 144 | 219.42 / 201.0 | 15.75 | 13.93 / 12.0 | 9/2268 | 0.9865 [0.927,1.045] | 1.0096 [0.981,1.036] |
| 64K (E5) | dense | 144 | 218.83 / 162.0 | 12.21 | 17.92 / 17.0 | 9/1758 | 1.0 [1.000,1.000] | 1.0 [1.000,1.000] |
| 64K (E5) | M3 + c0 | 144 | 205.82 / 156.5 | 11.88 | 17.33 / 16.0 | 3/1710 | 0.9514 [0.875,1.027] | 0.9785 [0.952,1.003] |
| 64K (E5) | M3 | 144 | 213.24 / 166.0 | 12.1 | 17.63 / 17.0 | 3/1742 | 0.975 [0.883,1.054] | 0.9822 [0.948,1.015] |
| 96K (E6+E6b) | dense | 66 | 289.24 / 192.5 | 12.38 | 23.37 / 21 | 67/817 | 1.0 [1.000,1.000] | 1.0 [1.000,1.000] |
| 96K (E6+E6b) | M3 + c0 | 66 | 241.53 / 200.5 | 10.98 | 21.99 / 20 | 23/725 | 0.9648 [0.770,1.200] | 1.0071 [0.936,1.078] |
| 96K (E6+E6b) | M3 | 66 | 267.56 / 220.5 | 11.68 | 22.9 / 21 | 39/771 | 1.0282 [0.852,1.237] | 1.0267 [0.961,1.105] |
| AIME (E7) | dense | 180 | 295.91 / 283.0 | 22.57 | 13.11 / 12.0 | 1/4062 | 1.0 [1.000,1.000] | 1.0 [1.000,1.000] |
| AIME (E7) | M3 + c0 + 2K gate | 180 | 295.44 / 282.5 | 22.47 | 13.15 / 12.0 | 3/4044 | 1.0003 [0.968,1.031] | 1.0057 [0.992,1.021] |
| AIME (E7) | M3 | 180 | 303.83 / 289.0 | 22.86 | 13.29 / 13.0 | 4/4114 | 1.022 [0.973,1.074] | 1.0075 [0.987,1.029] |

- Steps per block are about 13 (AIME), 14 (32K), 18 (64K) and 23 (96K) for dense; longer contexts take more steps
  per block. The method changes steps per block by at most about 2–3% in either direction (n.s.).
- Total steps vary with output length (blocks per request); 96K total-step means are pulled up by a few long requests.

### Step-count check (2026-10-01): is "no step inflation" itself noise?

Step ratio N = method decoder calls / dense decoder calls per cell (M3 R6 DP −ln2). Single-cell log SD ≈ 0.45 (×1.57).

| seed | 101 | 202 | 303 | 404 | 505 | 606 | 707 | 808 | 909 | pooled 9 seeds [CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| 32K N (24 items, v5; 101–303 from E3, 404–909 from E4) | 1.23 | 0.89 | 1.24 | 0.85 | 0.98 | 1.12 | 0.99 | 1.00 | 1.00 | 1.024 [0.973, 1.076] |

- The 3-seed set 101/202/303 sits high (1.10) and the 6 new seeds sit low (0.99). 13 of the 84 possible 3/6 splits
  give a gap at least as large, so this is ordinary seed-to-seed spread, not a difference between the seed sets.
- 64K pooled 9 seeds (traj_t1 on v4 with seeds 101–303, E4 on v5): N 1.015 [0.948, 1.085].
- E4 64K held-out items (12 × 6 seeds): N 0.955 [0.799, 1.099]; per-item N ranges 0.41–1.32. Its 3-seed subsets range
  0.87–1.05, so the older held-out 1.24 (v3 substrate, seeds 101–303) is at or beyond the edge of this spread.
- **Reading.** No significant step change; best estimate +1–2%. Request-level W from 6 seeds may be about 2%
  optimistic. Per-step S/N is stable across all batches.

## Panels on piecewise_v3 / v4 / v5 (seeds 101/202/303 unless stated)

| id | comparison | workload | result | caveat | source / protocol |
|---|---|---|---|---|---|
| L2 | final configuration, 8 arms | LB 64K / 32K / pool, 12 formal items × 3 seeds (36 cells) | 64K: M3 W 0.869 [0.805, 0.933], S 0.783, acc 23/18; B 0.864 [0.825, 0.905]. 32K: M3 0.902 [0.780, 1.015] (n.s.). Pool: ungated sparse 1.04–1.08; 32K gate = dense exactly. Fan plain 1.12–1.49× at every length | formal items chose −ln2 (selection bias); piecewise_v3 | `final_panel_f1/README.md`, `v27_final_lb_f1_79c81baf9ac23611` |
| L3 | final AIME, 8 arms | AIME26 30 × 4 seeds (120 cells) | M3 acc 65/65 (+9/−9), **Wc 1.008 [0.957, 1.063]**; B 58/65 (p = 0.27); Fan M1 Wc 1.058 [1.016, 1.104] (slower); gated arms = dense | use Wc. The README's W column (0.971) includes timed graph-capture pairs | `final_panel_f1/summary_aime_sensitivity.md`, `v27_final_aime_f1_6d5d56ee2ab4dc8d` |
| L4 | threshold sweep (M3 R6 DP, B, M1 DP) | LB 32K/64K formal, 36 cells | 64K: −ln2 0.868 [0.807, 0.932] best; +ln2 1.074 with NC 1.395; default 0.970 with NC 1.10 | piecewise_v2; same items as L2 | `threshold_sweep_s1/README.md`, `v27_long_lb_sweep_s1_322b24218397083a` |
| L5 | SparseD port vs ours | LB 32K/64K formal, 36 cells | 64K: M3 0.865 [0.804, 0.929], B 0.860, SparseD k10 0.905 [0.816, 0.997], k30 s10 1.005. 32K: all n.s. | port, not official code; first-step-dense variants | `sparsed_panel_c1/summary.md`, `v27_long_lb_sparsed_c1_72c5be17dd3fa43d` |
| L6 | 64K held-out (12 untouched items) | 36 cells | M3 W 1.026 [0.955, 1.110], S/N 0.833; B 0.986; N 1.24 for M3 | 3 seeds; superseded for the request verdict by L1; the per-step saving replicates | `holdout_panel_h1/summary.md`, `v27_lb64_holdout_h1_408ddcd9054f0148` |
| L7 | AIME fixed target sparsity 30/40/50% | AIME 30 × 3 seeds (90 cells) | no accuracy loss for our top-k or SparseD (47–49 vs 49); no speedup (W 0.99–1.09) | piecewise_v3 | `aime_sparsity_panel_t1/table.md`, `v27_aime_sparsity_t1_1e2ab3d77f6fb86e` |
| L8 | AIME 70/80% target | AIME 30 × 2 seeds (60 cells) | ours at 66% realized: 23 vs 34 correct (significant loss); SparseD 65%: 29 vs 34 | negative for V-aware selection | `aime_sparsity_panel_hi/table.md`, `v27_aime_sparsity_hi_8ab53dd5e5d30f6c` |
| L9 | M2c in the M3 R6 DP −ln2 configuration | LB 32K/64K formal, 36 cells (dlm2) | 64K M2c-config 0.847 [0.744, 0.937] vs M3 0.821; Fan M2c plain 1.44 | one host; different tokens from L2 (cross-host) | `m2_panel_c1/summary.md`, `v27_long_lb_m2_c1_24189fb947b680cf` |
| L10 | 96K (prompts ≤ about 95K) | 6 items × 2 seeds (12 cells) | M3 Wc 0.948, S/N 0.818; B Wc 1.042 | exploratory, 12 cells; 128K OOMs for every arm | `lb96k_panel_c2/summary.md`, `v27_long_lb96k_c2_ff3700adb8873eea` |
| L11 | 64K traj study (layer variants, −2ln2) | 24 items × 3 seeds (72 cells), piecewise_v4 | M3 0.952 [0.891, 1.016]; no_first 1.001; late3 1.022; −2ln2 0.964; per-canvas decode S/C: M3 0.866 [0.829, 0.899], B 0.861 | 3 seeds: N noisy (see L1) | `traj_panel_t1/summary.md`, `v27_lb64_traj_t1_af1f05697db9b9e3` |
| L12 | AIME carry/gates (E2) | AIME 30 × 2 seeds (60 cells), v5 | M3 S/N 1.020, W 1.017; carry4 S/N 0.976 but N 1.135; SparseD 50% acc 28 vs 33 | — | `aime_carry_panel_e2/summary.md`, `v27_aime_carry_e2_5602b0d5cc7f74f7` |
| L13 | 32K cross-canvas carry (E1) | 24 items × 3 seeds (72 cells), v5 | carry K=4: S/N 0.892 but NC 1.156, acc 40/47; B carry4 acc 34/47 | rejected | `lb32_carry_panel_e1/summary.md`, `v27_lb32_carry_e1_3a79c6c515285a31` |
| L14 | 32K single-change variants (E3) | 72 cells, v5 | per-canvas decode S/C: −ln2 1.018, obs2 0.968, −2ln2 0.969, R3 0.980, threshold 0 0.962 (acc 42/47), po 0.982 | 3 seeds; see L1 for request level | `lb32_steps_panel_e3/summary.md`, `v27_lb32_steps_e3_5d27443669d0888a` |

## Benchmarks and diagnostics (not panels; on the stated substrate)

| id | what | result | source |
|---|---|---|---|
| K1 | dense kernels at the GLOBAL decode geometry (256 q, 16/2 heads, hd512) | 64K: FA4 2.85 ms, FA4 all-kept 2.67 ms (bitwise equal), FlashInfer FA2 2.89, HF default path 6.16, D_c64 4.9–5.5 | `official_baseline/README.md` (`dense_sota_same_process.json`) |
| K2 | FA4 block-sparse scaling | 64K keep 50/25/10%: 0.45/0.27/0.14× dense | `official_baseline/README.md` |
| K3 | kernel → module → request at 64K (B / M3) | kernel 7.3× (keep about 10%), module 4.4× / 2.7×, request 1.16× / 1.15×; Fan plain module 0.31–0.66× | `kernel_bench/README.md` |
| K4 | per-forward and per-step time composition at 32K (dense, compiled) | forward about 32 ms: MoE GEMM 12.8 (HBM-bound), GLOBAL attention 7.2, dense GEMMs 3.5, LOCAL 0.8, other about 4; step adds about 4.6 ms sampler | `substrate/time_breakdown.json` (v2-era; KV concat since removed in v3) |
| K6 | batch scaling of one 64K decoder forward (B = 1/2/4) | prefix-attention share 26/22/25%; keep-0.12 saving 23/19/20%; B=8 OOM | `batch_scaling/README.md` |
| K5 | substrate correctness | eager-backend split bitwise equals eager; inductor numerics ≈ an exact-kernel swap | `substrate/README.md` §6 |

## Eager-substrate results (historical; do not compare with the above)

- v20 full-panel E2E vs HF native (M1/R2/R3 slower or no forward gain on AIME/RULER):
  `results/fan_m1_m3_multidataset_20260927/fan_today.md`.
- v23 bootstrap6, v24 direct cost, v25 aligned16: `results/m3_numeric_trajectory_bridge_20260927/v23_bootstrap6/`.
- v26 seven-arm panel: `results/m1_m2_m3_frontier_v26_20260928/seven_panel_report.md`.
- v27 Tier 3, long RULER, LB-long against `D_c64`:
  - `tier3_report.md`;
  - `long_ruler_panel/README.md` (shared support collapses quality at 64K);
  - `long_lb_panel/README.md`.
- FA4 panels on piecewise_v1 (`fa4_panel_v3/README.md`, protocol `v27_long_lb_fa4pw_v3_b5399ab5c46cd59e`) are
  graph-substrate results, but the substrate predates the v2/v3 fixes.

## Intake verification (2026-10-01, 22:45 UTC−5; CPU only)

- **Scope:** E4/E5/E6/E6b/E7/E8/E9/E10/E11, 1,338 question-seed cells across panels and 8,826 first executions.
  Same-host/GPU, substrate/protocol/model/source and scorer/run identity checks pass. No duplicate first outputs
  or unknown graph counters. All existing W/S/P/N/NC/T/SN/ST/Wc point estimates and correct counts reproduce.
  The historical CI estimator is unchanged; published CSVs and historical ledgers were not rewritten.
- **Known timing exclusions:** E7 and E8 each have three timed new-graph runs, already documented above; use Wc.
- **Dataset inventory, not model performance:** all 503 LongBench-v2 inputs were previously rendered with the pinned
  tokenizer/template. Min/median/max 10,334 / 107,706 / 5,174,028 tokens; 224 inputs ≤95,074. With an 8,192-token
  output reservation, 400 fit the model's configured 262,144-token positional range. These counts do not establish
  GPU-fit coverage or successful generation. No new accuracy or speed result is claimed.
- **Source:** `intake_audit_20261001/summary.json` and `README.md`; explanation `docs/INTAKE_AUDIT_20261001.md`.
  Source inputs are the existing private scored CSVs/closed ledgers and the prior full-dataset rendering log.
  GPU seconds for this review = **0**.

## Group-context verification (2026-10-01, 23:10 UTC−5; no new benchmark)

- User-authorized read-only search of the specified research group completed. Peer RULER/V-rank and step-inflation
  messages were read with relevant thread replies; HumanEval search and merge search produced no matches there.
  This does not establish that no peer HumanEval run or integration exists elsewhere.
- Source/background: `docs/GROUP_CONTEXT_20261001.md`. HERALD/PRR/SSV experimental boundaries and BRISK-DLM's
  abstract were checked against public primary sources. No peer figures enter this branch's W/S/N/accuracy tables.
- No new model output, protocol, run, score or GPU job. GPU seconds = **0**; A/B refs remain pending.

## V18 qualification failure (2026-10-02)

Attempt 001 has zero qualified timed results: phase observer used the external rather than
internal vLLM request ID. Dense longest-item warm-up ran; 241.607 reserved GPU seconds are
charged. This is an instrumentation failure, not a model accuracy or performance result.
Source: `vllm_v18_qualification/attempt001.json`.

V18 attempt 002: dense qualification passed, native failed independent clock/coverage checks.
No paired performance result. 355.475 reserved GPU seconds, source `vllm_v18_qualification/attempt002.json`.

V18 clock diagnostic: native executed 350 denoising forwards while the scheduler had retired
349; exactly 1750 GLOBAL calls and zero order errors. The extra forward is official async
queue work and must count in N. No speed claim. 158.557 GPU seconds, source
`vllm_v18_qualification/clock_diagnostic001.json`; total closed V18 work 755.639 s.

## V18 qualification 003 (2026-10-02 17:31 UTC-5)

Source: `results/m1_m2_m3_frontier_v27_20260929/vllm_v18_qualification/attempt003.json`.
Longest-input dense/native/all-kept pass actual async execution counting and zero timed
compile/capture. Main stops before generation because the unchanged frozen source hashes
are keyed by the prior deploy paths. No accuracy or performance conclusion. Reserved GPU
time for this closed attempt is 614.449 s; cumulative V18 closed work is 1370.088 s.

## V18 qualification 004 (2026-10-02 17:40 UTC-5)

Source: `results/m1_m2_m3_frontier_v27_20260929/vllm_v18_qualification/attempt004.json`.
Main passed source/config validation but OOM during longest 96K warmup at KV reservation
0.92. No timed main record. This is a feasibility negative at that reservation, not an
accuracy or timing result. Attempt reserved 144.065 GPU seconds; cumulative closed V18
work is 1514.153 s. V18b is a new protocol with common reservation 0.85 for every arm.

## V18b qualification 005 (2026-10-02 17:54 UTC-5)

Source: `results/m1_m2_m3_frontier_v27_20260929/vllm_v18_qualification/attempt005.json`.
Common KV reservation 0.85 passes the longest 96K input in all four arms, including main's
frozen effective method, actual async N and zero timed compile/capture. All four final
answers parse under the unchanged NeMo scorer; all four are wrong on this single item.
This validates the execution/scoring chain only, not accuracy or performance.
Reserved GPU time is 640.802 s; cumulative closed V18 work is 2154.955 s.
Formal V18b panel_001 is launched from the same deploy/binding; no formal result yet.


## Variant/fairness audit (2026-10-02 18:19 (UTC-5))

Historical counts and times are preserved. E13–E15 pooled main/dense correct counts are
267/274, 228/238, 93/76; old cell McNemar 96K p≈0.012 is exploratory, while the
11-question exact sign-flip p=0.125. Question-cluster intervals allow 32K/64K declines;
noninferiority is unproven. HF W excluded setup/cleanup; HF new_graphs counted Dynamo
compilations, not CUDA captures. Existing mass-only changes prefix ranking only.
R17 92K main discordance is corrected here to +0/-0, preserving 91/91 correct.
Source: `docs/VARIANT_FAIRNESS_AUDIT_20261002.md`; reproducible historical aggregate
audit: `scripts/v27_historical_accuracy_audit.py`. Audit GPU seconds: 0.
No new formal V18b result is claimed; the frozen campaign continues.


## Peer source update (2026-10-02 18:33 (UTC-5))

The user-supplied 10-page PDF resolves the missing C-gate source. HF state formulas
match Eq. (1)-(7), but P17 historical-QK/DP/R6/carry-first is not the full current-QK
retained-state method. Three existing C-gate CPU tests pass; GPU seconds = 0.
Peer high-sparsity RULER V gains are external reported evidence, not independently
reproduced branch results or a no-loss speed claim. Table 1/11 dense provenance needs
clarification. No new method or V18b configuration change. Source and interpretation:
`docs/PEER_PDF_UPDATE_20261002.md`. The private PDF is not uploaded.


## V28 campaign preparation (2026-10-02 18:54 (UTC-5))

User authorized worthwhile variants. Independent branch `research/vllm-variants-20261002`
starts at `9b027df8f`; parent V18b remains frozen. See `docs/VARIANT_CAMPAIGN_V28_20261002.md`.
This entry is specification/status only, not a result. New GPU seconds: 0.

## V28 CPU qualification — 2026-10-02 19:08 (UTC-5)

31 CPU tests passed (lifecycle ownership/exception cleanup, unsupported-variant guards,
HK=2 flat-view counterexample, Q64 coverage and regroup invariants). Source:
`tests/test_v28_adapter_lifecycle.py`, `tests/test_v27_vllm_adapter.py`,
`tests/test_v28_alias2_q64_bench.py`, `tests/test_v28_regroup_screen.py`.
KV-view rejected on layout grounds; request_clear remains an unmeasured standard
optimization. No new V28 GPU performance or accuracy result. See V28 campaign spec.

## V28 regroup screen and native qualification freeze — 2026-10-02 19:18 (UTC-5)

Real historical prefix-need snapshots: 144; natural Q64 work 794361 tiles.
Default bounded swap search accepted 7/144; gated load proxy ratio 0.9971768.
Expanded search (12 swaps/head, 16 candidate signatures/group) accepted 19/144;
ratio 0.9893759, total work +0.0402%. Aggregate max/p95 loads unchanged.
These are CPU prefix-only proxies, not GPU acceleration; no standalone regroup
request panel warranted. Source: `results/v28_20261002/regroup_screen/`.

Native q128/q64 request_clear qualification specs are frozen under
`results/v28_20261002/specs/`. Both use identical main parameters and source,
common 0.85 reservation, five GLOBAL layers, alias2 and lifecycle cleanup.
Each future preview uses first two E14 items per length bin x two repeats;
qualification first uses the longest selected item, one warm plus one timed.
Native dense/native-hook/all-kept controls retained. Wrapper records actual N,
untimed actual KV copy checks, effective q64 counters and allocator retry deltas.
30 CPU wrapper/lifecycle/adapter tests passed; 13 regroup and 3 component tests passed.

Component attempt 001 completed in 30.7184 reserved GPU seconds, but remains
diagnostic only: exact-length rather than nominal-bin sampling and non-native
Q strides were found during review. New attempt corrects these and strengthens
finite checks, disables TF32 for FP32 reference, and explicitly excludes first
list/split builds. Old run is preserved; no headline speed is taken from it.
LLaDA first upstream smoke loaded weights but failed RoPE JIT missing CCCL header;
isolated toolchain repair ongoing. Model sparse ports remain unimplemented.

## V28 native Q64 component result and request qualification — 2026-10-02 19:26 (UTC-5)

Corrected component002 passed finite IEEE-FP32 masked references on six historical
need snapshots with synthetic QKV, actual native tensor strides and random pages.
Q64/Q128 alias2 GPU-time geometric mean = 0.94575 (about 5.4% component reduction).
Selector, KV copies, first lists/splits and other model costs excluded; no request
or accuracy claim. Source: `results/v28_20261002/q64_component/`. Worker time
18.6512 GPU s; prior diagnostic001 30.7184 s. Dense component key corrected to
fixed-length diagnostic: it is not the official varlen serving baseline.

Native request qualification002 runs on mpk, frozen deploy e7b5ff061, one worker
at a time. Q64 method passed: actual N=66 (65 retired+1 unused), 330 GLOBAL calls,
60 q64-refined routes/list builds, five GLOBAL warm copy checks passed; no timed
CUDA captures/backend/inductor compiles or allocator retries/OOMs. Warm long
requests did show recoverable allocation retries, so request_clear does not
eliminate canvas allocation peaks. These one-request checks are not performance
or accuracy evidence. Main128, allkept, native-hook and dense follow serially.
Environment used OMP_NUM_THREADS=4; future performance preview should freeze 1
for all arms, following official serving warning. Do not compare qualifications
as a strongest-baseline speed panel. Further Triton/CuTe compilation coverage is
being audited independently from existing CUDA-graph/backend/inductor counters.

Held-regroup optimistic ablation now implemented with 8 CPU tests: natural Q64
versus token-major per-head gather, alias2 and scatter; at most one CPU-accepted
state per nominal bin. Search/first builds are excluded deliberately to ask whether
amortization could pay even in this favorable setting. No GPU result yet.
LLaDA CUDA13 closure mismatch identified (runtime13.4 headers with nvcc13.0);
isolated CPU small-kernel compile now passes with matching headers. No sparse
model port or qualified new-model accuracy result yet.

## V28 seed expansion, held-regroup negative and canvas release — 2026-10-02 19:44 (UTC-5)

User explicitly reiterated large seed-dependent denoising-step variance. New
seed4 protocols freeze four independent engine seeds (28001–28004), two sequential
repeat labels per engine, six question clusters: 48 requests/arm. Repeats are not
extra independent seeds; same engine seed is not a matched random trajectory.
Keep per-seed N distributions, paired geometric W/S/SN/N and correctness, with
question-cluster intervals. This remains preview; expand question coverage for
request-level claims. Existing qualification002/V18b protocols are untouched.
Common OMP_NUM_THREADS=1; new requests record pre-deduplication monitor Triton/CuTe
JIT event deltas as well as CUDA capture/backend/Inductor counters. Nonzero/missing
timed monitor receipts fail the new run. Events can include cache loads/failed
compiles, exclude autotuning/other workers/earlier aliases; zero is not universal
absence of compilation. Existing log warnings use warning_once, so the original
panel's zero post-warm warnings are only a lower-bound audit. Its block-0 main
had 30 recoverable post-warm allocation failures, retained in W.

Held-regroup component003 completed: natural-Q64-relative total times
1.04498/1.00462/1.03343 at nominal32/64/96K, geomean1.02753, even excluding all
search/initial-build cost. Three accepted states only; reject this unfused held
implementation for request expansion, not all grouping/fusion hypotheses.
Source: `results/v28_20261002/regroup_held/`; 45.8499 reserved GPU seconds.

All five qualification002 request arms passed (q64/main128/allkept/native/dense),
with existing graph/backend/Inductor checks and intended-path receipts. One item
per arm is implementation qualification, not performance or quality evidence.
A q64 alias1/2/4 component sweep is now specified and CPU-tested; independent
adapter per split avoids the split-cache identity-key pitfall. GPU run pending.

New standard optimization `canvas_buffers=release_after_invalidate` releases all
old contiguous adapter KV and prefix views only after successful encoder hooks
and same-CUDA-stream guards. Core observation/projection uses that stream;
async selector reads independent arrays with existing record_stream protection.
Carry and split maps are retained; no synchronization or empty_cache added.
Defaults remain legacy. Must qualify real release counters, numerics and allocator
behavior; no measured memory/performance claim yet. q128/q64 plus matched allkept
use the same option in separate frozen specifications. 92 relevant CPU tests pass.

LLaDA official SGLang JointThreshold dense smoke now passed, three toy checks,
forward count unavailable. This is environment qualification only; none of our
sparse variants are ported yet. Source: `results/v28_20261002/llada_dense_smoke/`.
Six attempts reserved345.5495GPU seconds including failures/stops. I-DLM own-stack
CPU setup/toolchain qualified and first dense smoke is compiling/running; no
new-model sparse or benchmark result claimed.

## V28 qualification and four-seed family — 2026-10-02 20:04 (UTC-5)

Qualification004 closed five request workers on the same dcfdb8730 source and OMP1.
Q128 legacy/release each N72; Q64 legacy/release each N66; allkept release N81.
Timed monitored JIT events and allocator retry/OOM deltas were zero. Release
counters show5 canvas invalidations for each method (9,731,891,200 cumulative
bytes released, not peak memory saved). Process peaks were nearly unchanged;
no peak-memory or speed advantage is established. Reserved cost867.9727GPU seconds
includes the14.3026-second alias sweep. Qualifying one item does not establish accuracy.

The six-arm family spec freezes dense, native-hook, allkept release, main legacy
canvas, main release and Q64 release. All use request_clear and shared generation
source. Four engine seeds28001–28004, two sequential repeats each, six questions:
48timed/variant; 288timed+144warm in24fresh engines. Variant order reverses in
alternating blocks. Paired W/S/P/SN/N and correctness use question-cluster intervals
and per-engine-seed step distributions. These are preview data, not a powered
noninferiority test. Family spec: `results/v28_20261002/specs/v28_seed4_family.json`.

Alias1/2/4 initial sweep uses three historical need states and synthetic QKV.
Alias1/2 ratio1.29397; alias4/2 ratio0.99264 with 64K slower1.01820 and96K faster0.96739.
Keep alias2; extend to all144 correlated snapshots with descriptive per-bin
aggregates, not question-level CI or request speed claims.

I-DLM chat follow-up executed successfully but all3calls hit512tokens despite
answer-presence checks3/3. Source and CPU EOS checks found no obvious stop-set
mismatch; actual output was not retained so repetition/think-closure diagnostics
are unavailable. Baseline stopping/quality remains unqualified. All new-model
attempts reserved723.0345GPU seconds. No sparse ports implemented; model-specific
clock, causal mask, rollback and all-kept controls are prerequisites. See
`docs/NEW_MODEL_PORT_AUDIT_V28_20261002.md` and
`results/v28_20261002/idlm_chat_smoke_followup/`.

## V28 preview launched and strict summary ready — 2026-10-02 20:14 (UTC-5)

seed4_preview001 launched on mpk from unchanged dcfdb8730 generation deployment,
using the six-variant family spec committed at8c9cd31ae. Four independent engines
per variant, shared seeds28001–28004, six questions, two sequential repeats per
engine:288timed+144warm total. One GPU worker at a time; immutable new run directories.
The full144 alias component run ended and released the GPU before preview launch.
Alias4/2 mean0.98836, with32K slower1.01731,64K0.98315,96K0.96533; keep alias2.
Reserved component cost27.0913GPU seconds. No request/accuracy claim from this sweep.

The new CPU summary validates all24workers and288records, exact bound source/input
bytes, protocol/config/host/software/seed identities, intended execution counters,
actual N, zero observed timed monitor JIT events and CUDA captures, and private
completion joins before unchanged NeMo scoring. Outputs contain aggregates only,
including W/S/P/SN/N, correctness, by-seed N distributions and length-stratified
question-cluster95%intervals. The main_legacy/dense extra comparison is labelled
descriptive; other seven comparisons are frozen family pairs.17CPU tests passed.
Qualification004 release/legacy private output equality holds for each Q128/Q64
pair (one item each); this is a limited numerical control, not model quality evidence.

CPU-only new-model event prototypes and12tests cover absolute-position identity,
prefix/layout epochs, edits including A-to-B-to-A, request reset and rollback.
They install no native hook, measure no forwards, and implement no sparse attention.
Unchanged token IDs do not prove unchanged hidden states/QKV or safe support reuse.
New-model adaptation audit states all unimplemented GPU/mask/KV/stream boundaries.

## I-DLM natural stopping diagnostic003 complete — 2026-10-02 20:33 (UTC-5)

All3official dense toy calls naturally stopped at EOS with closed thinking sections,
at868/549/1200 output tokens. Actual API IDs verify one EOS at the final position;
8gram excess repetition fractions0/0.00738/0.01006. This supports the earlier512
budget being too short for this toy; it is not a matched causal experiment because
later RNG trajectories change with earlier output length. Oneengine seed0, not3seeds.
Answer-substring presence does not establish exact final-answer or benchmark accuracy.
No sparse method or verified actual-forward count is implemented for this model.
Source: `results/v28_20261002/idlm_stopping_diagnostic_003/`. Reserved193.1752013GPU
seconds; I-DLM all attempts570.6601686; all new models916.2096868. Failed prelaunch
transfer check used0GPU seconds. Worker ended, GPU released, private output/statistic
recomputation and frozen-byte checks passed. Earlier reports remain unchanged.

## Triton permutation negative and completion pipeline ready — 2026-10-02 20:52 (UTC-5)

Frozen e69b59033 pure permutation ran on dlm2. All exact GPU checks passed;
Triton/Torch event-span geomean1.626493 (three bins1.643810/1.620939/1.614872).
Both allocate per call, warm8 and100rotated repeats. Event spans include any
host-dispatch gaps; no isolated device-kernel or request-speed interpretation.
Keep this negative. Full held-attention GPU follow-up was stopped at the stage
gate before launch; its prepared deployment is retained, no worker ran. Source:
`results/v28_20261002/regroup_triton_permutation/`. Reserved40.2940346GPU seconds;
internal post-selection/import body span1.6872869s is a different accounting scope.

The one-shot seed4 completion helper is prepared in private coordination as
`finish_seed4_preview001.py`; it has not yet been armed. Separate CPU scoring
source f0d3cd000 is deployed, using the pre-qualified registry CPU interpreter
and pinned read-only NeMo; real correct/incorrect toy joins plus17scorer tests
passed. Generation environment remains unchanged. Full24worker/inventory/source/
receipt/gold checks precede score; only anonymous aggregates are downloaded.
Publication requires unchanged expected local/remote V28 HEAD, clean staging and
only known ignored-artifact dirt, then updates HANDOFF/docs/STATE and non-force
pushes. Any failure retains private evidence and stops; no GPU launch/retry or
scientific conclusion is automated. Offline failure, closed-accounting and mock
publication checks pass. Fetch avoids writing the parent worktree's FETCH_HEAD.

Preview frozen question identities were checked privately: six distinct IDs,
zero cross-bin duplicates. Four engine seeds and two sequential repeats do not
increase the independent question count beyond six.

## Completed parent V18b and seed interpretation — 2026-10-02 21:02 (UTC-5)

Read-only source: parent branch commit `e1d2f89c29b6a713ffd3419cf7149d1a34abbd50`,
568 timed requests,59 questions,2 independent engine seeds, four repeat labels.
Formal reserved10654.113GPU seconds; with qualification12809.068. No merge.
Main/default-dense W ratios .7814/.7439/.8070 at32K/64K/96K, but main/native
W on the small matched subset is1.1377/1.0065/1.0446.32K S/N is slower versus
dense; changed N contributes substantially. Native's own lower W accompanies
lower N, not lower S/N; do not label it a proven engineering-only gain.
64K main/all-kept W .9255 [.8969,.9668] is a lead, with7/16 versus8/16 correct.
Equal64K/96K main/dense accuracy counts do not establish noninferiority.
See `docs/V18B_INTERPRETATION_V28_20261002.md` for sources, CIs and limitations.

Frozen V28 preview unchanged:4 engine seeds x2 repeats x6 distinct questions;
report N by seed and cluster by question. It is still a preview. Confirmation
needs more independent questions and seed blocks with expanded matched controls,
frozen accuracy margin and analysis. Do not silently add per-request RNG resets.
One-shot existing-campaign completion helper will be armed after this commit
is pushed; private status is authoritative. It uses separately qualified CPU
scorer f0d3cd000, checks frozen identities and all workers, then publishes only
aggregates behind unchanged-local/remote-HEAD guards. No new GPU launch or
automatic favorable scientific conclusion. Any failure stops for review.


## V29 expansion and fused-copy qualification specified (2026-10-02, UTC-5)

See `docs/CONFIRMATION_CAMPAIGN_V29_20261002.md`. New branch from V28 e5e1ffcc8;
no change to its running preview/finalizer.42 CPU copy/address/adapter tests pass.
GPU copy/component qualification and expanded dataset panels pending; no new
performance or accuracy result. Standard copy optimization must apply to matched
all-kept/main controls. Regroup O-scatter fusion is a separately named candidate.

## Fused paged-copy component001 passed (2026-10-02 21:53, UTC-5)

Source3a3a3c52d;10 synthetic native-stride GPU cases bit-exact,100 alternating
samples after8 warm. Long tail copy ratios .55754/.55732/.55517 at32/64/96K;
full-copy ratios .18225/.16055/.15203. Event spans include host dispatch and
original intermediate allocation. Real-model qualification and W/accuracy pending;
standard optimization must also apply to all-kept. No request-speed claim.
Reserved3.867908GPU seconds on dlm2; run closed. Sources:
`results/v29_20261002/paged_copy001/{summary.json,receipts.json,README.md}`.


Independent32K profiler (10 CPU tests) and mapped alias2 merge prototype (11 CPU tests) ready for separate GPU qualification. Fixed component protocol: historical3 accepted states, seed2903, warm8,32 rotated samples, natural_fused matched reference, frozen order/search unchanged; GPU Torch LSE and IEEE FP32 masked oracles before timing; online construction remains unmeasured. Merge is tolerance-qualified, not bit-exact. Source docs: docs/V29_32K_COST_AUDIT_20261002.md and docs/V29_REGROUP_REVIEW_20261002.md. No GPU result for these diagnostics yet.

## Regroup fused writeback component001 closed (2026-10-02 22:05 (UTC-5))

Sourcee07aec657; three fixed accepted states passed GPU numeric oracles.
Held_fused/natural_fused1.00566/.97856/1.01061, geometric.998177: approximate
tie and no general regroup speed evidence. No request expansion for this held
search. Standard natural merge fusion alone gives.90340/.92598/.90358 ratios;
keep as shared implementation candidate, not regroup novelty. Not bit-exact.
Reserved51.200860GPU seconds, including36.475206CPU selection; source
`results/v29_20261002/regroup_merge001/`. No W/quality evidence.

Adapter `merge_backend=triton` opt-in now CPU-qualified alongside copy backend;
default remains torch. Identity constructed from known CPU indices, no D2H
validation; count builds/calls, reuse only immutable index geometry, clear at
request boundaries. Apply to main/all-kept equally; all request initialization
cost stays in W. Real-model/GPU adapter qualification remains pending.
Independent32K profiler updated with denoise/encoder scope where exact existing
CPU phase is available; default FULL missing scopes stay unknown.11 CPU tests.
Combined copy/merge/profile/adapter suites67 tests pass.
