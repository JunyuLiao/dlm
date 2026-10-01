# Results ledger (verified results only)

Rules for this file:
- An entry needs a scored, frozen panel or a documented benchmark whose summary is in the repo.
- Dense reference is `D_fa4_allkept` unless stated.
- Ratios are method / dense: paired geometric mean with a question-clustered bootstrap 95% CI. < 1 means faster.
- W = request wall, including prefill (end to end). S = decode span, excluding prefill (generation time).
  S/N = amortized per step. NC = steps per canvas. T = output tokens.
- Accuracy is strict-correct cells, method vs dense.
- Paths are relative to `results/m1_m2_m3_frontier_v27_20260929/`.
- Protocol ids are the frozen protocol identities; frozen files are private (`E:/dlm/v23_private/`).
- Arm shorthand: **M3** = `M3_R6_A64_fused_dp_async_m1ln2_fa4` (M3 R6 DP −ln2). **Fan plain** = `M1_R1_A8_fa4` /
  `M2c_R1_A8_fa4` / `M3_R3_A8_fa4`.

## Current headline (piecewise_v5)

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
