# v27 Tier 3 frozen comparison and the strong-dense correction

## 1. What ran

Frozen before any Tier 3 output (`specs/v27_t3_*.json`, protocols `v27_t3_lb_d16064966ed362d5`, `v27_t3_aime_bb523138c0851cf1`, `v27_t3_ruler_f9f2cbe44b1fa1f7`). The deploy is `f66bfbad`, which is also the source of Tier 2.

- **Execution:** 888/888 executions completed; every warm run was accepted.
- **One resumed block:** one AIME block on mpk stopped at the per-stage cap and was resumed from the same ledger (`_resume1` receipt), with no gaps and no reruns.
- **Tasks:**
  - LB: six new ids × seeds 101/202.
  - AIME: eight new ids × 2 seeds.
  - RULER: the thirteen v20 tasks (exposed) × 2 seeds.
- **Arms:** native, D_matched, fresh T, M1, compact M2, M3 R6/A64 (pre-declared primary), M3 R3/A64 (pre-declared secondary), B A64 (hold-only), plus the M3 R3/A8 reference on LB.
- **Per-cell data** (no gold, no text): [tier3_cells.csv](tier3_cells.csv).
- **Summary tables** with question-clustered bootstrap CIs, per-seed and leave-one-question-out ranges: [tier3_summary_tables.md](tier3_summary_tables.md).
- **Redacted generations** of all 1,176 Tier 2 and Tier 3 executions: `generation_records_v27/`.

## 2. Results (first output = quality; accepted warm = time; ratios vs native, paired per cell)

**LongBench v2** (12 cells; native 9/12):

| arm | correct | W / native [95% CI] | N / native | (W/N) / native [CI] |
|---|---:|---|---:|---|
| M3 R6/A64 (primary) | 8/12 (1 cap) | 1.227 [0.997, 1.563] | 1.335 | 0.919 [0.889, 0.943] |
| M3 R3/A64 (secondary) | 9/12 | 0.962 [0.781, 1.221] | 1.020 | 0.943 [0.911, 0.975] |
| M3 R3/A8 (reference) | 10/12 | 1.182 [0.936, 1.500] | 1.255 | 0.942 |
| B A64 | 9/12 | 1.193 [1.041, 1.352] | 1.322 | 0.902 |
| compact M2 | 10/12 | 1.273 [1.014, 1.508] | 1.331 | 0.956 |
| M1 | 8/12 (1 cap) | 1.405 [1.104, 1.788] | 1.448 | 0.970 |
| fresh T | 9/12 | 1.108 | 1.136 | 0.976 |
| D_matched | 9/12 | 1.051 | 1.081 | 0.973 |

**AIME 2026** (16 cells; native 7/16 with 9 caps):
- Every arm costs 1.00–1.03 per call.
- The primary M3 R6/A64 keeps 7/16. The secondary R3/A64 and compact M2 give 5/16. B gives 6/16, M1 7/16, fresh T 9/16.
- Request-time ratios are 0.91–1.08 with CIs spanning 1.

**RULER-4K** (26 cells): every arm scores 24/26, per-call cost is 1.00–1.03, and request time is 0.99–1.08.

## 3. Reading

1. **The per-call amortized ranking of Tier 1/2 reproduces on new LB questions** (B < M3 A64 < M2c < M1 < fresh T), with tight CIs.
2. **The call count does not reproduce.**
   - In Tier 2 the M3 arms stayed near native calls. On the new LB questions most sparse arms took ×1.25–1.45 calls.
   - The primary M3 R6/A64 is therefore **slower per request than native (1.23)**.
   - Only M3 R3/A64 stays near native calls (1.02) and is request-neutral (0.96, CI across 1).
   - With 6 questions × 2 seeds the call counts are dominated by a few long trajectories; see the leave-one-question-out ranges.
3. **Quality.** No arm shows a clear loss on LB or RULER. On AIME, the primary keeps native's 7/16; the secondary and compact M2 lose two cells. Caps dominate AIME (9–10 of 16 in most arms).

## 4. Strong-dense correction (applies to every per-forward number in v27)

A same-state microbenchmark on H100, at the GLOBAL geometry of 16 query heads, 2 KV heads, head_dim 512 and 17.5K keys, showed:
- The native model path (SDPA with `enable_gqa`) takes 5.24 ms per GLOBAL layer.
- The same dense attention with K/V repeated to 16 heads takes 2.33 ms.

`D_fast` ([`v27_fast_dense.py`](../../experiments/numerical_qk_reuse/v27_fast_dense.py)) uses the repeated-KV call for GLOBAL layers only; the LOCAL layers stay native, where native is fastest. It is the strongest dense path we have. Complete-forward results on the same states ([fast_dense_attribution.csv](fast_dense_attribution.csv)), per forward vs native:

| state | D_fast | best sparse (M3 R6/A64, one shared GLOBAL bitmap) | B A64 | D_matched |
|---|---:|---:|---:|---:|
| LB odd-K c0 | 0.926 | 0.926 | 0.938 | 0.986 |
| LB odd-K c6 | 0.917 | 0.912 | 0.917 | 0.982 |
| LB KDIV8 c0 | 0.919 | 0.922 | 0.938 | 0.982 |
| AIME c0 | 0.994 | 1.008 | 1.014 | 1.007 |
| AIME c12 | 0.974 | 0.998 | 1.009 | 0.990 |

**Consequences:**
- The ~8–10% per-forward gain that v27 reported "vs native" is almost entirely the difference between the slow `enable_gqa` dispatch and a better dense call. It is not a sparsity gain.
- Against D_fast, the best sparse variant is within ±0.5% on LB (14–17K keys) and 1.5–2.5% slower on AIME.
- The cause is kernel efficiency.
  - Our pre-QK Triton consumer needs 4.49 ms with every tile kept, vs 2.33 ms for the repeated-KV SDPA.
  - At the real P0 support (38–51% of tiles kept) it needs 1.7–2.1 ms, which leaves ≤ 0.6 ms per layer to save before any selection cost.
- The bootstrap calls (B0/BO) of every sparse arm still use the slow native dispatch. Giving them the D_fast call would move sparse arms by at most about 2/N of a call's GLOBAL time; it does not change the conclusion.

## 5. What this leaves

- **Structure-level results that survive:**
  - Cross-layer bitmap sharing removes most selection overhead (M1 0.989 → 0.940; M3 R6/A64 0.933–0.950 → 0.913–0.926 vs native).
  - LOCAL-layer sparsity is not viable: a LOCAL native call is 0.04–0.07 ms, while our consumer needs ≥ 0.13 ms even at 10% kept.
  - Short contexts have no attention headroom.
- **For a real per-forward gain against strong dense, one of two things must change:**
  - (a) A block-sparse kernel whose per-tile cost matches dense SDPA. At a 40% kept fraction that would save ≈ 1.4 ms per GLOBAL layer, about 5% of a forward at 16K.
  - (b) Longer contexts (32K–128K), where the GLOBAL share, and hence the ceiling, grows.
- **Request-level:** call-count changes (×1.0–1.45) dominate at this sample size and in the unfavorable direction for most sparse arms on new LB questions.
