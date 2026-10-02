# P17 receipts: C-gate preview (2026-10-02)

- Spec `specs/v27_cgate_preview_p17.json`, protocol `v27_cgate_preview_p17_f0b80167ef28fb6e`, deploy `v27_p17_96a43f4`,
  run dir `v27_p17_001`.
- Before launch, all v27 GPU tests passed on dllm (240 passed).
- 176/176 runs ok on dllm, mpk and dlm2, one host per cell, substrate piecewise_v5. GPU time about 3,200 s.
- Data: AIME 10, LongBench-v2 64K 8 and 96K 4 items × seeds 404/505, so 44 cells per arm. **This is a preview.**

**The C gate is collaboration, not our contribution.** It is a group member's query-sensitivity design, ported from
their design note (shared 2026-10-02; their branch is not yet available).
- Code: `query_adaptive.State.enable_cgate`, tests `tests/test_v27_cgate.py`.
- It replaces T's per-position weights with s = clip(1 + β·h, 1, 1 + β), where:
  - h = 1 − g·(1 − q)·(1 − u) and g = 1 − exp(−r/τ);
  - q is an EMA (γ 0.65) of "renoised" (not accepted);
  - r is the stable-run counter, reset on a top-1 flip;
  - u = √(1 − p_top1); τ = 2.5, β = 3.
- The first call of each canvas is fully protected.

**Arms:**
- dense FA4;
- main (T, −ln2);
- C gate at −ln2;
- C gate at the sparser base threshold (no shift).

**Path receipts (per-cell counters, all three hosts).**
- All 88 C-gate cells report `sensitivity = cgate` in `effective_method`. The 44 main cells report `temporal_T`.
- Threshold shifts are as specified: `minus_ln2` for main and C gate −ln2; none for C gate base.
- No native or fresh-fused fallback calls in any arm.

## Vs dense (`summary.md`, `steps.md`)

| bin | arm | accuracy (+/-) | W | per step S/N [CI] |
|---|---|---|---:|---|
| AIME | dense / main / C −ln2 / C base | 15 / 16 / 14 / 15 | 1 / 1.013 / 0.911 / 1.057 | 1 / 1.014 / 0.966 / 1.011 |
| 64K | dense / main / C −ln2 / C base | 8 / 8 / 9 / 10 | 1 / 0.902 / 0.947 / 0.863 | 1 / 0.815 / 0.873 / 0.837 |
| 96K | dense / main / C −ln2 / C base | 4 / 6 / 4 / 4 | 1 / 0.788 / 0.850 / 0.907 | 1 / 0.726 / 0.801 / 0.739 |
| total | dense / main / C −ln2 / C base | 27 / 30 / 27 / 29 (of 44) | | |

## Direct vs main (`direct_*_vs_main.md`, paired cells)

| bin | C gate −ln2: per step [CI] | C gate base: per step [CI] |
|---|---|---|
| AIME | 0.953 [0.795, 1.057] | 0.997 [0.789, 1.223] |
| 64K | **1.071 [1.063, 1.079]** | **1.027 [1.008, 1.050]** |
| 96K | **1.103 [1.078, 1.125]** | **1.017 [1.002, 1.044]** |

## Reading

- **The C gate costs speed at long context.** At the same −ln2 threshold it is 7–10% slower per step than T at
  64K/96K. Its weights are high wherever a position has no stable accepted history, which is most of the canvas, so
  more key tiles are kept.
- **At the sparser base threshold it nearly matches T per step** (+2–3%). Accuracy is within noise there: 29 vs 30
  of 44 overall, 64K 10 vs 8, 96K 4 vs 6.
- **No accuracy gain to trade the speed for.** None of the differences is significant (McNemar p ≥ 0.5).
- **Verdict:** not adopted as the main sensitivity. No full panel is planned unless the group member wants one; it
  would need about 18 seeds to resolve differences of this size.
