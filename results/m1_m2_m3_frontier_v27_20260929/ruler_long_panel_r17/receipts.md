# R17 receipts: RULER 32K / 64K / 92K with V-dimension variants (2026-10-02)

- **Spec and run:**
  - spec `specs/v27_ruler_long_r17.json` (spec commit 6064109), protocol frozen before generation;
  - deploy `v27_r17_6064109`, run dir `v27_r17_001`;
  - 3,120 / 3,120 runs scored, on dllm, mpk and dlm2, one host per cell, substrate piecewise_v5;
  - run 13:36–15:12 UTC−5, about 16,900 GPU-seconds (sum of the three worker spans).
- **Data:** all 13 RULER tasks × 10 samples = 130 rows per length.
  - Fresh pinned-RULER generation: seed 1717, 64-token chat-template buffer, length mode total.
  - Lengths 32,768, 65,536 and 94,208 tokens; seed 404.
- **Why 92K instead of 96K:** the 96K rows are 97.5K–98.2K prompt tokens. Our HF-based substrate OOMs above about
  95K for every arm, dense included (c2 panel: 95,074 ran, 98,282 OOMed). Every 92K row is ≤ 95,074.
- **Arms:**
  - dense FA4 all-kept;
  - main (M3 R6 DP −ln2 + `carry_first`);
  - fixed GLOBAL sparsity 88% (`risk_topk` k12) with projected-V rank 32 (default), 8, 4 and mass-only (no V term);
  - fixed 95% (k5) with rank 32 and mass-only.
- **Endpoint: accuracy only.** RULER answers need about 5 decoder calls and one canvas, so speed is not measured.

**Path receipts (per-cell counters, all three hosts).**
- Every arm reports its specified `risk_topk` / `proj_rank` / `risk_value` / `threshold_shift` in `effective_method`.
- No native or fresh-fused fallback calls.
- Each request is one canvas: no carried first call; 5 fused observations per cell, one per GLOBAL layer.

## Accuracy (correct of 130; +/− and exact McNemar p vs dense)

| arm | 32K | 64K | 92K |
|---|---|---|---|
| dense | 110 | 97 | 91 |
| **main** | **110** (+3/−3, 1.0) | **98** (+3/−2, 1.0) | **91** (+1/−0, 1.0) |
| 88%, rank 32 | 105 (+1/−6, 0.13) | 98 (+4/−3, 1.0) | 91 |
| 88%, rank 8 | 102 (+1/−9, **0.021**) | 97 (+3/−3, 1.0) | 91 |
| 88%, rank 4 | 103 (+1/−8, **0.039**) | 95 (+1/−3, 0.63) | 91 |
| 88%, mass-only | 109 (+3/−4, 1.0) | 96 (+2/−3, 1.0) | 91 (+1/−1) |
| 95%, rank 32 | 101 (+0/−9, **0.004**) | 96 (+2/−3, 1.0) | 91 |
| 95%, mass-only | 107 (+2/−5, 0.45) | 97 (+2/−2, 1.0) | 90 (+0/−1) |

**V term, direct paired comparisons** (`direct_*.md`):
- At 88%: mass-only vs rank 32 is 109 vs 105 (32K, p 0.22), 96 vs 98 (64K, p 0.5) and equal at 92K.
- Rank 8 / rank 4 vs rank 32 is slightly lower in every case, never significantly.
- At 95%: mass-only vs rank 32 is **107 vs 101 at 32K (+6/−0, p 0.031)**, 97 vs 96 at 64K, 90 vs 91 at 92K.

**Where the fixed-sparsity arms lose (32K, items dense solved).**
- Almost all losses are `cwe` (common-words extraction, which aggregates over the whole context): 5–7 of the
  losses in each V-term arm, 3–4 in the mass-only arms.
- The rest are 1–2 `qa_2` / `niah_multivalue` items.
- The main arm loses 3 and wins 3.

## Reading

- **The main method keeps RULER accuracy at every length**, including 92K (110 / 98 / 91 vs dense 110 / 97 / 91).
  - It confirms R16 on a new, larger generation with a third length.
  - This is the adaptive threshold, not a fixed budget.
- **Fixed high sparsity hurts only at 32K.** At 32K the same kept fraction covers fewer absolute tiles, and
  aggregation tasks (`cwe`) need broad coverage. At 64K / 92K no fixed arm differs from dense.
- **The V term does not help on RULER; mass-only is as good or better.**
  - At 95% sparsity, adding the projected-V term is significantly worse than mass-only at 32K (p 0.031).
  - Lower V ranks (8 / 4) do not change this.
  - This matches E8 (AIME), E9 / E11 (LongBench 64K at 88% / 95%) and R16. The V-direction idea is a clean negative
    across all four benchmarks.
- **Dense ceilings:** 92K dense fails 39 / 130 and 64K fails 33 / 130, so the long lengths still have headroom.
