# v27 LongBench-v2 32K / 64K long-generation panel (first-only, 432 executions)

**Panel.** Protocol `v27_long_lb_b881bf31c3ce1fa6` (`../specs/v27_long_lb.json`), frozen before generation.
- 12 LongBench-v2 items per bin, at natural length with no truncation: [28K, 40K) and [56K, 76K) tokens with thinking ON.
- Seeds 101/202, 9 arms, native adaptive stopping, 8192-token budget, 2 H100 hosts.
- 420/432 executions succeeded. The 12 failures are CUDA OOM for the plain A8 M1/M3 arms on the longest 64K items (about 75K tokens): their FP32 score caches for all 5 GLOBAL layers do not fit next to the model.
- **Scoring.** v15 LongBench-v2 contract, run on mpk, with sha-pinned bin gold.
- **Timing.** From the first receipts, paired by (item, seed) against D_c64:
  - S is the decode span, which excludes the initial prefill;
  - N is the number of decoder calls;
  - S/N is amortized per call;
  - W is request wall.
- **Intervals.** Question-clustered bootstrap. Files: `summary.md` and `summary.csv`.

**Baseline caveat.** These ratios are against D_c64, our 64-row Triton kernel. The official FlashAttention-4 kernel is about 1.9× faster than D_c64 at these shapes (`../official_baseline/README.md`), so every per-call ratio here must be re-based on FA4 before it is claimed.

## Findings

1. **Cross-layer shared support (`one`) collapses long-generation quality.** Both shared arms (M1 A64 and M3 R6 A64, shared + fused) do this:
   - 32K: 4/24 correct vs 20/24 for D_c64, worse in 16 paired cells and better in none. Decoder calls ×2.7–3.3, with 10–15 of 24 capped at the length budget.
   - 64K: 6–7/24 vs 12/24.
   - Their per-call savings (0.81–0.90) are real but irrelevant.
   - This matches RULER 64K (`../long_ruler_panel/README.md`).
2. **The unshared B/A64 fused arm keeps quality and is faster end to end at 64K.**
   - 64K: 13/24 (+3/−2 vs D_c64). Per call 0.935 [0.919, 0.953], calls 0.99, decode span 0.926, wall 0.940 [0.869, 1.018].
   - 32K: 18/24 (+2/−4). Per call 0.98, but calls ×1.21 and wall ×1.16.
3. **Fan's plain M1/M2c/M3 (A8, 64-row consumer, pipelined exact selector).**
   - Per call: 1.05–1.13 of D_c64 at 32K and 1.05–1.18 at 64K.
   - Correct: 16–18/24 at 32K; M2c 14/24 at 64K (+3/−1).
   - The M1/M3 cells at 64K exclude 6 OOM failures each.
4. **Quality noise floor.** The two dense paths already differ by 4 correct answers per 24 cells: D_native 16 vs D_c64 20 at 32K, and 16 vs 12 at 64K. Totals within ±4 are therefore not evidence of a quality change; the paired per-cell counts are shown for that reason.

## Consequence

- **Keep:** unshared support (B-style held maps, M3 with long hold), per-layer decisions, and a cheap decision call.
- **Drop:** cross-layer sharing.
- **Next test:** the unshared follow-up panel (`../specs/v27_long_lb_unshared.json`), re-timed with sparse execution on FA4 against D_fa4.
