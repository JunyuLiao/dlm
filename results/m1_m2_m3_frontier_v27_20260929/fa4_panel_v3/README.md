# v27 LB-long panel on FA4 + piecewise_v1 (v3), 32K stage (408/816 executions)

**Panel.** Protocol `v27_long_lb_fa4pw_v3_b5399ab5c46cd59e` (`../specs/v27_long_lb_fa4pw_v3.json`), frozen before generation.
- LongBench-v2 32K bin, 12 items x seeds 101/202, first-only.
- One H100 (dllm), substrate `piecewise_v1`.
- Every sparse arm runs through FA4 block-sparse. Scoring uses the v15 LB contract (mpk, sha-pinned gold).
- Summary: `scripts/v27_fa4_panel_summary.py`. Ratios are paired geometric means with question-clustered 95% bootstraps.

**Columns.**
- W: request wall.
- S: decode span, which excludes the initial prefill.
- N: decoder calls.
- S/N: amortized per-call cost. It is not a direct per-forward price; that comes from common-state timing in `../substrate/`.
- S per output token.

## 32K findings (vs D_fa4_allkept, 24 cells)

- **Quality.** Every arm is within the noise floor: the two exact dense paths D_native and D_fa4 already differ by 3 correct, and the largest paired swing is +3/-6. Fan plain M1 has the most correct answers, 21 vs 18.
- **Speed is small at 32K.**
  - Best amortized per call: M3 R6 A64 pairs 0.933 [0.920, 0.945], G75 F 0.947, M3 R6 A64 0.957.
  - Request walls lie at 0.94-1.03 and every CI crosses 1.
  - Fan plain M1/M2c/M3 (A8) are 1.15-1.42x per call: per-call selector and score costs exceed the attention saving at 32K.
  - G75 S15/S30 are about 1.34x per call because their LOCAL layers must stay eager.
- **HF SDPA (D_native) is about 2x FA4** (S 2.22), so an HF-path baseline would inflate any gain.
- **D_fa4_allkept vs D_fa4.** Identical outputs (same calls); decode is 1.3% faster.
- **Pending.** The 64K stage, the quality panel (AIME 30 + LongBench 12, piecewise_v2), the supplement (async route, M3/M2c + DP) and v4 (these arms on piecewise_v2).
