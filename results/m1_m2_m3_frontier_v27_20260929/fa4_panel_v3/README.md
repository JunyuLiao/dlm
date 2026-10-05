# v27 LB-long panel on FA4 + piecewise_v1 (v3): 32K and 64K stages (816/816 executions)

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

## 64K findings (vs D_fa4_allkept, 24 cells; `summary_full.md`, `summary_full.csv`)

- **Request-wall gains whose 95% CI excludes 1:**

  | arm | W [95% CI] | decode S | S/N (amortized) | correct (dense 12) |
  |---|---|---:|---:|---:|
  | B A64 fused | 0.865 [0.785, 0.947] | 0.790 | 0.855 | 14 (+4/-2) |
  | M3 R6 A64 fused | 0.892 [0.825, 0.948] | 0.832 | 0.897 | 13 (+3/-2) |
  | M3 R6 A64 pairs | 0.909 [0.825, 0.997] | 0.864 | 0.865 | 14 (+2/-0) |
  | G75 L0 | 0.921 [0.857, 0.995] | 0.880 | 0.890 | 14 (+2/-0) |

- **CI crosses 1:** M1 R1 A64 fused DP, W 0.932 [0.856, 1.020], S/N 0.857.
- **Fan plain methods are slower than dense at 64K too.** Request W: M1 A8 1.30, M2c A8 1.35, M3 R3 A8 1.05. Their per-8-step observation refresh costs more than the attention they skip.
- **Other slower arms:** M3 R3 A64 (W 1.02), M1 and M2c R1 A64 with the sequential route (1.08, 1.22), and G75 L15/L30 (1.15, 1.03; LOCAL layers stay eager).
- **Quality.** Every arm is within the noise floor (the largest swing is +4/-2 of 24).
- **Dense references.** D_fa4 is 2.1% slower per call than D_fa4_allkept with identical calls, and D_native (HF SDPA) is 2.19x in request wall.
- **Why 64K gains and 32K does not.** GLOBAL attention is 9% of a dense 32K request wall but 14% at 75K (`../substrate/time_breakdown.json`), and held calls already sit at that Amdahl ceiling.
- **Pending.** The quality preview and panel (AIME + LongBench, piecewise_v2), the supplement (async route, M3/M2c + DP) and v4 (these arms on piecewise_v2).
