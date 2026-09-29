# v27 Tier 2 development panel (exposed; quality screen, not a validation)

**Frozen protocols:** `v27_dev_lb_b1c3c6e4860982b3` (216 executions) and `v27_dev_aime_e3b3edc987a4fde5` (72). Deploy `f66bfbad`. Every run completed; there were no failures and every warm run was accepted.

**Design.**
- LB: the six exposed v20 ids × seeds 101/202, first + warm.
- AIME: four new pool ids × two seeds, first only.
- Arms: native, D_matched, fresh T, M1, compact M2, M3 R3/A8, M3 R3/A64, M3 R6/A64 and B A64 (hold-only).
- All use P0, GLOBAL-only, native bootstrap and the aligned16_odd grouped-Q contract.
- Hosts are balanced by (question + seed) mod 2, and every arm of a block runs on one GPU.
- Per-cell data (no gold, no text): [tier2_dev_cells.csv](tier2_dev_cells.csv).

**Reproduction check.** The native, fresh T, M3 R3/A8 and M1 LB rows reproduce the earlier v23/v24 bootstrap panel exactly (6/12 with 2,360 calls, 6/12 with 2,223, 6/12 with 2,415, 5/12 with 2,623). The v27 kernel refactor and new options therefore did not change those outputs.

## LongBench v2 (12 cells; geometric means of paired per-cell ratios, warm wall)

| arm | correct | calls | W / native | N / native | (W/N) / native | (W/N) / D_matched |
|---|---:|---:|---:|---:|---:|---:|
| native | 6/12 | 2360 | 1 | 1 | 1 | 1.041 |
| D_matched | 5/12 | 2944 | 1.029 | 1.071 | 0.961 | 1 |
| fresh T | 6/12 | 2223 | 0.907 | 0.908 | 0.998 | 1.039 |
| M1 R1/A8 | 5/12 | 2623 | 1.049 | 1.069 | 0.982 | 1.022 |
| compact M2 R1/A8 | 6/12 | 2113 | 0.899 | 0.926 | 0.970 | 1.010 |
| M3 R3/A8 | 6/12 | 2415 | 0.984 | 1.040 | 0.946 | 0.984 |
| M3 R3/A64 | 5/12 | 2445 | 0.963 | 1.026 | 0.939 | 0.977 |
| M3 R6/A64 | 5/12 | 2360 | 0.928 | 0.991 | 0.937 | 0.975 |
| B A64 | 5/12 | 2996 | 1.111 | 1.232 | 0.902 | 0.938 |

## AIME 2026 (8 cells, first-only; wall is cold and includes first-request effects)

| arm | correct | cap | calls | W / native | N / native | (W/N) / native |
|---|---:|---:|---:|---:|---:|---:|
| native | 6/8 | 0 | 1563 | 1 | 1 | 1 |
| D_matched | 7/8 | 0 | 1314 | 0.900 | 0.896 | 1.005 |
| fresh T | 8/8 | 0 | 1475 | 1.017 | 1.009 | 1.007 |
| M1 | 5/8 | 0 | 1711 | 1.141 | 1.107 | 1.030 |
| compact M2 | 8/8 | 0 | 1259 | 0.851 | 0.822 | 1.035 |
| M3 R3/A8 | 7/8 | 1 | 1404 | 0.917 | 0.900 | 1.018 |
| M3 R3/A64 | 8/8 | 0 | 1231 | 0.807 | 0.794 | 1.017 |
| M3 R6/A64 | 6/8 | 1 | 1376 | 0.904 | 0.893 | 1.012 |
| B A64 | 6/8 | 1 | 1658 | 1.093 | 1.086 | 1.006 |

## Reading

1. **Per-call amortized cost reproduces the direct Tier 1 ranking on LB.**
   - Order: B 0.90 < M3 A64 ≈ 0.94 < M3 R3/A8 0.95 < M2c 0.97 < M1 0.98 < fresh T ≈ 1.0.
   - Against the same-consumer dense D_matched: M3 R6/A64 is 0.975, B is 0.938.
   - On AIME every method is 1.005–1.035 per call. There is no per-forward gain at short context, as predicted.
   - W/N is amortized. The direct per-forward numbers are in `direct_cost_report.md`.
2. **Held-only B takes many more calls.**
   - B's calls are ×1.23 on LB and ×1.09 on AIME.
   - Its whole-request time is therefore the worst on LB (1.11) and on AIME (1.09).
   - Periodic M1 redecision (M3) keeps the call count near native at a small per-call premium over B. This is the mechanism Fan's M3 is meant to exploit, seen on these exposed cells.
3. **Whole-request time is dominated by call counts.** One cell can differ 2–3× between arms (D_matched used 1,056 calls on one LB cell).
4. **Quality differences are one or two cells** (5/12 vs 6/12, 6/8 vs 8/8). This is a screen, not non-inferiority.

## Freeze for Tier 3 (taken before any Tier 3 output)

- **Primary M3: R6/A64.** Announced before AIME dev scoring, as the cheapest M3 per call in Tier 1 and in LB dev.
- **Secondary (pre-declared): R3/A64.** It shares the matched hold-only B A64.
- **Also kept:** D_matched for attribution; M3 R3/A8 as an LB-only reference sub-panel.
