# Measurement status (current as of v27, 2026-09-29)

`results/m1_m2_m3_frontier_v26_20260928/missing_measurements.md` is a **historical slice for v21–v25b**. Its statements "M2 never implemented/run" and "A16 never measured" were true of those records only. This table supersedes it.

| item | status | where |
|---|---|---|
| M2 row-wise reference (`pooled`) | implemented, qualified against an independent reference, run end to end (7-arm panel, 6 inputs, seed 101) | v26 `m2_contract.md`, `seven_panel_report.md` |
| M2 compact (`pooled_compact`) | implemented and GPU-tested; not yet run on complete answers | `m2_compact_contract.md` |
| A16 (M3 R3) | 7-arm panel run (6 inputs, 1 seed); not yet directly profiled | v26 panel; v27 profile running |
| R6, A64, B16, B64 (hold-only) | implemented and clock-tested; direct profile running; no complete answers yet | `v27_contract_audit.md` |
| threshold shifts −ln2 … +4ln2 | implemented; realized sparsity and direct per-phase cost being measured (M3 A8 R3) | v27 `v27thr` profile |
| direct full forward / denoising step for the new points | running (N24 from canvas start: LB odd-K c0 and c6, LB KDIV8 c0, AIME c0 and c12, RULER c0) | `direct_cost_breakdown.csv` when done |
| pure sparse execution with the bitmap given for free | running (`prepared_support_floor`, model_forward) | same |
| physical QK/PV work per phase | running (counter twin; B0/BO now counted as dense) | same |
| 30-point screen | pending: composed from measured per-phase costs and the real clock (conditional estimate) | `policy_screen.csv` |
| multi-question, multi-seed adaptive comparison | not done | Tier 2/3 |
| W attribution trace | not done | — |
| Q16 production, parallel prefix-summary builder, full runtime graph migration | **not implemented**. This is not a negative result and not a prerequisite. | — |
