# v14 gate decisions (recorded 2026-09-26 ~06:20Z, BEFORE any pilot answer score was read)

Evidence: `cost_budget.csv`, `complete_forward_profile.json`, `adaptive_canvas_diagnostic.json`.

## Gate 5.1 — direct complete-forward cost (AIME, production mode)
- Real states /2 canvas 1 (prefix 365), /2 canvas 8 (2157), /14 canvas 24 (6275); teacher-forced 8–10 real steps.
- Phase-weighted c (native canvas-length mix, each state with its own times):
  CVM_T 1.020 / 1.009 / 0.989; B8_P 1.015 / 1.008 / 0.980; T_P 1.021 / 1.016 / 1.016; T_G 1.015 / 1.014 / 1.012.
- Request-level estimate (canvases mapped to the nearest measured state): CVM_T 1.003, B8_P 0.998, T_P 1.017, T_G 1.014.
- Regime diagnostic (LongBench-v2 16–18K prompts, first canvas, NOT scored): CVM_T 0.929 / 0.942, B8_P 0.901 / 0.917.
- **Verdict: not encouraging on AIME** (no request-level forward saving); headroom exists only at long prefixes.

## Gate 5.2 — adaptive behavior (5 identical canvas-start states, one trajectory each)
- Calls to native stop (sum over 5 states): D 54, T_G 49, T_P 43, B8_P 43, CVM_T 46; no cap; stable-not-confident steps ≤1 per arm.
- Executed prefix fraction: CVM_T ≈ T_P (e.g. 0.72 vs 0.70 late) ≫ B8_P (0.53): live restoration brings the support back to roughly fresh-T density.
- ~90% of rows that diverge from the native argmax first diverge while s=1 (T bootstrap iterations 1–2 or no prior flip); B8_P ≡ CVM_T through step 1 by construction.
- **Verdict: no adaptive-work blow-up (descriptive, n=5)**.

## Decisions
1. **Extension (/1,/6,/16,/25; 80 executions): NOT RUN.** Rule requires both gates encouraging; gate 5.1 fails on AIME. The IDs stay committed in `frozen_protocol_extension.json`.
2. **Repair: NONE.**
   - A2 requires "CVM-T is cheap but fails adaptive behavior with failures localized to stale anchor margins". CVM-T is not cheap at AIME request level. The divergences localize to the bootstrap iterations, which use a fresh anchor, not stale margins.
   - The fused margin check requires "the planner launch dominates". It does not: the CVM−B8_P ordinary-call gap is ≤0.085 ms/layer at the early state, and it is dominated by restored QK/PV work at late states. The larger loss is a fixed ~1.7–2.0 ms/step outside GLOBAL attention that all T-using arms pay; T_G included.
   - No supported diagnosis, so no repair is invented. The spare budget goes to isolating that fixed overhead with a no-op binding control (profile only, no answers), plus the pending bounded sanitizer and launch-inventory checks.
