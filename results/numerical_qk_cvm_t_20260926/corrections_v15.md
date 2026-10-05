# Scoped corrections to the v14 reports (appended 2026-09-26 during v15; originals left unchanged)

1. **LongBench phase weighting (source-level).**
   - `scripts/v14_cost_budget.py` builds the native canvas-call histogram from the v13 AIME development receipts, and uses that one histogram to phase-weight every captured state, including `lb/21` and `lb/40`.
   - The LongBench `phase_weighted_c_native_canvas_mix` values in `cost_budget.csv` / `complete_forward_profile.json` are therefore an **AIME-canvas-mix counterfactual**, not the observed LongBench phase mix. Those values are CVM_T 0.929 / 0.942 and B8_P 0.901 / 0.917.
   - The teacher-forced `sequence_c` values (CVM_T 0.932 / 0.945, B8_P 0.907 / 0.923) remain their own measured 10-step diagnostic; they are not complete-request times.
   - This does not affect the scored AIME pilot.
2. **Universal wording.** "No method in this family can show a measured AIME E2E execution saving" should read: *the tested frozen configurations did not demonstrate such a gain on the AIME development pilot.*
3. **Divergence cause.** "~90% of divergent rows first diverge while s = 1" describes the first argmax divergence only. It is not evidence of the cause of any final failure. Token equality, local output error, convergence behavior and task correctness are different endpoints.
4. **Geometric vs pooled.**
   - Lines of the form "CVM_T/B8_P time 1.23 [1.08, 1.50] = calls ×1.27 × per-call 0.998" mix a paired geometric ratio (1.23) with a pooled decomposition.
   - The exact pooled identity is summed-time ratio 1.267 = pooled call ratio 1.270 × pooled time-per-call ratio 0.998 (the latter is amortized request wall per call, not pure GPU latency).
   - The geometric ratio is reported separately.
5. **Overhead within drift.** "Binding + guard + fast-T controller cost ~0" should read: *unresolved within the observed ~2% profile drift (not shown to be zero).*
