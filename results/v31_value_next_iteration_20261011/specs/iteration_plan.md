# V31 online value-aware next iteration

This is an accuracy-first, audit-only development panel. Timing remains paused and resumable. The unchanged reference is `current_v31_control`, whose first map is the exact native attention-mass qblock budget. All variants alter selector logic only; native adaptive denoising, GLOBAL-only scope, LOCAL dense execution, reuse clocks, and sparse consumer are unchanged.

## Gates and arms

1. **Fixed-budget mass-gated value selection.** Build the V31 mass map first, then exchange at most a fixed number of tiles within each `(KV head, 128-row query block)` budget. `value_mass_gated_online_swap8` and `...swap16` use the V2 preserve-mass projected residual for the exchange score. `value_mass_gated_full_swap16` uses the final-output omission score (true full-dimensional V in the offline snapshot gate; live Gaussian32 summary is labelled projected-full-information and is not promoted from timing alone).
2. **Swap saturation and phase protection.** `value_mass_gated_online_phase_swap16` uses zero swaps for the first observation of a canvas and swaps on refresh/held maps. This tests whether early maps are already reliable and later maps benefit from value reallocation.
3. **C_gate.** `value_mass_gated_online_cgate_swap16` applies a denoising-derived row sensitivity as a ranking weight while retaining exactly the same per-query-block budget. It does not multiply risk or alter K, so SM work remains balanced.
4. **Intersection diagnostic.** `value_intersection_online` and `value_intersection_full` retain only support in the exact V31 mass map intersected with the V2 map. This is intentionally a diagnostic of useful support, not a production candidate if the retained set falls below budget.
5. **Drift/trajectory follow-up.** Every arm records map age, refresh phase, denoising-call count, and audited physical tile counts. The age-stratified certificate is a promotion gate for any future drift-triggered refresh; no timing claim is made here.

The five gates are evaluated first on the frozen 15-item development schedule. Arms with a complete receipt and no validation errors are scored by the official scorer against the matched control. Only an accuracy-positive or statistically compatible arm proceeds to the 503-item official audit. Clean timing is a separate, later resumable phase and is not launched by this panel.

## Selection contract

The prefix budget is the realized eligible count capped at the current control budget. Mandatory protected tiles remain mandatory. Tie-breaking is stable by physical tile id. Invalid/non-finite statistics force dense retention. The mass candidate pool is twice the budget and all ranking is on-device in the CUDA path; the configured swap count is the only scalar candidate loop. `C_gate` changes ranking only. Intersection arms intentionally may retain fewer than the fixed budget and are reported with their realized denominator.
