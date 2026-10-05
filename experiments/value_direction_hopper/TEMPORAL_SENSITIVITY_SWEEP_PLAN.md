# Frozen temporal-sensitivity sweep (β, γ)

Question: with the existing Gaussian-32 value-direction risk and unchanged
native DiffusionGemma decoder, how does temporal query weighting alter the
number of denoising calls at matched physical tile sparsity?

## Controlled grid

- β = {0, 1, 3, 6}; γ = {0, 0.5, 0.9}. Because β=0 makes γ irrelevant,
  evaluate it once at γ=0; evaluate all nine β>0 combinations. The
  β=3, γ=0.5 point is the previous policy.
- Targets: 50% and 70% pooled physical skipped-tile rates, with local and
  global rates matched to the previous frozen T calibration anchor.
- Reuse the previous disjoint 26-calibration/26-validation manifests, the
  previously examined 130 final prompts, model/kernel/projection seed, and
  all decoding settings. Do not use final score or call count to choose a
  policy.
- The first two calls use the prior safe target-specific early local/global
  thresholds and unit weights for every grid point. From call 3, the causal
  weight is `1 + β u_i` where `u_i ← γ u_i + (1−γ) flip_i`; the flip is a
  deterministic top-1 argmax change. History resets per canvas. No other
  routing/decoder semantics change.

## Per-grid-point calibration

For each (β,γ,target), run complete generation on all 26 calibration prompts
for eight predeclared common offsets to the previous T late log-threshold
pair: `−1.2, −0.8, −0.4, −0.15, 0, 0.2, 0.5, 0.9`. If no candidate passes,
test six small local/global refinements around the best measured candidate:
`(−.2,0),(.2,0),(0,−.2),(0,.2),(−.1,−.1),(.1,.1)`.

Use the **same** previous guardrail function: all three complete-trajectory
rates within ±2 points of the T anchor's measured calibration rates; first
two call ceilings 55/57% for target50 or 75/77% for target70 (whole/each
stratum); mean calls ≤5.5 or ≤8; p90 ≤8 or ≤12; no caps or at most one cap.
Rank feasible points by smallest maximum whole/global/local rate deviation,
then fewer calls, then fewer executed tiles. Validate up to four ranked points
on the separate 26 prompts, permitting ±3 rate points but retaining the
trajectory gates and requiring score no more than 3 points below matched
dense. Freeze the first validation pass. If none passes, freeze the closest
measured boundary point and label it unattainable under guardrails. Report
actual rates for all final runs; never relabel a miss as a target hit.

## Execution and outputs

1. Freeze config/source hashes and audit the reused manifests. Smoke β=3,
   γ=0.5 against the previous output under its exact frozen thresholds, and
   β=0 against unweighted routing.
2. Calibrate/validate every grid point with one resumable H100 worker.
3. Run all grid points on the same 130 prompts with one resumable worker,
   preserving failed points separately. No selection on these final scores.
4. Audit shard identities, prompt/seed parity, step counters and source
   hashes; summarize score, whole/global/local rates, calls (mean/median/p90),
   cap rate, and executed tiles. Plot mean calls as β×γ heatmaps for both
   targets, annotate actual sparsity and guardrail failures, and show accuracy
   and sparsity companion plots. Compare with native/matched dense and the
   original β=3,γ=0.5 operating point.

Use `/home/exouser/ljy/dlm/results/query_adaptive_temporal_sweep_v1` as a new result root.
The 130 prompts have been studied repeatedly; all findings remain
development-benchmark evidence. No speedup claim follows from fewer calls
without separate untraced timing.
