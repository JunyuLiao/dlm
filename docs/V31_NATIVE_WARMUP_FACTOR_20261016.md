# Native comparison warmup factor

This execution diagnostic follows the first process-matched panel failure.
The four workload smoke gates passed, but one of nine AIME control bookends
drifted on the first long, value-active group. The failed attempt is preserved
and excluded from method inference. No accuracy result selects this factor.

First run two original V31 engines on indices 0, 10, 20, seed 42, with six
consecutive repetitions per input. This tests the same initial input history
without value interventions. Stable controls do not prove that value
initialization caused the failed comparison.

Then freeze a two-policy, two-engine-per-policy warmup factor in ABBA order:

1. Four explicit long-input V31 warmup requests.
2. Four explicit warmup requests on the identical input and seed, one per
   promoted candidate: full swaps, directional residual, raw residual,
   combined swaps and directional residual.
3. Repeat policy 2 in a fresh engine.
4. Repeat policy 1 in a fresh engine.

Both policies also receive the unchanged benchmark's implicit V31 warmup.
The explicit warmup input is AIME index 10, seed 42. After warmup, every engine
runs all nine cells from the failed shard in their original order: V31,
four rotated candidates, V31. There are 58 recorded requests per engine,
including four explicit warmups; 54 are measured comparison requests. Inputs,
request counts, source, sampler, K, and runtime are equal between policies.
Only explicit warmup arms differ. The same qualified request switcher runs.

The hypothesis is that first active value computation changes persistent
allocation or initialization state. The existing runtime lazily builds per-head
O-projection Gram matrices and, for guarded correction, a float32 projection.
These caches survive request reset. This observation motivates the factor;
it does not establish a causal path to native output drift.

Validate every live arm, request index, fixed-budget/scope receipt, graph
capture count and input binding. Confirm that all-method warmup actually
exercises anchor events and the applicable correction. Compare output hashes,
N, C, T, RNG seed and decoded text for control bookends, and require inactive
candidates to match the control. Publish only sanitized comparisons, never
decoded text. Structural failures stop; parity failures are recorded while
the remaining predeclared factor jobs continue. Do not rerun an engine until
it happens to pass or select a favorable subset of the nine cases.

Interpretation is predeclared:

- Both policies stable: long-input warmup may suffice, or the drift is rare;
  there is no isolated evidence for all-method initialization as the cause.
- Only all-method warmup stable: supports testing that policy in a new frozen
  native campaign, while retaining per-cell control bookends.
- Any continuing drift: do not launch the old exact-parity accuracy protocol.
  Quantify repeatability and design a replicated comparison that includes
  execution variation, or repair a demonstrated runtime cause. Do not silently
  relax the existing gate.

This is diagnostic work, not clean timing or accuracy evaluation. Existing
independent-process scores and the failed process-matched shard remain held.
The three method formulas, native sampler and per-query-tile budget are
unchanged. Other paused campaigns remain paused.
