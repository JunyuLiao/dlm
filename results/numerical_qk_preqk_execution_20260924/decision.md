# v7 decision: real pre-QK execution built and measured; still slower than dense

## What was asked and what happened

1. **Correct the reporting errors.** Done, CPU-only, records untouched.
   `corrections.md` + `correctness_vectors.json`, enforced by
   `tests/test_preqk_correctness_vectors.py`. The load-bearing one: the v6
   claim that the recovered M1 matched dense *and* fresh T "same missed
   question" is false for T. Dense `[T,T,F,T]`, fresh T `[T,T,T,F]`, new M1
   `[T,T,F,T]`. Same 3/4 count, different questions; M1 shares a vector with
   dense only. Also corrected: `temperature=0` is the native-annealing
   sentinel not greedy decoding; the A2 kernel check was 12 uniform-T samples
   at two layers; Phase B's cell B was ALL-KEPT and its `rel_l2` normalizes by
   the approximate output; the warm profile was two-layer components, not a
   full-forward speedup.

2. **Remove the discarded PV.** Done. `route_only` runs the same `_route`
   arithmetic with no `_pv` launch and no per-query allocation, and carries a
   new per-tile malformed-score flag so dropping the PV does not drop the
   guard. Proven bit-identical to the pre-refactor path on the same inputs.

3. **Actually skip dropped-tile QK.** Done. `historical_route_preqk_current_output`
   forms current QK inside the output program for retained tiles only; a
   dropped tile issues no K load, no V load and no dot. Qualified at the real
   geometries (sliding D=256 GQA 16/8, global D=512 GQA 16/2) against both the
   incumbent path and an independent FP32 reference: never further from FP32
   than the incumbent (max gap 7.0e-08), the two BF16 paths agree to 2.3e-05,
   and in-kernel counters match the exact expected program count in all 14
   cases. Materialized current-QK elements fall 33%, to exactly the
   `cached_scores` level (anchors only).

4. **Quality held.** The bounded panel scores **3/4 with a correctness vector
   identical element-wise to the contemporaneous dense control**.

## The result that decides this round

**End to end it is not faster.** Mean request wall 115.73 s against
contemporaneous dense's 40.84 s (2.83x); pooled 0.4298 vs 0.1504 s per decoder
call. It is also worse than the v6 arm it optimized (97.92 s), even though
every matched per-step measurement says the new consumer is faster than that
arm. Trajectory divergence (BF16-scale differences change argmax over
thousands of steps; `aime26/14` alone is 251 s of the 463 s total) means four
questions cannot separate a real regression from noise here.

**The measured limiting cost has moved, and it is now the selector, not the
consumer.** From the scaling sweep, attention only, at 50% drop:

| geometry (share of layers) | nk | dense | preqk | route_only | preqk+route vs dense |
|---|---|---|---|---|---|
| sliding D=256 (25/30) | 1279 (saturated) | 0.435 | 0.379 | **1.146** | **3.51x** |
| global D=512 (5/30) | 2048 | 1.786 | 0.927 | 0.789 | **0.96x** |
| global D=512 (5/30) | 8192 | 6.433 | 3.353 | 2.848 | **0.96x** |

On global layers the method now reaches parity with dense. On sliding
layers -- 25 of 30 -- `route_only` alone costs 2.6x the entire dense attention
for that layer, on every decision-refresh call. The consumer this round
rebuilt is no longer the bottleneck.

## Honest status against the scientific objective

- Useful temporal information beyond a reused bitmap: **not established.**
  What is established is that reusing the historical *support* while always
  computing current *weights* preserves quality where reusing stale weights
  destroys it. Whether the historical numerical route beats a simple frozen
  mass/bitmap or a fresh value-aware selector at matched work is the study
  that would test the claim, and it was **not** launched.
- Acceptable task quality: same 3/4 and same vector as dense on four
  development questions. Not noninferiority.
- Net E2E improvement against strong compatible dense: **no.** 2.83x slower.

## One next action

Reduce `route_only` on sliding layers, where it is 2.6x a whole dense
attention. It currently re-reads the full cached FP32 score tensor and runs a
rank-32 sketch dot per tile on every decision-refresh call at
`decision_interval=1`. The two obvious, separately testable levers are (a)
decide less often on sliding layers, and (b) stop re-reading a [Q,K] FP32
tensor to make a per-tile decision. Either changes the method's cost model and
must be qualified on its own, not bundled.

Not done and not started: the 240-request matrix, extra seeds, M2/R3, mask
unification (native_mask remains a separately named, unresolved discrepancy),
and any retuning of thresholds, `score_period`, T/beta, sampler or budget.
