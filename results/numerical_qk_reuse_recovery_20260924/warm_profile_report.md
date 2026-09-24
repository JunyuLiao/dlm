# Phase C: warm decomposed forward-cost PROFILE

`scripts/native_reuse_warm_profile.py`, real captured state from a live
generation on `aime26/2` (canvas-1 call, prefix=365, nq=256, nk=621),
layers 0 (local) and 5 (global), 20 warmup + 100 timed reps per component
(CUDA events, separate process from any generation timing). Raw:
`warm_profile.json`.

| component (ms/call) | layer0 (local) | layer5 (global) |
|---|---|---|
| native_dense (baseline, single SDPA call) | 0.046 | 0.091 |
| score_producer (the only current-QK matmul) | 0.149 | 0.094 |
| v_lease_cold (brand-new source every call) | 0.172 | 0.173 |
| v_lease_warm (same source, repaired lease) | 0.160 | 0.161 |
| decision_and_pv_fresh (`_route`+`_pv`) | 0.281 | 0.310 |
| held_pv_only (`_held_eligible`+`_pv`) | 0.072 | 0.074 |

## Findings

1. **Native dense's single fused SDPA call is cheaper than any one component
   of the numerical pipeline alone**, let alone their sum. `score_producer`
   by itself (0.09-0.15ms) already exceeds `native_dense` (0.05-0.09ms).
   This quantifies real *execution overhead*, independent of correctness:
   the multi-kernel numerical architecture (separate matmul-based score
   producer, Triton route kernel, Triton PV kernel) has materially more
   per-call launch/dispatch cost than one fused native SDPA call, at this
   problem size. This matches the frozen v5 evidence's "amortized-cost
   factor 1.34x" for M1 even setting aside the call-count blowup.

2. **The Phase C repair's per-call saving is real but modest (~7-8%,
   ~0.01ms/call/layer), not dramatic**, at this prefix scale (365 tokens):
   `v_lease_cold` vs `v_lease_warm` differ by only ~0.01-0.012ms per layer
   per call. Kernel-launch overhead dominates the matmul cost at this size,
   so the win is mostly about *call count*, not per-call latency: production
   M1 previously paid the full-reproject cost on every decision-refresh call
   (~40/canvas per the frozen v5 evidence); after the repair it pays it once
   per canvas (at the first call after a commit) and the cheaper warm cost on
   the rest. Rough aggregate: ~0.012ms x 30 layers x ~39 non-first calls x
   ~30 canvases =~ 420ms saved per full-length answer -- real, but small
   next to M1's ~254s mean wall time. Do not oversell this as the primary
   fix; it is engineering hygiene, not the correctness fix.

3. **`held_pv_only` is ~4x cheaper than `decision_and_pv_fresh`** (0.07ms vs
   0.28-0.31ms): M3's held-decision reuse (and `routing_only_current_output`'s
   steady-state call) genuinely skips real work when a bitmap can be reused,
   unlike the V-lease case.

## What this means for `routing_only_current_output`'s real cost

Per reuse call (score age>0, ~7 of every 8 calls): production M1 pays
`decision_and_pv_fresh` only (~0.28-0.31ms/layer; `score_producer` is
correctly skipped -- that is M1's entire benefit). `routing_only_current_output`
pays `decision_and_pv_fresh` (to derive the bitmap from the stale score,
discarded) **plus** `score_producer` **plus** `held_pv_only` (the real,
current-score output) = ~0.50-0.53ms/layer -- roughly **1.8x** M1's
per-call attention cost on reuse calls, and roughly **6-7x** native dense's
per-call attention cost (summed over 30 layers this is attention-only; FFN
and other shared work are identical across arms and not measured here).

This is a real, charged cost, not assumed away. Whether `routing_only_current_output`
is net-better than dense end-to-end depends entirely on whether restoring
correct (non-collapsed) quality lets the model reach a valid stop in a
call count closer to dense's (~10-11 calls/canvas) instead of M1's inflated
~40+ -- that is a generation-level question the per-call profile above
cannot answer, which is exactly why Phase D runs the real bounded panel
rather than projecting from this profile alone.
