# v8: exact prefix-block-summary selector — decisions preserved, method not faster

## Identity / authority
- Sole spec: user-supplied v8. Continued from reviewed HEAD
  `e1f3c743513b54d1d560506f334eac0ab3045c20` (verified local == remote, clean
  tree). Branch `research/numerical-qk-reuse-native-20260924`.
- Fresh output root `results/numerical_qk_prefix_summary_20260924/`. Older
  roots untouched except inline `CORRECTED (v8)` notes. Peer refs read-only.
- Execution identity lives in per-run config receipts (their own fingerprints
  and source hashes), NOT in any inherited SHA in STATE.json.

## Corrections landed first (CP1, CPU only)
`results/numerical_qk_prefix_summary_20260924/corrections.md`:
- The v7 panel's "counters across the four requests" was **aime26/14 alone**.
  Correct sums: 32310 attention calls, 5100 score refreshes, 27210 pre-QK
  consumer calls, 3.2710819840e10 materialized current-QK elements
  ("materialized" excludes retained dots inside the fused consumer).
- **Blaming the slowdown on /14 is withdrawn.** Excluding /14, the other three
  requests are 211.447 s vs dense 99.178 s (**2.132x**) on **31% fewer**
  forwards (444 vs 648). It is a real per-forward cost.
- The scaling sweep is synthetic component stress data (random Z/ref, uniform
  T, random consumer bitmap, an arithmetic sum) — not "complete method 3.51x".

Profiling-harness repairs (all pinned by `tests/test_preqk_profile_contracts.py`):
A single armed step now feeds both captures with an asserted prefix (v7's
"long context" runs were actually short); B production causal T is installed in
every router (v7 profiled uniform T and called it the T path); C M3_held walks
to a genuinely held step and asserts it; D RNG restored, replay inputs digested
twice; E counters reported as per-repetition deltas with telemetry cleared.

## The optimization (CP2)
For a KV tile lying WHOLLY inside the immutable prefix, `z = logsumexp` and
`mu = sum w * (V_prefix R)` depend only on the frozen cached scores and frozen
prefix projected V, so they are constant between real score anchors. `_route`
now optionally stores them at the anchor and loads them afterwards — the reused
values are that kernel's OWN FP32 outputs, so reuse is exact by construction.
**Not cached:** alpha, risk, retained log mass, projected accumulator, bitmap;
all recomputed every step from live T and live full-V RMS, same scan order.
`Attention(selector='legacy_recompute'|'prefix_block_summary',
selector_layers='local')`, default unchanged, LOCAL layers only.

## Results (CP3)
- **Decisions preserved, exactly.** Real saturated states (local prefix 1023
  asserted, global 1923) with production T (spread 2.625): bitmaps/eligibility/
  malformed flags bit-identical, 0 disagreeing tiles. Same-generation-twice:
  tokens identical. Two full 8192-budget smoke requests: completion tokens
  **identical token-for-token** to the v7 receipts, same calls (102, 181) and
  canvases (12, 15), selector active (2075/3750 hits). 12 CUDA + 5 CPU cases,
  including a poisoning probe that proves the avoided reads are real.
- **Work avoided:** 75% of local and 86% of global KV tiles skip their score
  re-read, sketch read and tf32x3 dot per decision. Cost: 8.23 MB per local
  layer, **205.8 MB resident**, on top of the retained score cache — total
  memory increases.
- **Per call faster, per step not.** LOCAL `Attention.__call__` 1.392 -> 0.990 ms
  (**-29%**); complete denoising step **206.20 -> 206.23 ms (unchanged)**.
  Dense 183.24, cached_scores 205.07, routing_only 214.96.
- **Natural request time: not established.** Per the v8 stopping rule the
  four-question panels were NOT run, because complete-call profiling shows no
  net reduction. The two mandated smoke requests ran with `--diagnostic` ON, so
  their walls are not comparable timing.
- **Section 7 done:** `preqk_support_cells.py` ran (52 real cells, one common
  mask, real causal T). Age 0 calibrates exactly (Jaccard 1.000, errors
  0.0000). Beyond it, **stale weights cost ~an order of magnitude more than the
  historical support choice** (`O_hh/O_hf` 0.37-0.95 vs `O_hf/O_ff` 0.06-0.19),
  now without v6's all-kept/uniform-T confound. Support is less stable than
  previously implied (local Jaccard ~0.62 at age 1).

## Unexplained (do not paper over)
1. -0.402 ms x 25 local layers predicts ~-10 ms per step; the step did not move
   (minima 204.39 vs 204.36). Likely the tight per-call loop keeps one layer's
   buffers in L2 while a real step streams 30 — hypothesis, not measurement.
2. v7's ~430 ms per real forward vs ~206 ms replayed is still unresolved, and
   v8 established it is not an /14 artifact. Neither harness has located it.

## Not done / not started
240-request matrix, extra seeds, M2, another M3/R2 quality trade-off, mask
unification (`native_mask` remains a separately named discrepancy), any
retuning of thresholds/`score_period`/T/sampler/budget, backend migration, and
the frozen-mass vs fresh-value-aware comparison study. New primary outputs this
round: **2** (the mandated smoke requests); the panel allowance was left unspent
rather than used to fill rows.

## Next command
Locate the request-vs-replay per-forward gap before optimizing anything else:
it is the only lever large enough to matter (2.13x on the other three
requests), and both the kernel-level and step-level work above leave it
untouched. A bounded trace of one real early and one real late step,
partitioning GPU active work, launch/CPU gaps and allocator behaviour, is the
v8 section 4F instrument for it.
