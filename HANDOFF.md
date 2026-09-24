# v7: real pre-QK execution built and qualified; quality held, E2E still slower

## Identity / authority
- Sole spec: user-supplied v7. Continued from reviewed HEAD
  `237e671ebe4e3902d8350ec939c66aaf75ed059d` (verified local == remote, clean
  tree, before any work). Branch `research/numerical-qk-reuse-native-20260924`.
- New output root `results/numerical_qk_preqk_execution_20260924/`. Older roots
  are untouched except for inline CORRECTED notes (see below). Junyu/Haowei
  refs unchanged, read-only. Remote code_cp1 is a git clone pulled to each
  pushed commit; raw private receipts stay remote-only.
- **STATE.json no longer carries the inherited v5 `executed_model_source_sha`.**
  Per-run config receipts (`panel/config.*.json`, each with its own fingerprint
  and source hashes) are the authority for what actually ran.

## Corrections landed first (CPU-only checkpoint, records untouched)
`results/numerical_qk_preqk_execution_20260924/corrections.md`:
- **The v6 "same missed question" claim was false for fresh T.** Dense
  `[T,T,F,T]`, fresh T `[T,T,T,F]` (T answers /14 at 7276 tokens uncapped and
  misses /20 at 2777), new M1 `[T,T,F,T]`. Same 3/4 count, different questions;
  M1 shares a vector with **dense only**. `scripts/preqk_correctness_vectors.py`
  regenerates this from records and `tests/test_preqk_correctness_vectors.py`
  makes the wrong phrasing raise.
- `temperature=0` is the native 0.8->0.4 annealing sentinel, not greedy
  decoding; frozen dense/T receipts are labelled development-quality
  references, their walls are not matched timing controls.
- The A2 kernel check was 12 uniform-T samples at two layers: narrows, does not
  exclude, a defect. Phase B's cell B was ALL-KEPT (so B-vs-C is pruning at
  all), ran with uniform T, and its `rel_l2` normalizes by the approximate
  output, so >1 does not mean "bigger than the fresh signal".
- The warm profile was two-layer components, not a full-forward speedup.

## Built this round
1. **`route_only`** -- same `_route` arithmetic, no `_pv` launch, no discarded
   [B,H,Q,D] output. A new per-tile malformed-score flag keeps the guard the
   removed PV used to provide. Bit-identical to the pre-refactor path
   (`tests/test_numerical_reuse_route_only.py`, 25 cases, incl. a launch spy).
   Anchors keep the fused path, so the same QK is never observed twice.
2. **`historical_route_preqk_current_output`** -- current QK/PV formed inside
   the output program for retained tiles only; a dropped tile issues no K load,
   no V load, no dot. GQA by indexing, explicit strides (the model hands us
   transposed views), reference BF16 rounding path matched.
   Qualified at real geometries (sliding D=256 GQA 16/8, global D=512 GQA 16/2)
   in 14 cases: **never further from an independent FP32 reference than the
   incumbent (max gap 7.0e-08)**, BF16 paths agree to 2.3e-05, in-kernel
   counters exact everywhere. Materialized current-QK falls 33%, to the
   `cached_scores` (anchors-only) level.

## Measured outcome
- **Quality held: 3/4, vector `[T,T,F,T]`, identical element-wise to the
  CONTEMPORANEOUS dense control.** Dense also reproduced its frozen v5 call
  counts (146/132/438/370) and canvases exactly, walls within ~5%.
- **Full forward (matched state, capture B):** ordinary step preqk 198.74 ms vs
  routing_only 207.16 ms vs the `cached_scores` floor 197.66 ms vs dense
  174.40 ms. The earlier "local-layer regression" from capture A was a Triton
  specialization artifact (1527 ms first observation) and is withdrawn.
- **End to end: NOT faster.** Mean wall 115.73 s vs dense 40.84 s (2.83x);
  pooled 0.4298 vs 0.1504 s/call; also worse than the v6 arm (97.92 s).
  Confounded by trajectory divergence (BF16-scale differences move argmax;
  `aime26/14` alone is 251 s of 463 s). Four questions cannot separate that.
- **Limiting cost has moved to the selector.** Scaling sweep, attention only,
  50% dropped: at the saturated sliding geometry (nk=1279, **25 of 30 layers**)
  `route_only` alone is **1.146 ms vs 0.435 ms for the whole dense attention**
  of that layer, giving `preqk+route` **3.51x dense**; global layers at >=2048
  keys reach **0.90-0.96x dense**. The consumer rebuilt this round is no longer
  the bottleneck.

Reports: `decision.md`, `panel_report.md`, `preqk_qualification.md`,
`full_forward_profile.md`, `scaling_sweep.json`.

## Not done
- `scripts/preqk_support_cells.py` (K_f/K_h x current/cached cells under one
  common mask with real nonuniform causal T, storing both norms and support
  overlap) is committed and syntax-checked but **not yet run**.
- `native_mask` remains a separately named, unresolved support discrepancy,
  deliberately not bundled with this arm.
- Not started: 240-request matrix, extra seeds, M2/R3, any retuning of
  thresholds / `score_period` / T / sampler / budget, and the
  historical-route vs frozen-bitmap vs fresh-value-aware comparison study.
- 8 of 8 allowed new full requests used (4 preqk M1 + 4 contemporaneous dense).

## Next command
Reduce `route_only` on sliding layers. It re-reads the full cached FP32 [Q,K]
score tensor and runs a rank-32 sketch dot per tile on **every**
decision-refresh call at `decision_interval=1`. Two separately testable levers:
decide less often on sliding layers, or stop re-reading a [Q,K] FP32 tensor to
make a per-tile decision. Qualify either on its own; do not bundle with a mask
change or a threshold change.
