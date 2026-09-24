# v6 recovery: diagnose the negative numerical-QK smoke

## Identity / authority
- Sole execution spec: user-supplied v6, 2026-09-24. v5/HANDOFF-before-this-commit
  is historical provenance only, not a requirement (period8, cached-final-output
  choices are starting points, not proven-optimal).
- Branch `research/numerical-qk-reuse-native-20260924`. Continuing from checkpoint
  `c9f7808c3d89253b29480e2adc0dbd8eb87b733c` (verified: local HEAD == remote HEAD
  before any new work; working tree was clean).
- New output root: `results/numerical_qk_reuse_recovery_20260924/`. Old
  `results/numerical_qk_reuse_20260924/` is untouched historical provenance.
- Junyu base `053441c6c6741ada6728dfc97d8faa6ea2be72aa`, Haowei
  `b23f969a3c52417ba84a999ff7a31e3dd00bb697`: unchanged, read-only, no peer edits.

## Done this round
- **Phase A1 (native-mask contract + audit, DONE).** Read the actual installed
  `sdpa_attention_forward` (Transformers 5.11: ignores the `sliding_window`
  kwarg entirely) and `DynamicSlidingWindowLayer.update` (already caps a local
  layer's stored encoder prefix at `sliding_window-1`). Native ground truth =
  every key in the supplied (already-capped) prefix plus full canvas, no
  further per-query narrowing; DiffusionGemma decoder attention is
  bidirectional (`is_causal=False`). Legacy Junyu additionally applies a
  query-relative window bound. Audited with the real
  `_attention_validity` function at the real `sliding_window=1024`,
  `canvas=256`, `layer_types` (25/30 local, 5/30 global) from the executed
  model's own config: **legacy==native whenever local prefix<=768; beyond
  that, legacy silently drops up to ~25% of the native-legal prefix for the
  last canvas query.** This is real but does NOT alone explain M1 losing to
  T, which shares the same legacy support (T scores 3/4).
  Added `Attention(..., support='legacy_junyu_mask'|'native_mask')` in
  `experiments/numerical_qk_reuse/integration.py` (default preserves prior
  behavior exactly); threaded through `runner.py --support` and
  `native_reuse_smoke_batch.py --support`. Audit:
  `results/numerical_qk_reuse_recovery_20260924/native_mask_audit.{json,md}`.
  Test: `tests/test_numerical_reuse_native_mask.py`.
- **Phase C repair, code + invariance proof (DONE; NOT YET GPU-justified).**
  Source finding: the decision-refresh step recomputed the full V projection
  and RMS norm (prefix+canvas) on every M1 call. Restored Junyu's own
  `Sketches` lease (`experiments/value_direction_hopper/integration.py`,
  version/epoch/shape-keyed, reprojects only the unaligned tail once the
  encoder-owned prefix source is unchanged) into the decision-refresh block
  in place of the inline recompute -- reusing the existing class, not a
  reimplementation. `crop` is always 0 in the executed model regime (implied
  by the A1 audit), so passing the full V tensor to `Sketches.get` is exact.
  Proved output/phase invariance with 2 new GPU tests
  (`tests/test_numerical_reuse_prefix_lease.py`): leased output equals
  brute-force recompute across an unaligned (65-token) prefix boundary and a
  changing canvas, and a commit (new encoder forward) invalidates the lease
  even with bit-identical content. Details:
  `results/numerical_qk_reuse_recovery_20260924/prefix_lease_repair.md`.
- 34/34 targeted local tests pass under CUDA (local WSL RTX4060 laptop GPU --
  a separate, private device from the remote timing host; used only for these
  small synthetic-tensor unit tests, never for model/timing work).

## Not run yet (next steps, in order)
1. Phase A2: zero-temporal-approximation checks (all-kept current-scores vs
   high-precision reference; M1 `score_period=1` vs the Torch oracle in
   `reference.py`) on REAL captured layer states from ids 2 and 8.
2. Phase B: same-state/same-support 5-cell (A-E) diagnostic, real cache ages
   0/1/2/7, on a short (<=2 canvas) real dense-trajectory capture for ids 2
   and 8. Needs a new capture/diagnostic script against the real model on the
   remote host (not yet written).
3. Phase C: warm PROFILE (separate process) -- dense/T/full-refresh/M1-reuse/
   M3 decomposed forward latency, using replayed real captured shapes.
4. Phase D: select ONE successor from A-C evidence; bounded 4-question
   (ids 2/8/14/20, seed42) natural generation; reuse the existing frozen
   native_dense/fresh_junyu_T outputs as controls (unaffected by this repair,
   deterministic at temperature=0) rather than rerunning them.

## Remote
- `exouser@149.165.151.254`, GPU idle, no own PIDs, ~60GiB free, confirmed
  reachable this session. Old root `/home/exouser/dyh/numerical_qk_reuse_native_20260924`
  is read-only provenance; new work goes in
  `/home/exouser/dyh/numerical_qk_reuse_recovery_20260924` with its own
  immutable `code_cp`. Not yet created -- next action.
- `git@github.com:coconight01/dlm_test.git` over SSH works from this sandbox
  (no HTTPS credentials available here); origin remote URL was switched to
  SSH for this session.

## Next command
```
ssh exouser@149.165.151.254 'mkdir -p /home/exouser/dyh/numerical_qk_reuse_recovery_20260924'
# then sync this commit's source into a new immutable code_cp1 and write the
# Phase A2/B capture script under experiments/numerical_qk_reuse/ before any
# GPU job.
```
