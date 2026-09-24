# v6 recovery: bounded recovery complete -- quality restored, speed not yet won

## Identity / authority
- Sole execution spec: user-supplied v6, 2026-09-24. v5/HANDOFF-before-this-round
  is historical provenance only (period8, cached-final-output choices are
  starting points, not proven-optimal or verbatim requirements).
- Branch `research/numerical-qk-reuse-native-20260924`, continued from
  checkpoint `c9f7808c3d89253b29480e2adc0dbd8eb87b733c` (local HEAD == remote
  HEAD verified before any new work; tree was clean).
- New output root: `results/numerical_qk_reuse_recovery_20260924/`. Old
  `results/numerical_qk_reuse_20260924/` untouched. New remote root
  `/home/exouser/dyh/numerical_qk_reuse_recovery_20260924/code_cp1` (git
  clone of this branch). Junyu/Haowei refs unchanged, read-only, no peer edits.

## Bottom line
The frozen v5 M1/M3 0/4 collapse is diagnosed with real GPU evidence and
**reversed on the frozen four-question panel**: `routing_only_current_output`
scores **3/4, exactly matching native dense and fresh Junyu T (same missed
question)**. It does **not** yet beat dense on wall time (2.37x slower).
Full numbers: `results/numerical_qk_reuse_recovery_20260924/paper_decision.md`.

## What was done, in order
1. **Phase A1 (native-mask audit, done, kept separate).** Real
   `sdpa_attention_forward` ignores `sliding_window`; `DynamicSlidingWindowLayer`
   already caps the stored prefix. Legacy Junyu's extra query-relative window
   diverges from native only once local prefix>768 (window1024-canvas256),
   dropping up to ~25% of native-legal prefix for the last canvas query --
   real, but shared identically by T/M1/M3, so it doesn't explain M1 losing
   to T. `Attention(support='legacy_junyu_mask'|'native_mask')` added;
   default unchanged. NOT combined into this round's successor (kept as a
   separate named item, per instruction not to bundle fixes).
   `native_mask_audit.{json,md}`, `test_numerical_reuse_native_mask.py`.
2. **Phase C repair (done, active by default, measured).** Restored Junyu's
   own `Sketches` prefix-V/norm lease (not reimplemented) into the
   decision-refresh step. Proven output/phase-invariant
   (`test_numerical_reuse_prefix_lease.py`). Measured: modest ~7-8%/call
   saving (kernel-launch overhead dominates at this scale); real win is
   call-count (once/canvas instead of ~40x), ~420ms/answer -- not the
   primary fix. `prefix_lease_repair.md`, `warm_profile_report.md`.
3. **Phase A2/B (done): real diagnosis.** `native_reuse_phaseAB_diagnostic.py`
   on the real model, real untouched dense trajectory, ids 2/8, 3 canvases,
   layers 0/5, 120 instrumented calls. A2: kernel-vs-oracle on real
   activations agrees to rel_l2<=0.00135 -- **not an implementation bug**.
   B: stale final scores (A-vs-E) reach rel_l2 0.33-0.69 by age 1 and >1.0 by
   age 5-7; routing/support alone (B-vs-C) stays 0.02-0.20 at every age --
   **the collapse is stale FINAL SCORES, not routing/support.**
   `phaseAB_report.md`.
4. **`routing_only_current_output` implemented (done).** Reuse the
   stale-score-derived retained support (comparatively stable per B); always
   recompute current QK for the final softmax/PV. Real extra cost charged,
   not assumed free. `test_numerical_reuse_routing_only.py` (bitmap identical
   to `cached_scores` mode, output provably matches the Cell-C oracle
   reconstruction).
5. **Phase D panel (done).** Ids 2/8/14/20, seed 42, native adaptive,
   8192/thinkingON/EOS. Dense (3/4, 41.31s) and T (3/4, 39.88s) **reused**
   from frozen v5 (deterministic, untouched by this round). New:
   - **M1 `routing_only_current_output`: 3/4, mean wall 97.92s (2.37x
     dense), pooled calls/canvas 12.78** (down from broken M1's 40.55, near
     dense's 10.86).
   - M3 R2 `routing_only_current_output`: 2/4, worse than M1's schedule --
     kept as a real negative result.
   `phaseD_panel_report.md`, redacted receipts in `panel/`.

36/36 local tests pass under CUDA (local WSL RTX4060, a separate private
device from the remote timing host; used only for small synthetic-tensor
unit tests, never model/timing work).

## Not done / next, if this line continues
`routing_only_current_output` still runs a full discarded PV pass during its
stale-score routing call (only the bitmap is needed from it). Reducing that
is the concrete next optimization to try to convert the now-correct
call-count reduction into an actual E2E win over dense -- not started, not
promised. No 240-request expansion, R3, M2, ASR, or model change was
started; 8 of the 16 allowed new primary outputs were used (4 M1 + 4 M3).

## Remote
`exouser@149.165.151.254`, GPU idle after this session. New root/code_cp
listed above. `git@github.com:coconight01/dlm_test.git` over SSH (no HTTPS
creds in this sandbox; origin URL switched to SSH this session). Raw private
receipts (prompts/completions) remain remote-only under
`/home/exouser/dyh/numerical_qk_reuse_recovery_20260924/results/panel/`.

## Next command
None required to reproduce this state; if continuing, profile
`routing_only_current_output`'s discarded stale-score PV pass
(`experiments/numerical_qk_reuse/integration.py`, the `route = attention(...)`
call in the `plan.decision_refresh` branch) to see whether skipping its PV
output (keep only `.skipped`/`.eligible`) meaningfully reduces per-call cost.
