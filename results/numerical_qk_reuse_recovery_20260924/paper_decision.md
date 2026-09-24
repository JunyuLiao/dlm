# v6 recovery: decision summary

**Diagnosis (real, GPU-verified):** the frozen v5 M1/M3 0/4 collapse was
caused by reusing stale final attention scores for up to 7 calls
(`score_refresh_period=8`), not by a kernel/implementation defect (12
real-activation kernel-vs-oracle samples agreed to rel_l2<=0.00135) and not
primarily by the routing/support decision (holding scores current, the
support-pruning-alone error stayed 0.02-0.20 at every age vs 0.33-1.07+ for
stale scores alone). Full evidence:
`results/numerical_qk_reuse_recovery_20260924/phaseAB_report.md`.

**Repair applied and measured:**
1. `native_mask` support option + audit (`native_mask_audit.md`) -- a real,
   quantified, but separate correctness item (legacy==native only for local
   prefix<=768); not combined into this round's successor.
2. Restored Junyu's `Sketches` prefix-V/norm lease (real, invariance-proven,
   modest ~7-8%/call saving; see `warm_profile_report.md`).
3. **`routing_only_current_output`**: reuse the stale-score-derived retained
   support, always recompute current QK for the final softmax/PV. Selected
   by the Phase B evidence, not assumed a priori.

**Result on the frozen four-question panel (seed 42, native adaptive,
8192/thinkingON/EOS):**
- M1 `routing_only_current_output`: **3/4 correct, matching native dense and
  fresh Junyu T exactly (same missed question)**; pooled calls/canvas 12.78
  (down from the broken M1's 40.55, close to dense's 10.86); mean wall
  97.92s, **2.37x dense's 41.31s -- not yet a net E2E win**.
- M3 R2 `routing_only_current_output`: 2/4, worse than M1's schedule; kept
  as a real negative result, not discarded.

Full table and per-question detail: `phaseD_panel_report.md`.

**Bottom line:** the negative result is diagnosed and the collapse is
reversible with a real, evidence-selected fix restoring dense-parity
quality. It is not yet a speed win. No noninferiority claim is made from
four questions/one seed; no 240-request expansion was started.

**Next action:** if this line continues, the next question is whether
`routing_only_current_output`'s per-call cost can be reduced (its
stale-score routing pass still runs a full discarded PV) enough to convert
the now-correct call-count reduction into an actual wall-clock win over
dense -- a distinct, unscoped optimization, not started here.
