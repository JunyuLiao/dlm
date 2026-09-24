# v6 recovery: decision summary

> **CORRECTED.** See `../numerical_qk_preqk_execution_20260924/corrections.md`.
> Fixed inline: the new M1 shares a correctness vector with native dense
> only (not with fresh T); the A2 kernel check covered 12 uniform-T samples
> at two layers and does not exclude every bug; the Phase B `rel_l2`
> normalizes by the approximate output, so values >1 do not mean "error
> bigger than the fresh signal"; the "7-8%" is a component measurement, not
> a full-forward speedup.

**Diagnosis (real, GPU-verified, within the sampled scope):** the frozen v5
M1/M3 0/4 collapse is attributed to reusing stale final attention scores for
up to 7 calls (`score_refresh_period=8`). No kernel/oracle disagreement
appeared in the 12 sampled real-activation cases (two layers, uniform
sensitivity, rel_l2<=0.00135) -- that narrows but does not exclude an
implementation defect. With scores held current, the pruning-vs-all-kept
error stayed 0.02-0.20 at every age, while the stale-score cells were far
larger (0.33-1.07+, normalized by the approximate output). Full evidence and
exact cell definitions: `phaseAB_report.md`.

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
- M1 `routing_only_current_output`: **same observed 3/4 count as native
  dense and fresh Junyu T**; element-wise its vector `[T,T,F,T]` is
  identical to **dense's only** and differs from T's `[T,T,T,F]`. Pooled
  calls/canvas 12.78 (down from the broken M1's 40.55, close to dense's
  10.86); mean recorded wall 97.92s vs dense's historical 41.31s -- a
  cross-session observation, **not yet a measured net E2E win**.
- M3 R2 `routing_only_current_output`: 2/4, worse than M1's schedule; kept
  as a real negative result, not discarded.

Full table and per-question detail: `phaseD_panel_report.md`.

**Bottom line:** the negative result is diagnosed and the collapse is
reversible with a real, evidence-selected fix that restores the same 3/4
count as dense on these four development questions (vector identical to
dense, different from T). It is not yet a speed win. No noninferiority
claim is made from four questions/one seed; no 240-request expansion was
started.

**Next action:** if this line continues, the next question is whether
`routing_only_current_output`'s per-call cost can be reduced (its
stale-score routing pass still runs a full discarded PV) enough to convert
the now-correct call-count reduction into an actual wall-clock win over
dense -- a distinct, unscoped optimization, not started here.
