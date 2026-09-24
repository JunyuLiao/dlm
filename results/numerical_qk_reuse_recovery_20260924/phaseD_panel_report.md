# Phase D: bounded natural-generation panel with the selected successor

Same four frozen AIME26 IDs (2/8/14/20), seed 42, thinking ON, cap 8192,
native adaptive, EOS -- identical protocol to the frozen v5 smoke. Native
dense and fresh Junyu T are **reused as controls** from the frozen v5
results (unaffected by this round's changes: dense/T never touch
`experiments/numerical_qk_reuse`, and generation is deterministic at
temperature 0). Only the new arms below were actually re-run this round, on
the remote host, `support=legacy_junyu_mask` (same support as T; the
native-mask correction from Phase A1 is a separate, not-yet-combined item
per the instruction not to bundle a mask fix with a new output mode),
`score_refresh_period=8` (unchanged; Phase B evidence did not show a need to
shrink it), Phase C's Sketches lease repair active by default.

| Method | Correct | Mean wall (s) | Mean decoder calls | Mean canvases | Pooled calls/canvas |
|---|---|---|---|---|---|
| native dense (reused control) | 3/4 | 41.31 | 271.5 | 25.00 | 10.86 |
| fresh Junyu T (reused control) | 3/4 | 39.88 | 234.75 | 21.00 | 11.18 |
| M1 `cached_scores` (old, frozen) | 0/4 | 254.16 | 1247.0 | 30.75 | 40.55 |
| M3 R2 `cached_scores` (old, frozen) | 0/4 | 166.00 | 1135.5 | 25.75 | 44.10 |
| **M1 `routing_only_current_output` (new)** | **3/4** | **97.92** | **255.5** | **20.0** | **12.78** |
| M3 R2 `routing_only_current_output` (new) | 2/4 | 79.34 | 403.0 | 23.0 | 17.52 |

Per-question detail (new arms), redacted (no raw completion text or
extracted numeric answers -- only correctness booleans and termination
metadata, which do not disclose the gold value): `panel/quality.M1.json`,
`panel/quality.M3.json`. Frozen configs (fingerprints, source hashes, no
prompt/gold content): `panel/config.M1.json`, `panel/config.M3.json`. Full
receipts (private, contain raw completions) remain on the remote host only:
`/home/exouser/dyh/numerical_qk_reuse_recovery_20260924/results/panel/`.

| id | M1 routing_only | M3 R2 routing_only |
|---|---|---|
| aime26/2 | correct, eos, 164 calls, 77.3s | correct, eos, 148 calls, 28.9s |
| aime26/8 | correct, eos, 257 calls, 77.4s | correct, eos, 173 calls, 31.4s |
| aime26/14 | **wrong, capped@8192, 472 calls, 188.6s** | wrong, capped@8192+unparsed, 627 calls, 114.3s |
| aime26/20 | correct, eos, 129 calls, 48.4s | **wrong, capped@8192+unparsed, 664 calls, 142.7s** |

## Interpretation

**Quality is recovered.** `routing_only_current_output` with M1's schedule
(decision_interval=1) scores **3/4, identical to native dense and fresh T,
missing the exact same question (aime26/14)** that both controls also miss
by hitting the length cap. This is the clearest possible evidence that the
0/4 collapse was the stale-final-score problem diagnosed in Phase B, not
something intrinsic to reusing the routing/support decision: reusing that
decision (still derived from an up-to-7-call-old score, unchanged from
production M1) while always feeding current scores into the softmax/PV is
sufficient to restore dense-parity quality.

**Call count also recovered**, not just accuracy: pooled calls/canvas
dropped from the broken M1's 40.55 to 12.78 -- within range of dense's 10.86
and T's 11.18, versus the frozen v5 evidence's characterization of the old
M1/M3 canvases as "stalled low-confidence trajectories." Restoring current
scores for the final softmax removed the stall, not just the wrong answers.

**Wall time is not yet a net win.** 97.92s mean is 2.37x dense's 41.31s and
2.46x T's 39.88s -- consistent with the Phase C profile's finding that
`routing_only_current_output` pays ~1.8x M1's own per-call attention cost
(fresh QK + a real extra PV pass on top of the routing-decision kernel) even
though it needs far fewer calls than the broken M1 did. The paper's stated
bar of "net E2E improvement against compatible strong dense" is **not met by
this successor as configured**; what Phase D establishes is that the
correctness collapse is understood, diagnosed with real evidence, and
reversible -- not that this specific successor is the final answer for
speed.

**M3 R2 (`decision_interval=2`) is worse, not better: 2/4, and it loses a
question M1 routing_only got right (aime26/20).** Holding the routing
decision for 2 calls instead of 1 -- even with scores always fresh for the
softmax -- lets the retained support drift further from what a fresh
decision would choose, on top of the shared score_refresh_period=8 for the
decision cache itself. This is a real, reportable negative result, kept
per the instruction to finish the bounded panel even when it loses.

## What this does and does not license

This isolates a specific configuration (stale-score M1 restricted to
routing-only reuse) as quality-viable but not yet speed-viable. It does not
establish population-level noninferiority (four questions, one seed), does
not justify launching a 240-request expansion, and does not justify
combining the still-separate native-mask correction into this successor
without its own named checkpoint. The honest next question, if this line
continues, is whether the per-call cost of `routing_only_current_output`
can be reduced (e.g. by skipping the discarded stale-score routing pass's
own PV, which the current implementation still runs) enough to turn the
now-correct call-count reduction into an actual E2E win over dense -- that
is a distinct, not-yet-started optimization, not a claim made here.
