# Missing measurements

What is not established by any report under
`results/m3_numeric_trajectory_bridge_20260927/`, stated precisely so it is
not mistaken for a negative result.

## M2 never implemented / run

`results/m3_numeric_trajectory_bridge_20260927/design_freeze.md` line 77,
verbatim: "No R/A/tau/earlyphase/geometry/Pprecision sweep, **M2
deferred**." No arm named M2 appears anywhere in
`generation_records_v21_v25b/index.csv` (the recorded arms are
`D_native, D_matched_legacy, D_matched_new_numeric, T_scope,
M1_boot_aligned16, M1_native_bootstrap2_observe1, M3_R3_A8_incumbent,
M3_boot_aligned16, M3_boot_logical, M3_native_bootstrap2_observe1,
B_boot_aligned16, B_native_bootstrap2_observe1`). There is a `test_v26_m2_pool.py`
in `tests/`, but it tests an unrelated "M2 (pooled mu)" geometry-diagnostic
kernel (`experiments/numerical_qk_reuse/geometry_diagnostic._select`,
`pool=True`), not a method arm — it does not constitute an M2 method run.

## A16 / R6 / threshold sweep never measured

Every raw record's counters checked by this v26 study has `score_period: 8`
and `decision_interval` in `{1, 3, 8}` (M1/M3/matched-B respectively) —
confirmed across all 139 validated bootstrap-instrumented records in
`v26_clock_opportunity.json` (`validation.all_match: true`, and separately
by direct inspection of the raw counters). No record anywhere in the v21-v25b
generation set has `score_period: 16`, and no record has
`decision_interval: 6`. `design_freeze.md` explicitly deferred an
"R/A/tau/earlyphase/geometry/Pprecision sweep." **Every A16 or R6 row in
`v26_clock_opportunity.csv` is therefore this study's own estimate on
recorded canvas lengths, not a measurement** — it says only how many
B0/BO/A/D/H layer-calls that policy *would* spend on the canvas lengths
actually observed under A8/R{1,3,8}; it cannot say what the canvas lengths
themselves would have been under a different schedule, since decision
timing plausibly changes what gets generated and how long a canvas runs.

## No KDIV2 measurement

`v25b_memory_and_gate_audit.md` CP2, verbatim: "one warm repeat per cell and
one odd-K and one control question. This is a mechanism check, not a
population benchmark; **the KDIV-2 class was not measured**." The only two
K classes bridged are odd-K (`…067c5456`, K mod 16 = 7) and KDIV8
(`…067c4480`); no KDIV2 (K mod 16 = 2, 4, 6, ... within the "even but not a
clean multiple of a larger power of two" band) case was run.

## No attribution trace of the remaining time

`v23_corrections.md` point 4, verbatim: "**The all-kept arm does not settle
attribution.** Its extra calls (v20 3009 vs native 2360) show that a
no-pruning numerical path also perturbs trajectories. That does NOT
identify what fraction of sparse M3's extra work comes from numerics versus
pruning/history." No report decomposes bootstrap M3's residual cost gap
(vs. native or vs. matched B) into a numerics component and a
pruning/history component; `v24_direct_cost.md` gives isolated per-piece
CUDA-event timings ("not additive to a forward price") but does not
attribute the observed whole-request or whole-forward deltas to those
pieces.

## RULER/AIME breadth is small

- RULER: 13 task-balanced RULER-4K inputs total (`v24_direct_cost.md` §4),
  "RULER requests are about 4 calls, so they carry no timing signal."
- AIME: "Complete blocks used: 8" in the v23 panel
  (`bootstrap6_report.md`); the v25 pilot covers "AIME 1 × 2" (one question,
  two seeds), and that one AIME question hit the 8,192-token cap on every
  arm ("every arm hit the 8,192-token cap on `aime26/14`",
  `v25_route_storage_report.md`), so its quality signal is uninformative by
  construction.
- Neither dataset's breadth supports a per-dataset population claim; every
  quality comparison in the cited reports is explicitly labeled
  "descriptive only" or "not noninferiority evidence."

## Host/seed confounding in v23

`v24_errata_and_audit.md`, verbatim: "Because host = seed here (seed 101 →
149.165.151.254, seed 202 → 149.165.159.64), **host and seed effects cannot
be separated**. Pairing within a cell is still same-GPU and fair. Future
blocks should counterbalance host by question × seed; existing first runs
are not relocated." The same file shows the host-stratified ratios can go
in opposite directions on the two hosts (LB M3-boot/native 0.863 on mpk vs
1.109 on dllm), which is exactly the symptom of this confound. v25/v25b's
bridge is explicitly host-counterbalanced (`v25b_memory_and_gate_audit.md`
CP2: "Host is counterbalanced, and both arms of a cell run on the same
GPU."), but that only covers the two aligned16-vs-logical bridge questions,
not the v23 panel.

## No TBT (time-between-tokens) boundary

`v24_direct_cost.md` §1 defines exactly two timing boundaries:
`model_forward` (`full model.forward(...).logits`) and `denoising_step`
(native step including T/sampler/stop). No report defines or measures a
token-emission or time-between-tokens boundary; the only per-call phase
timings available are these two, decomposed by B0/BO/A/D/H phase at the
call level, not at the emitted-token level.
