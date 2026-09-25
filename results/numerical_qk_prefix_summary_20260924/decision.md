# v8 decision: the optimization is exact, and it did not make the method faster

## The five questions this round had to answer

**1. Did summary reuse preserve decisions?** **Yes, exactly.**
- Real captured states on the remote GPU with the production causal T in
  effect (spread 2.625), one LOCAL layer at a *saturated* 1023-token prefix
  (asserted, not assumed) and one GLOBAL layer at 1923: skip bitmaps,
  eligibility and malformed flags **bit-identical**, 0 disagreeing tiles.
- Same bounded generation run twice, once per selector: **tokens identical,
  no divergence**, same call counts.
- Two complete-answer smoke requests at the full 8192 budget: completion
  tokens **identical token-for-token** to the v7 legacy-selector receipts,
  with identical decoder calls (102, 181) and canvases (12, 15), while the
  selector was genuinely engaged (2075 and 3750 hits).
- 12 CUDA qualification cases plus the 5 CPU fixtures. The decisive probe:
  poisoning the cached scores *and* the projected V inside the summarized
  range leaves the optimized result unchanged while provably moving the
  legacy path; poisoning beyond that range moves both.

This is exactness by construction, not by tolerance: the reused z/mu are the
anchor kernel's own FP32 outputs.

**2. Which real work was avoided?** Per wholly-immutable prefix tile per
decision: the re-read of that tile's 128x64 FP32 scores and its 64x32 sketch,
and the 128x64 by 64x32 tf32x3 dot. At the measured saturated state that is
**15/20 local tiles (75%)** and **30/35 global tiles (86%)**. Cost: 33 FP32
scalars per row per prefix tile, **8.23 MB per local layer**, **205.8 MB
resident** across the model, on top of the retained score cache — so total
resident memory **increases**.

**3. Was the COMPLETE measured path faster?** **Per call yes, per step no.**

| measurement (remote GPU, real saturated state) | legacy selector | summary selector |
|---|---|---|
| `Attention.__call__`, LOCAL layer (D=256, nk=1279) | 1.392 ms | **0.990 ms (-29%)** |
| `Attention.__call__`, GLOBAL layer (summary off by design) | 2.097 ms | 1.995 ms |
| complete denoising step (30 layers + sampler) | 206.20 ms | **206.23 ms (no change)** |

For reference in the same run: native dense 183.24 ms, `cached_scores` 205.07,
`routing_only_current_output` 214.96.

**4. Did natural request time improve?** **Not established, and the panel
queue was deliberately stopped.** The v8 rule is that the four-question panels
run only if complete-call profiling shows a reproducible net reduction. It
does not. The two smoke requests were run (as mandated, to prove wiring) with
`--diagnostic` ON, so their wall times include per-step diagnostic
quantile/entropy work and are **not** usable as timing. No panel was run; the
allowance was not spent to fill rows.

**5. What remains unexplained?** Two things, both stated rather than papered
over:
- A measured −0.402 ms per local-layer call across 25 local layers implies
  roughly −10 ms per step, and the complete step did not move (206.20 →
  206.23, with tight minima 204.39/204.36). The most likely reading is that
  the tight per-call loop keeps one layer's buffers resident in L2 and so
  flatters whichever path reads fewer bytes, while a real step streams 30
  different layers; but that is a hypothesis, not a measurement.
- The v7 gap between ~206 ms replayed steps and ~430 ms per forward in real
  requests is still unresolved. v8's own correction showed it is not an
  aime26/14 artifact (the other three requests are 2.132x slower on 31% fewer
  forwards), so it is a genuine per-forward cost that neither the replay
  harness nor this optimization has located.

## Where this leaves the method

The v7 scaling sweep suggested the selector was the dominant remaining cost.
v8 flagged that inference as synthetic component stress data, and that caution
was correct: on the real model the numerical arms sit only ~23 ms above dense
per step (206 vs 183), so there was never ~65 ms of selector time available to
remove at the whole-step level, whatever the isolated kernel ratios showed.
Removing ~29% of the local-layer attention call was real and is now permanent
and free of any accuracy cost — it simply is not where the request-level gap
lives.

Per the v8 stopping rule: one optimization did not yield a useful systems
improvement. No refresh-frequency, threshold, mask or sampler substitute was
attempted in its place.

## Scientific follow-through (section 7)

`preqk_support_cells.py` ran for the first time, 52 real cells under one common
mask with real causal T. At score age 0 the fresh and historical selectors
agree exactly (Jaccard 1.000, all error cells 0.0000), calibrating the
comparison. Beyond age 0, **stale final weights cost roughly an order of
magnitude more than the historical support choice** (`O_hh/O_hf` 0.37-0.53
local and 0.77-0.95 global, versus `O_hf/O_ff` 0.06-0.19 local and 0.10-0.13
global). This re-establishes the v6 conclusion with the v6 confound removed
(its cell B was all-kept and its T uniform). Support agreement is *not* as
stable as previously implied: local Jaccard drops to ~0.62 at age 1. Details
and one unexpected observation (the historical selector sometimes retains more
current mass than the fresh one on local layers) in `support_cells.md`.

This does not establish novel selection. Comparing numerical current-V routing
against a frozen mass/bitmap and a fresh value-aware selector under common
execution and measured quality/time remains the study that would, and it was
not launched.

---
**CORRECTED (v9, 2026-09-25) — appended; the text above is preserved as recorded.**
- The full-step row labeled `native_dense` (183.24 ms) ran `_install_dense` with
  `attention_override=None`, which the BLASST registry dispatcher sends to
  `dense_eager_attention_forward`, NOT the untouched native SDPA function. It is a
  same-legal-mask eager dense row. v9's dispatch spy proves this (30/30 calls in
  `dense_eager`) — see `results/numerical_qk_request_timing_20260924/measurement_contract_and_dispatch.md`.
- v8's step replay restored RNG once per series, shared the native sampler /
  stopping criterion / capture-time T observer across repetitions, and copied
  fixtures inside the timer. Superseded by `scripts/v9_step_replay_profile.py`.
- The inference "206 − 183 ≈ 23 ms, so there was never room for a ~10 ms gain"
  is WITHDRAWN: the 183 ms baseline is mislabeled, the two contexts need not add,
  and replay-state/launch-overlap effects were unresolved. The L2-residency
  explanation remains an untested hypothesis. The raw v8 numbers stand as measured.
- `real_state_qualify.json` is the 512-token trajectory check, not the full-budget
  smoke receipt; the latter is now `results/numerical_qk_request_timing_20260924/v8_smoke_receipt_verification.json`.
