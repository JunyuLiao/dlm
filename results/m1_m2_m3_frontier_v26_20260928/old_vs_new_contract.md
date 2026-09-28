# Old vLLM FIXED16 G75/L30 vs. current native-adaptive GLOBAL-only bootstrap path

Source: `results/m3_numeric_trajectory_bridge_20260927/old_new_contract_matrix.md`
(a source and published-output audit, not a new execution) and
`results/m3_numeric_trajectory_bridge_20260927/geometry_contract.md`. Only
statements the reports make are reproduced; nothing here is inferred beyond
them.

## What the two things are

- **Old vLLM FIXED16 G75/L30 (`S30`)**: `S30` = GLOBAL F75 plus corrected
  LOCAL30, measured on 16 LongBench-v2 questions × 2 seeds × 2 independent
  loads, old vLLM production dense graph vs. the old F/S30 pre-QK graph,
  same old model family, BF16, canvas256. Source anchor: git commit
  `1aa6d0971a8857f5622c032c0be7ec68afe5b6e5`
  (`temporal_qk/{fl_i1_entry,fl_arms,revision_selector_device,nl_local_runtime,history_free_runtime,trajectory_fixed}.py`,
  byte-identical to that commit).
- **Current native-adaptive GLOBAL-only path (v20/v21/v23-v25b bootstrap)**:
  Hugging Face `google/diffusiongemma-26B-A4B-it` revision
  `f7f5b7f5fa82ffc52addd066915886d497f5517b`, native adaptive diffusion,
  native mask, scope `ALL_NATIVE_LEGAL` (v20) / `GLOBAL_ONLY_NATIVE_LOCAL`
  (v21+ bootstrap panel, per the raw `counters.scope` field seen on every
  v23-v25b bootstrap record).

## Selector geometry

| | Old FIXED_RS16 S30 | Current (v20 `G75L30_nativeQ128` extension) |
|---|---|---|
| GLOBAL support | 16-row decision = 8 query heads × 2 positions, shared within KV head, KV32; union of each live row's first argmax tile is mandatory; lowest maximum row probability mass dropped to `floor(valid_tiles × .75)` | Per physical query head × **Q128 × KV64** support; maximum row mass, mandatory first argmax tile for every row; wholly immutable prefix only, current canvas/boundary retained, native mask |
| LOCAL support | 2 query heads × 8 positions, KV32 inside the 1024-token window; only wholly immutable prefix tiles eligible, mandatory union, budget `.30` | Native (not sparsified in the scored bootstrap panel) |
| Fractions | GLOBAL `.75`, LOCAL `.30` | GLOBAL `.75`, LOCAL `.30` (same nominal fractions; `old_new_contract_matrix.md` is explicit this is "a transplant of ranking intent, **not** the old 16-row grouping or bit-equivalent support") |
| `geometry_contract.md` (v21b) framing | — | Compares the existing per-head **Q128/K64** selector, a per-head Q16/K64 selector, and the exact historical GLOBAL 8-query-head × Q2/K32 mapping; Q16/K32 is "a closure diagnostic, not a fourth policy candidate" |

## Runtime

| | Old FIXED_RS16 / adaptive fairness screen | Current v20/v23-v25b |
|---|---|---|
| Backend | Local `hf_text` export as both model and tokenizer, BF16 vLLM `TRITON_ATTN`, graph capture 256, one sequence | Native adaptive diffusion runtime on the current HF revision; the bootstrap-panel raw `counters` record `consumer: 'triton'`, `kernel_variant: 'generic'`, `guard_mode: 'fused'`, `support: 'native_mask'` |
| Output/consumer contract | Not stated for the old path beyond its BF16 graph | `fp32_scores_bf16_pv`: FP32 QK accumulation/scaling without the two explicit BF16 score round trips; BF16 probabilities for PV and all other consumer arithmetic retained (`geometry_contract.md`) |

## Caps and stopping

| | Old | Current |
|---|---|---|
| Input/seed/budget | Old FIXED16 S30: 16 LongBench-v2 + 8 AIME26 development questions plus the original RULER12 probe; seeds 17/29, two loads. FIXED_RS16 sets **K=H=16 per committed canvas**; natural EOS and original task caps LB1024/AIME8192 continue | v20/v23-v25b: frozen task-specific gold-free panels, seeds 101/202, **native adaptive 48**; RULER/AIME/LongBench caps differ by task |
| Stopping | FIXED16 bounds denoise work per committed canvas, not natural adaptive stopping | Native stopping (`termination_reason: eos` / iteration cap), per-canvas `decoder_calls` varies freely; the reference clock in this v26 study is built on exactly this per-canvas variability |

## Bootstrap/observation/refresh

| | Old | Current |
|---|---|---|
| Schedule | Encoder native; canvas step 0 fast dense; step 1 dense observation writes bitmap; step 2+ held pre-QK skip, with `PERIOD=64` or changed-prefix resetting observation. Adaptive max steps is 48, so the period does not itself cause a within-canvas refresh | Step 0 all-kept custom consumer (**B0**); step 1 observes current scores and produces sparse output (**BO**); later held map until canvas/source identity invalidation, with a periodic score refresh (**A**, period 8 in every record measured) and a decision refresh (**D**) at interval R (3 for M3, 1 for M1, R=A_period i.e. no D for matched B) — see `v23_bootstrap6/v24_errata_and_audit.md` for the exact B0/BO/A/D/H accounting this v26 study reconstructs and validates |

`old_new_contract_matrix.md` also records: `B_A8_matched` is a **v20** held-routing
control with score anchor period 8 and no R2/R3 redecision, and is
"distinct from the older v15 B8P method and from historical G75/L30; none
should be substituted for another."

## What is amortized vs. direct

- **Amortized**: the v23 panel's `amortized_ms/call` column
  (`sum_warm_wall_s / total_calls`, e.g. D_native aime26 145.7 ms/call,
  M3_boot aime26 148.8 ms/call) and the `warm geo ratio` contrasts — these
  are whole-request warm wall time divided over all decoder calls in the
  request, and include prefill, refinement, sampler, commit and stop for
  the old-contract analogues (`fl_i1_entry.py` whole-request synchronized
  wall).
- **Direct**: v24's per-call, per-phase `model_forward` ratios (§2 of
  `v24_direct_cost.md`, e.g. "BO is the most expensive call: +19-31% of a
  native forward on LB") and the isolated CUDA-event pieces in "where
  observation time goes" — these are fixed-state timing diagnostics with a
  native bracket per block, not scored or fixed-step methods, and are
  explicitly "not additive to a forward price."
- The two must not be combined naively: v24_errata's corrected-wording
  table states "each forward is ~5% cheaper" is wrong; the supported
  wording is "request wall per call is ~5% lower on LB (geoW/N 0.952);
  direct bootstrap forward cost was **not measured**" in that check. And
  v25b's correction 2 (quoted in `current_facts.md`) is the same warning in
  the other direction: an isolated BO/A-call percentage must not be
  multiplied again by the BO/A share to produce a whole-forward number.
- Old-contract timing evidence is exclusively whole-request wall
  (`fl_i1_entry.py` for the adaptive fairness screen; "generation" /
  "TPOB" / whole-request for the FIXED16 S30 confirmation) — the old
  contract matrix states explicitly "No isolated direct full-forward
  price" for the adaptive screen, so there is no old-contract analogue to
  compare against the current direct per-phase measurements.
