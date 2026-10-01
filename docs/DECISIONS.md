# Decisions and negative results (chronological)

Read this before proposing a direction. Each item gives the decision, the reason and its provenance. Commit hashes are
on `research/m3-output-numerics-20260927` unless another branch is named. Result paths are relative to
`results/m1_m2_m3_frontier_v27_20260929/` (v27) unless they start with `results/`.

**Substrate caveat.** Every item before 2026-09-30 01:00 ran on the host-bound **eager** substrate and/or against a
superseded dense baseline (`D_native`, `D_c64`, `D_fast`). Its end-to-end ratios do not transfer to the current
FA4 + piecewise substrates. Mechanism findings (what fails and why) usually still apply.

## Before v20 (history only)

- 2026-09-14 to 09-23 (`research/*` branches, author dyh): vLLM-era historical-map GLOBAL/LOCAL skipping (F, G75,
  S15/S30), TBT, canvas-size and refresh studies.
  - **Reusing one map across a whole canvas inflates denoising steps** (`research/faithful-value-history-factorial-20260918`).
  - The `research/adaptive-trajectory-characterization-20260921` branch established a dense-vs-dense null: across
    model loads, denoise counts differ by a median of 1.17× (p90 1.64×). Sparse inflation at G60/G75 was not
    separable from it, except AIME G75 (+11%).
- 2026-09-24 to 09-26: numerical QK reuse, Junyu frontier v18, value-direction work (`results/numerical_qk_*`,
  `results/junyu_frontier_v18_20260926/`, `docs/handoff_archive/`).
  - The v5 M1 contract (`method_contract.md`) used cached-score output. It was later replaced by current-QK output.

## v20–v26 (2026-09-27 to 09-28; eager substrate, HF native dense baseline)

| date | decision / finding | provenance |
|---|---|---|
| 09-27 | Fan's M1 and M3 (R2/R3) under native adaptive stopping fail the gate (faster than strong dense and fresh T, acceptable quality). LongBench forwards are 4–9% cheaper, but extra adaptive calls cancel the gain; AIME/RULER show no forward gain. | `results/fan_m1_m3_multidataset_20260927/fan_today.md` |
| 09-27 | v22 CP-A: Q16 query geometry saves no work at matched error. Fine-geometry promotion stopped. | 6b795ca6d, `results/m3_numeric_trajectory_bridge_20260927/q16_calibration.md` |
| 09-27 | v23 bootstrap policy `native_bootstrap2_observe1` (call 0 dense, call 1 dense + observe) removes call inflation; E2E ≈ native. Adopted for all later method arms. | ca297e05b, `results/m3_numeric_trajectory_bridge_20260927/v23_bootstrap6/bootstrap6_report.md` |
| 09-28 | v24 direct cost: no exact optimization found. | 9ded5f804, `…/v23_bootstrap6/v24_direct_cost.md` |
| 09-28 | v25 `aligned16` route storage is bit-exact with request effect ≈ 0 (LB 1.001). v25b odd-K bridge: about 2.6% request saving with identical tokens. Correction: v23 B was 4/12 on LB, not equal quality. | af4ddad39, 41f5e462b, `…/v25_route_storage_report.md`, `results/m1_m2_m3_frontier_v26_20260928/current_facts.md` |
| 09-28 | v26 seven-arm panel (M1, M2 pooled, M3 A8/A16, B, native, fresh T): M2 implemented and run; 1 seed, exploratory. | 3d48ebcdd, `results/m1_m2_m3_frontier_v26_20260928/seven_panel_report.md` |

## v27 (2026-09-28 to 10-01)

| date (local) | decision / finding | provenance |
|---|---|---|
| 09-28 | Added A/R/B-hold/threshold configuration points and an authoritative `effective_method` receipt. A v26 receipt-field contradiction for A16 was report-only. | aafe61721, `v27_contract_audit.md` |
| 09-29 | Tier 3 on eager: superseded. Strong-dense correction to `D_fast` (errata on Tier 1/2). | c269052a5, 1d16820b7, `tier3_report.md` |
| 09-29 | `D_c64` (our 64-row Triton dense) as strongest dense: **superseded the same day by FA4** (1.9× faster). | 2bdcebc61, 57468b448 |
| 09-29 | **Cross-layer shared support collapses quality** (long RULER 64K, LB-long). Rejected. Unshared variants keep quality. | efd7cb1a4, `long_lb_panel/README.md`, `long_ruler_panel/README.md` |
| 09-29 | **FA4 (vLLM fork, SM90 hd512) is the official SOTA dense.** All-kept block-sparse is the fastest dense (4–6% kernel, bitwise equal); FlashInfer is tied. Headline baseline = `D_fa4_allkept`. | 741a73075, `official_baseline/README.md` |
| 09-29 | **Eager pipeline is host-bound.** All v21–v27 eager E2E ratios hold for eager only. Moved to piecewise CUDA graphs plus FA4. The B0-based "0.78×" was corrected (B0 is call 0 and hits fewer experts). | 9ab6bcc10, `substrate/README.md` |
| 09-29 | Selector latency was the root cause of weak M1 decision calls. Pipelined route (bit-identical), M1-DP (dense-prefix risk, 0.27 ms/layer) and async observation route (bit-identical) adopted. | f3d0cef4d, 7547f3cfd, 5601ad114, 3b5ea89f8 |
| 09-29 | Cross-canvas carry of B maps dropped: about half of the next canvas's kept tiles were not kept by the previous map. | 95905e9b1, `substrate/map_drift_75k.jsonl` |
| 09-29 | Density gate `ent0.05` turns most calls dense (entropy < 0.05 by about step 7). Not pursued. | 985062fb3, `exploration_plan.md` |
| 09-29 | HF default path (`D_native`) is about 2.1× slower than FA4 at the 64K kernel. The earlier "5.8–7.5×" measured SDPA's math backend: **corrected**. | 44a9e3769, `official_baseline/README.md` |
| 09-30 | Substrates v2 (FA4 prefill/append) and v3 (no per-call KV concat; tokens identical to v2). First significant 64K request gains (B, M3 R6, G75 L0). | c4d31e5da, 3bb1d7b59, 2437f9e45, `fa4_panel_v3/README.md` |
| 09-30 | Threshold sweep: sparser costs denoising steps (+40% per canvas at +ln2). The E2E optimum is **−ln2**, chosen on the formal items. | bdb3ca37a, `threshold_sweep_s1/README.md` |
| 09-30 | AIME: no speed room (GLOBAL attention ≤ about 5% of a step). Ungated sparse arms are net negative in q1v3 (B significant). Final AIME panel: M3 R6 DP −ln2 is accuracy-neutral (65/65). | e3648b07d, 81ad82338, `quality_panel_q1v3/README.md`, `final_panel_f1/README.md` |
| 09-30 | **AIME W corrected for pairs that captured new graphs while timed:** M3 R6 DP −ln2 0.971 → 1.008 (Wc). | da72ec843, 823aa214e, `final_panel_f1/summary_aime_sensitivity.md` |
| 09-30 | Layer variants (first layer or layers 5+11 dense, late3), tail-dense after step 24 and risk-budget selection: none beat plain −ln2. | 44c0fc280, 2f449826a, `variants_v5/README.md` |
| 09-30 | **SparseD port head-to-head:** no significant speed difference from SparseD keep 10% at 64K. Treated as a collision (B ≈ SparseD). Its paper default (dense first 20% of steps) is ≈ dense under adaptive stopping. | 7ae185180, `sparsed_panel_c1/summary.md` |
| 09-30 | AIME high sparsity: our risk top-k drops accuracy at 66% realized (23 vs 34); SparseD holds 29. **Negative for "V-aware selection is better".** | 78d516f1b, `aime_sparsity_panel_hi/` |
| 09-30 | 128K OOMs in dense prefill on one H100; 96K fits only prompts ≤ about 95K (12 cells, exploratory). | 900f8d6c9, 2d49253fa, `lb96k_panel_c2/summary.md` |
| 09-30 | 64K held-out (12 untouched items, 3 seeds): the per-step decode gain (−17%) replicates; the request gain did not (1.026). **Superseded by E4** (more seeds): the step inflation was trajectory noise. | 5674e8811, `holdout_panel_h1/summary.md` |
| 09-30 | AIME substrate defect: LOCAL recompile-limit eager fallback on short prompts. **Fixed in piecewise_v4.** Dense AIME 25% faster; v3-era AIME absolute times are inflated, but paired ratios are roughly fair. | 0a75f3177 |
| 09-30 | piecewise_v5: compiled post-attention tail of encoder canvas appends (all arms). Dense amortized ms per call on AIME −7%. | 97e520d83, `progress_20260930.md` update 9 |
| 10-01 | 64K traj (24 items, 3 seeds): keeping early GLOBAL layers dense does not stop step or length inflation and gives up per-step savings. Rejected. | d5ef60997, `traj_panel_t1/summary.md` |
| 10-01 | AIME carry and gates (E2): at most 2–3% per step, always with more steps or lost accuracy. **AIME is no longer a speed target**, only an accuracy check. | d5ef60997, `aime_carry_panel_e2/summary.md` |
| 10-01 | **Cross-canvas carry K (E1) rejected.** Cheaper steps (S/N 0.934 → 0.892 at K=4), but stale maps add steps per canvas (1.16–1.23) and cost accuracy (40/47, B 34/47). | d5ef60997, `lb32_carry_panel_e1/summary.md` |
| 10-01 | E3 32K single-change variants: a monotone trade-off between per-step saving and steps per canvas. Per-canvas decode cost (S/C) best ≈ 0.97 (obs2, −2ln2), CI crossing 1. Output protection did not help: the 32K step increase is within canvases, not output length. | d5ef60997, `lb32_steps_panel_e3/summary.md`, progress update 12 |
| 10-01 | **E4 large-seed confirmation (6 new seeds, 144 cells per arm per bin): request gains are significant and step inflation vanishes.** 64K M3 R6 DP −ln2 W 0.879 [0.820, 0.926]; 32K 0.940 [0.897, 0.980]; accuracy not lower. Supersedes the 09-30 held-out "request gain not robust" conclusion and the E3 "32K break-even" reading. | 7d724de18 (spec), `lb_confirm_panel_e4/summary.md` |
| 10-01 | **E5: first-call carry `carry_first` adopted.** Canvas call 0 reuses the previous canvas's decision; call 1 still observes. M3 + c0 vs dense: 64K W 0.853 [0.802, 0.895], 32K 0.907 [0.858, 0.952]; vs M3 0.971 / 0.963; accuracy not lower. **The `stable1` dense-confirmation gate is rejected** (slower; −5 at 32K). | 56de98fe2, adc2056d7, `lb_overhead_panel_e5/summary.md` |
| 10-01 | **E6 96K (6 fitting items × 6 seeds):** M3 + c0 W 0.744 [0.597, 0.903], per-step −24%; part of the request gain is (noisy) fewer steps. The per-step gain grows with context: −18% at 64K, −24% at 96K. | `lb96k_confirm_panel_e6/summary.md` |
| 10-01 | **E7 AIME (30 × 6 seeds, 180 cells per arm):** no speed gain on AIME. Best low-overhead variant (M3 + `carry_first` + 2K gate) W 1.005 [0.967, 1.042], accuracy 99 vs 99; Fan plain 7–12% slower; no significant accuracy difference. AIME stays an accuracy check. Also froze **E6b**: the 5 other fitting 96K pool items × 6 seeds, to pool with E6. | `aime_confirm_panel_e7/summary.md`, `specs/v27_lb96k_extend_e6b.json` |
| 10-01 | **E8 V-term ablation (AIME, fixed 70% sparsity, 180 cells per arm):** no evidence that looking at V helps. No V term differs significantly from projected rank 32 (p ≥ 0.30); ranks 32/16/8 score lowest (86–87 vs dense 99; rank 16/8 significantly below dense), mass-only 91, M2 tile-mean 92, rank 4 94, SparseD 94. The "V-aware selection" novelty claim is not supported on AIME; E9 tests it at 64K. | `aime_vterm_panel_e8/summary.md`, `receipts.md` |
| 10-01 | **E6b + pooled 96K (11 items × 6 seeds, 66 cells):** M3 + c0 W 0.822 [0.703, 0.944], S 0.727, per-step −25%, N 0.96 (n.s.), accuracy 30 vs 25. E6b alone had N 1.19 where E6 had 0.81: the 96K step count swings with the item set, so the 96K request claim rests on the pooled panel. | `lb96k_pooled_e6_e6b/summary.md` |
| 10-01 | **Batching does not make attention dominate at 64K.** For B = 1 → 4 the prefix-attention share of a forward stays 22–26% and the keep-0.12 saving about 20%: FA4 is under-occupied at B=1, and MoE becomes compute-bound. The "serving makes sparse attention pay much more" expectation is not supported at this context length. | 02eead519, `batch_scaling/README.md` |

## Group-internal evidence on the V term (not ours; cite, do not absorb)

- JunyuLiao's value-direction routing rank sweeps (branch `ljy/value_aware`, commits 8b29f942d and 47c47d9d7;
  copies in `results/diffusion_gemma_jl_*` and `results/diffusion_gemma_ruler8k_gaussian_rank_sweep_v15/`).
  - His selector uses fresh current QK plus projected V on the eager path.
  - **RULER8K at about 75% sparsity:** projected V rank 32 scores 47.4%, full-dimensional V 49.8%, attention-mass
    only 21.3%, rank 2 2.5%.
  - **At 50%:** every variant is about 92%.
  - **AIME (2048-token cap) and LongBench at 50%:** no rank winner.
  - Accuracy is not monotone in rank (r16 < r8 on RULER).
- Whether this transfers to our historical-QK M1/M3 selector was tested by E8 (2026-10-01): on AIME at 70% sparsity it
  does not (no V term beats attention mass; see RESULTS_LEDGER L1e). E9 repeats the test at 64K.

## Corrections to our own earlier claims (keep; do not repeat)

- "Three independent reproductions" of the 64K gain was overstated. Same-host runs generate identical tokens, so they
  were speed reproductions only (d11e356d7).
- An attribution of a classmate's AIME speedup to an "HF SDPA baseline" was unsupported. Their slides do not say
  which kernel they used (d11e356d7).
- A cross-environment FlashInfer-faster-than-FA4 reading was an environment artifact (`official_baseline/README.md`).
- B0-based per-forward ratios are biased: call 0 hits fewer MoE experts (`substrate/README.md` §5).

## Collisions and boundaries

- **B ≈ SparseD** (collision). M3 periodic refresh ≈ PulseCol's periodic refresh.
- The value of V-aware risk over score-only selection is **unsupported** by current data.
- Generic hygiene is never a contribution: CUDA graphs, compile, KV-concat removal, FA4 prefill, compiled append.
  These are applied to every arm.
- Query-sensitivity protection against step inflation belongs to a group member (`query-sensitivity-aware-v3`). It is
  a collaboration candidate only.
