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
| 09-30 | 64K held-out (12 untouched items, 3 seeds): the per-step decode gain (−17%) replicates; the request gain did not (1.026). **Superseded by E4** (more seeds); see the step-count correction below (pooled 9 seeds: +1–2%, n.s.). | 5674e8811, `holdout_panel_h1/summary.md` |
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
| 10-01 | **E9 V-term ablation at 64K (M3 + c0, fixed 88% sparsity, 96 cells per arm):** attention-mass-only ranking is as accurate (54 vs dense 51) and as fast (W 0.845; mass / −ln2 0.999 [0.961, 1.037]) as any V term; no difference is significant. With E8, the "V-aware selection" claim is dropped; the V term can be removed without loss. | `lb64_vterm_panel_e9/summary.md`, `receipts.md` |
| 10-01 | **E10 M2 vs M3 under the same optimizations (32K/64K, 96 cells per arm per bin):** tie. M2c / M3 0.987 [0.914, 1.063] and 0.983 [0.940, 1.024]; accuracy n.s. Together with E8/E9: neither the V term nor M2 vs M3 is a contribution. Also measured the v5 per-step time breakdown (GLOBAL 2% / 19% / 32% of a step at AIME / 32K / 64K; LOCAL 2–3%). | `lb_m2opt_panel_e10/`, RESULTS_LEDGER L1g |
| 10-01 | **E11 preview, 64K at 95% sparsity (48 cells):** still no V advantage (mass / rank 32 1.004 [0.916, 1.099]) and no accuracy loss for any arm; per-step 0.76 but W ≈ 0.89, not better than the 88% operating point. The V question is closed for LongBench 64K. | `lb64_vterm_hi_panel_e11/` |
| 10-02 | **User rules:** plain M1/M2/M3 no longer need to be rerun in every panel; run only the better configurations, keep comparisons fair (same substrate, deploy, host per cell, items, seeds), use more seeds, report the full metric set, take care with multi-host launches (pair within cells, report per host), and flag bugs or unfair comparisons in existing code promptly. | user |
| 10-02 | **GPT session review.** Commits 073d87ece and add23afa9 (pushed) harden `v27_fa4_panel_summary.py` (one first output per cell/arm, host/GPU/provenance identity, Wc excludes unknown graph counters). Recomputed E5, E7, E9, E10 here: 0 differences from the committed summaries. Its uncommitted HumanEval work used `sudo bwrap` on the shared hosts; replaced by unprivileged bwrap that only mpk allows (verified isolation) before committing (8ad870bf2). It also noted that `proj_rank` masks columns of the 32-wide bank, so rank-16/8/4 speed ratios do not reflect a cheaper projection (accuracy comparisons are unaffected). | `docs/INTAKE_AUDIT_20261001.md`, 8ad870bf2 |
| 10-02 | **chw/value_aware regrouping.** Their stack (JAX gemma + FA3/TRT-LLM BLASST kernels, `_skip_regroup.py`, `gate_reorder_value.py`) cannot be git-merged into ours; the idea (reorder query rows so rows with similar skip sets share a kernel query tile) is ported as an offline measurement first. Our FA4 call takes Q per head, so per-head row orders are expressible. Credit chw; any integration is a collaboration candidate. | `HANDOFF.md` Running |
| 10-02 | **Regrouping measured: not worth integrating.** Within 128-row FA4 query tiles it keeps only ~4% fewer prefix tiles (< 1% end to end). Finer query tiles are the larger lever (−22% kept work at 64 rows, −39% at 32), but need a fast head_dim-512 kernel with small query tiles; ceiling about 1–2% per step at 64K. | `regroup_diag_1002/README.md` |
| 10-02 | **E12 HumanEval (984 cells per arm):** no accuracy loss in any arm (dense 947, main 957, 70%-forced arms 948–963) and no speed gain (main W 1.023; short context, 9.2 steps per block). No V term differs (p ≥ 0.29), so the V term is not needed on HumanEval either. | `results/humaneval_ruler_v27_20261002/humaneval_e12/` |
| 10-02 | **E13 q64: non-negative, small.** 64-row FA4 keep maps (the regrouping lead) cut per-step cost by 0.7% at 64K (CI excludes 1, all hosts agree), not significant at 32K/96K or end to end; accuracy unchanged. Kept as an optional named variant, not a contribution; further regrouping within 64-row groups not pursued (~0.15% ceiling). | `lb_q64_panel_e13/` |
| 10-02 | **Kernel bench: q64 −9 to −11% per sparse GLOBAL call; q64r and split-KV dropped.** On identical real states, within-block row regrouping adds a Q gather and output scatter that cost more than the 4–6% of tiles it removes (q64r ≈ the 128-row map). FA4 `num_splits=2` is 4–5% slower on every map. E14 runs q64c (q64 + 64-row carried call-0 map, opt-in key `q_carry64`) with six fresh seeds. | `q64_bench_1002/README.md`, `specs/v27_lb_q64c_e14.json` |
| 10-02 | **Next speed levers after the kernel bench.** (1) `observe_carried` (c01): the observation call is about half of the GLOBAL attention time per canvas at 64K, and its dense PV can be replaced by FA4 sparse on the carried map while the observation stays exact (own variant; accuracy-first preview P15 queued after E14). (2) Cross-head (GQA) row grouping as an FA4 MHA reshape: measured offline first from real need matrices (`v27_regroup_offline`); implemented only if it removes clearly more tiles than its permutation costs. (3) A group member's design note (shared by the user 2026-10-02, branch not yet available) describes a newer query-sensitivity "C gate" (flip stability + renoise history + confidence) with fewer steps per canvas on AIME than the temporal T we use; a candidate cited port, not ours. | `specs/v27_c01_preview_p15.json` |
| 10-02 | **Expansion (user decision):** add LLaDA2.1-mini (2026-02, 16B MoE, 32K) and I-DLM-8B (2026-04, causal SDAR, 40K) with their official SGLang serving paths (FlashInfer backend) as dense baselines, in separate dyh envs. UltraLLaDA is dropped (2025 model, whole-sequence recompute makes 64K–128K impractical). Add LongBench Pro (2026-01, 8K–256K) and RULER 4K/8K (comparable to SparseD/PulseCol). New-model panels stay within each model's window (≤ 32K), like the dLLM papers. | `docs/EXPANSION_PLAN_20261002.md` |
| 10-02 | **Baseline gap flagged.** The DiffusionGemma dense reference (`D_fa4_allkept`: our HF + piecewise substrate with the official FA4 kernel) was compared with the HF official compiled path (slower). It was never compared end to end with vLLM's native DiffusionGemma serving (official, 2026-06, FA4). A vLLM 0.30.0 env is added to the setup; the per-step and request comparison at 32K/64K/96K follows. The per-model baseline protocol is in the expansion plan. | `docs/EXPANSION_PLAN_20261002.md` |
| 10-02 | **E14 (6 fresh seeds) and the regroup/c01 diagnostics.** The main result holds over 12 seeds: M3 + c0 W 0.925 / 0.860 / 0.801 at 32K / 64K / 96K with no significant accuracy change. q64 gives −0.4 to −0.7% per step (significant at 64K), with no end-to-end effect; it stays an optional named variant. q64c adds nothing per step over q64 and shows more steps at 96K, so it is not adopted. Offline regrouping (incl. cross-head and chw's set key) never beats natural 64-row tiles by more than about 5%, so the line is closed as a clean negative. c01 call-1 saving is 0.45–1.7 ms per layer, about 0.5–1% per step (earlier 3–4% estimate corrected); P15 tests it. | `lb_q64c_panel_e14/`, `regroup_offline_1002/`, `observe_split_1002/` |
| 10-02 | **P15 preview: c01 passes the accuracy-first check** (no drop vs dense on AIME, 32K, 64K or 96K). c01 / q64c per step: AIME 0.91, 64K 0.949 [0.913, 0.983]; the rest are n.s. 64K steps per canvas are 1.107. E15 (c01 on the main M3 + c0, AIME + LB 32K/64K/96K × fresh seeds 1616–2121, mpk + dlm2) is launched. | `c01_preview_p15/`, `specs/v27_c01_e15.json` |
| 10-02 | **R16 RULER 32K/64K: accuracy preserved by every arm**, including a fixed 88% GLOBAL sparsity. The V term equals mass-only (33 vs 33) at both lengths. The group member's RULER finding (V matters at high sparsity, 4K/8K, current-QK selector with LOCAL sparsity) does not carry over to our historical-QK GLOBAL-only selector at long context. Small n; dense near ceiling. | `ruler_long_panel_r16/` |
| 10-02 | **Baseline gap confirmed: vLLM's official DiffusionGemma serving is faster than our dense control.** Per step 0.76× (32K), 0.90× (64K), 0.98× (96K); prefill 0.44–0.57×; same FA4 attention kernel, so the gap is MoE, CUDA graphs, sampler and prefill. Existing speed ratios hold against our HF-based substrate only. Next: port M3 + c0 into vLLM's DiffusionGemma FA4 call (FA4 varlen accepts page_table + block_sparse_tensors) and re-measure against vLLM dense. | `vllm_dense_check_1002/README.md` |
| 10-02 | **vLLM port probe: FA4 block-sparse over vLLM's paged KV is wrong** (max abs about 22 vs the masked reference; contiguous is exact to 5e-4; dense paged is exact; page size 16 does not run). Port options in order: contiguous view of contiguous pages, fix the paged block-sparse path (report upstream), FlashInfer BSR, gather. | `vllm_port_probe_1002/README.md` |
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
  does not (no V term beats attention mass; see RESULTS_LEDGER L1e). E9 at 64K agrees (L1f).

## Corrections to our own earlier claims (keep; do not repeat)

- "The query sensitivity T is 1 in every run" and "re-decisions barely change the held map (≈ select once per canvas)"
  (2026-10-01 docs and deck v6) were wrong (corrected 2026-10-02). `NativeReuseState('T')` sets T = clamp(1 + 3·EMA of
  argmax flips, 1, 4) per canvas position every step; measured re-decisions change 44–71% of the first decision's kept
  tiles. Lesson: verify code-path claims with a receipt or a measurement before writing them into slides.
- "The 3-seed step inflation was trajectory noise and E4 shows no inflation" was too strong (corrected 2026-10-01).
  Per-seed step ratios range 0.85–1.24; the old and new seed sets are both draws from that spread (13 of 84 random
  3/6 splits give a gap at least as large). Pooled over 9 seeds the step count is +2.4% at 32K and +1.5% at 64K, not
  significant. Say "no significant step change, best estimate +1–2%", and note that E4's request W may be about 2%
  optimistic. See RESULTS_LEDGER "Step-count check".
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

## Intake decisions (2026-10-01, 22:45 UTC−5)

- User update: follow-up panels may select only stronger relevant arms. Keep `D_fa4_allkept` and a matched optimized
  reference; plain M1/M2/M3 stay in the historical ledger but are no longer mandatory in every panel. Pool same-host
  cell ratios across machines and report host-specific diagnostics.
- The standalone FA4 summary had a latent last-record-wins join and lacked its own same-host/scorer-provenance
  checks. The upstream runner and qualified scorer already enforce assignments. Hardened the independent join and
  tested it; the 8,826 first runs in E4–E11 pass the checks and all reported point metrics/correct counts are unchanged.
  This is a reporting guard repair, not a speed improvement or a new sparse method.
- Record the user's classmate clue as a hypothesis: RULER needs V dimensional information; AIME/LongBench may not;
  HumanEval is unknown. Full V in actual attention has never been removed. Existing peer RULER8K evidence supports
  V direction over scalar mass, but rank 32 already improves on mass-only, so it does not prove full rank is
  required. Do not transfer E8/E9/E11 negatives to RULER. See `INTAKE_AUDIT_20261001.md` for a proposed matched test.
- Prefer the RULER scope/freshness-controlled check over automatically enlarging already tied E10/E11 panels.
  HumanEval is a proposed coding quality test, not a presumed speed target. No new dataset or GPU run was launched.
- Full LongBench-v2 coverage has two constraints: current prefill memory and configured positional context. Existing
  all-item tokenization yields 400/503 inputs within context after reserving 8,192 output tokens. Chunking alone cannot
  give native, untruncated coverage of all 503. A complete truncated/retrieval evaluation must use a separate protocol.
- A/B integration remains pending the classmates' exact branch refs and work. No peer branch was absorbed or merged.
- Fixed a stale research-context sentence that still described batch scaling as unmeasured; the committed 64K
  diagnostic already found no increase in attention share.

Sources: audited base `6b13fe178`; `scripts/v27_fa4_panel_summary.py`,
`tests/test_v27_fa4_panel_summary.py`, `results/m1_m2_m3_frontier_v27_20260929/intake_audit_20261001/summary.json`.

## Group-context decisions (2026-10-01, 23:10 UTC−5)

- The user authorized read-only research search of the specified four-person group. Record only sanitized method
  context (`GROUP_CONTEXT_20261001.md`), not raw messages/contact information. No messages or shared edits.
- Peer RULER evidence supports V direction; it does not establish full-rank necessity. The peer BLASST control
  differs from mass-only. Keep their figures and adaptive-step observations out of this branch's benchmark claims.
- Query-sensitivity/trajectory protection remains a collaboration candidate. Acceptance-threshold changes alter
  native decoding and require a separately named frozen arm. A/B refs are still pending; no merge.
- Before any integration, check that each predictor exists at regroup/selection time. Historical same-forward S/TS
  was explicitly unavailable to pre-forward regroup; previous-step predictors have a different timing contract.
- HERALD's accuracy uses adaptive acceptance but its performance fixes T=20 and selects peak feasible batch sizes.
  PRR and SSV concern different AR sparse/speculative substrates. Cite overlap, never import their speedup as a v27
  result. Current map-drift/RULER/HumanEval/LongBench follow-up ordering is unchanged; no GPU workload launched.
