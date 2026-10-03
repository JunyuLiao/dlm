# Decisions and negative results (chronological)

Read this before proposing a direction. Each item gives the decision, the reason and its provenance. Commit hashes are
on `research/m3-output-numerics-20260927` unless another branch is named. Result paths are relative to
`results/m1_m2_m3_frontier_v27_20260929/` (v27) unless they start with `results/`.

## V18 intake (2026-10-02 16:45 UTC-5)

- Explicit fetch of the working remote branch confirms `b9e0700d8`; no foreign branch merged.
- Freeze the E14 eligible 24/24/11 items x four repeat labels on native vLLM, with default
  dense plus main and matched native-hook/all-kept controls. See `VLLM_PANEL_V18_20261002.md`.
- Correct smoke measurement before a formal panel: scheduler phases count actual denoising
  N separately from commit forwards, synchronization only at request boundaries, adapter
  initialization/cleanup charged to W, all shapes warmed, real graph/compile counters checked.
- Four repeats do not establish trajectory pairing or accuracy noninferiority. Report item-clustered
  confidence intervals and retain an inconclusive finding if that is what the data supports.
- Final CPU qualification: 112 tests passed on the registered dllm interpreter, no GPU job,
  including closed-worker, frozen-settings, phase-count and receipt checks.
- No new variant, upstream post, peer-branch integration, shared-slide edit or message.

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
| 10-02 | **dyh-only incident, fixed.** SGLang wrote `~/.cache/sglang` on import (it ignores XDG_CACHE_HOME; needs `SGLANG_CACHE_DIR` and `SGLANG_JIT_CACHE_DIR`), and one Triton entry landed in `~/.triton/cache` during the I-DLM env check. Both were created by our runs (timestamps) and removed; the cache hygiene list is in the expansion plan. LLaDA2.1-mini SGLang smoke is blocked on a full CUDA 13 toolkit in dyh (SGLang JIT kernels fail to link with the pip CUDA package). | `docs/EXPANSION_PLAN_20261002.md` |
| 10-02 | **FA4 paged block-sparse bug fixed locally:** the SM90 block-sparse producer passed logical KV-block indices to the paged TMA load. A 30-line patch translates them through the page table; paged now equals contiguous exactly at page 64. Page 32 (vLLM's GLOBAL default, cp.async path) is still unsupported. Upstream report pending user go-ahead. | `patches/README.md` |
| 10-02 | **E15: c01 not adopted; the main result holds over 18 seeds.** c01 / main per step is 0.983 (AIME) and 0.995–1.004 on LB, so the call-1 saving is negligible once spread over 14–18 steps per canvas. Its 64K accuracy is 70 vs dense 83 (p about 0.007; vs main p 0.096). Main / dense over E13+E14+E15: W 0.950 / 0.867 / 0.818 at 32K / 64K / 96K, accuracy not lower (96K higher, p 0.012). | `c01_panel_e15/receipts.md` |
| 10-02 | **P16: the 64K accuracy gap is mostly selection noise.** On the 3 worst E13–E15 items, the −20 pp gap shrinks to −8 pp on 12 fresh seeds (14 vs 17 of 36, p 0.58). Over 6 items main equals dense (45 vs 44). Output protection (38, N 1.035) and −2ln2 (44, per step 0.852 vs 0.808) are not adopted. | `lb64_item_check_p16/receipts.md` |
| 10-02 | **P17: C gate not adopted** (group member's query sensitivity, ported; collaboration). At −ln2 it is 7–10% slower per step than T at 64K/96K, because it protects most of the canvas. At the base threshold it is +2–3%. Accuracy is 27 / 29 vs main 30 and dense 27 of 44, none significant. Preview only (44 cells per arm). | `cgate_preview_p17/receipts.md` |
| 10-02 | **R17 RULER 32K/64K/92K: main keeps accuracy at every length** (110/98/91 vs dense 110/97/91). Fixed 88%/95% sparsity loses only at 32K (`cwe`). **The V term is closed as a negative:** mass-only is as good or better, significantly so at 95% / 32K (107 vs 101, p 0.031). 96K was replaced by 92K because 96K RULER rows exceed the one-H100 fit of our substrate. | `ruler_long_panel_r17/receipts.md` |
| 10-02 | **FA4 varlen block-sparse bug found (second, independent of paging).** The SM90 kernel builds `SeqlenInfoQK` without `tile_n`, so the varlen per-q-block list offset uses ceil(seqlen_k / 128) while head_dim 512 runs tile_n 64. Lists laid out at that stride are exact (3e-4); the natural layout is wrong. **This does not affect batch-1 work:** the fixed-length paged path is exact and every panel and the vLLM comparison run batch 1. The one-line fix is needed only for batch > 1 serving; an upstream report waits for the user. | `docs/VLLM_PORT_NOTES_20261002.md` |
| 10-02 | **Dense-baseline correction and first in-vLLM result.** vLLM's dense GLOBAL call runs FA4's dynamic-causal path with an effective 2-way split-KV: 0.92 / 1.71 / 2.42 ms at 32K / 64K / 94K, vs 1.49 / 2.96 / 4.17 ms for the num_splits=1 configuration our HF panels used as `D_fa4_allkept`. So the HF-substrate speed ratios overstate the gain vs the strongest official dense; speed claims move to vLLM. A 2-way page-table-alias split (same kernel, exact LSE merge) gives the sparse consumer the same parallelism. Method inside vLLM vs vLLM's default dense serving, smoke: per step 0.96x (32K), **0.85x (64K)**; adapter all-kept = dense. Next: a vLLM panel with many items and accuracy. | `docs/VLLM_PORT_NOTES_20261002.md` |
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

- 2026-10-02 16:50 UTC-5: launched V18 96K qualification on dllm (four serial arms, fresh run directories); no full panel yet. GPU seconds pending terminal receipts.

- 2026-10-02 16:58 UTC-5: V18 preflight 001 failed closed on vLLM internal request-ID rewriting (no qualified records; 241.607 GPU reserved seconds). Fixed attribution without changing inference; 125 CPU tests pass. Requalification will use a new deploy/binding/run.

- 2026-10-02 16:58 UTC-5: attempt 002 launched on dllm under new immutable deploy `eeba8b6ec` and new binding/run directories. All arms use the globally longest eligible input. No formal panel yet.

- 2026-10-02 17:07 UTC-5: attempt 002 dense passed (N=146, commit=12, zero timed compile/capture); native failed adapter coverage/clock validation before a timed record. No speed conclusion. Attempt 002 reserved 355.475 GPU seconds; total closed attempts 597.082 s. Independent diagnostic saves scheduler and adapter counters without changing inference. Qualification rows are now explicitly barred from the formal summary; 127 CPU tests pass.

## Async execution-count correction (2026-10-02 17:16 UTC-5)

The independent diagnostic found 349 retired denoising steps versus 350 actual native
forwards, 1750 GLOBAL calls, 29 encoder calls (6 prefill + 23 commits), and zero order errors.
The official async batch queue had executed the next canvas's first forward before retiring
the final commit. This is not an adapter scheduling defect. Earlier V18 dense qualification's
146 count was a scheduler-retired count, not yet a verified actual execution count.

The tracker now keeps references to vLLM's existing independent CPU sampled-count snapshots,
tagged using the sampler's CPU draft flag, and reads them after the request-boundary sync.
It adds no tensor operation, GPU synchronization or change to native async scheduling.
N includes all executed denoising work. Retired N and unconsumed speculative N are also
reported, with exact decomposition checked. Prefill counts must agree between both sources.
151 CPU tests pass. Closed reserved GPU seconds total 755.639; formal panel still not launched.

- 2026-10-02 17:17 (UTC-5): 96K qualification attempt 003 started on dllm from 088dc8184, after 151 CPU tests. It counts actual async executions using existing CPU snapshots. No formal panel launched; mpk and dlm2 idle.

- 2026-10-02 17:31 (UTC-5): Qualification 003: dense, native and all-kept pass actual async N and zero timed compile/capture; main stops before generation because frozen source-hash keys name the old deploy. Core bytes are unchanged. Rebind paths only after old/new byte equality checks, then use a new frozen binding and run. Formal panel not launched. All GPUs idle. Attempt 003 reserved 614.449 GPU seconds; closed V18 total 1370.088 s.

- 2026-10-02 17:35 (UTC-5): Source-path rebind fix passes 164 CPU tests and real frozen production-config validation. Old and new files must match original hashes; only source path keys and derived fingerprints change. All method fields remain frozen. Worker now validates the effective config before GPU initialization. New deploy/binding qualification 004 is next, main first. Closed GPU seconds remain 1370.088.

- 2026-10-02 17:36 (UTC-5): Qualification 004 launched from fdca7bf02 with a new frozen binding and strictly byte-validated config path rebinding; main runs first. Three matched references follow if main passes. No formal panel yet. mpk and dlm2 idle. 164 CPU tests and the actual production-config CPU guard pass.

- 2026-10-02 17:40 (UTC-5): Qualification 004 reached real main execution but OOM during longest 96K warmup at common KV reservation 0.92 (256 MiB request with only 50.19 MiB free). No timed main record. New committed V18b spec lowers KV reservation equally to 0.85 for all four arms; method and generation parameters unchanged. Requalify every arm under a new binding/run. All GPUs idle. Attempt 004 reserved 144.065 s; closed V18 total 1514.153 GPU seconds.

- 2026-10-02 17:41 (UTC-5): V18b qualification 005 launched from 8704cd072 with fresh frozen materials, deploy and binding. All arms use KV reservation 0.85; main runs first, then all-kept/native/dense. Method and native generation parameters remain frozen. No formal panel yet; mpk/dlm2 idle.

- 2026-10-02 17:45 (UTC-5): V18b main passes the longest 96K input under common KV reservation 0.85, with valid effective-method/actual-forward receipts and no timed compile/capture. Three references are still running/pending; this is not a completed accuracy or speed panel. Read-only expansion check found I-DLM CUDA 12.8 versus upstream SGLang CUDA 13.0, so the latter's shim cannot be treated as I-DLM-qualified.

- 2026-10-02 17:54 (UTC-5): All four longest-96K V18b qualification arms pass execution/config/count/graph guards and unchanged NeMo scoring. All four outputs parse and all four are incorrect on this one item; this is scorer qualification only, not accuracy evidence. Qualification 005 reserved 640.802 s; total closed V18 work 2154.955 GPU seconds. Formal V18b panel_001 launched from the identical 8704cd072 deploy/binding: 24/24/11 items at 32K/64K/96K, four repeats, dense/main plus matched controls, 568 timed requests. Expected 2-4 hours; mpk CPU scorer ready, both other GPUs idle.


## Variant/fairness audit (2026-10-02 18:19 (UTC-5))

- Preserve historical raw artifacts and append explicit corrections: repeated-seed cell p values
  are exploratory, noninferiority unproven, HF W excludes setup/cleanup, HF new_graphs is Dynamo-only,
  and mass-only is prefix-ranking-only. R17 92K main is +0/-0 at unchanged 91/91.
- Reject unsupported C/density gate configurations at vLLM adapter construction. Frozen
  8704cd072 main and ongoing V18b are unchanged. New guard has 6 CPU tests.
- Read-only CHW/PPT/Slack review distinguishes full-QK/PV-only optimization, JAX baseline,
  unmatched density/cost and different thinking/budget/protection protocols. No peer code merged.
- Keep P17 not-adopted as a result for its tested configuration, not a universal C-gate rejection.
  No evidence currently justifies aborting V18b or reopening closed regroup/V claims.
- Next method study needs a new branch, independent calibration, matched realized density and
  selector costs, native stopping, strong dense and optimized references. See audit document.


- 2026-10-02 18:25 (UTC-5): configured a single-campaign completion stage for formal V18b.
  After all eight workers close, score privately and publish validated aggregates only.
  Pin the local and remote branch head; stop on concurrent changes or invalid runs.
  No autonomous positive interpretation and no additional GPU launch. See HANDOFF.


## Peer source update (2026-10-02 18:33 (UTC-5))

The user-supplied 10-page PDF resolves the missing C-gate source. HF state formulas
match Eq. (1)-(7), but P17 historical-QK/DP/R6/carry-first is not the full current-QK
retained-state method. Three existing C-gate CPU tests pass; GPU seconds = 0.
Peer high-sparsity RULER V gains are external reported evidence, not independently
reproduced branch results or a no-loss speed claim. Table 1/11 dense provenance needs
clarification. No new method or V18b configuration change. Source and interpretation:
`docs/PEER_PDF_UPDATE_20261002.md`. The private PDF is not uploaded.


## V28 campaign preparation (2026-10-02 18:54 (UTC-5))

User authorized worthwhile variants. Independent branch `research/vllm-variants-20261002`
starts at `9b027df8f`; parent V18b remains frozen. See `docs/VARIANT_CAMPAIGN_V28_20261002.md`.
This entry is specification/status only, not a result. New GPU seconds: 0.

## V28 CPU qualification — 2026-10-02 19:08 (UTC-5)

31 CPU tests passed (lifecycle ownership/exception cleanup, unsupported-variant guards,
HK=2 flat-view counterexample, Q64 coverage and regroup invariants). Source:
`tests/test_v28_adapter_lifecycle.py`, `tests/test_v27_vllm_adapter.py`,
`tests/test_v28_alias2_q64_bench.py`, `tests/test_v28_regroup_screen.py`.
KV-view rejected on layout grounds; request_clear remains an unmeasured standard
optimization. No new V28 GPU performance or accuracy result. See V28 campaign spec.

## V28 regroup screen and native qualification freeze — 2026-10-02 19:18 (UTC-5)

Real historical prefix-need snapshots: 144; natural Q64 work 794361 tiles.
Default bounded swap search accepted 7/144; gated load proxy ratio 0.9971768.
Expanded search (12 swaps/head, 16 candidate signatures/group) accepted 19/144;
ratio 0.9893759, total work +0.0402%. Aggregate max/p95 loads unchanged.
These are CPU prefix-only proxies, not GPU acceleration; no standalone regroup
request panel warranted. Source: `results/v28_20261002/regroup_screen/`.

Native q128/q64 request_clear qualification specs are frozen under
`results/v28_20261002/specs/`. Both use identical main parameters and source,
common 0.85 reservation, five GLOBAL layers, alias2 and lifecycle cleanup.
Each future preview uses first two E14 items per length bin x two repeats;
qualification first uses the longest selected item, one warm plus one timed.
Native dense/native-hook/all-kept controls retained. Wrapper records actual N,
untimed actual KV copy checks, effective q64 counters and allocator retry deltas.
30 CPU wrapper/lifecycle/adapter tests passed; 13 regroup and 3 component tests passed.

Component attempt 001 completed in 30.7184 reserved GPU seconds, but remains
diagnostic only: exact-length rather than nominal-bin sampling and non-native
Q strides were found during review. New attempt corrects these and strengthens
finite checks, disables TF32 for FP32 reference, and explicitly excludes first
list/split builds. Old run is preserved; no headline speed is taken from it.
LLaDA first upstream smoke loaded weights but failed RoPE JIT missing CCCL header;
isolated toolchain repair ongoing. Model sparse ports remain unimplemented.

## V28 native Q64 component result and request qualification — 2026-10-02 19:26 (UTC-5)

Corrected component002 passed finite IEEE-FP32 masked references on six historical
need snapshots with synthetic QKV, actual native tensor strides and random pages.
Q64/Q128 alias2 GPU-time geometric mean = 0.94575 (about 5.4% component reduction).
Selector, KV copies, first lists/splits and other model costs excluded; no request
or accuracy claim. Source: `results/v28_20261002/q64_component/`. Worker time
18.6512 GPU s; prior diagnostic001 30.7184 s. Dense component key corrected to
fixed-length diagnostic: it is not the official varlen serving baseline.

Native request qualification002 runs on mpk, frozen deploy e7b5ff061, one worker
at a time. Q64 method passed: actual N=66 (65 retired+1 unused), 330 GLOBAL calls,
60 q64-refined routes/list builds, five GLOBAL warm copy checks passed; no timed
CUDA captures/backend/inductor compiles or allocator retries/OOMs. Warm long
requests did show recoverable allocation retries, so request_clear does not
eliminate canvas allocation peaks. These one-request checks are not performance
or accuracy evidence. Main128, allkept, native-hook and dense follow serially.
Environment used OMP_NUM_THREADS=4; future performance preview should freeze 1
for all arms, following official serving warning. Do not compare qualifications
as a strongest-baseline speed panel. Further Triton/CuTe compilation coverage is
being audited independently from existing CUDA-graph/backend/inductor counters.

Held-regroup optimistic ablation now implemented with 8 CPU tests: natural Q64
versus token-major per-head gather, alias2 and scatter; at most one CPU-accepted
state per nominal bin. Search/first builds are excluded deliberately to ask whether
amortization could pay even in this favorable setting. No GPU result yet.
LLaDA CUDA13 closure mismatch identified (runtime13.4 headers with nvcc13.0);
isolated CPU small-kernel compile now passes with matching headers. No sparse
model port or qualified new-model accuracy result yet.

## V28 seed expansion, held-regroup negative and canvas release — 2026-10-02 19:44 (UTC-5)

User explicitly reiterated large seed-dependent denoising-step variance. New
seed4 protocols freeze four independent engine seeds (28001–28004), two sequential
repeat labels per engine, six question clusters: 48 requests/arm. Repeats are not
extra independent seeds; same engine seed is not a matched random trajectory.
Keep per-seed N distributions, paired geometric W/S/SN/N and correctness, with
question-cluster intervals. This remains preview; expand question coverage for
request-level claims. Existing qualification002/V18b protocols are untouched.
Common OMP_NUM_THREADS=1; new requests record pre-deduplication monitor Triton/CuTe
JIT event deltas as well as CUDA capture/backend/Inductor counters. Nonzero/missing
timed monitor receipts fail the new run. Events can include cache loads/failed
compiles, exclude autotuning/other workers/earlier aliases; zero is not universal
absence of compilation. Existing log warnings use warning_once, so the original
panel's zero post-warm warnings are only a lower-bound audit. Its block-0 main
had 30 recoverable post-warm allocation failures, retained in W.

Held-regroup component003 completed: natural-Q64-relative total times
1.04498/1.00462/1.03343 at nominal32/64/96K, geomean1.02753, even excluding all
search/initial-build cost. Three accepted states only; reject this unfused held
implementation for request expansion, not all grouping/fusion hypotheses.
Source: `results/v28_20261002/regroup_held/`; 45.8499 reserved GPU seconds.

All five qualification002 request arms passed (q64/main128/allkept/native/dense),
with existing graph/backend/Inductor checks and intended-path receipts. One item
per arm is implementation qualification, not performance or quality evidence.
A q64 alias1/2/4 component sweep is now specified and CPU-tested; independent
adapter per split avoids the split-cache identity-key pitfall. GPU run pending.

New standard optimization `canvas_buffers=release_after_invalidate` releases all
old contiguous adapter KV and prefix views only after successful encoder hooks
and same-CUDA-stream guards. Core observation/projection uses that stream;
async selector reads independent arrays with existing record_stream protection.
Carry and split maps are retained; no synchronization or empty_cache added.
Defaults remain legacy. Must qualify real release counters, numerics and allocator
behavior; no measured memory/performance claim yet. q128/q64 plus matched allkept
use the same option in separate frozen specifications. 92 relevant CPU tests pass.

LLaDA official SGLang JointThreshold dense smoke now passed, three toy checks,
forward count unavailable. This is environment qualification only; none of our
sparse variants are ported yet. Source: `results/v28_20261002/llada_dense_smoke/`.
Six attempts reserved345.5495GPU seconds including failures/stops. I-DLM own-stack
CPU setup/toolchain qualified and first dense smoke is compiling/running; no
new-model sparse or benchmark result claimed.

## V28 qualification and four-seed family — 2026-10-02 20:04 (UTC-5)

Qualification004 closed five request workers on the same dcfdb8730 source and OMP1.
Q128 legacy/release each N72; Q64 legacy/release each N66; allkept release N81.
Timed monitored JIT events and allocator retry/OOM deltas were zero. Release
counters show5 canvas invalidations for each method (9,731,891,200 cumulative
bytes released, not peak memory saved). Process peaks were nearly unchanged;
no peak-memory or speed advantage is established. Reserved cost867.9727GPU seconds
includes the14.3026-second alias sweep. Qualifying one item does not establish accuracy.

The six-arm family spec freezes dense, native-hook, allkept release, main legacy
canvas, main release and Q64 release. All use request_clear and shared generation
source. Four engine seeds28001–28004, two sequential repeats each, six questions:
48timed/variant; 288timed+144warm in24fresh engines. Variant order reverses in
alternating blocks. Paired W/S/P/SN/N and correctness use question-cluster intervals
and per-engine-seed step distributions. These are preview data, not a powered
noninferiority test. Family spec: `results/v28_20261002/specs/v28_seed4_family.json`.

Alias1/2/4 initial sweep uses three historical need states and synthetic QKV.
Alias1/2 ratio1.29397; alias4/2 ratio0.99264 with 64K slower1.01820 and96K faster0.96739.
Keep alias2; extend to all144 correlated snapshots with descriptive per-bin
aggregates, not question-level CI or request speed claims.

I-DLM chat follow-up executed successfully but all3calls hit512tokens despite
answer-presence checks3/3. Source and CPU EOS checks found no obvious stop-set
mismatch; actual output was not retained so repetition/think-closure diagnostics
are unavailable. Baseline stopping/quality remains unqualified. All new-model
attempts reserved723.0345GPU seconds. No sparse ports implemented; model-specific
clock, causal mask, rollback and all-kept controls are prerequisites. See
`docs/NEW_MODEL_PORT_AUDIT_V28_20261002.md` and
`results/v28_20261002/idlm_chat_smoke_followup/`.

## V28 preview launched and strict summary ready — 2026-10-02 20:14 (UTC-5)

seed4_preview001 launched on mpk from unchanged dcfdb8730 generation deployment,
using the six-variant family spec committed at8c9cd31ae. Four independent engines
per variant, shared seeds28001–28004, six questions, two sequential repeats per
engine:288timed+144warm total. One GPU worker at a time; immutable new run directories.
The full144 alias component run ended and released the GPU before preview launch.
Alias4/2 mean0.98836, with32K slower1.01731,64K0.98315,96K0.96533; keep alias2.
Reserved component cost27.0913GPU seconds. No request/accuracy claim from this sweep.

The new CPU summary validates all24workers and288records, exact bound source/input
bytes, protocol/config/host/software/seed identities, intended execution counters,
actual N, zero observed timed monitor JIT events and CUDA captures, and private
completion joins before unchanged NeMo scoring. Outputs contain aggregates only,
including W/S/P/SN/N, correctness, by-seed N distributions and length-stratified
question-cluster95%intervals. The main_legacy/dense extra comparison is labelled
descriptive; other seven comparisons are frozen family pairs.17CPU tests passed.
Qualification004 release/legacy private output equality holds for each Q128/Q64
pair (one item each); this is a limited numerical control, not model quality evidence.

CPU-only new-model event prototypes and12tests cover absolute-position identity,
prefix/layout epochs, edits including A-to-B-to-A, request reset and rollback.
They install no native hook, measure no forwards, and implement no sparse attention.
Unchanged token IDs do not prove unchanged hidden states/QKV or safe support reuse.
New-model adaptation audit states all unimplemented GPU/mask/KV/stream boundaries.

## I-DLM diagnostic003 launched — 2026-10-02 20:23 (UTC-5)

Frozen private spec verified byte-for-byte before launch on idle dlm2. One
engine seed0, three sequential toy requests, official chat/thinking/sampler
unchanged, max4096. Raw Engine text/output IDs remain in private own directories;
only scalar stop/closure/repetition diagnostics will be published. An initial
prelaunch newline-transfer mismatch was caught before GPU launch (0GPU seconds);
the failed prelaunch directory is preserved. This is baseline diagnosis only.

D2H metadata audit: existing CPU scheduler/sample-count fields cannot replace
the full exact phase/step/sequence-length tuple. CPU length is an upper bound;
next commit state and actual/retired execution identity differ. Async copy plus
a wait at prepare would merely relocate synchronization. No routing optimization
or shadow trace implemented; frozen preview unchanged. See
`docs/V28_CPU_METADATA_AUDIT_20261002.md`.

## I-DLM natural stopping diagnostic003 complete — 2026-10-02 20:33 (UTC-5)

All3official dense toy calls naturally stopped at EOS with closed thinking sections,
at868/549/1200 output tokens. Actual API IDs verify one EOS at the final position;
8gram excess repetition fractions0/0.00738/0.01006. This supports the earlier512
budget being too short for this toy; it is not a matched causal experiment because
later RNG trajectories change with earlier output length. Oneengine seed0, not3seeds.
Answer-substring presence does not establish exact final-answer or benchmark accuracy.
No sparse method or verified actual-forward count is implemented for this model.
Source: `results/v28_20261002/idlm_stopping_diagnostic_003/`. Reserved193.1752013GPU
seconds; I-DLM all attempts570.6601686; all new models916.2096868. Failed prelaunch
transfer check used0GPU seconds. Worker ended, GPU released, private output/statistic
recomputation and frozen-byte checks passed. Earlier reports remain unchanged.

## Regroup permutation-backend follow-up specified — 2026-10-02 20:37 (UTC-5)

The existing held Torch result remains negative. A separately named Triton
permutation backend is worth testing because generic gather/scatter consumes
more than the consumer saving. Spec: `results/v28_20261002/specs/v28_regroup_triton.json`.
First test exact same-host Torch/Triton permutation; then full gather/FA4/scatter
on one host. No adding timings from different hosts. CPU and GPU equality are
prerequisites; no new performance result yet. Existing request preview untouched.

## Triton permutation negative and completion pipeline ready — 2026-10-02 20:52 (UTC-5)

Frozen e69b59033 pure permutation ran on dlm2. All exact GPU checks passed;
Triton/Torch event-span geomean1.626493 (three bins1.643810/1.620939/1.614872).
Both allocate per call, warm8 and100rotated repeats. Event spans include any
host-dispatch gaps; no isolated device-kernel or request-speed interpretation.
Keep this negative. Full held-attention GPU follow-up was stopped at the stage
gate before launch; its prepared deployment is retained, no worker ran. Source:
`results/v28_20261002/regroup_triton_permutation/`. Reserved40.2940346GPU seconds;
internal post-selection/import body span1.6872869s is a different accounting scope.

The one-shot seed4 completion helper is prepared in private coordination as
`finish_seed4_preview001.py`; it has not yet been armed. Separate CPU scoring
source f0d3cd000 is deployed, using the pre-qualified registry CPU interpreter
and pinned read-only NeMo; real correct/incorrect toy joins plus17scorer tests
passed. Generation environment remains unchanged. Full24worker/inventory/source/
receipt/gold checks precede score; only anonymous aggregates are downloaded.
Publication requires unchanged expected local/remote V28 HEAD, clean staging and
only known ignored-artifact dirt, then updates HANDOFF/docs/STATE and non-force
pushes. Any failure retains private evidence and stops; no GPU launch/retry or
scientific conclusion is automated. Offline failure, closed-accounting and mock
publication checks pass. Fetch avoids writing the parent worktree's FETCH_HEAD.

Preview frozen question identities were checked privately: six distinct IDs,
zero cross-bin duplicates. Four engine seeds and two sequential repeats do not
increase the independent question count beyond six.

## Completed parent V18b and seed interpretation — 2026-10-02 21:02 (UTC-5)

Read-only source: parent branch commit `e1d2f89c29b6a713ffd3419cf7149d1a34abbd50`,
568 timed requests,59 questions,2 independent engine seeds, four repeat labels.
Formal reserved10654.113GPU seconds; with qualification12809.068. No merge.
Main/default-dense W ratios .7814/.7439/.8070 at32K/64K/96K, but main/native
W on the small matched subset is1.1377/1.0065/1.0446.32K S/N is slower versus
dense; changed N contributes substantially. Native's own lower W accompanies
lower N, not lower S/N; do not label it a proven engineering-only gain.
64K main/all-kept W .9255 [.8969,.9668] is a lead, with7/16 versus8/16 correct.
Equal64K/96K main/dense accuracy counts do not establish noninferiority.
See `docs/V18B_INTERPRETATION_V28_20261002.md` for sources, CIs and limitations.

Frozen V28 preview unchanged:4 engine seeds x2 repeats x6 distinct questions;
report N by seed and cluster by question. It is still a preview. Confirmation
needs more independent questions and seed blocks with expanded matched controls,
frozen accuracy margin and analysis. Do not silently add per-request RNG resets.
One-shot existing-campaign completion helper will be armed after this commit
is pushed; private status is authoritative. It uses separately qualified CPU
scorer f0d3cd000, checks frozen identities and all workers, then publishes only
aggregates behind unchanged-local/remote-HEAD guards. No new GPU launch or
automatic favorable scientific conclusion. Any failure stops for review.


## V29 expansion and fused-copy qualification specified (2026-10-02, UTC-5)

See `docs/CONFIRMATION_CAMPAIGN_V29_20261002.md`. New branch from V28 e5e1ffcc8;
no change to its running preview/finalizer.42 CPU copy/address/adapter tests pass.
GPU copy/component qualification and expanded dataset panels pending; no new
performance or accuracy result. Standard copy optimization must apply to matched
all-kept/main controls. Regroup O-scatter fusion is a separately named candidate.

## Fused paged-copy component001 passed (2026-10-02 21:53, UTC-5)

Source3a3a3c52d;10 synthetic native-stride GPU cases bit-exact,100 alternating
samples after8 warm. Long tail copy ratios .55754/.55732/.55517 at32/64/96K;
full-copy ratios .18225/.16055/.15203. Event spans include host dispatch and
original intermediate allocation. Real-model qualification and W/accuracy pending;
standard optimization must also apply to all-kept. No request-speed claim.
Reserved3.867908GPU seconds on dlm2; run closed. Sources:
`results/v29_20261002/paged_copy001/{summary.json,receipts.json,README.md}`.


Independent32K profiler (10 CPU tests) and mapped alias2 merge prototype (11 CPU tests) ready for separate GPU qualification. Fixed component protocol: historical3 accepted states, seed2903, warm8,32 rotated samples, natural_fused matched reference, frozen order/search unchanged; GPU Torch LSE and IEEE FP32 masked oracles before timing; online construction remains unmeasured. Merge is tolerance-qualified, not bit-exact. Source docs: docs/V29_32K_COST_AUDIT_20261002.md and docs/V29_REGROUP_REVIEW_20261002.md. No GPU result for these diagnostics yet.

## Regroup fused writeback component001 closed (2026-10-02 22:05 (UTC-5))

Sourcee07aec657; three fixed accepted states passed GPU numeric oracles.
Held_fused/natural_fused1.00566/.97856/1.01061, geometric.998177: approximate
tie and no general regroup speed evidence. No request expansion for this held
search. Standard natural merge fusion alone gives.90340/.92598/.90358 ratios;
keep as shared implementation candidate, not regroup novelty. Not bit-exact.
Reserved51.200860GPU seconds, including36.475206CPU selection; source
`results/v29_20261002/regroup_merge001/`. No W/quality evidence.

Adapter `merge_backend=triton` opt-in now CPU-qualified alongside copy backend;
default remains torch. Identity constructed from known CPU indices, no D2H
validation; count builds/calls, reuse only immutable index geometry, clear at
request boundaries. Apply to main/all-kept equally; all request initialization
cost stays in W. Real-model/GPU adapter qualification remains pending.
Independent32K profiler updated with denoise/encoder scope where exact existing
CPU phase is available; default FULL missing scopes stay unknown.11 CPU tests.
Combined copy/merge/profile/adapter suites67 tests pass.


## 32K diagnostic launched and third vLLM host ready (2026-10-02 22:14 (UTC-5))

Four-arm independent32K profile running on dllm, frozen39e08c521, same one
question/engine28001, source/input/CPU-test guards passed. First preparation
attempt used wrong system Python lacking Torch; failed before GPU work (0s),
preserved. Attempt002 uses qualified own vLLM interpreter. Each arm new engine
and run/cache, GPU-idle gate,11 profiler CPU tests pass. No speed inference from
profiled records. Fifth PIECEWISE-dense-without-hooks diagnostic is specified in
`docs/V29_DENSE_CONTROLS_20261002.md`, not yet launched and not part of formal
panel. Main/Q64 already run in vLLM; component candidates have narrower scope.

dlm2 own vLLM0.30/Torch2.13cu130/Triton3.7.1/FlashInfer0.6.18.post1 installed,
202 runtime distributions match mpk, pip check/CPU imports pass, FA4 paged-patch
bytes match qualified mpk. Model606shards validated read-only. Setup GPU work0;
GPU numerical qualification remains required. No shared environment modified.

## Latest audit and diagnostic status (2026-10-02 22:35, UTC-5)

First independent32K diagnostic closed four arms successfully, reserved1081.689752
GPU seconds. Source39e08c521; result `results/v29_20261002/cost32k_002/`.
GLOBAL/LOCAL name classification and partial GPU associations prevent a valid
GPU cost breakdown. Preserve the original diagnostic; fix/test the profiler and
add separately qualified event scopes before repeating. No profile speed claim.

Source audit confirms main fused observation shares one QK pass between complete
V output and rank32 summaries; observation itself is custom Triton, while the
sparse output consumer and native dense use FA4 through different dispatches.
See `docs/V29_BASELINE_CODE_AUDIT_20261002.md`. Main has no2K length gate; short
prompt panels cannot inherit the HF AIME gated-method conclusion. New BF16-mode
oracle protocol adds20cases to4existing CUDA tests, source core unchanged;
GPU correctness execution pending, see `docs/V29_FUSED_OBSERVATION_AUDIT_20261002.md`.

Expanded panel infrastructure remains under CPU review and is not launched.
Separately, V28 generation closed24workers; its CPU scoring stopped on a
snapshot-symlink comparison bug, fixed on that branch7c8c608ed with18CPU tests.
Original generation/bindings are unchanged; repaired CPU rescoring is pending.

## New diagnostic and expanded-panel freeze (2026-10-02 22:45, UTC-5)

Expanded runner/scorer CPU validation passes33tests, including canonical model
snapshot paths, complete scorer-only artifact mirrors, original byte pins,
implicit Q128 defaults, native task contracts and same-host qualification proof.
Committed specs select engine seeds29001-29008, all59LongBench/all30AIME/all164
HumanEval questions and all4arms (8096timed plus8096warm total); suites assigned
dllm/dlm2/mpk respectively. No formal or qualification GPU worker is launched
yet; full qualification and real scorer toys remain required. Main stays Q128,
legacy canvas, torch copy/merge, no2K gate. New code never silently changes it.

Corrected independent profiler has16CPU checks: actual model.layers naming,
copy API wait caveat, optional same-stream event spans and fused-observe/DP-build/
DP-route leaf scopes. New four-engine PIECEWISE diagnostic (dense-nohook/native/
allkept/main) is prepared, not yet launched. Default FULL result remains unknown
for layer breakdown. Timings are diagnostic only, never formal panel evidence.

BF16 observation correctness protocol now covers actual scale1.0 and the prior
scale512**-.5:40BF16 cases plus4original=44. First CPU preflight found no pytest
and stopped before GPU work (0s). A separate own runner dependency directory
now supplies pytest; qualified vLLM environment unchanged. New attempt002 GPU
correctness awaits this source freeze. No tolerance was changed.

V28 complete preview was repaired/scored/pushed on its own branch91db65396.
See `docs/V29_V28_PREVIEW_INTERPRETATION_20261002.md`: Q64 request ratio.94245
against matchedQ128, but S/N1.00516 and seed/length signs vary; six questions
only. Canvas release W1.01228[.99035,1.03470], no supported gain. Main/allkept
W.99576[.91015,1.08942], no demonstrated increment. Keep matched native whose
preview accuracy is43/48 versus main40/48 and Q6441/48. No noninferiority claim.

## GPU correctness passed; event diagnostic active (2026-10-02 22:51, UTC-5)

Fused observation44GPU tests pass on H100, source54163f4e7:40BF16 FP32-oracle
cases plus4original STORE/LOAD tests. Includes actual scale1.0, full V output,
GQA, splits1/2 and boundary tails. Fixed tolerances unchanged; synthetic numeric
qualification, not task accuracy, bit-exactness or speed. Reserved32.737210GPU
seconds. First missing-pytest attempt stopped on CPU (0GPU); second used isolated
runner pytest directory, leaving the qualified vLLM environment unchanged.
Source `results/v29_20261002/fused_observe_bf16_001/`.

Corrected four-PIECEWISE-arm event diagnostic003 is active on dllm, frozen
54163f4e7,16CPU checks passed, idle gate passed. It includes native dense with
no hooks and GLOBAL/LOCAL plus fused observation/DP leaf event spans. Profiler
overhead remains excluded from formal evidence; no breakdown result yet.

LongBench59x8x4 source/binding frozen on dllm, no GPU qualification launched.
First CPU preparation missed an archived provenance spec and stopped (0GPU);
new002directory includes the committed E14 source, all parent input/source
hashes pass, strict rebind passes and full1888timed/1888warm inventory verifies.
Generation-host CPU checks and scorer qualification remain pending. AIME and
HumanEval CPU deployment preparation proceeds on their assigned hosts; no formal
panel launched. No existing run or frozen binding was edited.

## Expanded qualification and provenance freeze (2026-10-02 23:14, UTC-5)

HumanEval four-arm qualification is running on mpk from frozen54163f4e7,
following33CPU checks, real correct/wrong scorer and bwrap toys, private-cache
and single-idle-GPU checks. LongBench qualification is queued on dllm behind
independent diagnostic003; no formal panel has launched. The four diagnostic
workers closed successfully; complete anonymous event aggregation is pending.
Existing generation sources and bindings remain unchanged.

Explicit external provenance rebinding is CPU-qualified:42tests run,41passed
and1Windows symlink-permission skip. Default behavior is unchanged. Optional
complete mappings require unchanged original bytes in a new own deployment's
.external_sources; method/runtime fields stay fixed. This prepares AIME CPU
deployment without moving gold or executing inherited historical binaries.
Actual v21 validation and all42Linux checks are required before GPU qualification.
See docs/V29_EXTERNAL_PROVENANCE_REBIND_20261002.md. No new speed/quality result.

## Completed event diagnosis and current qualifications (2026-10-02 23:17, UTC-5)

Diagnostic003 closed all four PIECEWISE arms at source54163f4e7, captures0 and
adapter ordering errors0; reserved1092.569460863946GPU seconds. Complete scope
counts match actual N/C/prefill. Matched native/all-kept/main denoise GLOBAL
span/N is4.494699/4.894036/4.329062ms and LOCAL6.461866/6.618996/6.611120ms.
These instrumented, different-trajectory observations are not paired speed gains.
No-hook dense phase attribution remains unknown. Main's40 fused observations
sum99.534944ms (19.99% of inclusive GLOBAL span), including normal dense output:
this is not incremental observation overhead. DP build40 and route105 match
receipts; side-stream overlap and nested consumer/selector ranges prohibit
adding these scopes as wall time. Full source/caveats:
results/v29_20261002/cost32k_events003/. No new task-quality result.

HumanEval four-arm qualification remains active on mpk. LongBench qualification
001 passed host CPU tests and one dense worker, then stopped at the next idle
check during transient NVML activity; preserve its241.732GPU seconds. New002
qualification on dllm reruns all12 workers at unchanged54163f4e7/source/binding,
with bounded stable-idle checks; any foreign compute process stops the launcher.
No formal panel has launched. Score full qualification before formal work.

Automatic approval review rejected AIME cross-host input transfer, then its
private config/provenance-only subset; both stopped before SSH/transfer,0GPU.
Do not retry those transfers. Prepare same-machine AIME using existing mpk
inputs/config/pins. New immutablev29_aime_confirmation002 spec changes only
name/protocol_id and all8block hosts from dlm2 to mpk; original001 retained.
Seeds/questions/method/arm order/budget/scoring are unchanged. CPU bind and GPU
qualification remain pending. Queue behind current mpk worker, never overlap.

## User clarified private host transfers (2026-10-02 23:20, UTC-5)

The user explicitly authorizes transfers needed for research between their own
three machines. Private data/configuration/artifacts still must not enter Git
or public outputs. Writes stay inside the user's own dyh directories; chw/ljy
and all other collaborators' files are read-only and must not be deleted or
overwritten. Prior auto-review rejections remain recorded with zero execution;
new attempts must use normal approval review with this updated authorization.

Resume original AIME001 on dlm2 at frozen1931db5a6, with exact-source provenance
mirrors, original byte checks, actual v21 validation and Linux42 tests before
GPU qualification. AIME002 same-machine fallback is retained but not launched.
Gold can remain on its existing scoring host; generation/scorer artifact mirrors
are now explicitly authorized. HumanEval/LongBench qualification continues at
frozen54163f4e7. Formal launch still requires complete scored qualification proof.

User also requests timely commit/push and reasonably sized traces/measurement
records, not only conclusions. Full diagnostic trace privacy/size audit is
in progress; publish sanitized compressed traces and detailed numerical receipts
where suitable, retaining private originals unchanged.

AIME002 fallback CPU preparation did run on mpk before the latest switch: strict
rebind, actual v21 validation and all42Linux tests passed without CUDA initialization.
No binding, generation input/gold transfer or GPU launch was performed there.
This preserved CPU check does not qualify a full model request.

## Detailed evidence published and qualification progress (2026-10-02 23:34, UTC-5)

Publish the actual per-arm profiler outputs, sanitized request measurements and
terminal receipts from both closed diagnostics, not just their conclusions:
results/v29_20261002/cost32k_002/raw/ and cost32k_events003/raw/.
There are24measurement files,80,316uncompressed bytes and28,647bytes of gzip
copies, plus manifests/README. Root independently verified all24 gzip roundtrips,
public artifact digests and private-field/path scans. Full numeric measurements,
counts, method/allocator/layout receipts remain; private identities and data are
removed. No timeline was originally exported, so none is reconstructed. Resource
accounting uses each supervisor's complete reserved span, not inner terminal
clock values. These profiled requests remain excluded from formal speed/quality.

AIME CPU setup passed actual v21 validation and all42Linux tests on both original
mpk and destination dlm2; public preparation receipts and test inventory retained.
Destination frozen1931db5a6 now has a verified AIME001 input binding. An initial
minimal-input transfer stopped at strict byte verification due to Windows CRLF
versus Linux LF serialization (0GPU). New002 transfers original bytes, validates
both the byte pin and parsed30rows, and passes freeze/read_frozen; original001
is preserved. No gold contents were transferred. AIME GPU qualification pending.

HumanEval qualification generation closed4/4 with no failure; strict CPU scoring
and formal proof remain pending. LongBench qualification002 has closed5/12 with
no failures and is running64K native; frozen54163f4e7 unchanged. Its closed-worker
reserved time currently944.897625s; active-worker time remains unclosed. No formal
panel has launched. Preserve the separate previous241.732s idle-guard attempt.

## HumanEval qualified; trace exporter frozen (2026-10-02 23:51, UTC-5)

HumanEval's full four-arm qualification passes strict54163f4e7 scoring, original
bwrap sandbox, public correct/wrong toys and the actual formal-launch guard.
All four single-question outputs pass official tests; this is pipeline evidence
only, not accuracy noninferiority or speed. Publish detailed numeric requests,
receipts, workers and aggregate tables under humaneval_qualification001/.
Reserved outer GPU time672.373426352s; inner worker663.996s is not additive.
Dlm2 CPU scoring stopped on unavailable OS sandbox; mpk CPU scoring uses the
unchanged isolation after an authorized pinned-gold copy. An independent exporter
integer-key bug was repaired without changing the strict proof or GPU generation.
Formal launch is held until all three suite qualifications/scoring are ready, to
avoid heavy mpk CPU scoring contention during HumanEval formal timing.

LongBench qualification002 has11/12 workers closed without failure, final96K
method running,1992.535591133 closed-worker GPU seconds. Resource correction:
qualification001 outer reserved time is244.358443453s; earlier241.732s is its
inner worker terminal span. Preserve earlier records and use the complete outer
value in current accounting, never add both. A read-only original-source-ID audit
confirms59 distinct questions (24/24/11), with zero overlap between length groups.
Eight seeds remain repeats within question clusters; these are previously used
questions, not a newly held-out question pool. Audit receipt is published.

AIME qualification001 completed dense,239.994346702 outer GPU seconds, then its
launcher stopped at transient post-exit NVML activity. Dense artifact stays valid;
new002 runs only the remaining3arms at the same1931db5a6/input002 binding and merges
the preserved dense by exact byte pins. Four launcher CPU toys cover bounded
stable-idle wait, occupied GPU rejection, timeout and worker failure. Formal pending.

The independent trace wrapper and9 new CPU tests plus16 unchanged-profiler tests
pass. Both wrapper and original profiler must be source-pinned; spec/CLI flags and
ordinal agree; output is a separate wholly new private directory with no overwrite.
Trace export is off by default and occurs after the original request boundary.
No existing GPU deployment is changed, no GPU trace run has occurred, and no trace
is automatically published. See docs/V29_TRACE_EXPORT_20261002.md.

## Three formal panels launched (2026-10-03 00:12, UTC-5)

All three complete qualifications passed strict scoring and the actual frozen
formal-launch guard. LongBench12/12 and AIME4/4 detailed measurements, per-request
receipts and worker accounting are published under longbench_qualification001/
and aime_qualification002/. HumanEval4/4 detailed evidence was already published.
Wrong, unparsed and capped outputs remain failures; qualification is only pipeline
validation. In particular all four AIME qualification outputs capped/unparsed.
No accuracy or speed conclusion follows from these singleton qualifications.

The three formal panels now run: LongBench59 questions on dllm, AIME30 on dlm2,
HumanEval164 on mpk; each has8 engine seeds29001-29008 and4 arms,32 workers per
suite,8096 timed plus8096 warm requests total. LB/HE source54163f4e7, AIME1931db5a6.
Main stays frozen Q128/alias2 with legacy torch copy/merge and no2K route gate.
All qualification proofs/source pins, GPU identity/idleness and CPU scorer idle
checks passed. Private coordinator17CPU tests pass. mpk launched last; a shared
CPU-scoring barrier remains active until all three strict32/32 inventories close.
No existing source, binding or run directory was changed. Any failed worker stops
its supervisor and requires a new run directory; retain failed-attempt evidence.

Closed qualification reserved GPU seconds: HE672.373426352, LB2413.676399794
(including original interrupted001 outer244.358443453), AIME780.407498523
(including reused dense001239.994346702 exactly once). Together3866.457324669s.
With prior V29 components/diagnostics2262.065190672s, pre-formal closed total is
6128.522515341s. Formal workers are active; their unclosed time is excluded.
Use outer supervisor spans; inner worker times are not additional consumption.

Current new-model status clarified: LLaDA2.1-mini and I-DLM-8B official dense toy
GPU executions have completed. Sparse main/variant adapters on these models are
NOT implemented; CPU event-prototype tests do not qualify GPU integration.
See docs/NEW_MODEL_PORT_AUDIT_V28_20261002.md. Older pending-smoke notes are history.

Next: preserve all formal records, complete strict full-family scoring after the
three-suite barrier, export anonymous per-request numbers/receipts plus paired
geometric means and question-cluster95% intervals. Completion/scoring automation
is being prepared; no automatic interpretation or noninferiority claim.

## Detailed evidence index and receipt correction (2026-10-03 00:20, UTC-5)

Public evidence navigation: results/v29_20261002/EVIDENCE_INDEX.md, with executed
source commits, saved artifact links and interpretation limits. Both old diagnostic
campaigns saved measurement aggregates/records, not individual Chrome timelines.
Do not reconstruct event timing from their aggregates. The44GPU tests comprise
40BF16 independent FP32-oracle cases and4original TF32x3 STORE/LOAD comparisons.

AIME now also publishes path_execution_receipts.jsonl with actual adapter,
effective-method, KV-layout and method counters. The separate correction explains
the earlier README error: only dense has null receipts; native/allkept/method do
have them. Previously published per-row availability flags were already correct.
Old files and original generation are preserved; strict validation had checked
the real receipts all along. This is an export/documentation correction, not a
method change or new accuracy/speed evidence.

All three formal panels remain in their first full-question worker,0closed and
0failed at00:19 snapshot; active GPU time is not yet closed. Completion/scoring
helper is being CPU-tested before arming; current source/bindings remain frozen.

## Performance follow-ups and Junyu source audit (2026-10-03 00:44 (UTC-5))

The new same-state cost benchmark, raw component timing helper and sparse-cycle
Q movement prototype are CPU-qualified, not GPU-qualified.47 focused public CPU
tests pass (15 cycle,12 homogeneous,5 timing,15 engineering specs). Frozen public
protocols and six legacy/fused real-model qualification specs are committed with
the source. No new model/request gain is claimed. Component samples retain every
arm/repetition/order and host/event duration; no Chrome timeline is fabricated.
The official JIT monitor must activate before FA4/consumer imports, numerical
oracles precede warm timing, and all monitored timed JIT/capture deltas must be0.
See docs/V29_PERFORMANCE_NEXT_20261003.md and the new public specs.

Junyu ljy/value_aware was reviewed read-only at859c0c8fc2a4509a148a3270915f79961e565dee.
Its latest three commits only change READMEs; the newer Hopper/C_gate sources and
tests linked there are absent from that ref. Historical current-QK retained-state
routing is distinct from our dense-prefix surrogate and temporal reuse. Its old
quality/budget/backend results do not establish performance against vLLM/FA4.
The documented C_gate is a cooperation candidate, not a verified port or new
contribution of this branch. See docs/JUNYU_VALUE_AWARE_REVIEW_20261003.md.

The user's dense/native equivalence challenge requires a separate audit. Native
hooks currently call original FlashAttentionImpl.forward, but graph mode/hook and
RNG-trajectory equivalence are not proved by that fact. V18b warmed59 questions
for dense/main and only12 for the control subset, without per-request reseeding;
matched labels were not matched random states. Preserve its measurements but do
not causally attribute default/native W or N differences to PIECEWISE, sparse
quality, or seed noise alone. V29 already uses identical full warm inventories
for all arms; more seeds still do not establish graph/hook equivalence.

Current formal panels remain immutable and running. Closed workers at this
snapshot: LB1/32,AIME3/32,HE4/32; no reported failures.
Their outer closed-worker GPU seconds are recorded in STATE.current; active
unclosed time is excluded and inner worker times are not added. One-shot strict
completion/scoring is armed (17 CPU tests passed), waits all32/32/32 workers,
then scores HE/LB/AIME serially and exports detailed sanitized evidence for
review. New component and engineering queues are still being qualified; not yet
armed in this commit and no new GPU work overlaps the formal campaign.

## Dense/native accounting correction and component queue (2026-10-03 00:57 (UTC-5))

The public raw002/003 records were re-tabulated by
scripts/v29_compare_dense_diagnostics.py into
results/v29_20261002/dense_native_accounting001/ (8request rows,27event scopes).
These are original profiled singleton durations, explicitly excluded from formal
performance claims.003 includes PIECEWISE dense without adapter hooks: N374,
output4761,profiled S/N30.029ms. Same-mode native-hook has N99,output2107,
S/N32.657ms; main has N115,output2006,S/N33.248ms. Thus003 alone has slower
main amortized S/N, despite its lower GLOBAL span/N. Different trajectory and
instrumentation prevent causal kernel/request claims. The intermediate no-hook
control is missing from the formal panel, not from all previous diagnostics.

The native374-to99 change between002/003 remains unexplained. Do not dismiss it
as seed noise or assume PIECEWISE explains a same-PIECEWISE discrepancy. Warm
request N/output length and sampling RNG state were not published, so the new
table marks them unknown. Equivalent dense math with identical random inputs
and deterministic execution should reproduce; current engine-seed labels alone
do not establish those conditions. Investigate graph dispatch, hooks, sampling
state and first numerical divergence in a separately frozen diagnostic.

Correct a small old narrative transcription error without altering original
artifacts: raw native LOCAL639.6247041076422ms/99=6.460856ms/N, not6.461866.
All nested/side-stream leaf spans remain non-additive; no fake wall breakdown.

The two frozen component jobs are now armed on source1342ec0b193a929295f08964c132aee96c28c5a1,
waiting for all current formal generation and strict serial CPU scores before
any remote deployment/GPU launch.21 private queue tests passed; own-directory,
registered GPU/idleness, source, full numerical-oracle, every raw timed sample,
zero JIT/capture and failure/resource preservation gates are enforced. No new
component GPU result exists yet. The real-model fusion queue is still being
prepared; no source/binding/run of the current formal campaign is changed.

Source-audit supplement: docs/V29_DENSE_NATIVE_AUDIT_20261003.md documents the
actual installed vLLM sampler and runtime dispatcher. Native forwards the original
attention arguments and sampler result unchanged; six CPU wrapper-contract tests
pass. This does not rule out GPU execution, graph, hook or RNG bugs.002/003 inputs
and core source bytes match, while profiler instrumentation differs. Official
sampling uses global GPU random draws without a supplied request generator; no
saved RNG boundary state identifies the first divergence. Default initialization
FULL_AND_PIECEWISE is not evidence of each forward's actual graph mode. A separate
three-condition graph/RNG diagnostic is being implemented; no new GPU validation.

## Dense reproducibility diagnostic frozen; engineering queue armed (2026-10-03 01:16 (UTC-5))

New public worker/specs: scripts/v29_dense_reference_diag.py and three
v29_dense_reference_diag_* specs. Same original32K index0, two engine seeds
28001/28002, two fresh engines per seed/condition,12workers. Conditions are
default dense, PIECEWISE no-hook dense and PIECEWISE native-hook. Preserve the
original sampler; read only default-device CUDA-generator state at existing
request boundaries, retain fingerprints privately, publish only comparisons.
Record actual graph modes per execution phase, warm and timed N/C/P/output
length, and original counts/JIT receipts. No new per-step synchronization or
profiler is added; boundary instrumentation is diagnostic, never formal W.
Twenty-one public CPU tests pass. Read-only actual cost003 spec comparison
confirms exact equality with all three public protocols after make_spec; old
input, method and deployment are unchanged. The private12worker coordinator
passes20CPU toys and honors the frozen block/repeat/variant launch order.
It is not armed yet in this source-freeze commit. GPU validation remains pending.
See docs/V29_DENSE_REFERENCE_DIAG_20261003.md.

Engineering fusion qualification is now armed after30private CPU tests. It
waits three closed predecessors: original formal generation+strict scoring,
two component jobs, then the complete12worker dense diagnostic. It will run
only40warm+40timed qualification requests and strict score all six families;
no full variant panel or automatic public claims. Its source remains1342ec0b1.
Component source also remains1342ec0b1. No current formal deployment changes.

New telemetry note distinguishes sparse-path execution from unknown per-tile
keep fractions, and documents observation storage: exact32K prefix geometry
needs268MiB summary storage per GLOBAL layer (FP32z/mu32+int8flags),1340MiB for
five layers, excluding DP/output/allocator costs. This is capacity arithmetic,
not measured traffic or causal latency. Original003 summary_peak_bytes is
1569751040. QK sharing does not make statistics storage/DP/full output free.
See results/v29_20261002/dense_native_accounting001/TELEMETRY_LIMITATION.md.

## Dense diagnostic and performance follow-ups rearmed (2026-10-03 01:26 (UTC-5))

The first component coordinator failed during a local Windows atomic status
replacement (PermissionError), before deployment or GPU launch. Engineering
stopped at its dependency gate; the first dense coordinator failed check-only.
All original scripts, plans, statuses and failed evidence are preserved.
Successor002 coordinators use new local/remote directories and bounded retry
for transient status-file replacement contention, with retry/exhaustion tests.
All three successor queues are armed and waiting; no new GPU evidence exists.
See results/v29_20261002/queue_recovery002/status.json for sanitized receipts.

Order remains original formal generation and strict CPU scores, then two
component jobs, then12 fresh-engine dense diagnostic workers, then only the
engineering fusion qualification and its strict scores. Component/engineering
science stays at1342ec0b1; dense diagnostic source isbdd59ee97. No current formal
deployment or scientific protocol changed. Public diagnostic CPU tests:21;
private successor tests: components26, dense23, engineering35.
Only source/status evidence is committed here, not hypothetical GPU results.

## Junyu newly published implementation audited (2026-10-03 01:39 (UTC-5))

Read-only peer pin b890ff49488474c5d476df44019597dc045a5969 supersedes the
previous859c0c8fc publication boundary. All15 missing linked implementation/test
targets and the named report test are now present;118 changed Python files
parse, with no peer code import/test/build/GPU execution. New diff has127files
and no results changes. No peer branch merge, source absorption or modification.
See docs/JUNYU_VALUE_AWARE_CODE_REVIEW_20261003.md and the numeric publication
receipt results/v29_20261002/junyu_publication_update001/audit.json.

Confirmed static defect: ABI4 kernel is rejected by the ABI3-only debug_scores
guard used in same-logit qualification/tests. Ordinary no-debug calls are not
implicated by that check. Historical dense cache reuse has incomplete protocol
identity validation; this is a code risk, not proof that prior outcomes are wrong.
Peer kernel physically skips full PV but still computes current QK/rank32 risk;
its SDPA native adapter is not our vLLM/FA4 baseline. New C_gate agrees with our
P17 port mathematically at matched parameters, while complete routing systems
differ. P17 remains configuration-specific; current vLLM adapter still rejects
C_gate pending accepted-mask lifecycle support. Treat integration as a cited
cooperation candidate requiring separate qualification/calibration, not a speed
or accuracy result. Current formal deployments and follow-up queues unchanged.

## Additional peer kernel and timing boundaries

The peer kernel advances through KV tiles sequentially with per-tile cluster
votes (value_direction.cu:306–316,404–416,571–578). Its split variant separates
router/PV roles across CTAs (:623–666); this is not independent KV-range
splitting followed by an LSE merge. validate.py:63–83 measures graph replay,
excluding ordinary Python dispatch and the host-side TMA descriptor construction
in value_direction.cu:754–758. Cached-sketch component timing is not dynamic
request latency. All locations refer to the audited b890ff494 peer ref.

A cooperation candidate is register-fed rank32 computation and on-chip running
state (value_direction.cu:359–401): investigate whether our observation/DP
boundary can avoid some intermediate-summary materialization. Our all-dense-prefix
risk dependencies differ from the peer retained-state router. Preserve our
mathematics and independently qualify any experiment; no equivalence, port,
new variant or measured gain is claimed. Running queues remain unchanged.
