# Decisions and negative results (chronological)

## 2026-10-10 — Freeze AIME26/RULER128K value-selector performance inputs

- Freeze `results/v31_value_performance_20261010/` as a new, resumable
  performance panel. AIME26 uses all 30 problems × seeds 42/43/44 and the
  1,728-token fixed-budget control target (46.1964% pooled physical GLOBAL
  sparsity). RULER128K uses 104 deterministic cells, eight per each of the 13
  pinned RULER tasks, with the existing 8,192-token / 128-KV-block budget.
- Exact greedy remains excluded. The panel includes dense FULL/PIECEWISE,
  all-kept FA4, the current control, V1, V2, V3a, and approximate batch-8.
  V1/V2 thresholds must be calibrated separately per workload against measured
  fixed-budget GLOBAL work; no requested budget is reported as achieved
  sparsity.
- Both pools passed CPU binding checks. GPU generation is currently blocked by
  the execution environment (`nvidia-smi` has no device and PyTorch reports
  zero CUDA devices). No accuracy or timing conclusion follows until an
  authenticated H100 resumes the frozen shards.


## V31 LOCAL compact consumer (2026-10-06)

- The prior direct FA4 LOCAL consumer was slower than dense at the 512-token
  budget because its block-list setup dominated the short `(1023, 1023)`
  window. A named `local_compact_triton_q64` consumer now packs the existing
  Q128 selector map into Q64 retained-tile lists and performs paged K/V,
  window masking, and online softmax in one Triton kernel. The selector and
  budget are unchanged; `LOCAL_KERNEL=fa4` remains an explicit fallback.
- On one H100 with Q256, H16/Hkv8/D256 and 100 synchronized alternating
  samples, compact-vs-dense median speedups are 1.30x / 1.10x / 1.25x at
  prefixes 187 / 3500 / 8192. The maximum absolute difference from the old
  FA4 sparse output is 0.001953125. This is held-map kernel evidence only;
  no AIME accuracy or end-to-end claim follows. See
  `results/v31_20261006_local_kernel_opt/attempt001/`.

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

## V30 cooperative campaign active — 2026-10-03 02:17 (UTC-5)

Branch `research/cooperative-sparsity-20261003` starts from verified V29
`035b70c3f`. User authorizes attributed peer-document snapshots and isolated
execution; root owns algorithm/fairness review. Peer branches stay read-only.
See docs/V30_COOPERATIVE_CAMPAIGN_20261003.md.

Correction: profiled singleton32K cannot establish that main is slower at64K/96K.
Closed V28 preview main/native S/N ratios are0.902307 and0.864398; N ratios
1.197263 and0.955624; W ratios1.084097 and0.849983. Each bin has only two
questions, so these are preview point estimates. V18 controls have different
warm/request inventories; its four repeats are two seeds times two repeats.
Full accounting, numeric tables and five CPU tests are committed under
results/v30_20261003/closed_length_accounting001. Physical density was not
recorded in these requests; high HF sparsity is not evidence for their density.

Junyu b890ff494 docs copied with attribution and redaction. Original peer CPU
formula tests actually executed:31passed, no CUDA initialization, GPU seconds0;
per-test receipt in results/v30_20261003/junyu_cpu001. Kernel/E2E remains pending.
ABI4 debug guard defect needs an isolated patch before affected qualification.
Peer reproduction design is explicitly a draft, not launched or frozen.

Formal workers continue on their original sources and eight seeds. The previous
completion monitor failed on a read-only SSH timeout, not a GPU worker failure.
New completion002 retries only read-only transient transport errors. Components003,
dense003 and engineering003 are armed after strict formal scoring in that order;
new directories preserve failed originals. No scientific settings or deployments
were changed. Snapshot and source pins:results/v30_20261003/queue_recovery003/status.json.
Successor CPU checks passed28/26/23/35. Counts at snapshot are operational status,
not partial performance results. Three GPUs remain reserved for formal workers.

Next: isolate peer ABI/toolchain fixes and GLOBAL-only routing tests; compile
without touching shared environments; execute numerical qualification only on
an idle GPU. Await frozen dense12-engine diagnostic before attributing the large
dense trajectory discrepancy. New regroup candidates must include construction,
movement and restoration costs against the same optimized natural-order consumer.

## Peer build and bounded regroup screen — 2026-10-03 02:37 (UTC-5)

Root implemented the isolated build, ABI guard fix and GLOBAL selector wrapper.
Junyu b890ff494 CUDA source compiled successfully with own nvcc13.0.88 and
NVIDIA TensorRT-LLM headers796584295. Peer default flags all remain false;
fast-SFU/inline variants are not silently substituted. C++/CUDA source unchanged.
Only the Python debug buffer check now accepts both ABI3 and ABI4, preserving
shape/dtype/device/contiguity checks. Complete chosen header trees are privately
hashed; public receipt, source hashes and29.6KB compiler log are in
results/v30_20261003/junyu_build002. Preparation001 failed before compilation
on the archive root directory entry; it is retained, with0GPU seconds.
Compiler warnings include WGMMA serialization and register spills; they do not
establish a measured bottleneck. The built kernel has NOT run on GPU yet.

GLOBAL-only scope is implemented in peer_global_scope.py using a narrowed
adapter module selector, leaving LOCAL untagged and native call eligibility
unchanged. Four CPU tests cover exact module identities, pass-through arguments,
result identity, scope rejection and cleanup. ABI/build guards add three tests.
This is an HF peer-reproduction component, not a vLLM port or model parity proof.
Twenty-nine original peer kernel acceptance cases collected successfully with
CUDA hidden. Their new frozen spec rejects skips and changed parameter counts;
three queue-contract tests pass. GPU execution will follow the original formal
strict scoring barrier on idle dlm2; it is not armed in this source-freeze commit.
A matched ATen bridge and complete model/scale/scope qualification remain pending.

New independent regroup_key6 CPU prototype samples six fixed support bits,
stably groups inside each head/Q128, rebuilds exact unions, and accepts only
blocks with non-worsening total/max alias2 tile work. Five tests cover coverage,
bijection, ties, nonnested equal-count supports and adversarial/random supports.
Executed all144existing held-inventory snapshots:603/4608blocks accepted,
12.1179%rows moved, aggregate tile ratio0.997058, load-proxy sum ratio0.996582.
CPU construction median1.774874ms (range1.045137–3.961188), excluding loading.
These are prefix-only development proxies, NOT GPU/request gains. Improvement
is too small to justify claiming that row movement/build costs are covered.
Preserve raw anonymous per-snapshot values in results/v30_20261003/regroup_key_cpu001.
Haowei document snapshot pins6f7279c16 and is explicitly attributed.

## Peer kernel GPU qualification armed — 2026-10-03 02:43 (UTC-5)

The29-case peer C ABI acceptance queue is armed on sourcee2ec0ae4b, currently
waiting for all original formal strict scores. Prepared source/kernel bytes
and pinned SM90 were checked remotely; no GPU was started. It then requires two
idle checks and an exclusive own worker lock on dlm2. Launch is single-attempt;
unknown transport outcomes are preserved, never automatically relaunched.
Every frozen case must pass; skips/duplicates/wrong parameter counts fail.
Per-test receipts and outer reservation seconds are retained. No automatic
ATen/model/E2E rollout or public performance claim follows a passing kernel test.
See results/v30_20261003/peer_gpu_queue001/status.json. Original dllm component,
dense-equivalence and engineering queues retain their existing order and pins.
Formal snapshot:LongBench6/32,AIME18/32,HumanEval18/32,zero closed-worker failures;
these counts are operational progress, not partial scientific results.

## Dense divergence and M3 query-weight ports — 2026-10-03 03:16 (UTC-5)

Root reviewed installed vLLM sampling and adapter code. Native-hook prepare
performs a GPU-to-CPU metadata stack.tolist() each forward; no-hook dense does
not. The official compiled sampler draws from CUDA RNG without a request-local
generator argument. These facts do NOT prove the cause of divergent steps.
The new independent sync-only arm isolates the metadata read. Separate invasive
first-sample protocols retain initial RNG/canvas and first-sample input/output
privately; traces change synchronization and cannot establish timing gains.
Four arms, two engine seeds and two fresh-engine repeats per protocol (32
engines across sync-only/trace protocols); original V29 queues remain pinned.
New diagnostics have CPU tests, but are NOT deployed/armed in this commit.

Two named query-weight variants now have vLLM code: unit_v30 and confidence_v30
(previous completed call's processed p_top, 1+3*sqrt(1-p_top); first call fully
protected). This ports the confidence prior, NOT Junyu's C_gate or fresh
value-aware router. Preserve M3 DP/R6/A64/Q128/carry0 and torch copy/merge.
C_gate remains rejected in vLLM because the actual accepted mask is unavailable.
All three modes have new full-inventory eight-seed specs for LongBench (59),
AIME (30), HumanEval (164). Qualification/freeze/deploy/launch remains pending;
no variant GPU or E2E performance result exists. Short-task accuracy is required.

Remote CPU regression: 25 passed, no skips, CUDA uninitialized, 0 GPU seconds.
Per-case evidence: results/v30_20261003/sensitivity_cpu001. Local diagnostic,
privacy-comparator and spec/receipt tests: 12 passed. Existing adapter guard,
lifecycle and merge tests also passed. Last direct GPU check found one active
original worker on each of dllm/dlm2/mpk; no new GPU launch. Verified parent:
ed711c8b4afd5bf93a457f03bfc5714a5b36dcb9. Do not mislabel the new source/config freeze as a byte-identical rebind.

## V30 source-freeze guard correction — 2026-10-03 03:21 (UTC-5)

First CPU-only preparation on dlm2 stopped before GPU: the original parent
provenance also pins v21.py, so a byte-identical parent rebind correctly rejected
the reviewed wrapper change. Preserve failed campaign001. New config migration
now verifies every old pin, explicitly allows only reviewed v21.py/adapter.py
changes at any nesting, keeps all other dependencies byte-identical, and asserts
all mathematical fields unchanged except the named sensitivity. Two CPU tests
cover nested fingerprints, old-byte drift, unreviewed kernels and basename
spoofing. New immutable preparation is pending; no GPU was started.

## V30 configuration qualification and final shape guard — 2026-10-03 03:33 (UTC-5)

All three hosts successfully CPU-froze three sensitivity bindings each and dllm
also froze eight diagnostic bindings at26f3f03b6; CUDA stayed uninitialized.
No GPU execution. These unused campaign002 bindings are retained and superseded
by the next source freeze adding a pre-routing confidence geometry check.
A within-canvas Q-size change now fails before sparse selection; a new canvas
may change Q normally. Updated CPU regression26/26passed, no skips,78stages
CUDA-uninitialized,0GPU seconds. Per-case receipts and source pins:
results/v30_20261003/sensitivity_cpu002. New immutable deployment follows.

The prepared private continuation has32dense diagnostic workers and60pipeline
qualification workers (three modes x four arms x five datasets), then nine
complete strict CPU scores. Full eight-seed specs are committed; qualification
is one longest input/dataset, one warm+one timed and is NOT a quality/speed claim.
It waits for original formal strict scores, earlier component/dense/engineering
queues, and the29-case peer kernel qualification. It has six local guard/payload
tests but is NOT armed at this commit. No frozen prior queue is modified.

## V30 follow-up queue armed — 2026-10-03 03:38 (UTC-5)

The new continuation is actually armed on94134a91b, phase waiting_prior_queues,
0newGPU seconds. All three own-host campaign003 CPU preparations passed, with
nine full-inventory sensitivity bindings and eight dense diagnostic bindings;
CUDA uninitialized. Source CPU tests33passed; private queue/payload guards6;
remote sensitivity/config/adapter regressions26. Earlier source freezes and
running worker deployments are unchanged. Receipt:
results/v30_20261003/campaign_queue001/status.json.

After original formal strict scores, original component/dense/engineering work,
and peer29-case qualification finish, run32fresh-engine dense diagnostics,
then60sensitivity pipeline qualification workers and nine complete strict CPU
scores. No automatic formal launch or quality/speed claim. Full eight-seed
specs exist for59LongBench,30AIME,164HumanEval questions per mode. Raw first
logits/canvas/RNG traces and hashes stay private; sanitized phase/count/graph
traces and boolean comparisons are retained. The first sampled state only
locates an initial difference; it cannot rule out a later divergence.

Original formal snapshot: longbench 8/32, aime 25/32, humaneval 25/32;
zero closed-worker failures. GPU resource seconds per suite remain in the
receipt; active worker spans are not yet included. Diagnosis is pending.

Operational continuation: private campaign_queue001.py/PID file, inputs in
prepare_*_003, scorer campaign_score001.py. Do not edit these now-pinned helper
files; a correction requires a new queue/run. Inspect status, preserve failures,
then review first-divergence comparisons and all nine strict qualifications
before choosing the matched engineering baseline for full confirmation.

## Independent Junyu vLLM adapter checkpoint — 2026-10-03 04:13 (UTC-5)

User-facing concise status: [V30_WORKING_STATUS_20261003.md](V30_WORKING_STATUS_20261003.md).
Added separate no-M3 GLOBAL adapter source: bare peer dense, projected all-kept,
fresh value-only, previous-confidence and previous-temporal variants. Official
FA4 remains the external dense reference; the peer consumer is explicitly named.
Ten CPU tests passed, no skips/CUDA initialization, 0 GPU seconds. Per-case
evidence/source hashes: results/v30_20261003/standalone_cpu002. Real peer CUDA,
paged-KV numerical qualification, finite policy binding and E2E runner remain
pending; this adapter is NOT armed. Do not mistake earlier M3 sensitivity or
original peer kernel queues for its execution. Current peer SHA remains b890ff494;
uniform attained policy JSON absent, README zero is a placeholder. Public old
RULER8K policies may only be separately labeled transfer candidates.

Read-only 04:09 formal snapshot: LB10/32, AIME28/32, HumanEval29/32; 67 closed,
0 failed, 41710.304596 closed GPU seconds. Earlier frozen queues unchanged.

## Paused by user after short-suite completion — 2026-10-03 04:40 (UTC-5)

AIME32/32 and HumanEval32/32 completed generation, unscored. LongBench11/32
complete; b2_native interrupted in warm-up (0 timed rows), 20 workers unstarted.
All 75 closed workers succeeded. Three GPUs confirmed idle; all six local
follow-up/scoring coordinators stopped. NO automatic resume. Older queue
armed/active fields are preserved historical snapshots, not current liveness.

Published 6,857 sanitized request rows, per-worker accounting, effective_method
and numeric counters; 0 timed CUDA captures. Private originals and completion
files backed up locally and retained remotely. Evidence/resume instructions:
results/v30_20261003/pause_after_short001. Closed GPU seconds46574.813729;
interrupted-worker age643.62s recorded separately as approximate reservation.
Three exporter privacy/format tests passed. No new score or speed conclusion.

On explicit resume, verify pins and reuse completed workers; rerun interrupted
and unstarted workers in new directories, form a new complete family, then
strictly score. Do not simply restart original supervisors/frozen queues.

## 2026-10-06 — LOCAL sparse FA4 result

A new LOCAL router was qualified on the H100 using the native 1023/1023
window, zero-copy 64-token views of paged K/V, and the pinned FA4 Q128/K64
block-sparse consumer. The full AIME26 panel used GLOBAL budget 1728 and
LOCAL budget 512. GLOBAL physical sparsity was 45.90%; LOCAL was 25.33%
over all calls and 32.81% on sparse-only calls. The clean consumer timing
was 0.0531 ms dense versus 0.0701 ms sparse at prefix 8192 (0.757x), so
there is no actual LOCAL speedup at this budget. The complete end-to-end
arm was 0.833x dense speed and matched dense at 51/90 exact answers. Keep
this as a negative kernel result; do not tune the AIME panel to force a
positive claim. Source and receipts: `results/v31_20261006_aime_global_local/`.

## 2026-10-06 — AIME26 GLOBAL budget sweep with compact LOCAL consumer

The frozen V31 AIME26 panel ran all 30 problems at seeds 42/43/44 for a dense
FULL PR #51994 reference and MAGE + LOCAL arms with GLOBAL budgets 1024, 2048,
and 4096 and LOCAL budget 512. Thinking, native adaptive stopping, and all
sampler settings were preserved. The compact Triton LOCAL consumer was used
for every sparse arm on one H100.

The achieved count-weighted overall GLOBAL+LOCAL physical sparsities were
37.82%, 31.99%, and 22.30% for GLOBAL 1024/2048/4096. Pooled exact-match
accuracy was 52.22% dense, then 57.78%, 52.22%, and 55.56%. Arithmetic-mean
end-to-end speedups versus dense were 0.783x, 0.764x, and 0.809x; decode-only
speedups were 0.783x, 0.763x, and 0.809x. No positive end-to-end speed claim is
supported. Full receipts and the paired official comparison are in
`results/v31_20261006_aime_global_budget_sweep/`.

## 2026-10-06 — Dense-LOCAL control for GLOBAL budget sweep

Run a fresh matched AIME26 panel before attributing the prior slowdown to the
LOCAL consumer. Keep the native dense LOCAL path by omitting `LOCAL_KV_BUDGET`
and use the same V31 MAGE GLOBAL selector at budgets 1024, 2048, and 4096.
The complete result is `results/v31_20261006_aime_global_dense_local_budget_sweep/`.

The dense reference scores 51/90 (56.67%). GLOBAL-only sparse scores 51/90,
53/90, and 54/90; arithmetic end-to-end speedups are 0.920x, 0.931x, and
0.917x. GLOBAL physical sparsity is 58.05%, 41.33%, and 17.81%; LOCAL is 0%.
The matched control therefore remains slower than dense even with native dense
LOCAL layers. The result does not support a positive end-to-end speed claim.
# Independent V31 selector qualification — 2026-10-08

The new study independently implements the requested V1/V2/V3 formulas and
keeps the inherited qblock-max control unchanged. A completed 15-item
development control receipt at source `205e3502` measures 76.45235% GLOBAL
decode sparsity, including dense observations and current canvas tiles. This
measured denominator is the online calibration target. No fixed-k constraint
is applied to V1/V2, and no target threshold has been frozen.

Singleton scores are computed once. A support-constrained descending ranking
preserves every live row when unconstrained top-k would jointly remove its
support; ties favor smaller retained tile IDs. Independent CPU/H100 regression
tests pass. Exact greedy retains its cumulative scoring loop; batch8 is a
separate approximation. These are qualification findings, not benchmark claims.
Frozen failed attempts and existing evidence are retained. Private scorer
records and caches stay outside Git. Exact RULER v33 and HumanEval inputs remain
unavailable; no replacement prompts or seed schedules are invented.

## 2026-10-08 — Restore permitted calibration and separate clean target timing

The15-cell preliminary reference in the preceding qualification note is
superseded and excluded from calibration. Eleven items belong to the inherited
holdout. The restored original S1 schedule contains32 items at seeds1,2 and
matches historical schedule SHA
`d0f0f7724d3350028091d6d15a6d8a0bdc4054b8764eec0097941a494e812401`.
The15-cell pilot remains diagnostic; it never froze a target threshold.

All nine model arms passed one-cell qualification at `d63d8b1e`, including the
correct native FULL/PIECEWISE references and all-kept consumer. Subsequent
source `01f37fe3` passes63 focused H100 tests. The new clean switch removes GPU
kept-tile accounting without changing split lists or output: the paged consumer
parity test is bitwise equal. Full503-cell target audits are frozen separately
from full503-cell clean timing. Audit-only native geometry and actual graph-mode
receipts provide denominator and kernel-path checks. Uniform online calibration
remains gated on a completed audited64-cell original development control.
No target performance or accuracy claim is made from these qualifications.

## 2026-10-08 — Accept coarse online thresholds for the first effectiveness experiment

The user accepted V1 trial004 and explicitly removed the exact sparsity-match
requirement. Its uniform threshold0.027000000000000003 achieved80.250158%
GLOBAL sparsity on the original64-cell development schedule; the completed
control reference achieved83.459100%. No accuracy or target answers selected
this threshold. Trial005 remains preserved, and the finer trial006 was stopped
at26/64 cells with zero new CUDA captures or invalid rows.

Freeze that same threshold for V2 and perform one development audit to measure
its realized sparsity, without further matching. Report actual sparsity
differences for all target arms. Full503-cell LongBench audit and clean timing
plans, official inputs/seeds, controls, sampler, reuse settings, LOCAL scope,
and fixed budgets remain unchanged. Protocol execution attempt003 records the
amendment and explicit source binding. The selector and generation source
hashes are unchanged; only execution-protocol selection and reporting labels
were updated. These are calibration/protocol facts, not target results.

The one V2 development audit completed64/64 cells at the accepted shared
threshold, measuring84.529368% GLOBAL sparsity and zero LOCAL sparsity. Its
receipt validates all source/layer/graph checks with zero timed captures.
V1, V2 and control generation/selector/consumer source hashes match exactly,
despite the later execution/reporting-only protocol amendment. The frozen
first-experiment parameter file records these measured differences before
any target generation; no additional threshold search is performed.
# 2026-10-08: user drops future exact greedy from the value-selector experiment

The direct user confirmation is: "Drop exact greedy; retain its completed
diagnostics." Freeze the remaining eight-arm scope in
`results/v31_value_selectors_20261008/protocol_execution_attempt004_no_exact_20261008.json`.
Retain the completed128-cell exact audit, runtime diagnostics and small exact
reference/tests. Do not launch further exact generation or recompute fresh
full-context exact masks offline. Preserve the old nine-arm frozen protocol and
attempts; do not label them complete. Reuse only completed nonexact shards whose
generation bytes, model/pool, host/GPU, runtime, thresholds and exact schedules
match. The13 validated predecessor shards supply1,664 audit cells. Separate
audit/clean coverage is8x503 per pass; V1/V2 threshold0.027 stays fixed.
Nine focused continuation tests pass; this decision reports scope and validated
reuse, not target accuracy or clean latency.
# 2026-10-08: canonical JSON comparison for value-selector coordinator restarts

The first eight-arm coordinator serialized its arm tuple as a JSON list but
compared that list with the original tuple on restart. Use a separately pinned
corrected entrypoint and execution attempt005, preserving attempt004 and its
completed records. The new attempt002 may reuse completed matching shards from
the original nine-arm family and the first eight-arm family. Its generation
sources, threshold0.027, budgets, sampler and consumer remain unchanged.
Ten continuation tests pass, including repeated metadata roundtrip without
rewriting the saved config and rejection of changed execution identity.

## 2026-10-09 — Pause clean timing after complete official audit scoring

The user requested that timing stop for now. Preserve all completed clean
shards and the incomplete current-control shard; the resumable stop marker is
`results/v31_value_selectors_20261008/longbench/target_no_exact/attempt002/timing_stop_20261009.json`.
The clean stage has no publishable latency result. The eight-arm audit stage is
complete and official scoring covers 503/503 cells per arm: native FULL 247,
PIECEWISE 247, all-kept FA4 252, current V31 control 268, V1 247, V2 266,
V3a 253, and batch8 242. No value-aware arm exceeds the control. Audit GLOBAL
sparsities are 0, 0, 0, 83.16, 80.04, 84.11, 83.08, and 84.97 percent in
that order; LOCAL sparsity is zero. These are historically examined LongBench
cells and descriptive paired cluster intervals, not noninferiority evidence.

## 2026-10-09 — Accuracy diagnosis for value-aware selectors under reuse

Official 503-item scoring does not show a value-aware gain over the current V31
qblock-max control: control 268/503, V1 247, V2 266, V3a 253, and batch8 242.
The accuracy analysis is recorded in
`results/v31_value_selectors_20261008/reports/accuracy_analysis_no_exact_20261009/`.
The evidence points to a cross-step objective mismatch: value selects a bitmap
from current projected V but the bitmap is held across later Q/K/V states; the
rank-32 PV residual is not a downstream logit or sampler objective; and every
sparse arm changes the native adaptive trajectory. Do not interpret this as
proof that value direction is generally useless or as a timing result. The next
accuracy panel should use matched snapshots and a mass-anchor hybrid/fresh-vs-held
ablation before timing resumes.

## 2026-10-10 — V2 online refinement: mass anchors, jcgate, and full-V gate

The fresh 15-item seed-1 development panel is under
`results/v31_value_v2_refinement_20261010/` and is development evidence only.
The control scored 7/15, V2 preserve-mass 6/15, mass-anchor32 7/15, jcgate
7/15, and combined jcgate+anchor16 7/15. Mean denoising forwards were 146.9,
144.2, 159.7, 122.9, and 180.3 respectively. Jcgate is the only refinement
kept for an accuracy-first follow-up; it lowers calls but also lowers GLOBAL
sparsity to 71.68%, so no speed claim is made. The anchor combination is
rejected pending new evidence.

The native-mass anchor sweep improved a contiguous masked-operator diagnostic
monotonically as anchors increased, but also increased retained work and caused
anchor64 long-context smoke OOM warnings. The streaming full-dimensional-V
snapshot diagnostic found projected/full-V IoU 0.858 and full-V/control IoU
0.156; at threshold0.027 full-V error 0.3298 exceeded mass control error
0.2467. Thresholds near the control's tile count did not close the gap on the
initial snapshot. Do not build or time a full-V kernel until a matched-budget
operator advantage is demonstrated. Clean timing remains paused.


## 2026-10-10 — Full V2+jcgate audit rejects row-weighted refinement

The complete 503-item official audit is under
results/v31_value_v2_refinement_20261010/scoring/audit/. Current V31 control
is 268/503 (53.3%) and value_v2_online_jcgate is 256/503 (50.9%), -2.39
percentage points. The paired discordance is 44 control-only versus 32
jcgate-only (exact two-sided McNemar p=0.207). Jcgate has 78.07% GLOBAL
sparsity versus 83.16% for control and 242.59 versus 279.71 mean denoising
forwards. Because the candidate both keeps more GLOBAL work and loses
accuracy, it is rejected before clean timing; the timing stop remains
resumable.

The evidence favors a fixed-budget mass-gated value selector for any next
accuracy experiment: use native mass to constrain the control-sized support
and use value residuals only for within-budget swaps. Do not launch a
full-dimensional-V kernel or timing until a matched-work operator or accuracy
advantage is measured.

## 2026-10-10 — V3 full-row selector diagnosis

The focused V3 analysis is recorded in
`results/v31_value_selectors_20261008/reports/v3_accuracy_diagnosis_20261010/`.
V3a's sampled fresh age-0 mask IoU with the V31 control is 0.703, but its
singleton deletion objective replaces roughly one sixth of the fixed-budget
support and is slightly worse in max-row/full-dimensional error; held age-7
max-row error is 0.661 versus 0.638 for control. V3b batch8 is not exact
greedy: its non-submodular residual objective is updated only once per eight
deletions, and early batch choices cannot be restored. Its max-row error becomes
worse than control on 62.5% of age-1 and 74.2% of age-7 snapshots; mean N is
+61.6 calls. The partial exact-greedy diagnostic (58/128 versus V3a 62 and
control 67) shows batch approximation is not the sole cause. A separate
six-snapshot full-dimensional-V check found projected/full-V IoU 0.858 but
full-V/control IoU 0.156 and full-V error 0.330 versus mass-control 0.247,
so rank32 alone does not explain the support divergence. Keep the next panel
fixed-budget and mass-gated; use value residuals only for bounded swaps after
matched-snapshot operator validation.

## 2026-10-10 — AIME26/RULER128K value-selector audit

The frozen accuracy panel under
`results/v31_value_performance_20261010/` was run on the H100 host
`megakernel` and scored with the official AIME and RULER scorers. All 90 AIME
cells (30 problems × seeds 42/43/44) and all 104 RULER cells (eight samples per
task across 13 tasks) completed with clean receipts and no validation errors.
Clean timing remains paused and resumable.

AIME exact-match scores were control 53/90 (58.89%), V1 50/90 (55.56%), V2
49/90 (54.44%), V3a 53/90 (58.89%), and V3b 52/90 (57.78%); dense FULL and
all-kept FA4 were 52/90 and dense PIECEWISE was 46/90. RULER scores on the
frozen 104-cell subset were control 86.394, V1 86.154, V2 85.673, V3a 86.298,
and V3b 81.490. Thus value-direction information did not improve accuracy over
the attention-mass control in this cross-step reuse setting. V1/V2 moved the
support and denoising trajectory, while V3a tied the control without improving
it; V3b's larger sparsity and call trajectory is not a matched comparison.

The long-prefix V1 calibration initially failed before writing records. The
online selector was corrected to process fixed 128-tile chunks while carrying
the exact log-mass/output/support recurrence. Unit tests passed (28 passed, 20
skipped), and the failed attempt remains preserved. The accuracy evidence
supports a next experiment that keeps the native mass support as a control-size
envelope and permits only bounded value-based swaps; no timing claim is made.

## 2026-10-11 — Fixed-budget online value swaps do not yet improve the official audit

The new accuracy-only panel is recorded under
`results/v31_value_next_iteration_20261011/`. The implementation anchors every
query block to the exact V31 attention-mass budget and allows bounded
Gaussian32 value swaps. Online swap8 was the best 15-item development arm at
9/15 versus control 7/15, but its complete 503-item official audit scored
264/503 versus 268/503 for control (-0.795 pp, exact McNemar p=0.741,
paired-bootstrap 95% CI [-4.97,+3.38] pp). It used 249.86 versus 279.71 mean
denoising forwards; this is an accuracy/audit observation, not a timing gain.

Online swap16 tied the control in development, phase protection and C_gate were
not better, projected full-information scoring inflated calls, and the
V31∩V2 intersection retained too little support and inflated calls. The
offline true full-dimensional-V score likewise did not beat the mass map.
Reject all these arms before timing. The next permitted refinement is a
confidence-gated mass-anchored online swap with a fixed per-query-block budget
and an age/drift-triggered no-swap fallback; require a new paired accuracy gate
before a clean timing run.

## 2026-10-11 — Requested 50/75% follow-up remains accuracy-first

The follow-up campaign is under `results/v31_value_followup_20261011/` with
separate resumable attempts. At the existing s50 budget, AIME official scores
were V2+jcgate 56.67%, mass-gated online swap8 51.11%, and C_gate swap16
53.33%, versus the inherited V31 control 58.89%. AIME swap8 at the reduced
s75 budget scored 52.22%. RULER128K swap8 at s50 scored 86.88 on 104 cells.
The fixed arms realized approximately 46% GLOBAL tile sparsity at s50 and
62% at s75; the thresholded V2+jcgate arm realized 30%, so threshold 0.074 is
not a matched-sparsity setting for that arm. No timing claim is permitted.

The auxiliary quick pool and LongBench-v1 run are diagnostic: the current
generic short/long launcher reports dense physical accounting despite the
requested selector budget. Threshold calibration and a matched rerun are
required before comparing those auxiliary accuracy numbers as sparse methods.

## 2026-10-11 — Full two-regime matrix queued

The complete requested matrix is frozen in
`results/v31_value_matrix_20261011/`: eight methods, two provisional target
regimes, 90 AIME cells, 104 RULER cells, 100 samples for each short benchmark,
and 100 LongBench-v1 samples split evenly across the five requested tasks.
The 144-job queue runs one H100 worker sequentially and persists completion
state. Timing remains paused. Provisional thresholds must be replaced or
accepted only after each receipt's realized physical sparsity is audited.

## 2026-10-11 — Matrix queue retries transient GPU preflight races

The first restart of the full matrix stopped after AIME s75 control shard002
observed the H100 still occupied by the previous vLLM process. No attempt
directory or records were created for that shard, so its failed state was
removed and the persisted queue resumed it. The queue runner now retries a
nonzero return only when no attempt directory exists, waiting up to five
minutes; once an attempt directory exists, the return is preserved as a real
failure. The resumed s75 control shards completed cleanly. This is an
execution-reliability fix and does not change selector behavior or scores.

## 2026-10-11 — Matrix paused for threshold-strategy update

An audit of the provisional full-matrix queue found a builder routing error:
AIME online arms were assigned the long-context thresholds, producing clearly
under-target physical sparsity for V1/V2. Those attempts are retained as
diagnostics and are not publishable comparisons. A corrected queue root was
created with AIME short-threshold routing, then paused at the user's request
before launching the remaining jobs; its partial attempt is preserved and
timing remains paused.

## 2026-10-12 — V2 threshold sweep does not yield one equal-work threshold

- The complete campaign is `results/v31_value_v2_equal_work_20261011/`. The
  unchanged `value_v2_online_preserve_mass` selector was swept at all eleven
  requested thresholds on the original 64-cell development panel. Realized
  GLOBAL work, eligible work, and native calls were non-monotonic with the
  threshold because online reuse and native stopping change the trajectory.
- Work-first calibration selected `tau_total=0.015` and `tau_early=0.021` at
  the common C=20 prefix. No single natural threshold met both the recommended
  total (±2%) and early (±3%) development tolerances, so both named official
  arms were run without adding a quota to V2.
- A fresh current-V31 503-cell control scored 256/503 (50.9%), with
  5,691,974,080 physical GLOBAL kept tiles and 134,521 denoising forwards.
  `v2_work_match_total` (`tau=0.015`) scored 263/503 (52.3%) and +1.392 pp
  paired (bootstrap [-2.187,+4.970] pp; McNemar p=0.5203), but retained
  5,879,477,757 tiles (1.0329x control) and over-allocated C=20 work by 13.29%.
  `v2_work_match_early` (`tau=0.021`) scored 274/503 (54.5%) and +3.579 pp
  paired (bootstrap [0.000,+7.157] pp; McNemar p=0.0630), matched C=20 work
  within 1.19%, but retained only 0.9221x total tiles. These accuracy results
  are descriptive audit evidence, not a noninferiority claim.
- Report physical FA4 consumer tiles, selector discovery/evaluation, native
  forwards, and audit wall time separately. Do not call either arm equal
  end-to-end work or a speedup. The next selector experiment should use a
  fixed-budget mass anchor plus an explicit early-work guard; clean timing
  remains paused.
# 2026-10-12 — V31 cached-statistics partial refresh

- The next selector is a selector-only cached-statistics refresh.  A dense V31
  observation caches raw per-row prefix-tile log-mass and resets ages; every
  two reuse calls refreshes retained QK statistics and combines them with the
  cached discarded support.  The held map produces the current FA4 output and
  a changed map is eligible only on the next call.
- Global top-K and bounded swaps are separate policies.  Bounded swaps use a
  total top-M frontier (`M=160,256,512`) and `B=8`; every query block keeps
  exactly `K=128` KV tiles.  Stable physical tile ids define ties.  The first
  decay arm uses lambda `log(2)/2` and floor `0.25`.
- Value information is a near-mass tie signal only.  C_gate sensitivity is
  likewise a ranking/tie signal and never changes K or SM work.  Strict
  intersections are diagnostics; filled intersections are separately named
  matched-budget variants.  Full-dimensional value omission is offline only.
- A non-finite retained statistic or drift above the frozen operator threshold
  `10,000,000` keeps the current output map and schedules a dense refresh on
  the next call.  The threshold is fixed from operator stress replay and is
  not fitted to target answers.
- The H100 smoke completed after a resource-setting correction and exposed a
  Python-loop selector overhead.  The bounded-swap implementation is now
  tensorized on CUDA; the follow-up smoke reduced audited selector time from
  47.36 s to 4.55 s on the same one-cell trajectory.  These are audit costs,
  not clean timing; matched controls and target scoring remain pending.

## 2026-10-12 — Cached-statistics refresh panel is complete but not promotable

The runnable native matrix is complete under
`results/v31_cached_refresh_20261012/`: 16 arms × 154 cells, with the primary
bounded `M=256` arm in `attempt009` and the remaining 15 arms in `attempt010`.
All records have complete receipts, official subset scoring, matching frozen
source hashes, zero validation errors, zero unexpected LOCAL routing, and zero
timed CUDA captures. The current V31 control scored 53.1% on the 64-cell
LongBench-v2 subset and 54.44% AIME avg@k. Global-top-K variants stayed near
V31's mean denoising count but still exceeded V31 on some cells; bounded swaps
had 103–120 cells with higher denoising counts. No cached arm improves the
LongBench subset control, and none passes the fixed-trajectory promotion gate.

The native panel is therefore complete as an accuracy/operator study, not a
promotion or timing result. Separate online-value-only, full-information,
strict-intersection, and filled-intersection variants remain operator-only
because native map wiring was not frozen. Clean timing remains paused.

## 2026-10-13 — Correct first cached-refresh interpretation before refinement

Source audit confirms that the cached dense layout used a reshape in place of
row/tile transposition; the CUDA bounded no-op branch can clear a retained tile;
the supposedly last-dense cache was mutated by observations; dense counts were
end-state sums, and periodic/progress scheduling overlapped. The first panel's
scores are executed-code measurements, not clean tests of intended policies.
Do not use them to conclusively reject bounded, value-aware or C_gate ideas.
Prior map IoU compares consecutive masks. Three global M arms were equivalent.

The new study preserves historical artifacts and uses an independently derived
opt-in stable path. Parameter selection is operator-only on real trajectories;
no benchmark answers enter the hysteresis/guard grid. Aggregate N must remain
within the predeclared 2% provisional limit on each workload with cap/tail checks.
A single longer problem is not an automatic rejection, and the 154-cell requested
panel is distinct from a later 503-cell expansion. Clean timing remains paused.


## 2026-10-13 — Stable cheap refresh fails the development promotion screen

The corrected 13-arm study completed 2,002 native cells and 52 clean shards,
real matched-state replay, repeated instrumentation invariance, official
scoring and workload-separated paired intervals. Operator selection froze
hysteresis 0, persistence 2, event/cumulative caps 2/8 and a 0.025 row guard
before native scoring. Stable maps reduce churn substantially; this does not
ensure native accuracy or lower N.

Fresh V31 scores 34/64 LongBench and 49/90 AIME (mean N 240.27 / 275.50).
Stable progress s2 scores 31/64 / 49/90 and increases LongBench N 13.15%.
Stable no-progress s8 has the best candidate point accuracy, 35/64 / 55/90,
but N rises 22.29% / 3.04%, N/C rises on LongBench, and the upper tail worsens.
Its accuracy differences have clustered intervals spanning zero. No tested
candidate passes all frozen aggregate/tail/cap/accuracy criteria. Retain V31;
do not expand or launch timing. This is a negative result for these settings,
not a general rejection of stable repair, C_gate or value-aware methods.

All no-progress cheap arms match the no-cheap hold AIME correctness vector,
and AIME GLOBAL sparsity is below 0.004% with K=128. Do not attribute that
AIME gain to cheap re-ranking. Full/Piecewise controls are bitwise identical
on the panel; all-kept FA4 remains a distinct consumer control. Audit selector
overhead, duplicated QK and memory are separate from absent clean timing.
Source: results/v31_cached_refresh_stable_20261013/reports/final_report.md and
its final_complete.json. Failed trajectory/scoring attempts are preserved;
the scoring correction changes only schedule-key interpretation, not policy,
criteria, native generation or official extraction.


## 2026-10-14 — Isolate refresh factors before selecting full LongBench arms

The user questions the reliability of the 64-problem development subset and
requested existing native analysis plus complete operator replay before
choosing 503-problem arms. The source native study remains immutable. Its
8/16 method settings do not support a matched progress-on/off comparison
with cheap refresh: on uses s2, off uses s1/s4/s8. Compute only one-factor
paired contrasts that exist; do not fit unbalanced pooled main effects.

Frozen factor source `be9e23dc` uses the previous operator-only stability
choice unchanged: zero hysteresis, persistence 2, event cap 2, cumulative
cap 8, removal tolerance 0.025. Raw does not activate those constraints.
All 16 operator settings and both held references completed on identical
Q/K/V. Each has the same 570 post-anchor frames and 480 common sparse
comparison frames. CPU NumPy map checks agree on 2,550 refreshes; 26 new
and 38 CUDA/FA4 tests pass. Physical budget, stable caps and actual capture
counter checks pass. Fresh QK/score discrepancies are at most 1.91e-6 /
1.64e-6. This remains conditional on source V31 Q/K/V and progress events,
not a native outcome prediction.

Progress improves common sparse-call operator error by 0.89-1.57%, while
stable reduces churn but has higher error than matched raw. Stable progress
s8 performs zero swaps: each of its 15 nonempty anchor epochs has only one
cheap observation, insufficient for two-event persistence. Frequency alters
effective stable response delay and interacts with dense-refresh resets.
Native stability effects vary by interval and do not establish a general
win; the s8 no-progress advantage over raw is a rescue relative to a weak
raw arm, only +1 correct over V31 on the subset, with N inflation.

Proposal, not launch: four raw/stable x progress-on/off s2 arms, add stable
no-progress s1/s4/s8, retain both held references and native-consumer controls.
The user chooses the 503 scope next. Clean timing remains paused. Complete
sources and report: results/v31_refresh_factor_isolation_20261014/.


## 2026-10-14 — Full503 refresh comparison and current-step extreme baseline

User approved the seven factor-preserving refresh variants, V31 progress and
no-progress hold controls, and an additional no-reuse baseline: current full
V31 qblock-max mass selection at every GLOBAL call, then native sparse FA4
output from that same current mask. The dense observer output is discarded.
No cached mass, carried map or prior-map bonus influences this baseline. Its
extra full QK/PV plus duplicate tail-QK work is counted separately. This is
an accuracy/trajectory extreme, with no clean timing claim. Existing hold
keeps per-canvas initialization and carry; sampler, K=128, LOCAL, prefill
and commits remain fixed. Fresh FULL fix51994/PIECEWISE/all-kept controls
bring the matrix to 13 arms x 503 seed-1 cells. Frozen source 98a37a14, study
results/v31_full503_refresh_20261014/. Code and all inputs are copied/frozen
inside the study before qualification/target generation. Prior paused V2
campaigns and completed cached-refresh studies remain untouched.

## 2026-10-14 — Full503 dense-mode diagnostic and resource wait

The approved 13-arm native study remains frozen at generation source
`98a37a14` in `results/v31_full503_refresh_20261014/`. Stop and preserve the
first PIECEWISE leaf after 18 records: every output hash differs from the
126-record FULL shard, while all 18 PIECEWISE outputs match the earlier valid
dense controls. No scoring or selector conclusion is derived from this
partial comparison. FULL cell 0 repeats exactly at memory utilization 0.85,
so the original 0.80 setting does not explain the discrepancy. The baseline
notes document valid FULL/PIECEWISE free-running differences; qualify the
actual persistent int32 causal-buffer alias during capture and runtime before
resuming the unchanged generation source.

An external H100 worker started during this diagnostic period. Wait for idle
resources and do not terminate or modify it. The qualified resume driver
monitors at 15-minute intervals; failed diagnostic and native leaves remain
immutable. Timing stays paused. The raw/stable selector and observation
modules match the completed independent factorial replay by hash; optional
every-step adapter dispatch and benchmark metadata changes have separate
fresh parity/operator qualification. See `DIAGNOSTIC_PAUSE_20261014.json` and
`qualification/factor_replay_source_audit.json`. This is a pending runtime
qualification, not a completed full503 result.

## 2026-10-14 — Dense causal qualification passed; unchanged full503 resumed

Read-only causal-alias diagnostic attempt006 completed two native FULL cells
with 28,410 metadata checks (four capture and 943 runtime preparation calls),
zero dtype/alias errors, and exact output hash/N/T/C agreement with the ordinary
FULL repeat at memory utilization 0.85. The diagnostic itself used frozen 0.80.
The persistent int32 causal buffer is shared by capture/runtime metadata. The
FULL/PIECEWISE free-running difference does not imply the unfixed causal flag;
cross-mode bitwise equality was never required by this protocol, and valid mode
differences are documented in the baseline notes. Do not infer selector effects
from this partial dense comparison. Receipt: full503 study qualification/
dense_mode_resolution.json and causal_alias_full_attempt006/receipt.json.

The user reconfirmed H100 availability. After the external evaluator exited,
the resource guard admitted this diagnostic at 13:30 UTC following sustained
idleness. Startup succeeded with 11.56 GiB available KV cache; earlier failed
leaves remain intact. At 13:34 UTC coordinator_attempt002 resumed the original
generation source `98a37a14`, reusing the validated 126-cell FULL shard and
starting a new PIECEWISE attempt. All 13 arms, manifests, seeds, native sampler,
source hashes and output-consumer paths remain frozen. Monitor every 15 minutes;
no final target accuracy or clean timing claim exists.

## 2026-10-15 — Publish three separate research directions

- Publish the existing value-aware V1/V2/V3 selectors, V2 threshold studies,
  and cheap cached-refresh work in three distinct integration/reproduction
  documents under `docs/research_week_20261015/`.
- Include the pending chunked online Triton scan, device-side hybrid ranking,
  launcher changes, missing experiment tools and tests. Registration of a
  prototype is not a promotion or a completed native evaluation.
- Export only selected sanitized aggregates with original/exported SHA256s;
  leave frozen runs, private inputs/generations/tensors and live worker state
  local. Keep the separate full503 campaign running under its own protocol,
  the V2 local threshold campaign paused, and clean timing paused.
- Supersede the historical lifecycle-ID early-work interpretation using the
  verified ordinal/visible-canvas correction. Official C=20 deltas are +7.74%
  for tau .015 (50 common items) and +3.07% for tau .021 (45); no exact
  total-plus-early work match is claimed. Entire-run work and scores are intact.


## 2026-10-15 — User requests shard02 boundary and interim official report

Finish the final FULL dense shard02 worker, pause before shard03, and report
13 arms on the same 252 seed-1 LongBench-v2 problems (3,276 cells total).
This explicitly replaces the earlier instruction to continue to all 503 cells.
Suspend only the queue controller while its current native worker finishes;
preserve all completed and failed leaves and require explicit user instruction
to resume. Freeze a separate CPU analysis protocol for shards01/02. Official
scoring, paired problem-cluster intervals and native trajectory metrics apply
to these 252 problems only; do not create a full503 completion marker.

A CPU reporting filename defect was found before any full503 report ran:
the official scorer rejects bare completions.jsonl because it has no arm label.
Use run_<arm>.private.jsonl in the separately recorded root reporter and record
its source hash. The operational post-generation report command now uses that
reporter. Frozen native source bytes, inputs, sampler and all generation paths
are unchanged. This amendment authorizes no additional GPU work or resume.


## 2026-10-15 — Two-shard interim report complete; generation paused

Completed the requested boundary: 13 arms x 252 problems, 26 validated shards,
3,276 officially scored cells, generation source 98a37a14 unchanged. Native
sampler/source/input/runtime and completed-artifact checks pass, with zero budget,
LOCAL-routing, order or request-capture errors. No shard03 launched, all study
workers stopped naturally, and the queue controller terminated after the boundary.
The H100 is idle. Resume requires explicit user instruction; 251 problems per
arm remain. Scoped completion: full503 study reports_shards01_02_attempt001/
partial_complete.json. No full-study final_complete.json exists.

Fresh every-step scores 134/252 versus V31 130/252, mean N -7.76%, but the
accuracy difference interval spans -3.57 to +6.75 pp and N increases on 129
problems. No cheap-refresh arm exceeds V31's observed accuracy. Stable no-progress
s2 leads cheap accuracy at 127/252; N is -4.15%, output length -8.16%, mean N/C
+1.10%, and p95 N rises from 727.15 to 811.65. The four crossed s2 arms show
conditional stability/progress effects, with stability signs reversing between
shards; every accuracy-contrast interval includes zero. Treat these as interim
measurements, not superiority/noninferiority or full503 evidence. Selector audit
cost, native trajectory and physical output-consumer versus observation work
remain separate. Clean timing is intentionally absent. See summary.md and
interpretation.md in the scoped report directory.


## 2026-10-15: Complete only V2 tau=0.021 on LongBench-v2; retain V31 as accuracy reference

The user resumed only the 503-item seed-1 V2 arm. Campaign
`results/v31_v2_tau021_longbench_20261015/` preserves the exact 886-file
execution snapshot and 24 inputs from source `ab05d324`; coordinator/spec
freeze is `910504b8`. Repeated V31/FULL/PIECEWISE/all-kept qualification cells
match all output hashes and N/T/C, allowing reuse of their complete matched
controls. Official scoring and independent source/receipt validation are
complete for all 503 items per arm, with zero invalid/missing/capture/LOCAL
errors. Original artifacts are unchanged.

V2 scores 254/503 (50.50%) versus inherited Yuhan V31 268/503 (53.28%) and
FULL/PIECEWISE dense 247/503 (49.11%); all-kept scores 252/503. The V2-minus-V31
paired accuracy delta is -2.783 pp, 95% bootstrap CI [-6.362,+0.795], exact
McNemar p=0.1654. No accuracy improvement or preservation is established.
V2 native calls fall 11.635% (247.59 versus 280.19/problem), but only 253/503
problems use fewer calls, 247 use more, and three tie. Kept GLOBAL work falls
10.114% (5,243,209,214 versus 5,833,185,760), while GLOBAL sparsity is 82.54%
versus 83.20%. This is an unequal-total-work native trajectory comparison.
Retain per-item and early-survivor-cohort receipts rather than inferring equal
work from sparsity. Selector discovery/selection cost stays separate, and
audit wall time is not clean latency.

All other thresholds, finer sweeps, timing and the separate refresh study
remain paused. The completed 0.021 arm may be referenced in place on an
explicitly authorized future sweep resume. Verified reports and completion:
`summary.md`, `interpretation.md`, `paired_comparisons.json`,
`receipts/independent_verification.json`, and `final_complete.json` in the
campaign above.


## 2026-10-15: Historical RULER128K reporting and source provenance limit

At the user's explicit choice, report the existing 104-item panel rather than
launch a new tau=0.021 run. CPU reaggregation verified 20 completed shards,
520 scored cells and all 104 paired prompt/seed/budget/runtime-geometry bindings
under `results/v31_value_performance_20261010/ruler128k/`. The report is
`results/ruler128k_historical_report_20261015/`; its completion marker certifies
reporting only, not a new native experiment or completion of the originally
planned dense/control family. No GPU generation or paused queue was resumed.

Official scores: V31 86.394, V1 86.154, V2 85.673, V3a 86.298, batch8 81.490.
V1/V2 use tau=0.027. V2 has 612 versus V31 577 native calls, one visible canvas
per item, and 96,830,509 versus 99,178,720 retained GLOBAL tiles. This is +6.066%
calls and -2.368% GLOBAL work, with no observed accuracy gain. The same
92,898,560 full-support-phase tiles dominate both arms. Official RULER uses
fractional target credit; scores are not binary correct/104 accuracy.

One recorded execution-source hash differs across the historical comparison:
`v31_value_kernels.py` is `f9b5dfdd...` for V31 and `192f027f...` for value arms.
The other four recorded source hashes agree. Therefore do not describe this
archive as a freshly verified byte-identical deployment comparison. No completed
dense/all-kept RULER arm was found in this 104-item panel. Literal CUDA launch
counts were not measured; retained attention tiles and logical adapter split-FA4
calls are distinct fields. Existing source records and published summaries
remain unchanged. Clean timing and the separate campaigns stay paused.

## 2026-10-15: Fresh seed42 RULER128K V31/V2 audit does not reproduce shorter V2 trajectories

The user authorized only V31 and exact online preserve-mass V2 at tau=0.021
on the same 104 RULER128K prompts. Freeze/spec commit `5859d086`; both arms
execute the identical qualified `ab05d324` byte snapshot, runtime and model.
Native panel seed42 replaces seed1; the subset, base31, projection rank32/
seed1729, native sampler and GLOBAL-only/LOCAL-dense policy remain fixed.
Eight 26-cell shards plus four qualification cells completed. No historical
generation, differently pinned control, additional threshold or paused queue
was reused. Campaign: `results/v31_v2_ruler128k_seed42_tau021_20261015/`.

V31/V2 official fractional scores are 86.250/85.769, with identical 81/104
full-credit sets. Delta -0.481 points, 13-task-cluster bootstrap 95% interval
[-1.442,0.000]; full-credit McNemar p=1.0. One problem improves, five worsen,
98 tie. The net task-average loss is common-word extraction, 70.00 to 63.75.
Do not interpret equal full-credit counts as equal partial-credit outputs or
the interval as evidence of noninferiority.

V2 has 586 versus 545 calls, +7.52%; both commit one canvas per problem.
Variable tracking alone adds 45 calls (106 versus 61), while all other tasks
combined remove four. Two trajectories add 28 and 10 calls. V2 therefore does
not show the shorter mean trajectory observed on LongBench. Retained GLOBAL
work falls only 1.05% (95,030,781 versus 96,039,360); overall work falls 0.34%.
GLOBAL sparsity is 50.33% versus 46.00%, partly from the longer trajectory's
larger denominator. Full-support phases are 95.7%/94.0% of retained V2/V31
work, limiting savings from smaller held/carried supports.

Independent CPU verification passes for 886 sources, 11 inputs, all 208 score
cells, 104 paired bindings, source/seed/physical/phase invariants and two
qualification repeats per arm. Invalid rows, request CUDA captures and LOCAL
skips are zero. The generic inherited `value_budget_policy` receipt label is
static; the executed `triton_log_mass_online` branch uses tau without a quota.
Selector discovery and evaluation each count 57,019,520 additional rectangles;
recorded audit time is not clean latency. Preserve the completed immutable
attempts, leave timing and other campaigns paused, and report this as a
paired seed on previously examined prompts rather than held-out confirmation
or a clean threshold ablation against the old seed-1 archive.


## 2026-10-15 — Independent value/decision/reuse iteration, diagnostics first

The user authorized the three-direction plan with fixed per-query-tile K and
paused timing. Implement bounded true-mass-neighbor full-V proposals, causal
query-direction error prediction, and separately labeled output reconstruction.
Preserve historical code/results and paused queues. The new plan and source
are frozen at 378e147a; whole-layer replay at 39a56aa0. No exact greedy.

All 36 H100 mathematical/parity checks passed. Independent same-state replay
covers all 620 saved frames, 570 usable frames, 670 verified files and all five
GLOBAL layers. Follow-up sums signed errors through actual O weights across
all 16 heads and both query tiles. Fixed K and capture checks pass. The tested
past-query extrapolator loses to the static predictor on every one of 100
sparse age-1 screened units, despite correct finite-difference derivatives;
do not promote it into accuracy runs. Bounded swap gains are modest and cached
correction gains have harmful tails, so operator error alone does not pass
the decision gate. All-kept short states are excluded from sparse-benefit tables.

Source-frozen last-GLOBAL-layer suffix diagnostics must reproduce native logits
and native trajectories with diagnostics disabled/enabled before their results
are accepted. Extra suffix forwards and privately coupled diagnostic Gumbel
noise are explicitly offline; the latter is not claimed as the compiled
native sampler's random realization. An isolated final-layer intervention
cannot qualify changes at earlier layers. Native accuracy and clean timing
remain unlaunched at this checkpoint.

## 2026-10-15 — Refine stale correction with causal value-direction trust

Whole-layer replay exposes harmful residual tails. The causal coefficient
`clip(1 - <St-S0,D0-S0>/||D0-S0||^2,0,1)` uses selected current output and
cached anchor outputs only. It cuts LongBench >2% max-row harm counts from
26/220 to 0/220 and 25/100 to 3/100 while preserving substantial mean-error
reduction. The norm guard is a conservative ablation. Promote these only to
same-state decision diagnostics; they have no accuracy/call-count result.
Complete scoped evidence is under `v31_value_decision_reuse_20261015/reuse_guard`.

Attempt003's suffix observer allocated new identity outputs across fixed-address
CUDA-graph boundaries and failed suffix parity with 256/256 top-1 mismatches.
Exclude its generation from scientific method comparisons. Attempt004 changes
observers to ordered read-only side effects on original tensors, and adds an
uninstrumented control to both repetitions. This strengthens qualification
instead of relaxing a failed parity threshold. Native accuracy remains gated.

## 2026-10-15 — Eager same-state qualification passes; broaden before accuracy

Attempt006 resolves the observation/compiler problem by an explicit eager
model diagnostic with unchanged native sampler and FA4 paths. Six runs
(uninstrumented, observer-disabled, active; twice) produce identical native
output hashes,117 calls,6 canvases and1438 tokens. Two active repeats each
record12 frames/120 offline suffix calls, no parity failures, unchanged RNG/
adapter counters, resolved graph NONE and zero captures. Source604446fb,
protocol and completed reproducibility receipt are under
`v31_value_decision_reuse_20261015/suffix_qualification_attempt006`.

On that one item, direction-guard KL is0.000920 versus mass0.001306, while
Top1 disagreements are51 versus56 out of3072 row comparisons. Moment merging
has lower KL0.000556 but worse acceptance disagreement16 versus14 rows.
These small, correlated counts are not accuracy evidence. Broad sourcebb2789ca
same-state panels now compare8 LongBench,6 AIME,8 MATH and13 RULER inputs,
selected independently of correctness. One uninstrumented/active pair per
new item follows the completed two-repeat instrument qualification. Every new
pair must again preserve native outputs/N/C/T and exact suffix parity. Added
fixed C_gate row permutation and bounded full-swap plus directional correction.
K remains fixed; no full task-accuracy or clean timing gate has been bypassed.

## 2026-10-15 — First complete multi-item decision panel and native readiness

The six AIME uninstrumented/active pairs preserve native output hashes,N,C,T.
Five requests provide35 sparse frames; one remains all-kept. Mass anchor has
26 top1/16 acceptance disagreements, direction guard24/12, full swaps21/14,
combined20/11. Full C_gate23/15 versus shuffled22/17 does not establish a
causal sensitivity benefit. The same-mask numerical floor is substantial;
these exploratory proxy counts do not establish task-accuracy improvement.
Native promotion waits for all four workload panels and exact-source H100 tests.

Pair admission remains a fixed-K unchanged-anchor batch proposal, with stable
disjoint ranking now performed on the device. A full-V ordered online-risk
kernel removes per-KV-block Python dispatch; its H100 qualification is pending.
Neither optimization is injected into the source-frozen diagnostic worker.

## 2026-10-15 — Complete initial decisions; require actual sparse coverage

The70 native diagnostic trajectories (35 paired items) pass invariance and
exact eager suffix parity. Initial sparse frames:99 LongBench,35 AIME,
14 RULER,zero MATH. The complete MATH job is not sparse-method evidence.
LongBench direction guard reduces top1 disagreements1068→792 and moments
→573; all eight items improve. RULER direction guard reduces24→12 top1
disagreements but raises acceptance disagreements4→8. Initial residual and
moment stop flips delay one dense-reference stop each (LongBench/RULER), not
premature stopping. Do not turn mean KL improvement into an accuracy claim.

Freeze a coverage extension before native promotion: eight unseen MATH indices
with at least12 canvases in prior completed V31 controls, and all14 remaining
RULER vt/cwe items. No correctness filtering; label this purposive exploratory
coverage. The native gate now rejects a workload with no sparse decision
frames. All70 current H100 correctness tests pass; receipt binds every native
refinement and test source, including the optimized full-V online kernel.

## 2026-10-15 — Coverage complete; select native development and cache ablation

Sourcea6f339fe coverage adds22 paired items with unchanged native output/N/C/T
and exact suffix parity. Seven MATH items supply75 sparse frames;14 additional
RULER vt/cwe items supply42. On MATH, mass/direction-guard/moment top1 counts
are107/98/97 and acceptance20/21/31. On additional RULER, counts are27/20/17
and52/34/42. The broad mean-KL gain from moments does not justify ignoring its
MATH acceptance regression. C_gate again does not beat shuffled weights clearly.

Freeze development inclusion to full_swap8, residual_direction_guard,
full_swap8_direction_guard, and raw residual as a mechanistic cache baseline.
The raw baseline is needed to attribute any benefit to the guard; it is not
certified as tail-safe or novel. All changes are restricted to the qualified
last GLOBAL layer. This is promotion to testing, not task-accuracy success.
See `native_inclusion.md` for all panel counts, exclusions and remaining risks.
Four fresh matched controls, actual sparse smoke activation, official scorer
binding and unchanged native sampler are required. Clean timing stays paused.

## 2026-10-15 — Correct inherited sampler metadata before native generation

The pre-launch model-config audit found ambiguous historical metadata:
`confidence:0.1`/`entropy:0.005` were the reverse of the actual pinned model
fields, and `thinking:true` could not describe non-thinking RULER inputs.
The generation code reads model defaults, which are confidence_threshold0.005,
entropy_bound0.1,stability_threshold1,48 steps and temperature0.8→0.4.
No native worker was launched from preparation001; preserve it as superseded.
Preparation002 reads the named defaults directly and binds the config hash.
Thinking/output budgets remain bound to each private input manifest. Selector
settings and inactive Gaussian32 metadata are not sampler settings and are
kept separate. This correction changes metadata, not native decoding behavior.

## 2026-10-15 — Native refinement expansion held for repeatability audit

The added intervention-exposure audit (analysis7f498574) found that inactive
trials in the completed first two AIME shards differ from matched V31:
full_swap8 4/4,direction guard3/3,raw residual5/5,and combined4/4. Inputs,
request RNG seeds,budgets and named decoding settings match. Native002's
`native_noop_precheck/` binds the complete shard records and sanitized counts.
No accuracy result was used to choose this diagnostic. This does not identify
whether the cause is inherent compiled repeatability or refinement integration.

The original smoke gate qualified activation,receipts and scoring but lacked
an exact same-seed repetition check. Hold accuracy interpretation and further
workload expansion; drain the final AIME controls and preserve all attempts.
The new frozen `native_noop_attempt001` (diagnostic coordinator dfd8f65e) will
repeat one identical cell twice per process,using two original V31 processes
and one full-swap process on the unchanged frozen execution source. Native
sampling,K and timing pause remain unchanged. Same-state eager evidence is
separate and is not invalidated by this unresolved compiled execution issue.


### 2026-10-15: Localize original V31 process-repeatability divergence

Native noop diagnostics003 and005 qualify their observers by exact observed/
unobserved repeats within each process. Across original V31 processes, first
sampling-call CUDA/CPU RNG and canvases match, but logits already differ.
Attempt005 has31/256 first-call argmax disagreements. Its embedding/layer0/
self-conditioning parameter and buffer hashes match (30 entries, limited
scope). First prefill and decode layer0 Q/K/V/attention outputs match exactly;
small differences appear by layer1 Q/K/V and amplify thereafter. This narrows
the first observed numerical difference to the intervening model computation,
not to sampling RNG or the first attention operator. It does not identify a
specific kernel. Source60d9f21f analysis and completion markers bind results.

Keep native accuracy interpretation held. Test a fixed cuBLAS workspace as a
single diagnostic runtime factor before attributing the effect to compilation
or MoE; unchanged sampler and compiled model path, two original processes,
observed/unobserved repeat qualification. Preserve failed observer004 and all
completed native panels. Any repaired runtime requires fresh matched controls.

### 2026-10-15: Qualify comparisons within a common compiled engine

The fixed cuBLAS workspace factor did not restore process repeatability.
Attempt007's finer observer captured no matrix-operation Python boundaries
under graph replay; its20 visible autotuner objects cover sampler-side kernels,
not all cached model kernels. No particular runtime defect is established.

Native_interleaved_attempt001 passes both frozen sequences. Engine1's five
V31 controls and four inactive candidates all reproduce the54-call trajectory.
Engine2's five V31 controls all reproduce the101-call trajectory despite
intervening active candidates (102,99,101,98 calls for full swaps, direction
guard, raw residual, combined). This is execution qualification on one
previously examined cell, not an accuracy result. It supports an incremental
comparison within each common engine while leaving cross-process variation
unresolved. All original AIME accuracy results remain interpretation-held.

The replacement panel will use the same frozen input cohorts and four
promoted methods, per-cell V31 control bookends, rotated candidate order, and
fresh independent dense FULL/PIECEWISE/all-kept references. Engine counts are
10 AIME,8 LongBench,10 MATH,13 RULER shards, with equal-size blocks. Report
process-cluster sensitivity alongside problem/task-cluster intervals. Preserve
all prior attempts; use a separately named protocol and source recipe. All
four workload smoke gates (active exposure, control bookends, inactive parity
where observed, exact budgets, scope and scorer binding) precede expansion.
No accuracy, denoising-trajectory benefit, or speedup follows from the single-
cell qualification. Clean timing remains paused.

### 2026-10-16: Retain baseline variability in a replicated comparison

All four process-matched smoke gates passed, but the first AIME accuracy
shard failed one of nine control bookends (426 vs 446 calls, same capped
length). Preserve the failed54-request attempt and its interpretation hold.
The subsequent original-V31-only history diagnostic completes two engines,
three inputs each, six repeats per input. One engine is exact; the other
changes its sixth long-input repetition from395 to393 calls and changes output.
This establishes that value intervention is not necessary for the drift.

Run the predeclared ABBA warmup factor without selecting favorable engines.
Prepare a separately named replicated native design with two engines per cell,
mean-of-two-controls comparisons, reversed candidate order, official repeated
scoring and both problem/task and engine-block uncertainty. Retain all
structurally valid outputs. The old exact-parity study remains held; do not
retroactively declare it valid. See V31_REPLICATED_NATIVE_PLAN_20261016.md.
Same-state evidence, fixed K and unchanged method/sampler contracts remain.
No new accuracy result or clean timing claim is established.

The ABBA warmup factor subsequently completed all four engines:4/18 drifting
bookends with control warmup,2/18 with all-method warmup, all on AIME item10.
Neither restores parity. Do not infer a warmup benefit from these small counts.
The new replicated study uses source b485866c, unchanged method code, and the
all-method policy preselected before these results. Its15 coordinator tests
pass; all four new workload smokes remain a launch gate for accuracy.

### 2026-10-16: Completed replicated AIME gives no joint improvement

All four replicated workload qualifications passed. The complete AIME cohort
then scores15 labels x90 cells with official coverage and independent hash,
repetition arithmetic, mean-N and tile-total verification. V31 is51.111% and
295.300 calls/trial. Full swaps52.222%/301.672, direction guard51.111%/296.911,
raw residual50.556%/296.861, combined51.111%/298.611. All fail the predeclared
joint AIME screen. The full-swap gain changes sign between engine replicates;
its problem-cluster score interval is[-2.778,5.000] percentage points.

The13/180 differing control bookends never change official correctness, but
their mean after-minus-before N is+2.467 (CI[0.194,5.878]). Across engines,
73/90 first-control hashes differ. Do not mistake a small operator improvement
or a single-engine score gain for robust native improvement. Continue the
remaining frozen workloads without retuning or excluding drifting cases.
All four current variants are ineligible for full503 under the joint screen
because AIME mean N is higher; do not relax that rule after seeing results.
Clean timing stays paused. Source b485866c, protocol config09e19e21.

### 2026-10-16: Full-value swaps reduce LongBench calls, accuracy unresolved

Complete replicated LongBench64 gives V3145.703%/251.039 calls and full
swaps46.875%/192.023. The score delta+1.172 points has problem-cluster interval
[-6.250,8.594]; no reliable accuracy gain is established. N drops59.016/trial
(23.51%), with problem interval[-143.250,-3.988] and engine-block interval
[-119.172,-14.945]. Both engine repetitions reduce calls.41/64 cells improve,
but the largest five reductions account for89.69% of the net saving; do not
present the mean reduction as a uniform per-question benefit.

Raw residual and guarded residual score43.750%; their N is327.094 and249.516.
Combined swaps/guard score47.656% but N rises299.438. The operator-error gains
of reconstruction do not establish a better native accuracy/call tradeoff.
Full swaps remain ineligible for the all-workload screen because AIME N rose.
Complete the unchanged remaining workloads. Keep clean timing paused and
report extra discovery work separately from retained consumer tiles.

The complete-cohort correctness/trajectory cross-tab further shows that one
always-wrong capped cell accounts for58.56% of net LongBench call savings:
2549→337 calls and16384→3026 tokens. Both candidate repetitions terminate
natively, both remain wrong, while all four controls hit the length cap.
Twenty-six always-correct cells save38.231 calls on average (17 fewer,8 more,
1 tie). This conditioned subset is descriptive only. Describe the headline
N gain as concentrated tail reduction, not proof of better reasoning.

### 2026-10-16: MATH100 gives small call savings with limited sparse exposure

The complete replicated MATH cohort has15 official labels x100 cells and
passes independent verification. V31 is95.0%/61.468 calls; full swaps95.0%/60.590,
guard95.5%/60.970, raw94.5%/61.145, combined95.0%/60.725. Combined saves0.7425
calls (1.21%); problem interval[-1.5775,-0.0575], engine interval[-1.4325,-0.1500].
There is no score improvement for this arm. Guard adds one correct trial out
of200 and its score interval includes zero. Full swaps' score/N changes reverse
sign across engine repetitions. Only46/200 candidate trials reach a sparse
anchor (45 correction-path trials); do not generalize this mostly all-kept
workload into evidence about broad value routing. One control bookend hash
differs, none in correctness; all100 first-control hashes differ across engines.
Finish RULER under the same protocol. AIME still blocks full503 promotion.

### 2026-10-17: Complete three-direction study; no full503 promotion

RULER completes the frozen replicated campaign. V3186.250/5.240 calls,
full swaps87.308/5.231, guard86.058/5.317, raw86.346/5.375, combined85.962/5.481.
All104 cells x15 score labels pass official coverage and independent arithmetic,
N-C-T,balanced13-task and work checks. Repetitions match exactly for all candidate
scores,output hashes,N/C/T. Full swaps improve two cells and worsen one:
multikey3 gains100 points on one cell; cwe changes+30 and-20. Task-average gain
1.058 has interval[0,3.077]; call reduction0.0096 has interval[-0.0577,0.0481].
This is a narrow signal, not a broad accuracy or speedup result.

All four workloads are complete:5,370 native generation trials on358 fixed
input/seed cells. None passes the unchanged all-workload screen because AIME
mean N rises. Do not launch full503 or relax the rule after seeing results.
Bounded full-value swaps remain the strongest research direction; correction's
large same-state operator gains do not produce a consistent native tradeoff.
The combined method loses swap benefits on LongBench/RULER. Further research
would require a better decision/lifetime admission objective and fresh frozen
validation; this iteration does not establish such a predictor or novelty claim.

Keep GLOBAL layer29 scope, prior examined inputs, runtime variation and extra
QK/PV discovery costs explicit. Timing stays paused. Final source/attempt audit
checks1,099 files,225 attempts,1,710 bound artifacts and5,370 records. Audit002
corrects the independent LOCAL-check field name; all records require the actual
native_local_skipped_tiles field and have zero skips. No generation changed.
GPU is idle, completed/failed attempts preserved, no further queue running.

### 2026-10-17: Publish complete method/setup/statistic and reproduction references

Added the dated README entry and three publication documents for every new
operator,decision and native variant. Earlier V1/V2/V3, threshold/equal-work
and cached-refresh families retain the October15 references and distinct results.
A new stdlib public exporter reproduces all native/diagnostic tables from
numeric evidence, without CUDA or private generations. It validates source-bound
reports/scorer outputs, complete cohorts, unequal repetitions, score arithmetic,
N-C-T/work and zero capture/LOCAL-skip/budget-error receipts. Three integrity
tests protect unequal control counts, required LOCAL field and changed score/work.
No new generation or timing run. Pending public artifacts are published without
upgrading held/partial evidence. Local private/runtime/external exclusions have
explicit inventory/ignore rules; external checkouts and frozen runs stay intact.

## 2026-10-17 — authorized full503 value-method performance study

The user selected full-value swaps and direction-guarded residual for fresh
LongBench-v2 full503 seed42 clean end-to-end evaluation against V31/dense.
The previous failed cross-workload promotion screen is unchanged. New source
ec87d397 uses the exact device backend, with86 passing H100 tests and256
matched saved-state units. Factored pair scoring was slower and changed one
mask, so it is not promoted. Reference selection incurred26 scalar reads and
10 nonzero calls per unit; device ranking/admission removes both, giving about
19% isolated selection savings. Native qualification and latency evidence are
still required. See results/v31_value_e2e_longbench503_20261017/README.md.

Native qualification subsequently passed all eight clean/audit engines and
official smoke scoring (14:09 Hong Kong checkpoint). The actual-state oracle
found no mask or correction-cache mismatch. Clean/audit trajectories can differ;
retain both independently scored passes. Full503 is running under ec87d397.

## 2026-10-17 — Pause full503 and reject continuation of current value methods

The user paused full503 pending a bounded investigation. The coordinator was
stopped before another launch; its active audit PIECEWISE shard003 finished
naturally and was finalized with frozen validation rules. All17 completed jobs,
private generations, attempts and source/protocol identities are preserved.
The guarded resume requires explicit new user authorization. Parent full503
is incomplete; its completion marker is absent.

The committed staged plan and completed evidence are under
results/v31_value_investigation_20261017/attempt001/. Independent verifier
c8feb508 binds CPU analysis, four crossed engines, one exploratory trace,
saved-state profiling and official scoring. Four tests pass;100 new measured
requests and25 warmups stay within45.55 conservative GPU minutes. GPU is idle.

Neither current method passes the joint accuracy/latency objective. On clean64
seed42, V31 scores30/64 at8.914s/request, swaps29/64 at14.242s, residual30/64
at11.420s. Accuracy intervals include zero. Previous savings and current
losses have large tail contributions; removing tails is descriptive only.
The crossed two-input experiment establishes seed-dependent swap behavior
with identical reference/device outputs in16/16 pairs. Residual has two audit
backend discrepancies of opposite signs; the final trace does not reproduce
them, so their cause remains unresolved.

The residual trace demonstrates continuous self-conditioning divergence before
sampled-token/support divergence and later changed RNG consumption. Better
saved-state operator error is therefore insufficient as an admission objective.
It does not prove lower current dense error caused the observed wrong answer.
Graphing the unchanged selector math passes64 saved units and reduces launch
cost, but duplicate QK/full-PV and pair evaluation still prevent a demonstrated
latency benefit. The graph is diagnostic only; no production variant changed.

No-go for resuming either current full503 arm. For swaps, a future complete
observation implementation would first need a source-qualified engineering
gate near/below70ms on representative states, then a predeclared multi-seed
native accuracy/latency screen. This is an optimistic allowance, not a speedup
claim. Directional residual needs a decision objective redesign before more
accuracy work. No further experiment is queued; full503 remains paused.


## 2026-10-18 — Qualify a bounded refresh-output/support factorial

- User authorized a new runtime-headroom and 2x2 diagnostic under a7200s GPU
  reservation cap; original paused campaigns remain paused. Frozen source6aab8632,
  plan docs/REFRESH_DIAGNOSTIC_PLAN_20261018.md, attempt
  results/v31_refresh_diagnostic_20261018/attempt001. All46 CPU tests pass.
- The six measured qualification runs on metadata-selected LongBench index422,
  seed42 match all pre-event fingerprints. Dense/fresh reproduces both V31
  controls exactly; current output and subsequent support are separated across
  all five GLOBAL layers, with no added native forwards or RNG consumption.
- Only V31/dense_fresh is correct in this one-item qualification. Dense/held
  reaches the unchanged output limit; both sparse-output branches use fewer
  calls but are wrong. Do not treat fewer calls as either quality or clean-speed
  evidence. No aggregate necessity or noninferiority claim follows.
- Qualification used471.963 GPU-reservation seconds. Runtime headroom is a
  separate clean/profile phase; the larger panel remains subject to the frozen
  budget and gates. See qualification/summary.md and native completion receipt.

## 2026-10-18 — Preserve profiling failure and recover with explicit provenance

Attempt001 completed qualification and all 20 clean headroom requests, then
engine006 failed before any request because the profiling hook accessed an
uninitialized phase during model startup. Preserve the failed attempt and its
70.189-second reservation. The total charged reservation is 1,521.995 seconds.
The repair makes inactive hooks pass through directly and adds startup tests.

Use a new committed-source successor with explicit, hash-validated references
to the five completed engines. Their original source/protocol identities stay
visible. Only profiling and coordinator/report tooling change; active native
and factorial code must remain byte-identical. The recovery addendum assigns
300 seconds of the existing reserve to headroom, with the 7,200-second total
and 3,900-second factorial limits unchanged. All attempts share one budget
ledger. No result or missing cell is imputed and no old full503 work resumes.

The clean controls demonstrate why consumer and trajectory controls are needed:
FULL and PIECEWISE differ in N on both selected inputs; all-kept FA4 also flips
the 41K answer relative to both. Fixed16's faster 41K request is incorrect while
dynamic V31 is correct. These observations prevent promotion from timing alone;
they do not establish benchmark-level accuracy differences.

The first recovery preparation (attempt002, source 35e2e9bc) failed its parent
campaign check before protocol freeze or GPU launch. The pool-copy loop had
shadowed the requested parent variable. Preserve this CPU-only failure, correct
the parameter and test the real preparation path for both ordinary and successor
cases. Attempt003 is a new preparation under the same recovery contract and
budgets; it references attempt001's completed engines directly.

## 2026-10-18 — Complete the bounded first-refresh factorial

Attempt003 source db755c43 completed all288 measured requests:48 six-arm groups,
eight LongBench questions and four RULER questions, two seeds and two engines.
All groups reached the eligible event; all-six recorded prefixes match. Current
output and future support separate as intended across five GLOBAL layers, with
fixedK, unchanged LOCAL scope/RNG/native-call counts and zero measured captures.
Read factorial_analysis/attempt001/mechanism_review.md for the qualified gates.

The current-output main N effect is+5.54%, but its conditional signs depend on
seed and question tails; extra calls sometimes accompany beneficial answers.
Future-support removal has a+12.88% main N effect, with stronger provisional
evidence under sparse current output. This is a one-event exploratory mechanism
screen, not a cheap-selector evaluation or deployable timing result. Technical
repeats are not independent accuracy observations.

Two LongBench A/A bookends differ in output and N (maximum9 calls), and one
sparse/fresh same-arm repeat differs by33 calls. Official scores repeat, and
RULER repeats exactly. Recorded-prefix equality is not a cloned full-engine
checkpoint; full self-conditioning is recorded only at the first post-event
state. The source audit identifies unrecorded layout/auxiliary/resolved-kernel
configuration inputs, without establishing the cause of variability. Preserve
these cases; causal promotion remains unresolved under the frozen A/A rule.

All clean timing controls completed, but all six Python-instrumented profile
requests differ from their clean trajectories. Progress refresh occupies about
1.5% of profiled request spans on the selected inputs, which is descriptive and
cannot establish a clean critical-path bound. The all-kept profile was skipped
under the headroom phase cap. A single external original-clean-worker trace uses
at most the existing600-second reserve; no original campaign is resumed.

## 2026-10-18 — Close bounded refresh execution without method promotion

The one-shot external trace (runner 2f5687d4, original worker 6aab8632) completed
under its 600-second reserve. All eight measured requests and both explicit
warmups reproduce the clean outputs, trajectories and native counters exactly.
A unique source-consistent synchronization alignment qualifies request/prefill
windows, with no GPU activity crossing boundaries. Independent parity and full
SQLite-summary recomputation pass. Monitored JIT warnings precede measurement;
zero new request-time graph captures and unchanged original caches are verified.

Dynamic-arm attention-named decode GPU activity is about 1.043/1.099 seconds on
the two inputs, while traced prefill accounts for 12.25%/44.27% of request time.
These heuristic activity categories do not isolate refresh, selection or commit,
and are not removable critical-path costs. The trace incurs overhead; it is not
clean timing. Thus the >=10% incremental refresh-benefit gate remains unresolved.
No batching or alternate-model benefit was tested.

Close execution at 5423.862923 GPU-reservation seconds (90.40 minutes), including
failures, warmups and external processing. All reservations are finalized and
GPU idle was verified. All 288 factorial and 20 clean measurements completed;
the all-kept profile remains missing, so the overall attempt stays partial.
No additional GPU work or full503 continuation is queued. The completed
factorial and external trace are recorded as diagnostic subpanels only.

Prioritize a bounded native-operator repeatability check and parity-preserving
refresh-phase attribution before a new method sweep. Future-support maintenance
is the provisional algorithmic lead if runtime headroom qualifies; neither dense
current-output necessity nor cheap refresh accuracy/latency is established.
Use answer quality, native decisions and physical work together, since fewer
calls can accompany a worse answer. See the consolidated analysis at
results/v31_refresh_diagnostic_20261018/analysis/attempt001/summary.md.

## 2026-10-18 — Numerical sample passes; measured refresh opportunity is too small

Completed the separately authorized 2700-second follow-up under native source
6aab8632, capture/NVTX source 62ae618b and recovered CPU replay source 9d864986.
Six first-refresh captures (three unique inputs) and 192 measured automatic-split
operator replays match exactly for output, observation z, active partial outputs
and partial LSE. Layout, input, RNG and recorded configuration checks pass in two
independent processes. The recaptured 70K event has the old geometry but different
Q/K/V bytes; this does not explain or erase the old full-engine control drift.

All eight clean and eight NVTX-measured requests reproduce the original clean
outputs/trajectories/counters. Qualified request windows, full attention scope
counts, zero ambiguous launch attribution and an independent 165-group arithmetic
reproduction support phase-specific cost accounting. No monitored runtime JIT
occurs after warmup and no measured request captures a new CUDA graph.

Dynamic V31's progress-refresh CPU-plus-GPU deletion screen is 2.97%/2.44% of
clean request W at 41K/112K. Even free replacement of bootstrap, initialization,
refresh and GLOBAL commit attention gives 8.14%/7.92%. Adding held/carried KV-buffer
upkeep without double counting gives 10.04%/9.13%, before replacement costs. These
are deliberately generous fixed-trajectory screens, not measured savings or
universal algorithmic bounds. The 41K extended ceiling leaves only about 3.5ms
per request beyond the 10% bar. No credible >=10% net opportunity is established.

Do not start another cheap-refresh selector sweep in these measured settings.
First qualify runtime margin in repeated sparse execution or a feasible long-
output/concurrency/model regime. Different trajectories and workloads require
new measurements. Preserve V31/fixed16 as controls; this result does not promote
fixed16 from a faster but incorrect answer or revive refresh-timing novelty.

Preserve two unstarted original replay leaves, the correctly bootstrapped but
CPU-slow partial replay, and its explicit CPU-only repair. Preserve the capture
index description correction (329 is36K; runtime339 is41K). Total reservation
1334.421052s=22.24min, within45min; all reservations closed and GPU idle. No selector
or older full503 continuation. Full report:
results/v31_refresh_repeatability_20261018/analysis/attempt001/summary.md.

## HumanEval164 K256 completed; separate audit and clean evidence (2026-10-19 Hong Kong)

User authorized a new HumanEval campaign, not an older LongBench resume.
Native source20e6c0be, protocol7b81c845fdce07743d3ed5dd91c4b6b4a54640e071b594ef7f690d6f400bd4d5,
results/humaneval_value_k256_20261018/attempt001. All164 tasks, seed42,
thinking on/off, fixed native output8192, V31/V2/V3a/full swaps/directional
residual and FULL/PIECEWISE/all-kept controls. K256 is four prefix blocks;
V2 is a variable threshold, calibrated on first-canvas audit work only.
V2/V3a Gaussian32 cover all GLOBAL layers; full-value modifications stay at29.
LOCAL remains dense. All58 engines completed without structural failures;
96 H100 tests,328 canonical controls and5,904 independent official rescores pass.

Clean thinking-on correct159/155/160/160/159/158 for V31/V2/V3a/swaps/guard/
dense FULL, with mean N51.787/56.537/49.774/50.232/52.793/48.104 and
W1.16967/1.32367/1.15001/1.83676/1.31692/0.95358 seconds. Thinking-off
correct155/154/154/154/154/157, N10.280/10.677/9.335/10.073/10.055/10.335,
W0.24978/0.25614/0.23545/0.30871/0.25756/0.22114. V3a is the closest clean
point, but both latency intervals include1 and its accuracy intervals include0.
All sparse arms are slower than dense FULL. No joint improvement is promoted.

Audit trajectories differ from clean even in dense controls; accuracy rankings
and call differences change across passes. Keep complete per-pass results.
Do not interpret audit masks as clean masks. First-canvas audit thresholds
0.119043883153/0.144224957031 match GLOBAL sparsity within0.085/0.115pp,
but transfer to clean is unverified. Whole-trajectory GLOBAL sparsity for V2
is29.98/8.37% versus V31 48.17/14.24%; first-canvas matching is not total-work
matching. The divergence cause is unlocalized. No additional GPU trial is queued.

GPU engine reservation15,032.143s; no measured captures, LOCAL skips or fixed-K
budget errors. Preserve all calibration and qualification outputs privately.
Report: analysis/attempt001/summary.md, source and final verification receipts.
Older full503 and refresh campaigns stay paused; legacy STATE.json is unchanged.

## Freeze V31 budget failure-map diagnostic (2026-10-19 Hong Kong)

User authorized the three-stage plan and bounded execution. Follow
`docs/V31_FAILURE_MAP_PLAN_20261019.md`. The14,400-second cap includes all
GPU startup, warmups and failures and has independent phase limits. Select16
LongBench development questions by metadata hash before generation; map
K8192/4096/2048 with unchanged native decoding and corrected dense controls.
Promote only same-type failures on both screening seeds; confirm against all
three dense consumers on two new seeds in two processes. Do not relax gates.
At most four confirmed cases enter an audit-only first-held-call output/support
factorial with same K, native prefix checks, V31 bookends and a sham. This
identifies possible repair mechanisms; it is not a deployable selector or
held-out quality claim. Older paused studies are not resumed.

## Qualify and preserve V31 failure-map continuations (2026-10-19)

Native source5dcade2c and protocol0ca7a002319049af7f0a7b379647ddca576af676909aa702ea78c0d9c31cb7e4
remain unchanged. All six qualification families passed. The cold first engine
used344.047s; original qualification stopped at774.110s under its predeclared
180s launch gate. Addendum71109d5b used650.094s of the existing recovery reserve
for missing dense families; no budget borrowing or expansion. Preserve attempt001.

An absolute-path invocation correction in77956613 preserves a zero-record,
returncode2 screen launch in attempt002, charging1.001s. Attempt003 carries
verified qualifications and identical protocol/source/inputs. Native generation
restarts only in new leaves; no completed experiment is rerun. This is an
orchestration correction, not a new selector or scientific protocol.

Qualification bookends show output/call drift on one of two inputs at K4096
and K2048, without score drift; K8192 matches. This does not diagnose clean
repeatability or justify a quality claim. Keep the frozen clean bookends,
new-seed/process replication gates, and common-prefix/sham checks for repairs.

## V31 budget screen completed without promoted failures (2026-10-19)

Native5dcade2c, analysis7cff0c2f, protocol0ca7a002319049af7f0a7b379647ddca576af676909aa702ea78c0d9c31cb7e4.
All320 screen requests and20 qualification requests complete;340 official
rescores and376 record checks agree. Frozen same-type two-seed candidates are
zero at all three budgets. Stop before confirmation/repair; no gate relaxation
or selector trial. The completed diagnostic is not a global noninferiority claim.

The important control result is all-kept meanN203.31 versus FULL332.66 and
V31K8192 201.41. Preserve consumer-path/output-length confounding. K2048 relative
to all-kept has−3.125pp accuracy and+12.74% N/C, with intervals including no
difference. Retain this exploratory regime and the mixed-type question462 case
for independent follow-up under a new frozen endpoint. Do not call them
confirmed failures or repair evidence. K4096 is a strong fixed-budget reference
for any future selector; fixed16 remains an explicit comparator.

Five of96 bookend pairs change output/calls and one changes score. All six screen
engines have one post-warmup fused-MoE JIT warning. Report raw times with this
limit and make no qualified warm-speedup claim. No measured graph captures,
fixed-K/tail violations or LOCAL skips. Total5355.761 seconds within14400;
GPU idle, older campaigns paused. Complete report: results/v31_failure_map_20261019/attempt003/summary.md.
Sanitized export: publication/attempt002; earlier qualification publication is
unchanged. MAGE's static unmasking and SparseD's early-dense protection remain
prior art; adaptive convergence is a research question requiring a demonstrated
failure mechanism, not established novelty by naming a new selector.

## Qualify compiler numerics before HumanEval K256/K128 (2026-10-19)

Investigation results/humaneval_reproducibility_20261019/attempt001 preserves
the original K256 study and both older LongBench pauses. Same-process audit
interventions agree while independent native processes drift. The reproduced
same-input operator contrast (evidence582bc68d) changes only R0_BLOCK512/1024
and reproduces the recorded fused-normalization output differences. It establishes
one numerical mechanism, not complete end-to-end repair or a selector advantage.
Read docs/HUMANEVAL_REDUCTION_MECHANISM_20261019.md for observer exclusions.

Require actual selected-kernel receipts, not an environment-variable claim:
TORCHINDUCTOR_DETERMINISTIC=1 alone still receipts40 nondeterministic kernels.
The next named runtime variant explicitly binds deterministic compiler options
to vLLM and the separately compiled native sampler. Sourceb59cd177 fails closed
on missing/ineffective receipts or changed graph mode. It preserves native sampler
formulas and selector budgets. Independent processes, audit/clean equivalence
and both thinking modes remain gates before full generation or latency claims.
No full matrix is launched while qualification fails; no accuracy-based tuning
or relaxed reproducibility criterion is authorized by diagnostic progress.

## HumanEval baseline numerics qualified; corrected panels released (2026-10-19)

Qualified source10341ab2 explicitly binds deterministic options to the model and
the unchanged native compiled helper callables. Actual53/53 kernel receipts pass.
Horizontal combo fusion/benchmarking are disabled; retain that policy identity
and matched native graph/FA4 controls when measuring latency. The environment-only
and model-only failures remain diagnostics, not successful runtime variants.

Two independent unobserved FULL engines match12/12 output/N/C/T pairs. Eleven
further dense/V31 engines, both thinking modes, pass all declared pre/post
sampler-state, valid-history, emission and RNG comparisons, including same-mode,
audit/clean and bookend contrasts. Baseline marker:
results/humaneval_reproducibility_20261019/attempt001/qualification/attempt001/baseline_runtime_qualified.json.
This is a six-question baseline qualification, not a full accuracy result.

H100 preflight passes122 checks at both prefix budgets; two additional CPU
reporting tests reject population/trajectory mismatches without requiring
different methods to agree. K256 execution10a222c6 is frozen in a new study;
work-only calibration precedes method qualification and full generation. Its
CPU output-metadata telemetry records emission intervals without new device
operations. K128 is gated on independently sealed K256 completion and fresh
threshold calibration. Keep original panels and older pauses intact. Diagnosis
10477.365s, preflight31.295s; the43200s shared ceiling and phase caps are unchanged.

## Residual-specific qualification failure holds both full matrices (2026-10-19)

In corrected K256 protocol0faecb13, all seven other methods match audit/clean
output/N/C/T/intervals on the12 thinking-on qualification cases. Direction-guarded
residual differs on question129,seed43: auditN167/T3295, cleanN171/T3302; final
emission interval6/10, preceding12 intervals equal. No first internal divergence
has been located yet. Exact first12 intervals do not prove identical earlier
tokens or internal states. This is not sampling-noise evidence or qualified
residual reproducibility. The six-question baseline diagnostic stays valid.

Finish the active thinking-off work-calibration job naturally (332 requests),
then halt the owned controller before another GPU launch. Preserve its closed
reservation and all attempt artifacts. Hold K128 followup; no full matrix starts
and no gate is relaxed. A new bounded diagnostic freezes both pass-specific
prefixes plus observed/unobserved residual repeats, with production-cache and
native full sampler fingerprints. Exclude independent oracle work from the
production-cache trace. Native formulas/budgets remain fixed; diagnostic copies
are not latency evidence. Initial ceiling1500s charges existing phasek256,
without expanding total43200s or borrowing from other phase caps.

## Residual trace pair is inconclusive; broaden the remaining variation (2026-10-19)

The two bounded engines (370.783/360.297s) qualify their residual observation:
both unobserved/observed/unobserved bookends agree. All184 sampler calls and all
production residual/cache fingerprints agree across passes, reproducing the
earlier clean result. The original residual difference was not captured.
Independent unobserved prefix comparisons change the audit residual case,
clean measured V31 question129,seed43 and V3a question116,seed42, plus a clean
full-swap warmup. Preserve all contrasts. The earlier baseline diagnostic is
complete within its declared cohort/context; mixed-method reproducibility is
unresolved despite actual deterministic Inductor kernel metadata.

Next freeze two identical clean mixed-prefix engines with per-arm bookends.
An integer-only two-seed128-bit all-logit fingerprint supplements the full
sampler observer. Three H100 tests compare an independent NumPy oracle, repeats
and a tail bit change, covering partial and multistage reductions. This is a
noncryptographic first-divergence locator, never equality or root-cause proof.
Any detected change needs actual tensor comparison/replay. Keep native formulas,
both full matrices and K128 hold unchanged; no compiler/library flag is promoted
without evidence. The bounded stage still charges existing phasek256.

## Full-cohort traces locate remaining differences before sampler updates (2026-10-19)

Mixed trace attempt002 completes877.086/867.093s. Two of72 independent observed
method/case contrasts differ: V3a116/42 at raw-logit input to call229, and V31
control-after116/43 at call128 (zero-based). Both have step5/11, valid canvas256,
and encoder_phase=False. Prior complete sampler/logit records match; the first
changed before-field is logits, with unchanged default CPU/CUDA RNG hashes.
This bounds the recorded change upstream of the sampler update. It does not
identify an attention, MLP, cache or selector operator yet.

Both engines fail some observed/unobserved trajectory checks. Preserve the
instrumentation limit and all native repeats; do not promote a repair or release
full accuracy. Next freeze attention/map/cache/progress fingerprints on only
those two cases, with identical original prefixes and unobserved repeats.
Observer imports follow native mask-fix setup and wrap the already patched
adapter path; output buffers are read only after their native forward. The
integer fingerprint implementation remains source-tested; tensor proof/replay
must follow a localized change. Two750s reservations charge existing phasek256;
no budget expansion. Existing43200s total and both campaign holds persist.

## Preserve observation/context negatives; test fixed BLAS workspace (2026-10-20 Hong Kong)

Attention attempt003 has matching per-layer/progress/cache/map/output and full
sampler traces with30-layer coverage, at525.730/500.678s. The original fault did
not recur, so no operator is localized by those matching traces. One-engine
crossover510.626s has first unobserved V3aN274/C20/T5089 and final unobserved
N272/C19/T4787; all four traced conditions match the latter. A V31 bookend pair
also differs. This excludes a simple tracing-only explanation, not every
timing/context mechanism. Retain failed observation checks.

Native FA4/selector definedness screen passes4 checks in22.915s. NaN-prefilled
observation outputs match reference/native outputs, and native-dimension scratch
poisoning, canaries, growth/shrink and returned-map ownership pass. Do not infer
global memory/race correctness from this bounded negative. A controller binding
failure before worker launch is preserved separately and charged a conservative
36.299s wall upper bound; active GPU/model work was zero in that attempt.

Sourceb13808d4 freezes a one-factor fixed-BLAS-workspace diagnostic, without a
Torch algorithm-guard, memory-fill, sampler, selector or graph-mode change.
Require actual configured bytes in the exit receipt. Two1000s full-cohort jobs
charge existing phasek256. This is a diagnostic candidate, not a promoted cause
or repair. Both full matrices and K128 hold remain; no accuracy or speed claim
is released. All caps and total43200s remain unchanged.

## Fixed workspace is insufficient; isolate guard compatibility (2026-10-19 UTC)

Attempt005's effective33554432-byte cuBLAS workspace does not eliminate native
variation: V2 question145,seed42 first changes raw logits at call529 and ends at
N586/591. Guarded definedness attempt003 passes4 operator checks. Native guard
attempt006 stops during warmup,95.009s, before saved question records. Its actual
failure is a BF16 source/FP32 destination indexed-assignment dtype error; later
missing-receipt errors are consequences of initialization not completing.

Pinned Inductor index_put_impl_ returns through its deterministic fallback before
the destination-dtype conversion present in ordinary lowering. Generated native
code passes BF16 soft embeddings into FP32 self-conditioning storage. Freeze a
bounded operator regression and narrow process-local compiler conversion adapter,
requiring an actual unpatched failure and exact explicit-cast reference match.
Preserve native Python helper identity. The compatibility fix is not yet a cause
or repair for the earlier intermittent model-logit divergence. Fresh compiler
cache namespaces are mandatory for any later adapter-enabled native attempt.
No full accuracy run is released; existing resource caps and holds persist.

## Guard compatibility succeeds, native repeatability still fails (2026-10-19 UTC)

The indexed-assignment operator regression passes, charging7.082s. Its narrow
compiler adapter retains native Python helper identity. Guarded native attempt007
completes997.189/897.178s, actual guard=True/fill=False/32MiB cuBLAS,42/42 recorded
kernels deterministic. All72 fully observed independent trajectories agree.
Unobserved repeats still fail1/72 and3/72, across V2, V31 and residual. Reject the
guard as a qualified repair. Do not claim observation alone is the cause.

Recorded source/name/config entries match across earlier failing engine pairs:
37 unique entries in each pair for attempts002/003/005, with no selected-config
differences. Next compare native graph boundaries and before-sampler logits with
asynchronous fingerprints, avoiding full observer per-step CPU tensor copies.
Two158-request engines include80 full-observed,72 light-observed,6 pure native
requests, with independent and within-engine contrasts. Keep native dispatch,
outputs, sampler and methods unchanged; disclose partial activation argument
scope and possible padded storage differences. A localized hash difference still
requires actual tensor capture and exact-input operator replay.

Five source-bound CPU tests pass; no CUDA/model operation. Two1400s reservations
charge existing phasek256, no total/phase expansion. Full matrices remain gated.
The adapter metadata source hash originally unwrapped functools.wraps to the
old fallback; immutable complete-file bindings are the actual source authority.
Current code fingerprints its code object instead, CPU-checked; no mathematical
change or rewrite of old qualification artifacts.

## Scalar diagnostic coverage and observer interpretation correction (2026-10-19 UTC)

Graph attempt008 stops165.015s on a zero-dimensional BF16 argument in diagnostic
fingerprinting. Preserve it, flatten diagnostic scalar views and zero-pad copied
byte extents to4 without changing native inputs or existing fingerprint kernels.
Six source-bound CPU checks pass10.939s, including scalar BF16/bool/byte cases.
Attempt009 is the new immutable native retry; phase/total caps remain unchanged.

Correct earlier shorthand about full sampler tracing's per-step CPU copies:
trajectory_worker retains GPU snapshots and hashes them on CPU at request finish.
Residual cache tracing separately does CPU copies inside residual forwards.
Graph-only mode avoids full snapshots and generator-state reads but changes GPU
work and allocation/lifetime. See HUMANEVAL_OBSERVER_SEMANTICS_20261019.md.
These are exact source distinctions, not a proved race/synchronization cause.

## Preserve partial graph evidence; optimize observation before continuation (2026-10-19 UTC)

Attempt009 expires1403.290s,113/158 rows. All33 completed seed42 full/light
contrasts match raw logits and native outputs/trajectories. Seed43 light cases,
pure repeats, second engine and exit compiler inventory are absent. The partial
analyzer verifies every failed-marker/source/file binding and refuses technical
qualification or full accuracy. Absence of later contrasts is never agreement.

Runtime-length checksum kernels and batched tiny-word CPU drain pass the final
12-check CPU/H100 preflight in13.898s. It includes independent NumPy/static
agreement, repetition, tail bit, input preservation, actual CUDA unsigned high
bits and seven wrapper/scalar/drain checks. Record one dynamic words/one fold
variant on tested sizes; no inference-speed or native-equivalence claim follows.
Earlier computational preflights remain preserved; final source/metadata gating
uses attempt003. No native selector/sampler/graph formula changes.

Explicit allocation_addendum002 reassigns3600s from unused K128 capacity to
execution continuation/K256:10800/19800/12600, same43200s total. All old limits,
ledgers, receipts and scientific gates remain. Attempt010 freezes two1500s
158-request engines after the combined checks. It is still a diagnostic; a
matching bounded trace does not repair prior native failures. No full matrix
or historical campaign resumes from this stage alone.

## Localize the remaining HumanEval drift before an operator claim (2026-10-19 UTC)

Attempt010 expires1503.675s at132/158 public rows. Of52 completed full/light
contrasts, two q129/seed42 cases differ in raw logits. The first changed graph
input is layer29's BF16 attention output before wrapper30; earlier recorded
boundaries match. Full swaps changes logits at100 without changing native
N/C/T/output; direction residual changes at154 and N153/159,T3168/3182/output.
These partial hashes do not establish equality of actual masks, paged KV,
cached moments or correction state. Pure repeats/engine2/inventory are absent.

Freeze a smaller two750s,84-request capture after exact-source CPU preflight.
Keep the original80-request prefix and only two q129/42 targeted probe/pure
contrasts. Observe actual FA4 operands/partials, merge, moments and cache at29;
read write-only outputs only after native return and call every original once.
Private raw operands support the next exact-input replay. No sampler/selector
change, qualification relaxation, full matrix release or total budget increase.
Plan: docs/HUMANEVAL_LAST_GLOBAL_CAPTURE_PLAN_20261019.md.

## Capture before changing the FA4 observation kernel (2026-10-19 UTC)

Source666f54a2 native013 engine1 completes84 requests610.873s. Full/layer-only
swaps and residual repeats match native metrics; pure swaps changes N156/187,
C13/16,T3307/3981, whereas residual pure repeats match. Complete captured
operands at the earlier changes100/154 match in this engine; the exact old
operator discrepancy was not reproduced by adding the target capture.

Both call100 swaps and154 residual are observing_dense_fwd anchors with
contiguous full KV, num_splits=0, pack_gqa=False, causal=True/dynamic=False.
These anchor outputs precede swaps or residual correction. This is useful
localization, not proof of an automatic-split defect. Prepare an exact-input
operator replay after engine2 closes: automatic/single-split, observing/plain,
freed FP32 scratch zero/finite/NaN factors; sixteen repeats, max240s. Compare
actual outputs to captured bytes and a separate FP64 oracle, record actual
kernel configuration and input preservation. All-stable replay is a negative.
No silent numerical-policy promotion or release of full accuracy follows.

Capture wrappers passed4 CPU checks10.771s. Operand restoration passed4 CPU
checks1.933s. Final comparison distinguishes logical used KV/active lists from
physical translation/padding and must pass its exact-source updated preflight.
011/012 never launched on GPU and remain immutable preparation/supersession
records. Plan: HUMANEVAL_OBSERVE_OPERAND_REPLAY_PLAN_20261019.md.

## Isolate the remaining numerical failure to observing FA4 overlap (2026-10-19 UTC)

Completed same-input replay00221.870s finds one BF16 output change in single-
split observing FA4 at K3532; automatic16-split and poisoned freed-scratch factors
are stable in that small screen. The controlled followup uses8192 calls, two
independent processes and reversed factor order (55.959/50.147s). On K3532,
observing/overlap=True varies4/510 automatic and4/510 single-split comparisons.
All2048 observing/overlap=False calls are stable, as are all4096 plain calls.
Attention-mass bytes match native for every observing call; Q/K/V are preserved.
First automatic change is2322 BF16 elements, one head/query64 tile,max0.0625.
This demonstrates an operator numerical failure, not a selector advantage or
score bug. Exact instruction-level mechanism remains unproved. Full report:
results/humaneval_fa4_overlap_screen_20261019/attempt001/publication/attempt001/.

Qualify the smallest supported correction: disable intra-warp-group overlap
only for mass-observing FA4, leaving plain arguments and native sampler intact.
Four source-bound CPU checks pass4.606s. Per-request class-level receipt capture
survives shared-worker rebinding. Attempt014's validator field-path mistake
stops after one valid warmup160.381s; preserve it as harness failure. Two new
source-bound regressions pass0.382s. Fresh015 source d62691b2 uses two1000s,
152-request engines, observed versus pure72 cases, independent seeds42/43.
Both full matrices stay gated. Native qualification must precede corrected
audit/clean/control qualification, calibration and official-scored full runs.
No silent slower-mode performance claim, budget expansion or old-campaign resume.


## 2026-10-19 — Qualify observing-only FA4 correction before corrected HumanEval panels

Native attempt015 (`d62691b2`, protocol `5163cde0`) completes both independent
152-request engines. Strict qualification passes all 72 complete sampler/cache
trace comparisons across processes and observed/pure native output and trajectory
comparisons; no warmup changes. The certificate binds the qualified helper SHA
`f13970b2`, all leaf markers and analysis hashes. This is bounded diagnostic
qualification, not full164 audit/clean or latency qualification. The earlier
operator factorial supports the observing-only non-overlap correction; the exact
instruction-level hazard remains unproved.

Freeze corrected K256 (`55370024`) and K128 (`187719fc`) from deploy `435511e4`.
Use all164 questions, seed42, both thinking settings, unchanged native decoding,
work-only V2 calibration, strict audit/clean equivalence, independent process
prefix replication and official independent rescoring. Share only newly run,
source-matched budget-independent dense FULL/PIECEWISE/all-kept controls across
budgets. Every selector arm remains fresh at its own budget. Full-value refinements
remain layer29-only; V2/V3a retain all five GLOBAL layers; LOCAL remains dense.

Allocation003 explicitly revises the investigation's self-imposed 43200s estimate
to a bounded combined 65831s including all earlier usage, with a shared 36000s
allowance for the already-requested corrected panels. The user supplied no numeric
hour ceiling. The completed original K256 campaign cost 15032.143s; its cost plus
the unexpected diagnosis explains the revision. Preserve original protocols,
allocation001/002, actual charges and all paused campaigns. No extra GPU or new
method sweep is introduced. Each reservation remains at most1800s, one worker,
with immutable attempts and a combined/phase accounting check.

## 2026-10-20 — Investigate native trajectory control before another sparse selector

Research-only review at HEAD `0b86cdd4` reads the DiffusionGemma report, pinned
native sampler, related sparse/trajectory/distillation literature, and existing
source-specific panels. Full assessment and source fingerprints:
`results/denoising_trajectory_research_20261020/`. No method implementation,
experiment, queue action, or new performance result accompanies this entry.

The hypothesis remains plausible: attention support can change logits, native
acceptance/re-noising, self-conditioning and the time at which unchanged native
stopping fires. Lower entropy or attention error does not establish correct
convergence. RULER128K supplies a one-canvas positive lead; LongBench reductions
must separate output length, within-canvas calls and consumer effects. Earlier
fixed-canvas results mitigated step inflation; clean64 value-method results were
negative. The completed small refresh factorial is a provisional support-lifetime
lead, not a confirmed cheap controller or universal benefit from holding masks.

Generic mass-mask reuse, refresh, temporal confidence and fewer denoising steps
already have close prior art. Recommend discussing a controller that predicts
the quality-constrained downstream value of fixed-budget support and its
lifetime, with all selection cost included. A small mask bank is one proposed
implementation form, not a tested method. A separate training route would learn
earlier reliable predictions on native on-policy states; DiffusionGemma SD·RL,
OPTD and discrete consistency learning are relevant precedents.

Any later diagnostic should first use corrected observing FA4 and reproducible
source-matched dense/all-kept/V31 controls, then test common-state interventions
and affordable decision features. Nonsignificant score differences, fewer total
calls, or an oracle intervention alone do not establish accuracy preservation,
faster within-canvas convergence, or clean latency improvement. No historical
protocol, paused campaign, result ledger entry or current execution state changed.


## 2026-10-20 — Corrected K256 is reproducible; retain negative value-method results

K256 protocol55370024 completes all164 tasks/seed42 in both thinking modes.
Full audit/clean and independent prefix replication pass for every method;
independent official rescoring agrees on5904 main and432 qualification records.
V2 has162/158 correct versus V31's159/157, but uses50.043/10.854 forwards versus
49.256/10.774 and takes1.14762/0.26408s versus1.12399/0.26218s. Whole-trajectory
GLOBAL sparsity25.618%/5.168% is lower than V31's48.325%/20.031%, so first-canvas
matching must not be described as whole-work matching. Full swaps155/156 and
residual158/156 do not meet the joint accuracy/latency objective; preserve those
negatives. Native FULL and all-kept controls remain separate implementations.
Do not infer seed robustness from seed42 or the six examined seed43 questions.
K128 now starts fresh calibration with only source-matched dense references
shared; no K256 selector result is reused at K128.

## 2026-10-20 — Narrow trajectory research to training-free support dynamics

Research-only follow-up at `9c0a90bf` incorporates the user-supplied stopping
analysis, primary literature and newly completed corrected K256 publication.
Full assessment, analytical derivations and evidence fingerprints:
`results/denoising_value_dynamics_research_20261020/`. No code, GPU evaluation,
queue action or automation accompanies this entry.

Keep native denoising and stopping fixed; training-free also excludes fitting a
selector. Investigate whether support can preserve useful evidence while reducing
harmful sensitivity to changing queries and current-canvas K/V. Equal mass and
current PV error do not fix that response. A response objective differs from the
earlier post-O reconstruction objective, but key-value covariance itself is
established prior art and does not supply novelty or a convergence guarantee.

Correct-to-wrong revisions and useful corrections should be measured separately
where external correctness is defined. Prefix deletion amplifies retained canvas
mass, so equal-K comparisons need per-row mass controls to isolate value effects.
A query-change support bound may delimit reuse within a fixed prefix; it cannot
certify accuracy or carry unchanged across canvases. Full covariance/gradient
cost, conservative bounds, false-confidence lock-in and lost correction remain
material risks. No candidate has been validated.

Native full-state forks and passive quality-versus-call accounting are proposed
diagnostics. Do not disable stopping, substitute RNG, or mix dense/sparse sampler
pathways in the primary study. Such changes would answer different questions.
Preserve source-matched FULL/PIECEWISE/all-kept/V31 controls, physical work
accounting, question/seed uncertainty and clean selector overhead. The corrected
K256 negative and the limited historical RULER lead remain separate evidence;
no new experimental result, ledger entry or execution authorization is implied.


## 2026-10-20 — Seal both corrected HumanEval budgets and separate mean gains from claims

Both frozen panels complete and reproduce all164 questions across audit/clean
for every method in both thinking modes. Independent engine prefixes and official
rescores pass; combined publication and independent readback/source receipt are
under humaneval_corrected_k256_k128_20261020 and humaneval_corrected_validation_20261020.
Do not reuse older unqualified accuracy/timing rankings for these corrected cells.

V2 is the strongest observed value-method lead atK128:158/160 correct vsV31's
157/156 and mean W1.22407/0.25270s vs1.29796/0.27420s, with N53.323/10.439 vs
57.415/11.488. The speedup intervals include1; seed42 and six examined seed43
questions are not robust confirmation. K128 non-thinking threshold matching
misses the0.25pp target (+0.53116pp); this stays explicitly unmatched under the
predeclared nearest-work rule. Whole GLOBAL kept work also differs. Preserve
V2's3 thinking-on output caps versus control1 and all per-item transitions.

Full swaps and guarded residual do not meet the joint objective across these
cells. Swaps save few model calls but add substantial full-PV/selection cost;
residual does not consistently shorten trajectories. Native FULL remains faster
than the tested sparse/value methods. No new variant or GPU study is authorized
or launched by this result interpretation. Keep model/native sampler, fixed
per-query-tile selector budgets, provenance and previous pauses unchanged.

## 2026-10-20 — Document the two reproducibility fixes and complete HumanEval results

User requested a committed/pushed README update and detailed reproducibility
documentation. At reviewed HEAD `09877bcb`, both corrected HumanEval budgets are
complete. Document reduction-configuration rounding and the separate intermittent
mass-observing FA4 overlap failure, actual implemented corrections, unsuccessful
partial fixes, native-decoding preservation and full audit/clean evidence.
Keep operational job ownership separate from numerical repeatability and avoid
attributing historical LongBench-v2 drift without a corrected-runtime replay.

The documentation report derives all tables from the immutable combined
publication: accuracy, N/C/T, S/W/S-per-N, GLOBAL/overall physical sparsity,
paired intervals, first-canvas matching, whole-trajectory kept work and tails.
Retain K128 V2's unmatched non-thinking calibration and output-cap tail, source
and layer-scope differences, shared dense references and seed42 limits. A mean
K128 improvement is a development lead, not established robust superiority.
README section `2026-10-20 update` links this report and the correction account.
No source/runtime change, GPU run, rescore or campaign resumption accompanies
the documentation. Existing frozen artifacts and separately owned edits remain.


## 2026-10-20 — Sparse V2 refresh prototypes have only a short synthetic lead

User-authorized kernel study at verified HEAD755b528c; artifacts under
`results/v2_sparse_refresh_kernel_20261020/`. Current published V2 returns
dense observing FA4 on refresh. New sparse-refresh variants return masked
current full-V output on refresh, retaining the inherited hold/carry path and
native sampler. Keep their identity separate from published selector-only V2.

Fourteen H100 operator checks pass, zero skips. Fused Q128 QK/PZ discovery,
normalized routing mass/squared risk, optimized summary scan and conditional
full-PV were implemented; D64/warp alternatives and failed mixed-shape cluster
lowering remain negatives. Synthetic model GLOBAL geometry at about50% sparsity
finds best complete sparse refresh160.17/490.42/998.86us versus V31
299.73/367.61/461.00us at1024/4096/8192 keys. Plain dense FA4 is faster at every
measured length. Source-frozen final timing has zero measured graph captures.
These are synthetic component diagnostics, not HumanEval or native latency.

Automatic approval review rejected native QKV persistence and GPU-only private
HumanEval attention replay pending explicit user authorization. Neither native
worker launched; a numeric-only replay approval question is pending. A source
commit was also rejected; immutable source snapshots retain tested code without
Git index/history changes. Do not promote AIME26 or LongBench503 from synthetic
results. New native first-canvas calibration, correctness/receipt/trajectory
qualification and matched HumanEval performance/accuracy remain necessary. No
old paused study or existing frozen protocol changed; no result ledger claim.


## 2026-10-21 — Reduce V31 observation overhead; retain real-online V2 as a negative kernel screen

At verified HEAD755b528c, user-authorized H100 work adds independent experimental
refresh implementations under refresh_kernel_optimization_20261021. Source is
frozen in attempt004, not installed as a native default. V31 reuses softmax
probabilities and fuses selection/list creation, with an underflow fallback.
All six V31 tests pass. Static complete refresh56.07/191.83/366.37us at
1024/4096/8192 keys replaces inherited299.94/372.00/468.48us. Plain dense is
46.90/172.94/358.71us. At8192, dynamic dispatch gives dense294.21us versus
optimizedV31306.52us; retain the dispatch-matched control.

V2 performs real ordered fixed-threshold decisions and skips full PV in the
same attention traversal, while charging fresh current-V projection/RMS. Lean
routing, qualified BF16x3 PZ and double-buffered votes improve the prototype.
Nine synthetic qualification cases and final timing shapes show no mask
mismatches; BF16 custom output differs from same-mask FA4 by about0.003 relative
L2 and is not native equivalence. At~50% GLOBAL sparsity, complete online
187.55/601.06/1158.78us is only tied or1–3% ahead of two-pass references and is
still slower than dense FA4. Do not promote a successful sparse-kernel claim or
launch conditional AIME/LongBench generation from these numbers. HumanEval is
also unlaunched for the new output variant. Held consumer and sampler stay fixed.

Alternative register layouts and schedules are not inferred successes;
failed overlap, split/TMA, compact/staged and deferred-P attempts remain recorded.
Compiler spills motivate the next register/pipeline redesign, but unavailable
Nsight counters preclude a measured stall breakdown. Borrower FastWAM workers
and dedicated supervisors were identity-verified and reclaimed as authorized;
MPS was not signaled. Exclude contaminated screen007 and conservatively006.
Final clean screen013 has measured graph captures0 and no foreign GPU client.
This is a synthetic diagnostic completion with a partial optimization objective,
not a native panel or result-ledger entry. Preserve inherited method provenance.


## 2026-10-21 — TMA/channel-router V2 reduces online refresh cost, still fails dense gate

Continue at verified HEAD755b528c in an independent source-frozen experimental
variant, results/v2_online_pipeline_20261021 (frozen/attempt003, exact compiled
source qualified_source/attempt002, final report/attempt003). Two channel groups
increase Q256/H16 CTA coverage and halve the output accumulator; the final WG0
Gaussian32 router removes one partial-risk barrier. TMA Q/K and retained-only V,
next-K overlap, acquire/release votes, retained-mass output and one-kernel list
compaction reduce complete refresh cost. Approximate division is explicit.
Selected schedule3 uses214 registers with zero spills. Duplicated QK/routing
across channel groups is measured, not hidden by a sparsity count.

Final clean synthetic complete-refresh GPU medians at1024/4096/8192 keys are
102.46/322.18/629.52us versus inherited real-online184.63/589.47/1147.74us;
the previous kernel with fast lists is165.04/569.15/1134.71us. Two-pass discovery
plus sparse FA4 takes185.38/609.74/1156.42us. The gain is real against those
measured implementations, but plain dense46.23/166.18/358.58us remains faster;
dynamic dense291.24us is stronger at8192. Do not expand the conditional native
campaign from this result. Kernels alone are also slower than plain dense.

Final diagnostic and production each pass six independent selector/FA4-output
cases and all nine H100 regressions pass. Timed capture counter0, five final
geometry masks exact, current-V projection/RMS and lists charged. BF16 custom
output~0.003 relative L2 is not native accuracy or trajectory equivalence.
Historical selector-only V2 dense refresh output is distinct from this online
sparse output variant. Preserve Yuhan and Junyu provenance and NVIDIA BLASST
reference. Native held/carry consumer, LOCAL/prefill and sampler are untouched.

Retain negatives: serial/TMA versus spilling overlap; ordered four-KV wavefront
DSM handoff; exp2 and full-sketch duplication; duplicated QK per WG; single-WG
channel splitting; two-WG channel overlap. Qualification034 exposed an invalid
exact-text decision-publication patch after adding a channel guard. The builder
now validates the decision patch; no failed-binary timing is used. The final
changing-support graph regressions pass. Source/report failures and contaminated
workers remain immutable with failure markers. Exclude screen004/005/008/009.

The user explicitly authorized preemption of GR00T gr00t_n15_venv/sprof_1020.
No matching live client appeared after that approval, and no signal was needed.
Final H100 is idle. Recorded completed-screen worker reservation is379.08s,
including initialization/JIT/warmup; this is neither total active GPU seconds
nor total campaign GPU use. No native HumanEval/AIME/LongBench generation or
private-input replay was launched. Do not infer matched first-canvas work,
accuracy, denoising calls or request latency from this synthetic calibration.

Final bounded continuation: reduced QK unrolling removes overlap spills but does
not beat the serial winner (screen026/027). The compiler reports WGMMA accumulator
serialization in the channel-router overlap variant; no hardware attribution is
claimed. Final builder/source freeze is attempt004, with byte-identical generated
winning CUDA verified against pre-timing attempt003. Final report is attempt004;
tested/timed binaries remain the original qualified_source/attempt002 artifacts.
Completed-screen worker reservation including these two controls is411.45s, with
the same incomplete-total accounting boundary. No native run was launched.


## 2026-10-21 — Q128/register-source online V2 improves refresh, dense parity still fails

At verified HEAD755b528c, source-frozen synthetic continuation in
results/v2_online_warpspec_20261021 (pre-timing frozen/attempt003, final
frozen/attempt004, exact binaries qualified_source/attempt002, report/attempt001).
Retain Yuhan inherited cross-step reuse and Junyu Gaussian32/Hopper provenance;
NVIDIA BLASST buffering/warp-specialization informs the execution variants.

The winner uses full Q128 within one CTA/two WGs, four128-channel output groups,
register-source BF16 PZ/PV, prepacked current Gaussian32 operands, double-buffered
one-barrier votes, WG-local synchronization, next-K/early-V overlap and base2
SFU/unnormalized folded-risk arithmetic. Explicit fast math is a numerical arm.
Schedule2 prefetches all V and skips full PV online; schedule3 also skips V
loads, separately timed. All projection/RMS, packing and list costs are charged.
233 registers, zero spills, do not confuse core+packing with a raw CUDA kernel.

Clean complete-refresh medians at1024/4096/8192/16384 keys are
84.10/239.20/460.20/892.71us; plain dense46.72/173.13/361.68/710.40us wins.
The corrected previous online control with matched fast lists is
104.31/327.51/635.47/1236.88us (19–28% improvement); this is not a task or
end-to-end speedup. All-kept FA4 and dynamic dense remain reported. Synthetic
physical GLOBAL sparsity is~50%, except mandatory-support-limited512 at37.5%.
No native first-canvas calibration, LOCAL accounting or task generation follows.

Diagnostic and production each pass9 independent ordered-selector/output cases;
13 H100 regressions pass with zero failures/skips. Timed captures0, changing
support graph replay, sticky/partial/extreme/currentV checks pass. Same-mask
FA4 relativeL2<=0.00307 does not imply native accuracy/trajectory equivalence.
Source and exact library copies are hash verified. No failed/FP16/TMA candidate
is promoted. Unique per-build temporary roots fix concurrent nvcc sandbox-PID
collisions; compiled libraries must export the required ABI.

Correct the prior comparison identity: channel-router builder's epoch insertion
silently missed output[64] because it expected output[128]. Old passing examples
do not establish race freedom. Frozen previous attempts remain unchanged;
the builder now validates a generic exact anchor and the corrected reference
is independently qualified and remeasured. Final comparisons use that control.

Independent router/PV-CTA and vector DSM designs are correct but much slower;
staged QK, QK lookahead and two-scorebank overlap also lose. The final two-group
256-channel design removes spills, halves duplicated QK, but loses at509.38us
for8192. Winning four-group routing/QK repeats four times to occupy more SMs;
half PV therefore does not halve total kernel work. Rejecting half the8K tiles
only cuts custom core451.74us to427.63us. The next design must reduce QK
replication and handoff latency while retaining ordered threshold semantics.
This evidence does not establish that dense parity is impossible.

Final GPU idle, no foreign preemption needed; own stopped-worker receipt kept.
Completed-screen process reservation638.12s includes JIT/warmup but omits tests,
qualification, builds, probes and failed/stalled processes, so not total active
GPU seconds. No native HumanEval/AIME/LongBench campaign was launched or old
paused campaign resumed. No native result ledger entry; diagnostic marker sets
objective_achieved=false. Keep native qualification and measured dense parity
as gates for the requested conditional benchmark expansion.


## 2026-10-21 — Explicitly requested optimized-kernel native benchmark campaign

User requests current optimized online V2/V31 versus dense across all30 AIME26
x seeds42/43/44, MATH500 full500, HumanEval164, LongBench-v2 full503. Other
suite seed42, thinking-on. Prior synthetic dense-parity gate is superseded by
this explicit evaluation request. New source/data/kernel/runtime-frozen attempt
results/optimized_refresh_benchmarks_20261021/attempt003; never resume old paused
studies. Fixed V31 budgets AIME/MATH1728, HumanEval128, LongBench8192.

Integrate frozen V2 binary7300f103369c804d schedule2 and optimized V31 FA4
observer/fused topk/list into existing native adapter lifecycle. V2 refresh
returns online sparse current-QKV output; no dense output or posthoc simulation.
Held/carry paged split FA4 is common. All projection/contiguous packing/list
costs charged. Audit correctly counts sparse V2 refresh PV; total QK/PZ is
additional repeated work and not equated with kept-PV matching. Preserve
Yuhan reuse and Junyu Gaussian32 provenance.

21 kernel/integration regressions pass, then2 audit-phase timing tests. Native
V31 and V2 smoke requests each pass zero capture/scope/finite/budget receipts;
no scored accuracy claim. Frozen attempts001/002 retain smoke evidence and
are superseded after report and native-dense first-canvas receipt additions.
Existing HumanEval K128 tau0.5600716206066294 and .021 for others seed bounded
first-canvas work-only calibration. Freeze nearest absolute GLOBAL PV-kept work
match and report mismatch if tolerances unattainable. Separate audit/clean
paired-native hash qualification before full shards; official scorers and
problem-clustered uncertainty. Dense FULL fix51994/PIECEWISE/all-kept controls.

Durable user service dlm-optimized-refresh-20261021-a003 launches accepted after
one initial automatic-review timeout (unit confirmed absent before retry).
One worker/H100, resumable <=32-cell shards, explicit failed attempts and live
state. No full benchmark completed yet; final marker gates claims. No sampler,
installed runtime or paused historical run changed. Private data stay private.

The first coordinator attempt003 completed four calibration control records,
then its immutable-source audit stopped on regenerated Python bytecode that
was mistakenly included when copying a smoke snapshot. No Python/CUDA method
source changed. New frozen attempt004 excludes generated bytecode and starts
unit dlm-optimized-refresh-20261021-a004; old attempt is retained as failed.


## 2026-10-21 — Pause benchmark; resident sparse execution and K128 short budgets

Explicit user instruction stops dlm-optimized-refresh-20261021-a004. Resource/
process/start-time/SIGTERM receipt and PAUSED_USER_REQUEST retained in attempt004;
85 completed job markers/1698 records remain immutable. Campaign partial and
not a complete benchmark. New short V31 budgets K128 for AIME26/MATH500/HumanEval;
LongBench8192. Old K1728 results supplementary; recalibrate each V2 workload
for40-50% actual held GLOBAL physical tile sparsity and report every phase.

ResidentMask/PagedContext precompute compact indices/splits/page tables/used
lengths and preserve output buffers. Direct paged FA4 skips per-held canvas KV
copy and page metadata allocation; no selection/list construction on held calls.
Six changing-QKV/mask/page-map graph tests pass. Initial50% synthetic resident
25.55/70.32/113.39/196.03us vs dense46.60/168.19/361.51/722.89us. Flex compiled
BlockMask107.01/777.82us at1024/8192 is correct but loses and is retained negative.
These are synthetic observations, not native/end-to-end performance evidence.
SparseD is an attributed execution reference, Yuhan reuse/Junyu Gaussian32
remain distinct. New bounded native001 qualification, no full panel resumed.

Required next: clean synthetic40/50/60 matrix, mask/page/output graph identity,
charged module/host execution, true-online refresh optimization and native
source/finite/budget/extra-forward/LOCAL/capture/audit-clean trajectory gates.
V2 remains online conditional full PV; projection packing fusion is an explicit
variant, not a hidden dense replacement. Relaunch only in a new frozen attempt.


## 2026-10-21 — Charged held execution matrix and preserved refresh negatives

Required synthetic12-cell held/refresh matrices completed under sparse_execution_
20261021/attempt001, source copied/hashed before timing. At50% physical GLOBAL
sparsity, warmed held bucket host medians33.62/101.35/128.41/208.20us at1/4/8/16K,
including Q staging, resident FA4/merge/replay/output delivery/synchronization;
plain dense62.74/175.12/365.39/726.83us, native paged dense141.90/206.70/294.96/
480.37us. Graph captures0 during timing. Compiled Flex prototype is correct but
slower. These are synthetic module observations, not native request speedups.

Exact Gaussian32 projection+packing fusion passes4 operand parity cases but
performance mixed; no promotion. No-spill167-register unique-QK packed producer
passes4 changing-input graph cases but complete8K refresh517.5us loses to the
selected459.3us. Preserve all sources/builds and failed harness/compile attempts.
V2 still true online conditional full PV and still slower than dense refresh.

Native001 one-request resident fixed-split audit/clean hash/calls/canvas intervals
agree exactly (204calls,16canvases);695held calls avoid all KV copies/selectors/
list construction, capture count0. Native002 explicitly primes static graph
buckets during warmup and audits zero captures on the measured request. Its
clean pair is pending. Split arithmetic can change trajectories; native001 vs
native002 request latency is not a matched execution-speed comparison. No full
benchmark relaunch, per-workload sparsity calibration or accuracy claim.


## 2026-10-21 — Static graph-bucket qualification and native paged V31

Native004 one-request K128 audit/clean output hash,49calls,5canvas intervals agree
with0measured CUDA captures and0Triton compilations, after explicit selector
specialization warmup. Native002 timing remains withheld due nearby JIT warnings.
These are qualification requests without matched end-to-end controls/accuracy.

Direct native paged V31 observation/recovery passes2 cases including extreme
logits and changed cache/page mapping. At8K302.55us vs native dense288.34us,
16K492.32us vs470.09us; named dispatch uses contiguous through4096, paged above.
Native005 is a previously examined long-output diagnostic to exercise this
branch, not a full panel or held-out calibration.

Packed unique router with retained-only probability queue compiles167regs/no
spills and passes4 changing-input graph cases. Rejected tiles advance routing
without output-consumer handshakes. Complete8K480.41us still loses to selected
four-group459.29us. Retain negative refresh005; two-group version484.74us also
negative. Synthetic performance gates do not establish a faster online refresh.


## 2026-10-21 — Native target-sparsity diagnostic and final qualification snapshot

V2 native006 tau0.05 achieves45.338% actual held GLOBAL tile sparsity on examined
HumanEval145; audit/clean hash,563sampler calls/32canvas intervals agree exactly.
0timed JIT events/Triton compilations/captures/extra forwards/LOCAL skips. Whole
GLOBAL sparsity45.087%, first canvas near dense. This is one diagnostic, not
workload calibration or benchmark accuracy. V31 long native005 audit/clean
529sampler calls/32canvas intervals agree with170direct paged refreshes.

40combined final regressions pass. Required12-cell synthetic resident/refresh
matrices complete, prototype/negative variants retained with separate frozen
sources. 3D-TMA variant passes13cases and complete8K449.02us vs~458us prior,
still slower than dense. Hoisted descriptors no material gain. Prefix projection
reuse within a canvas passes7bitwise cases, modest gain, not native-promoted.

Hardware profiling denied ERR_NVGPUCTRPERM; separate stage-clock12-cell probe
preserves output/mask bitwise, explicitly instrumented rather than clean timing.
QK/softmax/staging dominate measured CTA intervals; no hardware stall claim.
Do not relaunch full evaluation: V2 refresh parity, per-workload held sparsity
calibration and matched native controls still outstanding. Mutable old controller
now guards pause and rejects legacy full-run/preparation (2tests); frozen old
attempt untouched. Current source hashes under qualified_source/attempt001 and
all sanitized summaries under sparse_execution_20261021/attempt001.

## 2026-10-21: Further unique-routing refresh screen remains negative

Continue the user's ordered refresh-first task in
results/sparse_execution_20261021/attempt002. Q32 warp-MMA full-D output computes
routing once but spills and costs2968.58us at8K/~50%, vs357.76us dense. Q64
two-channel single-WG register-P removes half the duplicated routing and has no
spills, but459.98us complete loses to the previous449us best. Shared P677.46us
spills; hardware cluster barrier521.33us loses to release/acquire vote. Previously
qualified retained-only dual queue measures602.65us. Preserve every negative and
failed harness; no variant promoted and no benchmark gate relaxed. All are
synthetic current-Q/K/V single-traversal conditional-PV kernels. Generic unpacked
wrapper adds matching-held-map validation. The full per-workload calibration and
new native campaign remain outstanding. See attempt002/README.md and receipts.

Further attempt002 results: aligned-shape bounds specialization improves complete
8K refresh449.02->433.11us, but still loses to matched plain dense359.32us and
native paged288.48us. It retains generic masking for irregular shapes. All40
combined H100regressions pass, no skips/failures,41.54s. Unroll1 gives507.47us;
hoisted-descriptor combination434.10us, no further gain. Unique cooperative GPU
queue539.65us and current-PV/next-QK overlap515.70us are preserved negatives;
each passes80changed-input graph replays with exact masks and relL2<0.003.
Q32 probe001's capture count was inferred, not measured: its timing is weaker
diagnostic evidence. The script now uses an actual counter; no old receipt was
modified or retro-certified. The main refresh objective and later full panel
remain incomplete; no accuracy or native speedup claim.

## 2026-10-21 — Complete broader true-online refresh synthetic screen

Preserve new source-frozen attempts002-018 under sparse_execution_20261021.
Best interior/tail specialization200354cb completes all12required synthetic
cells at40/50/60% actual physical GLOBAL sparsity. At50%, complete refresh
80.58/222.77/427.90/860.36us at1K/4K/8K/16K; matched plain dense
46.95/180.42/363.58/717.77us and native paged137.16/199.65/287.47/471.89us.
8K improves prior449.02us by4.70% but remains17.69% above plain dense and
48.85% above native paged. Actual timed capture count0. All43combined H100
operator/projection/resident/V31/graph tests pass with0failure/skip in41.45s.

Unique Q128 queue598.60us, interleaved QK764.50/770.74us, corrected wide-QK
679.03us, unique split-key583.81us, QK/PZ lookahead614.42us, earlier K/V479.68us,
compact descriptors435.32/433.80us, Q32 occupancy1270.90us and register-Q queue
636.42us are retained negatives. Static descriptor addresses fail launch
qualification; that design is abandoned. No hidden dense PV or two-pass
substitution. Numerical qualification does not establish native task benefit.

The pending150-file local source/documentation commit is now5a5611e2. Native
combined qualification, matched controls, workload calibration and full scored
panel remain incomplete because the first performance gate still fails. Keep
old optimized_refresh_benchmarks attempt004 paused/partial; do not claim a
completed benchmark or relax sampler/budgets. See attempt018/summary.md.

## 2026-10-22 — Evaluate the current best refresh with qualified native controls

The user explicitly supersedes further refresh tuning and requests the final
multi-benchmark comparison with current best kernels. Preserve the negative
synthetic dense-parity result. Source checkpoint `06ee4806` freezes the native
best-library fused projection/pack integration, resident held consumer and
resident all-kept control. Full model content hashes verify; online/resident
22 tests, fused native/projection 9 tests, and resident all-kept 2 tests pass.

New frozen attempt `optimized_refresh_benchmarks_20261022/attempt004` runs via
durable service `dlm-optimized-refresh-20261022-a004.service`, initially at native
qualification. Universal request JIT counters cover dense as well as sparse
arms, per-input untimed native warmup precedes every measured request, and
clean rows must have zero capture/JIT events. Audit-only tile fields are
required on audit rows; their absence on clean rows cannot be treated as zero.
All-kept uses the selected resident consumer with full physical support.

Calibration must aggregate actual held GLOBAL eligible/kept tiles over full
native requests from four workload length strata; a 256-token first-canvas cap
can be dominated by mandatory canvas support on short prompts. Reject an
out-of-range 40-50% held calibration before full panel launch. The mandatory
full coverage includes the transductive calibration items and discloses them.
K128 applies to AIME/MATH/HumanEval; LongBench remains K8192. No official final
accuracy or latency claim until all completion markers and scores exist.

Preserve failed preparation attempt001, validator diagnostic attempt002, and
27-row partial qualification attempt003. The latter was terminated after the
active worker exited, with process identity and signals recorded; its frozen
sources and saved rows remain unchanged. The old 2026-10-21 attempt004 remains
paused/partial. No old campaign is resumed or promoted by this decision.

## 2026-10-22 — Correct resident all-kept receipt and freeze attempt005

Native attempt004 stops before calibration/panels on missing top-level
GLOBAL eligible/kept fields from the resident all-kept control. Its audited dense
phase has 133120/133120 tiles and zero LOCAL skips. Source 6ecdfbf6 adds exact totals
from the accumulated actual geometry, retaining numerical output and sampler;
2 GPU tests verify parity and exact counts at 641/8192 keys. Preserve attempt004
failed/partial, no retroactive completion.

New attempt005 freezes source 6ecdfbf6, full model hashes, protocol/kernel/input
identity, and shards ofat most 128requests to amortize engine startup. Its
resident all-kept native audit/clean pair passes with full support, zero K/V copies,
zero LOCAL skips,zero actual captures,zero universal JIT events and identical native
trajectory. Durable unit dlm-optimized-refresh-20261022-a005 runs remaining native
qualification, full-request held calibration, full audit/clean evaluation and
official scoring in that order. No completed panel or official score yet. All
prior attempts and the paused 2026-10-21 campaign remain preserved.

## 2026-10-22 — Bound V31 selector storage before benchmark relaunch

Preserve attempt005 as partial: 32 native qualification passes completed, then
LongBench V31 untimed warm-up failed with CUDA out of memory in `_select`.
The generated largest warm-up prefix exceeded the configured model limit and
the single-CTA row/tile tensor produced a 10 MB PTX selector. No calibration or
full benchmark panel ran. Do not report those qualification cells as accuracy.

Source f7318fa3 bounds repair scans to 128 tiles, splits selector statistics
above 512 tiles into row normalizers and tile scores before the unchanged
fixed-budget compact top-k, and caps warm-up at the actual context limit.
Short-context selection and the chosen attention kernels remain unchanged.
Nine H100 tests pass, including long-context exact budgets/ties/invalid rows;
all selector allocations and launches remain charged to refresh. Attempt006
receives a new frozen protocol/source identity and repeats native gates before
calibration or full coverage. User requests 15-minute process-status checks.

## 2026-10-22 — Apply user-fixed HumanEval and LongBench thresholds

User explicitly removes HumanEval and LongBench-v2 recalibration. Reuse the
saved K128 HumanEval threshold0.5600716206066294 from the full164 work-matched
receipt; use LongBench threshold0.021. These replace the earlier per-workload
40-50% calibration requirement for those two workloads. Only AIME26/MATH500
calibrate. Report actual held/refresh/carried/first-canvas/whole work and any
remaining V2/V31 mismatch without treating old matching as new qualification.

Preserve attempt006 partial after29 completed qualification passes. Let its
worker finish, then stop only the verified held coordinator; no calibration or
panel ran. Sourcec31224ae/new attempt007 explicitly permits completed paired
qualification evidence only if all execution sources, model/runtime/kernel,
inputs/budgets and threshold match; only orchestration/report source changes
are excluded from that equality. Changed-threshold V2 pairs run anew, and all
full panel generations are fresh. Three policy tests pass. Fixed-threshold
receipts contain no invented held sparsity or new calibration claim.

## 2026-10-22 — Extend MATH500 range after native gate pass

Attempt007 passes all20 workload/arm audit-clean qualification pairs, using
explicit source-identical evidence where available. AIME26 calibration selects
0.059460355750136064 with45.6943% actual held GLOBAL sparsity. MATH500's
largest candidate0.8 gives32.0452%, so no full benchmark panel launches.
Preserve attempt007 failed/partial and its complete calibration evidence.

Attempt008/source3db64cea fingerprints completed qualification evidence and
computation-only calibration records, reuses the AIME result, and extends only
MATH500 candidates to1.6/3.2/6.4/12.8 with the same eight refinements.
HumanEval0.5600716206066294/LongBench0.021 remain fixed by user. Five CPU
policy tests pass. Runtime/kernel/model/input identities and native sampler
are unchanged; all full panel generation remains fresh.

## 2026-10-22 — Accept demonstrated MATH500 ceiling as explicit target miss

Expanded MATH thresholds3.2/6.4/12.8 yield identical native output hashes,
denoising calls and held work. Actual kept70880 equals the mandatory tile
floor70880 out of115840 eligible:38.8122% sparsity. A per-canvas geometric
proof uses current canvas/boundary tiles plus mandatory first prefix tile and
is frozen with all measurement fingerprints. This proves saturation on those
four calibration trajectories, not the full500 population.

Source7bd6128e/attempt009 explicitly accepts this target miss at3.2 to complete
the user's requested panel. It does not relabel38.81% as40% or alter the kernel,
mask method, native canvas/stopping/temperature, prompts or output budgets.
AIME's passing0.059460355750136064 and user-fixed HumanEval/LongBench
thresholds remain. Reuse source-matched native gates and computation evidence;
no further calibration forwards. Seven CPU policy tests pass. Full panel
accuracy, work, clean timings and paired uncertainty still require fresh full
coverage and official scoring; all previous attempts remain partial.

## 2026-10-22 — Finish current MATH; replace LongBench with requested follow-up

User requests finishing the current MATH500 panel, stopping before LongBench,
retaining achieved HumanEval/AIME results, recalibrating lower MATH V2 sparsity
to40-50%, testing AIME V31K256, and adding balanced104 RULER128K withK8192
and approximately matched V2 work. A boundary controller arms the unchanged
pause mechanism after final MATH clean worker launch; no native worker is killed.

Full MATH records at3.2 show about77% GLOBAL sparsity, unlike the four short
calibration trajectories. The new protocol selects32 examined full-request work
strata, targets both held/whole40-50%, and keeps original results immutable.
New AIMEK256 is paired with preserved source-identical controls/K128. RULER
uses its same104 official prefilled/thinking-off prompts and seed42; fractional
score and13-task uncertainty remain intact. All new arms/thresholds receive
native audit/clean gates; all copies/launches and clean timing remain charged.
Ten CPU tests pass. Freeze new follow-up attempts only after MATH finishes and
the explicit stop is verified. No LongBench generation is authorized now.

## 2026-10-22 — Requested MATH boundary reached with no LongBench launch

Attempt009 completes and officially scores HumanEval164, AIME30x3 and
MATH500 across five arms. MATH all40 workers complete and trajectory pairs
agree. User pause is honored before next LongBench run directory/launch; the
coordinator exits on the expected explicit-pause exception. Preserve attempt009
as partial with completed-suite markers, not a completed four-benchmark panel.

MATH V2tau3.2:392/500,77.5753% whole GLOBAL sparsity,79.4255% held; V31K128
433/500, dense FULL/PIECEWISE449/500, all-kept450/500. The lower-threshold
follow-up uses32 full-request work strata and retains achieved controls/results.
AIMEK256 and RULER128K104 follow source80e7fa0c. No raw data committed.

## 2026-10-23 — Complete requested optimized-kernel follow-up and verify results

Completed source/protocol-bound consolidated report at
results/optimized_refresh_results_20261023/attempt003. Original short panels
remain preserved; MATH V2tau0.2 rerun and AIME V31K256 are complete, RULER104
all five arms officially scored. LongBench never started after requested stop.
Original campaign remains partial; do not create a whole-campaign completion.

MATH V2 improves392->438/500 after lowering3.2->0.2; final actual whole/held
GLOBAL sparsity47.6038/49.3625%. AIMEK25642/90 versusK12840/90, uncertainty
includes zero. V2 AIME50/90 vs V31K12840/K25642 and dense51; HumanEvalV2
159/164 vs V31160/dense161; MATHV2438 vs V31433/dense449. RULER official
fractional scoresV286.346,V3186.779,dense86.683,all-kept87.115.

No dense speedup: V2 mean clean request ratios1.233/1.239/1.180/1.004 on
HumanEval/AIME/MATH/RULER. RULER held sparsity94.64/93.49% for V2/V31,
but whole sparsity79.94/47.33%, so this is approximate held matching only.
Preserve failed first-canvas work matching and disclose absolute phase mismatch.
Paired CIs use problem clusters (AIME three seeds) and13 RULER task clusters.
All measured audit/clean output hashes/N agree; explicit zero clean captures/JIT
and LOCAL skips, selector-budget/extra-forward receipts pass. Runtime identity
matches, final service exits0 and H100 is idle. Report verifier sourcee2d2755c;
full source/runtime/model/protocol and field coverage are published sanitized.
