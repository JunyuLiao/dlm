# HANDOFF — current frontier (2026-10-01, local UTC−5)

Read `AGENTS.md` first. Stable context: `docs/RESEARCH_CONTEXT.md`. History and negatives: `docs/DECISIONS.md`.
Verified numbers: `docs/RESULTS_LEDGER.md`. The previous handoff (v27c, 2026-09-29) is archived at
`docs/handoff_archive/HANDOFF_v27c_20260929.md`.

- **Branch:** The current working branch is `research/humaneval-v27-20261001`. A GPT session created it from the v27
  checkpoint `research/m3-output-numerics-20260927` (`add23afa9`) on 2026-10-01 23:59; all work since then
  (HumanEval, step statistics, regrouping, docs) is on it. The old branch stays at `add23afa9` as the v27
  checkpoint; fast-forwarding it is the user's call.
- **Method code:** `9d8ae5e` (named keep k5) on top of `efea33024` (k12), `c36b1f933` (V-term ablation variants)
  and `56de98fe2` (`carry_first`). Docs and decks are newer commits; trust `git log` over this line.
- **Docs:** committed and pushed. Keep them current with every change; this is a user instruction.
- **Intake review (2026-10-01, 22:45 UTC−5):** audited base `6b13fe178`, confirmed as the live remote head.
  See `docs/INTAKE_AUDIT_20261001.md`. Follow-up panels may select the stronger relevant arms instead of repeating
  plain M1/M2/M3 each time (user update). Same-host paired cells and matched optimized references remain mandatory.
  Classmates' A/B branches are pending; their exact refs are unknown, so no merge has been attempted.
- **Group-context review (2026-10-01, 23:10 UTC−5):** user-authorized read-only search of the specified four-person
  research group completed on base `073d87ece`. See `docs/GROUP_CONTEXT_20261001.md`; peer claims remain separate
  from frozen v27 evidence. No HumanEval result or exact A/B refs was found in that search. GPU seconds = 0.
- **Two local checkouts.**
  - `E:/dlm/m3_output_numerics_20260927` is the working checkout used for all v27 work.
  - `E:/dlm/dlm_state_adaptive_router_20260828` is the same branch at an older commit (`3d48ebcdd`); pull before
    using it.

## Current conclusion

- **Best configuration now: M3 R6 DP −ln2 + `carry_first`** (E5, same 288 cells as E4).
  - 64K: request W **0.853 [0.802, 0.895]**, generation S 0.775.
  - 32K: W **0.907 [0.858, 0.952]**, S 0.878.
  - Accuracy 76 vs 75 and 92 vs 86. Versus M3 without carry: 0.971 / 0.963.
  - The `stable1` gate is rejected.
  - 96K (E6 + E6b pooled, 11 fitting items × 6 seeds = 66 cells): W **0.822 [0.703, 0.944]**, S 0.727, per-step
    −25%, accuracy 30 vs 25 (`docs/RESULTS_LEDGER.md` L1c2).
- **Long context: significant end-to-end gain, no accuracy loss.** E4 has 6 never-used seeds, 144 cells per arm per
  bin, on piecewise_v5, against FA4 all-kept dense.
  - **64K:** M3 R6 DP −ln2 request W **0.879 [0.820, 0.926]**; generation-only S 0.807.
  - **32K:** W **0.940 [0.897, 0.980]**; S 0.921.
  - Accuracy is equal or higher (64K 77 vs 75, 32K 86 vs 86). B is similar (0.876 / 0.941).
  - Prefill is not optimized by any arm (P ≈ 1.00).
- **Step count: no significant change, best estimate +1–2% (corrected 2026-10-01 evening).** Per-seed step ratios
  (M3 / dense, 32K, same 24 items and substrate) range 0.85–1.24. The old 3-seed set (101/202/303) sits high and the
  6 new seeds sit low; 13 of 84 random 3/6 splits give a gap at least as large, so both are draws from one
  distribution. Pooled 9 seeds: 32K N 1.024 [0.973, 1.076], 64K 1.015 [0.948, 1.085] (64K pools v4 + v5). E4's
  request W may therefore be about 2% optimistic; per-step costs (S/N) are stable across all batches.
- **Held-out items only** (never used to choose −ln2): 64K M3 0.85 [0.73, 0.96] significant; 32K M3 0.96 n.s.,
  obs2 0.95 [0.91, 0.98].
- **Fan plain M1/M2c/M3 are slower than dense** at every length (1.06–1.34× in E4): the selector and observation cost
  exceeds the skipped attention.
- **AIME has no speed room** (E7, 180 cells per arm, `docs/RESULTS_LEDGER.md` L1d). The best low-overhead variant,
  M3 + `carry_first` + 2K gate, is W 1.005 [0.967, 1.042] with accuracy 99 vs 99. Fan plain M1/M2c/M3 are 7–12%
  slower. No accuracy difference is significant. AIME is an accuracy check only.
- **Novelty is weak.** B ≈ SparseD (no significant speed difference at 64K). **E8 (AIME, fixed 70% sparsity,
  180 cells per arm) finds no evidence that looking at V helps:** no V term beats attention mass alone (all p ≥ 0.30
  vs rank 32), and projected V at rank 32/16/8 scores lowest (86–87 vs dense 99). **E9 (64K, M3 + c0, fixed 88%
  sparsity) and E11 (95%, preview) agree:** mass-only ranking is as accurate (54 vs 51) and as fast (W 0.845) as any V term. The V term is
  not a contribution in these settings and can be dropped (L1e, L1f).
  This conclusion is limited to the tested AIME/LongBench settings. The user relayed a classmate's hypothesis that
  RULER needs V dimensional information; HumanEval is unknown. Actual attention already uses full V. The question
  concerns selector inputs and should be tested on RULER, not generalized from the LongBench negatives.
- **M2 vs M3 under the same optimizations: tie** (E10, L1g): M2c / M3 32K 0.987 [0.914, 1.063], 64K 0.983
  [0.940, 1.024]; accuracy n.s.
- **Skip ratio is not set; it emerges from one fixed threshold.** All GLOBAL layers, heads and lengths share the log
  threshold −3.874 (frozen base −3.180, shifted −ln2). Skipped tiles in sparse calls (diagnostic fidelity_v6, 2
  requests each): AIME about 4%, 32K about 79%, 64K about 88%. Fixed ratios are used only in E8/E9/E11.
- **Within a canvas** (code facts, `docs/RESEARCH_CONTEXT.md` §3): call 1 observes and builds the prefix risk table;
  call 2 decides (map for calls 2–7), call 8 re-decides, and so on. A re-decision reuses the prefix risk table but
  applies the current per-position query sensitivity T = clamp(1 + 3·EMA(argmax flips), 1, 4), which is active in every
  run (corrected 2026-10-02; an earlier note wrongly said T = 1). Measured: re-decisions change 44–71% of the first
  decision's kept tiles (Jaccard 0.59–0.70), so the earlier "≈ select once per canvas" inference was wrong.
- **Per-step time on v5** (one real dense call each, `docs/RESULTS_LEDGER.md` time breakdown): GLOBAL attention 2% /
  19% / 32% of a step at AIME / 32K / 64K; LOCAL 2–3%; MoE experts 27–39%; sampler 10–16%. Single decoder forward
  in one request, dense → best variant: AIME 1.00, 32K 0.85, 64K 0.72.
- **The ceiling at batch 1 is low.** GLOBAL attention is about 21% of a 64K request and 16% at 32K; a step is
  dominated by reading about 46 GB of MoE weights. With `carry_first` the 64K gain (about 15%) is roughly 70% of that
  ceiling.
- **Model.** `google/diffusiongemma-26B-A4B-it` belongs to the Gemma 4 family (Gemma4Processor, gemma4_vision in its
  config); its text model is `diffusion_gemma_text`. Say "DiffusionGemma-26B-A4B", not "DiffusionGemma4".
- **Batching does not raise the attention share at 64K** (B=1→4: 26/22/25% of a forward; keep-0.12 saving about
  20%). See `batch_scaling/README.md`.

## Done (this session, 2026-09-30 to 10-01)

- Substrates piecewise_v4 (static LOCAL shape; fixes the AIME recompile fallback) and v5 (compiled encoder-append
  tail).
- Panels, all scored and in `docs/RESULTS_LEDGER.md`:
  - traj_t1, E1 (cross-canvas carry: rejected), E2 (AIME carry/gates), E3 (32K single-change variants): L11–L14;
  - E4 large-seed confirmation (L1), E5 `carry_first` (L1b), E6 + E6b 96K pooled (L1c, L1c2), E7 AIME (L1d);
  - E8 / E9 / E11 V-term ablations (L1e, L1f, L1h), E10 M2 vs M3 under the same optimizations (L1g);
  - step-count check over 9 seeds (correction: +1–2%, n.s.) and the v5 per-step time breakdown.
- New named variants: `observe_step`, `protect_output` (7786b0ef9); `stable1` (adc2056d7); `carry_first` (56de98fe2);
  `proj_rank`, `risk_value='mass'` (c36b1f933); named keeps k12 (efea33024) and k5 (9d8ae5e), each with unit tests
  (v27 subset passes on dlm2; the stale min_route_keys=4096 test was fixed).
- Agent docs: `AGENTS.md`, `CLAUDE.md`, `docs/RESEARCH_CONTEXT.md`, `docs/DECISIONS.md`, `docs/RESULTS_LEDGER.md`,
  this file, and the `current` block of `STATE.json`.
- Group-meeting deck (2026-10-01): `scripts/v27_build_deck_1001_compact.py` reads the pushed `summary.csv` files and
  the time-breakdown JSONL and writes both `ppt_sample/dlm_sparse_attention_20261001_compact_v6.pptx` (latest) and
  the markdown source `weekly_slides_20261001_compact.md`. `--md-only` refreshes only the markdown (use it when the
  pptx is open in PowerPoint). The older 9-page deck and its source (`weekly_slides_20261001.md`) are superseded.
- Read-only review of group work: Junyu's position protections and value-aware family (branch `ljy/value_aware`);
  his "vector_mean" result files are not committed anywhere we can read.
- Literature check of step and length inflation (SparseD, PulseCol, Focus-dLLM, Lil, LessIsMore, JoT, Prophet).

## Running (as of 2026-10-02 01:15 local)

- **Finished and scored:** E5 (L1b), E6 (L1c), E6b and the pooled 96K panel (L1c2), E7 (L1d), E8 (L1e), E9 (L1f), E10 (L1g), E11 (L1h), E12 (L1i), E13 (L1j),
  the batch diagnostic (`batch_scaling/`).
- **E12 HumanEval: done** (`docs/RESULTS_LEDGER.md` L1i): no accuracy loss in any arm, no speed gain (main W 1.023),
  no V-term difference. Scored on mpk with the unprivileged sandbox.
- **E13 q64: done** (`docs/RESULTS_LEDGER.md` L1j): per-step −0.7% at 64K (significant, all hosts), end to end n.s.,
  accuracy unchanged. Optional named variant `q_block=64`; not a contribution.
- **Nothing running.** All GPUs idle.
- **Regrouping diagnostic (chw idea): done** (`regroup_diag_1002/README.md`): ~4% fewer kept tiles within 128-row tiles,
  not integrated; finer query tiles would cut kept work by 22–39% but need a new kernel.

## Blockers

- None.

## Immediate next steps

1. Measure how much the held map changes between re-decisions (calls 2 vs 8 vs 14) at 32K/64K. This tests the
   inference that M3 R6 DP is effectively "select once per canvas" (about 40 minutes, one host).
2. Test the user's dataset-dependent V hypothesis on RULER: mass-only / rank-32 / full-dimensional selector V at
   matched tile budgets. Start with the current historical-QK GLOBAL-only setting; then separately test the peer's
   fresh-QK / GLOBAL+LOCAL setting as a collaboration candidate. Existing 32K/64K pools have only 13 items each;
   expand the official task inventory before a task-level claim. A full-dimensional selector is not implemented yet.
3. HumanEval quality pilot and full LongBench coverage are proposed follow-ups, not launched work. HumanEval needs
   a frozen code-extraction and executable scorer contract. LongBench needs a new shared chunked-prefill substrate,
   followed by a separately named truncation/retrieval protocol to cover all 503 items. New directions use new branches.
   E10/E11 extensions are optional if needed to resolve a specific question; do not enlarge tied panels by default.
4. Larger-gain directions:
   - make the encoder canvas append's GLOBAL attention sparse with the canvas's final map (about 1.4% at 64K);
   - chunked prefill so that 96K items above 95K tokens and the 128K bin run on one H100;
   - orthogonal step reduction (Prophet/JoT-style early exit) is not our contribution.
   - Batched serving did not raise the attention share at 64K (B = 1–4); see `batch_scaling/README.md`.
5. Note for E6b: its first bind call ran with `HOSTS=dllm` only, so the binding was merged from the three host
   fragments; the unused one-host binding is kept beside it.

## Intake audit (2026-10-01, 22:45 UTC−5)

- Three GPUs checked idle by SSH; this review launched no GPU worker, GPU seconds = 0.
- The independent FA4 timing join previously allowed later duplicate records to overwrite first outputs and did
  not recheck scorer/run identity. Added rejection of duplicates, omitted scored executions, mixed host/GPU,
  substrate/protocol/model/source, and unqualified scores. Unknown graph counters are excluded from Wc.
- Audited E4/E5/E6/E6b/E7/E8/E9/E10/E11: 8,826 first runs, same-host cells throughout, no duplicate first outputs,
  no point-estimate or correct-count changes. E7/E8 each retain the known 3 timed graph captures (use Wc).
- Final regression: 17 CPU tests pass on the registered dlm2 interpreter; all nine panels pass the final guards.
- Full LongBench-v2 length inventory: 503 inputs, median 107,706, max 5,174,028 rendered tokens. 224 inputs ≤95,074;
  400 fit the configured 262,144-token context with 8,192 output tokens reserved. Chunked prefill can address
  memory, not the 101 inputs whose prompt alone exceeds the context limit. No full-panel run has started.
- HumanEval: 164 Python tasks, 984 generations/arm with six seeds. Not yet a v27 dataset/scorer; proposed as a
  coding quality check. Read-only review of the shared `megakernel` deck was limited to retrievable slide text.
- Slack context: the user subsequently authorized the specified four-person group conversation. Read-only research
  search completed; no one-to-one DM was read. The peer RULER note supports V direction but not a full-rank necessity.
  Aggressive-sparsity step inflation is a collaboration clue, not a new v27 result. A/B refs remain pending.

## Operational notes (still relevant)

- **Hosts** (H100 80GB, user-authorized, write only under `dyh`):
  - dllm `149.165.159.64`;
  - mpk `149.165.151.254` (writes go to `/media/volume/dllm-1/dyh`);
  - dlm2 `149.165.168.28`.

  Interpreters and env are in `E:/dlm/v20_private/hosts.json` plus `E:/dlm/v27_lbfa4_env.json` (torch 2.12 FA4
  overlay; caches pinned inside `dyh`).
- **Panel pipeline.** Coordinator scripts in `E:/dlm`, not in the repo:
  1. Commit the spec in `results/…/specs/`.
  2. Freeze: `python -m scripts.v21_freeze_panel freeze --mode v27_panel --spec … --out-dir E:/dlm/v23_private/<name>_frozen`.
  3. Deploy: `python E:/dlm/v21_deploy_qualify.py --sha <HEAD> --tag v27_<x>_<sha7> --host <alias>`.
  4. Bind: `python E:/dlm/v23_transport.py --action bind --run-dir … --frozen-dir …`.
  5. Launch: `TAG=… RUN=… FROZEN=… STAGES=… bash E:/dlm/v27_lbfa4_host.sh <alias> <ip>` (waits for an idle GPU).
  6. Score: `python E:/dlm/v27_score_lb.py --tag … --run-dir … --label …`.
  7. Summarize: `python -m scripts.v27_fa4_panel_summary …`.

  Scored outputs land in `E:/dlm/v27_private/lb_scoring/<label>/`.
- A killed worker leaves no terminal receipt. Relaunch in a **new** run dir and merge at scoring.
- On Windows, stopping a task can orphan the bash chain. Kill leftover `bash` processes with `v27_` in the command line.
- Diagnostics that must match a frozen config's source hashes run with cwd = that deploy dir.
- Time breakdown of one real decoder call: `scripts/v27_time_breakdown.py --run-dir <bound run dir> --host <ip>
  --gpu-uuid <uuid> --stage <stage> --dataset <dataset> --index 0 --call <k> --arm <dense> --arm <method>`, run in
  that panel's deploy dir with the host env; outputs in `results/…/time_breakdown_v5/`.
- Unit tests on a host: in a deploy dir, `PYTHONPATH=<fa4 overlay>:.:src <python> -m pytest -q -p no:cacheprovider
  tests/test_v27_*.py` (CPU-only tests can run while a GPU job is active).
