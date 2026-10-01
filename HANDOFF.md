# HANDOFF — current frontier (2026-10-01, local UTC−5)

Read `AGENTS.md` first. Stable context: `docs/RESEARCH_CONTEXT.md`. History and negatives: `docs/DECISIONS.md`.
Verified numbers: `docs/RESULTS_LEDGER.md`. The previous handoff (v27c, 2026-09-29) is archived at
`docs/handoff_archive/HANDOFF_v27c_20260929.md`.

- **Branch:** `research/m3-output-numerics-20260927` (pushed to `origin`).
- **Method code:** `c36b1f933` (V-term ablation variants), on top of `56de98fe2` (`carry_first`).
- **Docs:** committed and pushed. Keep them current with every change; this is a user instruction.
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
- **Long context: significant end-to-end gain, no accuracy loss.** E4 has 6 never-used seeds, 144 cells per arm per
  bin, on piecewise_v5, against FA4 all-kept dense.
  - **64K:** M3 R6 DP −ln2 request W **0.879 [0.820, 0.926]**; generation-only S 0.807.
  - **32K:** W **0.940 [0.897, 0.980]**; S 0.921.
  - Accuracy is equal or higher (64K 77 vs 75, 32K 86 vs 86). B is similar (0.876 / 0.941).
  - Prefill is not optimized by any arm (P ≈ 1.00).
- **The earlier "steps inflate / request gain not robust" reading (3-seed held-out, E3) was trajectory noise.** With 6
  seeds, steps per canvas are 0.97–1.03 and output length 0.95–1.01.
- **Held-out items only** (never used to choose −ln2): 64K M3 0.85 [0.73, 0.96] significant; 32K M3 0.96 n.s.,
  obs2 0.95 [0.91, 0.98].
- **Fan plain M1/M2c/M3 are slower than dense** at every length (1.06–1.34× in E4): the selector and observation cost
  exceeds the skipped attention.
- **AIME has no speed room** (E7, 180 cells per arm, `docs/RESULTS_LEDGER.md` L1d). The best low-overhead variant,
  M3 + `carry_first` + 2K gate, is W 1.005 [0.967, 1.042] with accuracy 99 vs 99. Fan plain M1/M2c/M3 are 7–12%
  slower. No accuracy difference is significant. AIME is an accuracy check only.
- **Novelty is weak.** B ≈ SparseD (no significant speed difference at 64K). There is no evidence that V-aware
  selection beats score-only selection (it is worse at high AIME sparsity).
- **The ceiling at batch 1 is low.** GLOBAL attention is about 21% of a 64K request and 16% at 32K; a step is
  dominated by reading about 46 GB of MoE weights. The current gain is about 60% of that ceiling.
- **Batching does not raise the attention share at 64K** (B=1→4: 26/22/25% of a forward; keep-0.12 saving about
  20%). See `batch_scaling/README.md`.

## Done (this session, 2026-09-30 to 10-01)

- Substrates piecewise_v4 (static LOCAL shape; fixes the AIME recompile fallback) and v5 (compiled encoder-append
  tail).
- Panels traj_t1, E1 (cross-canvas carry: rejected), E2 (AIME carry/gates: no room), E3 (32K single-change
  variants), E4 (large-seed confirmation). See `docs/RESULTS_LEDGER.md` L1, L11–L14.
- New named variants:
  - `observe_step` (7786b0ef9);
  - `protect_output` (7786b0ef9);
  - `stable1` dense-confirmation gate (adc2056d7);
  - `carry_first` (56de98fe2).
- Unit tests pass on dlm2 (75 in the v27 subset).
- Literature check of step and length inflation (SparseD, PulseCol, Focus-dLLM, Lil, LessIsMore, JoT, Prophet). Group
  deck and branches were read; the query-sensitivity direction is a collaboration candidate.
- Progress doc updates 8–12 (`results/m1_m2_m3_frontier_v27_20260929/progress_20260930.md`). Updates 6 and 12 predate
  E4 and are superseded by it for the request-level verdict.

## Running (as of 16:15 local)

- **Finished and scored:** E5 (L1b), E6 (L1c), E7 (L1d), the batch diagnostic (`batch_scaling/`).
- **Running or queued** (background jobs started when each host's previous job closed).
  - **E6b, 96K extension** (`specs/v27_lb96k_extend_e6b.json`, protocol `v27_lb96k_extend_e6b_d1b1c0c74a1dc69d`).
    - The 5 other fitting items of the 96K pool (pool positions 13–24; prompts ≤ 95,074 tokens) × seeds 404–909 =
      30 cells per arm, same 7 arms as E6. 210 runs on dllm, mpk and dlm2. Pooled with E6: 11 items, 66 cells.
    - Deploy `v27_e6b_b805351`, run dir `v27_lb96k_e6b_001`, scoring label `lb96k_e6b`.
    - Runs after E8 on each host (dlm2 started first). The binding was merged from the three host fragments
      because the first bind call ran with `HOSTS=dllm` only; the unused one-host binding is kept beside it.
    - Item facts: LongBench-v2 has 32 items in the 84–104K bin, of which 14 fit one H100. The pool kept the first 24
      by sha256 (11 fitting); the other 3 fitting items are outside the frozen pool.
  - **E8, V-term ablation on AIME** (`specs/v27_aime_vterm_e8.json`, protocol `v27_aime_vterm_e8_bd293196ff730851`).
    - Runs after E7 on each host; same 180 cells and hosts as E7.
    - Fixed 70% target sparsity (risk top-k keeps 30%) on the M3 R6 DP selector. Only the V term changes:
      - projected V at rank 32, 16, 8 and 4 (`proj_rank`);
      - M2 tile-mean V (`pooled_compact`);
      - attention mass only (`risk_value='mass'`);
      - SparseD port at 70% as reference.
    - 1,440 runs. Deploy `v27_e8_021431b`, run dir `v27_aime_e8_001`, scoring label `aime_e8`.
    - Question: does looking at V help selection quality at matched sparsity?

## Blockers

- None technical. The meeting is 21:00 local; results are wanted by 20:00.

## Immediate next steps

1. When E8 is scored (label `aime_e8`): check `proj_rank` and `risk_value` in each arm's `effective_method`, and that
   the realized kept fraction is about 30% in every arm. Then compare accuracy at matched sparsity across projected
   V rank 32/16/8/4, M2 tile-mean V, mass-only and SparseD. Copy the summary into `results/…/aime_vterm_panel_e8/`.
2. When E6b is scored (label `lb96k_e6b`): report it alone and pooled with E6 (11 items, 66 cells), W and S.
3. Larger-gain directions, ranked (see `docs/RESEARCH_CONTEXT.md` §8):
   - (a) Make the encoder canvas append's GLOBAL attention sparse with the canvas's final map (about 1.4% at 64K).
   - (b) Chunked prefill so that 96K items above 95K tokens and the 128K bin can run on one H100.
   - (c) Orthogonal step reduction (Prophet/JoT-style early exit). Not our contribution.
   - Batched serving did not raise the attention share at 64K (B = 1–4); see `batch_scaling/README.md`.

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
