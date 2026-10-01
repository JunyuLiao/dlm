# HANDOFF — current frontier (2026-10-01, local UTC−5)

Read `AGENTS.md` first. Stable context: `docs/RESEARCH_CONTEXT.md`. History and negatives: `docs/DECISIONS.md`.
Verified numbers: `docs/RESULTS_LEDGER.md`. The previous handoff (v27c, 2026-09-29) is archived at
`docs/handoff_archive/HANDOFF_v27c_20260929.md`.

- **Branch:** `research/m3-output-numerics-20260927` (pushed to `origin`).
- **Last verified code commit:** `758723513` (E5 spec). Method code is at `56de98fe2` (`carry_first`).
- **Docs (AGENTS/CLAUDE/docs, E4 summary):** written 2026-10-01, uncommitted pending user review.
- **Two local checkouts.**
  - `E:/dlm/m3_output_numerics_20260927` is the working checkout used for all v27 work.
  - `E:/dlm/dlm_state_adaptive_router_20260828` is the same branch at an older commit (`3d48ebcdd`); pull before
    using it.

## Current conclusion

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
- **AIME has no speed room.** Accuracy is neutral for M3 (65/65 of 120). AIME is an accuracy check only.
- **Novelty is weak.** B ≈ SparseD (no significant speed difference at 64K). There is no evidence that V-aware
  selection beats score-only selection (it is worse at high AIME sparsity).
- **The ceiling at batch 1 is low.** GLOBAL attention is about 21% of a 64K request and 16% at 32K; a step is
  dominated by reading about 46 GB of MoE weights. The current gain is about 60% of that ceiling.

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

## Running (started 11:14 local)

- **E5 overhead/tail study** (`specs/v27_lb_overhead_e5.json`, protocol `v27_lb_overhead_e5_1c93b97271a3a266`).
  - Same 288 cells and host assignment as E4.
  - Arms: dense, M3, B, M3/B + `carry_first`, M3/B + `stable1`. 2,016 runs.
  - Deploy tag `v27_e5_7587235`, run dir `v27_lb_e5_001` on dllm, mpk and dlm2 (64K stage, then 32K).
  - Scoring is queued (label `lb_e5`). Expected about 14:00 local.
  - Fan plain M1/M2c/M3 for these cells are in E4; check that E5 dense tokens equal E4's.
- **Queued behind E5** (background jobs that start when each host's E5 ledger closes).
  - **E6, 96K large-seed panel** (`specs/v27_lb96k_confirm_e6.json`, protocol `v27_lb96k_confirm_e6_8ab3aa35d1874ca1`).
    - 6 fitting items × seeds 404–909 = 36 cells per arm.
    - Arms: dense, Fan plain M1/M2c/M3, B, M3, M3 + `carry_first`. 252 runs.
    - Deploy `v27_e6_8f9f188`, run dir `v27_lb96k_e6_001`, hosts dllm and mpk. Scoring label `lb96k_e6`.
  - **Batch-scaling diagnostic** on dlm2 (`scripts/v27_batch_step_bench.py`).
    - E4 run dir and deploy, 64K item 0, seed 404, decoder call 6, batch 1/2/4/8, keep 0.12/0.2.
    - Output: `<dlm2 dyh>/m3_output_numerics_v21_20260927/scratch_tests/batch_1001/bench64.jsonl`.
  - **E7, AIME large-seed panel** (`specs/v27_aime_confirm_e7.json`, protocol `v27_aime_confirm_e7_0515fa3125588ff0`).
    - 30 problems × seeds 404–909 = 180 cells per arm.
    - Arms: dense, Fan plain M1/M2c/M3, M3, and two low-overhead variants:
      - M3 + `carry_first` + 2K gate;
      - B at −ln2 + `carry_first` + 2K gate.
    - 1,260 runs. Deploy `v27_e7_ceefaf2`, run dir `v27_aime_e7_001`.
    - Starts after E6 on dllm/mpk and after the batch diagnostic on dlm2. Scoring label `aime_e7`.
    - Question: is any accuracy-safe AIME gain significant?

## Blockers

- None technical. The meeting is 21:00 local; results are wanted by 20:00.

## Immediate next steps

1. When E5 is scored:
   - verify the receipts: `carried_first_calls > 0` and bootstrap-dense = 1 canvas × 5 layers per request for c0;
     `gate_entries` for stable1;
   - check dense tokens are identical to E4;
   - report W and S per arm.
2. Larger-gain directions, ranked (see `docs/RESEARCH_CONTEXT.md` §8):
   - (a) **Batched serving.** Attention share grows with batch. Measure per-step time at batch 1/2/4/8 first; the
     runner is batch-1 only.
   - (b) Make the encoder canvas append's GLOBAL attention sparse with the canvas's final map (about 2% at 64K).
   - (c) Orthogonal step reduction (Prophet/JoT-style early exit). Not our contribution.
3. Commit the docs after the user reviews them. Copy E5 summaries into `results/…/lb_overhead_panel_e5/`.

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
