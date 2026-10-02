# AGENTS.md — entry point for coding agents (Codex, Claude Code, others)

## What this repository is

- The root is a general dLLM inference/evaluation framework (the `dllm` package in `src/dllm`; see `README.md`).
- **Branches.** The current working branch is `research/humaneval-v27-20261001`. A GPT session created it from the v27
  checkpoint `research/m3-output-numerics-20260927` (`add23afa9`) on 2026-10-01 23:59; all work since then
  (HumanEval, step statistics, regrouping, docs) is on it. The old branch stays at `add23afa9` as the v27
  checkpoint; fast-forwarding it is the user's call.
- This study (v27 checkpoint branch `research/m3-output-numerics-20260927`, continued on
  `research/humaneval-v27-20261001`) is about **block-sparse attention for the
  GLOBAL layers of DiffusionGemma-26B-A4B under its native adaptive stopping**.
  - The study's original core methods are M1/M2/M3; their historical comparisons remain in the ledger.
  - The goal is a real end-to-end gain against the strongest official dense baseline (FlashAttention-4), with no
    accuracy loss on AIME and LongBench-v2.
- Stable context is in `docs/`. The current frontier is in `HANDOFF.md`. Do not rely on chat history; the repo and its
  verified artifacts are the source of truth.

## Read in this order

1. `AGENTS.md` (this file): rules and terminology.
2. `HANDOFF.md`: current conclusion, what is done or running, next steps.
3. `docs/RESEARCH_CONTEXT.md`: model, method definitions (M1/M2/M3 and all comparison arms), substrates,
   workloads, metrics and comparison rules.
4. `docs/DECISIONS.md`: chronological decisions and negative results. Check it before proposing a direction.
5. `docs/RESULTS_LEDGER.md`: verified results only, each with its source.
6. Detail when needed:
   - `results/m1_m2_m3_frontier_v27_20260929/progress_20260930.md` (Chinese; detailed glossary and updates 1–12);
   - the per-panel `README.md` / `summary.md` files under `results/m1_m2_m3_frontier_v27_20260929/`.

## How to determine the current state

- Run `git log --oneline -20` on this branch. Commit messages name the panel, spec or variant they add.
- `HANDOFF.md` carries a date and a verified commit. If the log has moved past it, trust the log and the result
  files, then update `HANDOFF.md`.
- Panels are defined by specs in `results/m1_m2_m3_frontier_v27_20260929/specs/*.json`. A result counts only when
  its summary is in `results/…/<panel_dir>/` and is listed in `docs/RESULTS_LEDGER.md`.
- Some state lives outside the repo on the coordinator machine. It is private and not committed:
  - frozen protocols: `E:/dlm/v23_private/<name>_frozen/`;
  - scoring outputs: `E:/dlm/v27_private/lb_scoring/<label>/`;
  - deploy records: `E:/dlm/v21_private/`;
  - host registry: `E:/dlm/v20_private/hosts.json`;
  - coordinator scripts in `E:/dlm/` (see `HANDOFF.md`).

  If you cannot reach them, say so and do not guess.
- `STATE.json` is partly legacy (v18–v26 ledgers). Read its `current` block first; the older blocks are history.
- Several root files are legacy from earlier study versions and do not describe this study's current state:
  - `method_contract.md` is the v5 M1 contract: cached-score output, since superseded by current-QK output.
  - `DESIGN_DECISION.md` (v18), `TASKS.md` (v22), `RUN_MANIFEST.json` (v18) and `RESOURCE_AUTHORIZATION.md`
    (v18).

## Mandatory invariants

- **Arm selection (user update, 2026-10-01).** Follow-up panels may run only the stronger, relevant configurations;
  repeating plain M1/M2/M3 in every panel is no longer required. Keep the strong dense baseline and a matched
  optimized reference for incremental claims. Preserve historical plain-method results and name variants honestly.
  A selector that uses the current QK is fresh-T information, not M1.
- **Dense baseline** is official FlashAttention-4 (vLLM fork, CuTe DSL, SM90, head_dim 512).
  - The headline reference is `D_fa4_allkept`: FA4's block-sparse interface with every tile kept. It is bitwise
    equal to FA4 dense and the fastest dense configuration.
  - Also keep `D_fa4` (the plain FA4 path) where feasible.
  - Never headline against the HF default path (`D_native`), `D_c64` or `D_fast`.
- **Fair comparison.** All arms share:
  - substrate, deploy commit, host per cell, items, seeds and warm-up;
  - sparse arms run through the same FA4 kernel as dense, differing only in skipped tiles.

  Every arm of a cell runs on one host. Timed runs must not capture new CUDA graphs: report `Wc`, or check the
  counter.
- **Native decoding is fixed.** The runner asserts the official generation config: canvas 256, max 48 steps,
  confidence 0.005, stability 1, entropy bound 0.1, temperature 0.8→0.4, thinking ON.
- **Freeze before generation.** Workflow: spec (committed) → frozen protocol → deploy commit → bind → launch → score.
  - Never edit a frozen protocol, a run directory, a ledger or a published result.
  - A relaunch or resume after a killed worker needs a new run directory.
  - Never reset historical ledgers.
- **Every new variant gets three checks before its numbers are reported:**
  - unit tests;
  - a receipt check that the intended path ran (counters such as `observe_step`, `protected_routes`, decision
    interval, thresholds);
  - fairness of the comparison.
- **Reporting.**
  - Report kernel, attention-module and end-to-end levels separately.
  - `S/N` is an amortized per-step cost, not a per-forward price.
  - Attribute gains against the strong substrate; generic graph, compile or fusion hygiene is not a contribution.
  - Clean negatives beat forced positives. Never present interpolated or partial-panel numbers as results.
- **Group members' work** is read-only: cite it, do not absorb it. Overlapping directions are marked as
  collaboration candidates. Example: query-sensitivity protection against step inflation (branch
  `query-sensitivity-aware-v3`, git author JunyuLiao) is theirs.
  - The user is considering integration with classmates' A/B branches, which are still pending. Do not infer the
    branch names or merge now. Read-only review and an integration plan may precede a separately identified merge.
- **Privacy.** Never commit:
  - prompts, gold answers or unredacted generated text;
  - credentials or keys;
  - private absolute paths beyond what the repo already documents.

  Generation records are published only redacted.
- **Remote GPU hosts.**
  - Write only inside the user's `dyh` directories. Pin all caches there.
  - Run one worker per GPU. Check that the GPU is idle before launching.
  - Use the host interpreters recorded in the private registry.
  - GPU time needs no approval and budgets are not stopping points, but record GPU seconds.

## Terminology agents most often confuse

| term | meaning |
|---|---|
| M1 / M2 / M3 | Fan's methods. Definitions are in `docs/RESEARCH_CONTEXT.md` §3; they are not interchangeable with variants. |
| Fan plain / 原版 | `M1_R1_A8`, `M2c_R1_A8`, `M3_R3_A8` without our engineering variants |
| fresh T (`T_scope`, `fresh_fused`) | Junyu's selection from the CURRENT QK + projected V. Not M1. |
| B | bootstrap or anchor bitmap held without redecision (`hold_only`); ≈ SparseD |
| A8 / A64 | QK re-observation period: every 8 calls / once per canvas (fused into a dense call) |
| R1 / R3 / R6 | decision interval in calls (M1 = R1) |
| DP | dense-prefix risk (named variant of the M1 risk; parallel decision) |
| −ln2, −2ln2, 0 | threshold shift on the log-risk threshold. Negative keeps more tiles. Default −3.180, so −ln2 gives −3.874. |
| shared support | cross-layer bitmap sharing: rejected, it collapses quality |
| D_fa4_allkept / D_fa4 / D_c64 / D_fast / D_native / D_matched | dense controls (see `docs/RESEARCH_CONTEXT.md` §4) |
| eager vs piecewise_v1…v5 | execution substrates. Numerics and speed differ; never compare across them. |
| cell | one (item, seed). Each arm contributes its first output per cell. |
| W, Wc, S, N, NC (=N/C), T, S/N, S/C | request, decode, steps and per-step ratios vs dense (`docs/RESEARCH_CONTEXT.md` §6) |

## After substantial work

- Update `HANDOFF.md` with the date, verified commit, conclusion, done, running, next.
- Append to `docs/DECISIONS.md`. Add only verified results to `docs/RESULTS_LEDGER.md`, with source and protocol id.
- Update the `current` block of `STATE.json` with verifiable values only.
- Commit docs together with the work they describe, then push. Studies live on their own pushed branch (no merge, no
  PR); reports reference branch names.

## Handing over to another agent

A ready-to-paste Chinese handover prompt is in `docs/AGENT_PROMPT.md`. Keep it consistent with this file.
