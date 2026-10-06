# AGENTS.md — dLLM sparse-attention research

This file has two layers. The main body contains repository-wide rules for the
current `combined_end2end` worktree. The user-specific section records local
preferences for `/home/exouser/ljy/dlm`. The final section preserves Yuhan's
inherited context for collaboration; it is reference material and does not
silently redefine the current branch.

The first cross-step-reuse implementation in the current merged end-to-end line
was created by Yuhan. Preserve that provenance when describing the inherited
baseline, and name subsequent improvements as separate variants owned by this
study. The user's earlier work includes the value-direction-aware Gaussian32
router and query-sensitivity protection; keep those method identities distinct.

## What this repository is

- The root is a general dLLM inference/evaluation framework (the `dllm`
  package in `src/dllm`; see `README.md`).
- The current study is block-sparse attention for DiffusionGemma-26B-A4B under
  its native adaptive denoising sampler. The active execution path is vLLM
  with the pinned SM90 FlashAttention-4 implementation on Hopper.
- The current merged method performs cross-step reuse: it observes an exact
  map, selects a balanced fixed-budget set of eligible tiles, holds that map
  across calls, optionally carries it to the next canvas, and re-observes when
  a named progress clock fires. Sticky retention, candidate-pool
  re-observation, budgets, and selection granularity are separate variants.
- Gaussian32/value-direction-aware routing and query-sensitivity/C-gate
  protection are prior method families in this repository. A projected
  diagnostic or a reused map is not by itself evidence that full-dimensional
  attention work was removed.
- Historical M1/M2/M3 results remain in the ledger. Do not use a historical
  method name for a new selector.
- Stable context is in `docs/`. `HANDOFF.md`, committed specs, frozen run
  metadata, completion markers, receipts, and result summaries are the source
  of truth. Do not rely on chat history or partial workers.

## Read in this order

1. `AGENTS.md` (this file): shared rules, local preferences, and inherited
   collaboration context.
2. `HANDOFF.md`: current conclusion, what is done or running, and next steps.
3. `docs/V31_BASELINE_ROOT_CAUSE_20261003.md`: current V31 dense baseline,
   execution, and progress-aware method record.
4. `docs/RESEARCH_CONTEXT.md`: model, method definitions, substrates,
   workloads, metrics, and comparison rules.
5. `docs/DECISIONS.md`: chronological decisions and negative results.
6. `docs/RESULTS_LEDGER.md`: verified results and their sources.
7. The relevant panel `README.md` and `summary.md` under `results/`.

## How to determine the current state

- Run `git branch --show-current` and `git log --oneline -20` before making a
  claim. Commit subjects identify panels, specs, variants, and fixes.
- `HANDOFF.md` carries a date and verified commit. If the log has moved past
  it, trust the log and result files, then update `HANDOFF.md`.
- A panel counts only when its frozen spec, source identity, complete run
  family, summary, and ledger entry agree. A calibration, diagnostic, or
  partial panel is labeled as such.
- If a required private coordinator, model cache, scorer, or host registry is
  unavailable, say so and do not guess its contents.
- `STATE.json` is partly legacy. Read its `current` block first; treat older
  blocks as history unless the current protocol explicitly names them.
- Several root files are historical rather than current study contracts:
  `method_contract.md`, `DESIGN_DECISION.md`, `TASKS.md`, `RUN_MANIFEST.json`,
  and `RESOURCE_AUTHORIZATION.md`. Read them only when an older panel points
  to them.
- Keep historical experiments, source pins, and published results intact. A
  new source, protocol, runtime, calibration, or relaunch receives a new named
  arm and run directory.

## Shared mandatory invariants

- **Arm selection.** Follow-up panels may run only stronger, relevant
  configurations; repeating every historical arm is unnecessary. Keep the
  strongest dense baseline and a matched optimized reference for incremental
  claims. Preserve historical plain-method results and name variants honestly.
- **Dense baseline.** The current headline reference is vLLM's FULL decode
  path with the exact DiffusionGemma mask fix corresponding to upstream PR
  #51994. Keep matched PIECEWISE dense and all-kept sparse-consumer controls.
  Historical HF controls and the unfixed FULL path are diagnostic or
  substrate-specific references, not headline baselines.
- **Fair comparison.** All arms in a cell share substrate/runtime pins,
  deploy commit, host/GPU, questions, seeds, warm-up, sampler settings, and
  scoring rules. Disclose exact kernel paths. Native dense and sparse paths
  may use different consumers, so retain all-kept controls to expose that
  implementation difference rather than hiding it.
- Every arm of a cell runs on one host. Seed-pair trajectory comparisons when
  possible and account for question clusters in uncertainty estimates.
  Nonsignificance does not establish noninferiority.
- **Native decoding is fixed.** Preserve the official proposal, acceptance,
  re-noising, self-conditioning, stopping rule, temperature schedule,
  thinking setting, and output limits. The native canvas, maximum calls,
  confidence, stability, and entropy settings come from the frozen protocol.
  Do not add hidden forwards or replace native stopping with a fixed-step loop.
- **Freeze before generation.** Use spec (committed) → frozen protocol →
  deploy/source pins → bind → smoke/receipt → launch → clean timing → score.
  Never edit a frozen protocol, run directory, ledger, or published result.
  A relaunch or resume after a killed worker needs a new run directory; retain
  the failed attempt.
- **Every new variant gets three checks before reporting numbers:** relevant
  unit tests; a receipt proving that the intended selector, layer scope,
  budget, thresholds, and kernel path ran; and a fairness audit against the
  matched dense/all-kept controls.
- **Timing and accounting.** Check an actual CUDA capture counter; timed runs
  must not capture new CUDA graphs or include routing/scoring instrumentation.
  Run synchronized clean timing separately from the audit pass. Report kernel,
  attention-module, and end-to-end levels separately.
- **Metrics.** Report accuracy, physical eligible/skipped tiles by GLOBAL and
  LOCAL scope plus the stated overall denominator, denoising calls `N`, calls
  per canvas `N/C`, output length `T`, per-forward cost `S/N`, generation time
  `S`, and request/end-to-end time `W`. `S/N` is an amortized per-step cost,
  not a per-forward price. Never infer speed from sparsity or a requested
  budget alone.
- Use the official scorer and paired or item-clustered uncertainty intervals.
  Separate held-out confirmation from transductive calibration, previously
  examined prompts, fixed-canvas diagnostics, and partial panels. Clean
  negatives beat forced positives.
- **Collaboration.** Cite peer algorithms and inherited code; do not silently
  attribute them to this study. Peer branches and files remain read-only unless
  the user explicitly authorizes an isolated snapshot or execution. Do not
  merge a peer branch merely to make a comparison convenient.
- **Privacy.** Never commit prompts, gold answers, token IDs, credentials,
  keys, raw completions, or unredacted generated text. Publish only sanitized
  generation records, aggregates, source fingerprints, and execution receipts.
- **Remote GPUs.** Run one worker per GPU, check that the device is idle, pin
  caches and interpreters, and write only inside user-owned remote directories.
  Record GPU seconds. Do not modify shared environments or another
  collaborator's worktree.

## User-specific preferences (this worktree only)

This section belongs to the user's `/home/exouser/ljy/dlm` worktree. It is not a
portable instruction for another agent, collaborator branch, or repository.
Do not copy it into another agent's `AGENTS.md`.

- **Canonical results:** write generated results, reports, receipts, and plots
  under `/home/exouser/ljy/dlm/results`, normally referenced as `results/...`
  from the repository root. Do not create a competing result root or use a
  groupmate-owned result directory. Keep `/home/exouser` otherwise clean.
- **Campaign directories:** use `results/<study>_<YYYYMMDD>/`, for example
  `results/v31_20261003/` or `results/v31_20261005_aime_halfkv/`. Use a new
  date/attempt directory for a new frozen protocol or a relaunch; never reuse
  an old run directory.
- **Arm names:** use lowercase `snake_case` names that state the method and
  material configuration, such as `dense_full_fix51994`, `dense_piecewise`,
  `allkept_fa4`, `lean_k8192_progress`, or
  `lean_k8192_progress_sticky`. Do not use ambiguous names such as `main`,
  `new`, `final`, or `latest` without a manifest that defines them.
- **Attempt and shard names:** use zero-padded identifiers such as
  `attempt001`, `seed03`, and `shard02`. Keep the arm, seed, host, source
  hash, and attempt in structured metadata even when they are also in a path.
- **Standard result files:** prefer `README.md`, `config.json`,
  `receipt.json`, `summary.md`, and `final_complete.json`. A summary is
  publishable only when the completion marker and source/protocol identity are
  present.
- **Evidence style:** publish compact sanitized traces, numeric measurements,
  source hashes, and execution receipts. Keep prompts, gold answers, token
  IDs, raw generations, credentials, and private host paths out of Git.
- **Run behavior:** preserve native decoding, use resumable shards, inspect
  long-running jobs about every 30 minutes, and retain failed attempts rather
  than overwriting them. Calibration, diagnostics, and previously examined
  prompts must be labeled as such.

## Yuhan-specific inherited context (collaboration reference)

This section preserves the original inherited material that describes Yuhan's
branch history, cross-step-reuse lineage, and older coordinator layout. It is
useful when reviewing or comparing that work. It does not override the current
branch, the shared invariants above, or the user's local result conventions.

### Historical branch overlays

- The V30 cooperative campaign was based on V29 and allowed attributed
  peer-document snapshots and isolated execution of peer algorithms. Peer
  branches and `chw/ljy` files were read-only; frozen V29 experiments and
  source pins were preserved. Multiple questions and eight seeds were required
  for that confirmation campaign; small diagnostics were not claims.
- The V29 continuation used branch `research/vllm-confirmation-20261002`,
  from V28 `e5e1ffcc8`, and read
  `docs/CONFIRMATION_CAMPAIGN_V29_20261002.md`. Its private transfers were
  limited to user-owned `dyh` directories; prompts, tokens, gold, credentials,
  generated text, and private paths were excluded from Git.
- The V28 continuation used branch `research/vllm-variants-20261002`, from
  `research/humaneval-v27-20261001` at `9b027df8f`, and read
  `docs/VARIANT_CAMPAIGN_V28_20261002.md`. It was independent of the parent
  worktree and its frozen V18b campaign.
- Those branch-specific coordinators and frozen campaigns are historical. Do
  not infer their paths, liveness, or result status for the current branch.

### Historical v27 study and terminology

- The older v27 checkpoint studied block-sparse attention for the GLOBAL
  layers of DiffusionGemma-26B-A4B under native adaptive stopping. Its core
  methods were M1/M2/M3, with historical comparisons retained in the ledger.
  Its goal was an end-to-end gain against official FlashAttention-4 without
  accuracy loss on AIME and LongBench-v2.
- Historical details used these documents and locations:
  `docs/RESEARCH_CONTEXT.md`,
  `results/m1_m2_m3_frontier_v27_20260929/`, and its per-panel
  `README.md`/`summary.md` files. The older `E:/dlm/v23_private/`,
  `E:/dlm/v27_private/`, `E:/dlm/v21_private/`, and `E:/dlm/v20_private/`
  paths were private coordinator locations for that work, not current output
  roots. If they are unavailable, do not guess.
- Historical terms remain meaningful when reading old panels: M1/M2/M3,
  Fan plain, fresh T, B/hold-only, A8/A64 observation periods, R1/R3/R6
  decision intervals, DP, threshold shifts, shared support, dense-control
  labels, eager/piecewise substrates, cell, and `W`, `Wc`, `S`, `N`, `N/C`,
  `T`, `S/N`, and `S/C`.
- For the older panels, use these definitions: M1/M2/M3 are Fan's methods;
  Fan plain means `M1_R1_A8`, `M2c_R1_A8`, and `M3_R3_A8` without our
  engineering variants; fresh T (`T_scope`, `fresh_fused`) selects from the
  current QK and projected V; B is a bootstrap/anchor bitmap held without
  redecision; A8/A64 are QK re-observation periods; R1/R3/R6 are decision
  intervals; DP is dense-prefix risk; negative threshold shifts keep more
  tiles; shared support is cross-layer bitmap sharing; and `D_fa4_allkept`,
  `D_fa4`, `D_c64`, `D_fast`, `D_native`, and `D_matched` are distinct dense
  controls. Eager and piecewise substrates must not be compared as if they
  were numerically or temporally identical.
- The initial Yuhan cross-step-reuse baseline should be cited as inherited
  work. The current merged method may reuse its machinery, but every changed
  selector, progress clock, budget, sticky rule, candidate pool, or consumer
  path receives a new name and a new receipt.

### Older collaboration and handoff notes

- Historical peer work was read-only and attributed. Overlapping directions,
  including query-sensitivity protection and value-aware routing, were marked
  as collaboration candidates until separately integrated and audited.
- Historical workflows required prompt commits/pushes, reproducible evidence,
  sanitized traces, preserved private originals, one worker per GPU, idle GPU
  checks, and no modification of peer files. These remain compatible with the
  shared rules above.

## After substantial work

- Update `HANDOFF.md` with the date, verified commit, conclusion, done, running,
  and next steps.
- Append verified decisions to `docs/DECISIONS.md`. Add entries to
  `docs/RESULTS_LEDGER.md` only for complete, protocol-identified results with
  their source.
- Update the `current` block of `STATE.json` with verifiable values only when
  the current study still uses that ledger.
- Commit documentation together with the work it describes and push the
  research branch when the user requests it. Do not reset historical ledgers.
