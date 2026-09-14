# Value-aware BLASST, 128×64

This is an emulated attention research experiment, not a custom kernel or a
speed benchmark. Canonical output: `results/diffusion_gemma_value_aware_128x64/`.
Run from the repository root with the `ljy_dlm` environment and `PYTHONPATH=src:.`.

```bash
python -m pytest tests/test_diffusion_gemma_value_aware.py -q
python -m experiments.diffusion_gemma_value_aware.run prepare
python -m experiments.diffusion_gemma_value_aware.launch screen
```

The launcher first requires the CUDA smoke, then collects one shared dense
calibration/development trajectory per prompt. It checks its actual child
process, progress, failure count and GPU health every 900 seconds. All shards
are atomic and fingerprinted; resume never silently mixes code versions.

## Frozen data and operator contract

- AIME26: all 30; original calibration IDs 2,8,14,20,23,30. Report both full30
  and heldout24. These six calibration problems are also the AIME development
  pool; their accuracy is not confirmatory evidence.
- LongBench: the exact previous 50 prompts, ten each of qasper, hotpotqa,
  gov_report, trec, passage_retrieval_en. Separate next two per task for
  calibration, next one per task for development (deterministic hash ordering).
- Existing official prompts/budgets/scorers, pinned BF16 model revision,
  seed per prompt, native temperature schedule, thinking off, canvas256 and
  maximum48 denoising steps. Temperature0 is a native-schedule sentinel, not
  greedy decoding. No old sparse shard with changed settings is reused.
- 128 query ×64 KV physical tiles; prefix and canvas both eligible. Structural
  support, native GQA and native logit scaling are preserved. Retaining a
  physical tile retains all its valid positions. Counts include mandatory
  retention, mixed prefix/canvas boundary tiles and partial sequence tiles.

## Criteria

New streaming variants use the retained softmax state **before** the candidate
block. The reference BLASST variant instead preserves the existing maximum over
all preceding blocks (even skipped blocks). This distinction matters for its
aggressive lambda>1 extension. Initial physical decisions require all valid rows
to pass; an aggregate-risk rule exists only as a separately labeled ablation.

For Sol and diagnostic rankings, final nonempty repair is separately versioned
in `ranking_guards.py`: select the highest declared signal on each empty row's
structural support. It never gives nominally proxy/QK-only routing access to
exact attention mass. The historical initial screen's generic mass repair
affected30.33% of AIME query rows at Sol Gaussian90%, so its Sol numbers must
remain labeled historical. Corrected probes are collected in the queued dense
refinement replay, with their own CUDA smoke and an unresolved-issue freeze gate.

Value metadata includes max, mean, RMS, p95 of token norms, norm of the mean
vector, the mean vector itself, radius around that vector, and token norms.
The reference magnitude is RMS token norm across all currently valid KV tokens
in that layer/native KV head, shared across pooling choices. This requires a
metadata reduction before routing, not the current dense attention output.
Centered risk uses the output of preceding retained blocks. Its units are
normalized by the reference value scale so a dimensionless scalar can be fitted.

Only complete, content-verified unchanged prefix blocks reuse summaries.
Canvas and mixed boundary blocks refresh each call. An actual implementation
would replace the explicit equality audit by cache-write/version notifications.

Compensation keeps exact block softmax mass/denominator and substitutes mean V
for the block contribution; its control substitutes zero. Both omit full PV on
selected blocks but do **not** delete attention tiles or skip softmax. No raw
QK,V operator or value-multiplied raw logits are used. Dense FP32 diagnostics
are separate from routing, and BF16 retained operators match dense eager order.

## Screening, calibration, and final evaluation gates

1. Test operators and CUDA parity/coverage before screening.
2. Screen 21 criteria/pooling combinations on shared dense states at first
   observed calls for steps0,4,12,24, all30 layers/all16 heads/both Q tiles.
   Fixed-budget risk ranking is a **screening diagnostic**, not deployment.
   Collect physical all-row log-risk values for calibration. Screen QK/mass/
   contribution references and plain/value Sol Gaussian/top-k on those states.
3. Select pooling from calibration-only matched-budget mass/output-error and
   selection diagnostics. Fit independent local/global thresholds per target.
   New criteria use scalar thresholds. BLASST retains the repository's
   inverse-valid-KV-length scale and a separately labeled cap1 reference.
4. Verify/refine thresholds on sparse calibration trajectories, examine
   separate LongBench development accuracy, and test targeted revisions only
   when diagnostics justify them. Freeze candidates before heldout evaluation.
5. Every final condition requires all30 AIME26 and all50 LongBench samples.
   Evaluate targets25/50/75/90; diagnostic top-p has adaptive density. Confirm
   promising frontier points with paired seeds when practical.
6. Regenerate tables, physical-count-weighted local/global/whole statistics,
   row-weighted retained dense mass, relative FP32 output error, positional
   token agreement and official task scores from complete shards only.

The implementation includes sparse calibration, separate development reporting,
frozen final evaluation, audited report generation, and paired-seed confirmation.
Implementation does not imply those GPU stages are complete: consult the bundle's
`WORK_STATUS.md` and process-verified monitor. Never infer completion from a log
filename or a successfully launched worker.

## Resume and subsequent stages

Use `/home/exouser/miniconda3/envs/ljy_dlm/bin/python` with `PYTHONPATH=src:.`,
`HF_HUB_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false`, and `OMP_NUM_THREADS=8`.
CUDA/process checks require the environment's escalated CUDA access. Do not run
another worker beside the live calibration or its queued refinement screen.

The2026-09-11 calibration-only pruning decision is recorded in the bundle's
`screening_pruning.json`: the original running-output centered criterion failed
all6 AIME calibration problems at49.15% measured deletion, with all outputs
length-terminated. Its completed LongBench50 point was preserved; remaining75%
tests were stopped. Do not restart the default Cartesian sweep and silently
reintroduce this pruned work. Previous-output/exact-mass centered refinements
are separate candidates and remain in scope. A subsequent output-risk50 failure
(AIME0/6 at47.43% measured deletion; LongBench.1339 versus dense.4732 at49.03%)
also pruned the remaining base `risk`75/higher-sparsity development. Its two
partial AIME75 shards are preserved and are not a complete result. See
`risk_screening_findings.md` and `interventions/20260911_risk_pruning_restart.json`.
Mean compensation50 subsequently failed LongBench calibration at48.78% PV
omission (all ten outputs ended with only `thought` text); its AIME0/6 result
overshot50 and is labeled accordingly. Remaining compensation75/higher-omission
development was also pruned, preserving all partial data. See
`compensation_screening_findings.md` and the linked intervention record.
Zero-PV50 is now complete on both benchmarks. AIME failed0/6 at49.90%PV,
but LongBench scored.4766 versus dense.4732 at52.80%PV. An initial higher-omission
stop was therefore revised after the LongBench quality audit: retain this control
and finish its interrupted LongBench75 calibration. AIME75 is already complete
and must be reused. See `zero_pv_screening_findings.md` for the exact ordering
and caveats; do not call the whole zero-PV family permanently pruned.

The refinement screen and summary audit are now complete:12 refinement smoke
cases,14 corrected-routing cases, and21 exact dense replays. The corrected
signal-only routing review is resolved. See `refinement_screening_findings.md`.
The deferred zero-PV75 calibration is now complete. LongBench scores0.00045
at76.21%PV, so its near-dense50% calibration score does not extend to75%.
The original-BLASST cap1 repair completed successfully and passed the raw-shard
audit. LongBench global50 verifies at51.18%, with local independently fixed at
lambda1 (24.76%, unattainable target). Equivalent joint-lambda1 checkpoints are
reused with explicit provenance. Exact-mass calibration50/75 is now complete.
Bounded development50 and its stricter audit are complete (eight explicit
methods, five disjoint LongBench prompts each). Candidate identities are chosen
in `candidate_decision.json`. All40 required policies are now complete and passed
the raw-shard audit; a bounded recovery reused checkpoints after four transient
GPU-contention OOMs. Historical errors remain preserved.
The **currently running stage is the frozen final sweep**, launched2026-09-12
07:10 UTC:47conditions×80prompts, with dense first and mass_exact/mass/mass_value
as the selected candidates. `final_contract.json` pins477 evidence sources;
heldout data were not used for selection. `job.json` and its900-second supervisor
track the live worker. Do not launch competing work or overwrite frozen policy,
selection or development evidence. The queued finalizer runs the CPU report
after the sweep and requires its complete audit; final results, paired-seed
confirmation and empirical conclusions remain pending.
Original BLASST90 already reuses the proven joint-lambda1 calibration boundary
with explicit raw-shard/configuration proof (`boundary_reuse_audit.json`), so no
identical inference is repeated. Both type targets remain labeled unattainable.
The completed control's checkpoint-resume command was:

```bash
python -m experiments.diffusion_gemma_value_aware.launch calibrate --methods zero_pv --targets 0.75 --max-rounds 3
```

Original/aggressive BLASST, mass-only, aligned, value and mass_value50/75,
plus centered50, risk50, compensation50 and zero-PV50 policies are already cached.
Higher-sparsity centered/risk/compensation partial data remain historical, not
complete or target-attained results. Use explicit method lists in subsequent
development/report commands; completed50 diagnostics for the pruned families,
if needed, must be a separate
`--methods centered risk compensate --targets 0.5` run. Final selection still
requires complete evidence for every selected candidate and all final references.

```bash
python -m experiments.diffusion_gemma_value_aware.calibration
python -m experiments.diffusion_gemma_value_aware.analysis
python -m experiments.diffusion_gemma_value_aware.launch refinement-screen
python -m experiments.diffusion_gemma_value_aware.refine_run summarize
# This reference repair is complete and cached:
python -m experiments.diffusion_gemma_value_aware.repair_original
# Screening priority, not a final candidate freeze:
python -m experiments.diffusion_gemma_value_aware.launch calibrate --methods mass_exact --targets 0.5 0.75 --max-rounds 3
# This bounded development pass is complete/cached:
python -m experiments.diffusion_gemma_value_aware.launch development --methods mass_exact blasst_original blasst_aggressive mass aligned value mass_value zero_pv --targets 0.5
python -m experiments.diffusion_gemma_value_aware.development_report --methods mass_exact blasst_original blasst_aggressive mass aligned value mass_value zero_pv --targets 0.5
```

The refinement screen is evidence-driven: exact online block mass isolates the
loose max-logit mass bound; previous-step centered output isolates the running
partial-output estimate. The combined criterion tests both. These operators have
a separate source SHA and require their own two-benchmark CUDA parity smoke.
Refinements are **not** silently included in the default Cartesian sweep. After
their screen, explicitly name promising variants (`mass_exact`,
`centered_previous`, `centered_previous_exact`) in calibration/development commands.
All use the same generator and sample caches through explicit versioned dispatch.
Completed shared-state evidence prioritizes `mass_exact`: LongBench50 error
improves0.2323→0.2190 versus the BLASST ranking reference at identical counts;
AIME50 does not improve versus BLASST. Previous-output centered variants do not
improve overall error versus exact mass alone. This motivates targeted sparse
verification, not an asserted accuracy win or an exhaustive refined-method sweep.

The legacy initial calibration selector requires the documented independent
cap-one repair before final freeze. The new `policy_search.py` fixes selection
for future runs; it does not rewrite the already-running worker's in-memory code
or its completed traces. The repair archives previous policies and verifies new
joint settings with each unattainable attention type independently fixed at1.

For new value/mass risks, `rank_calibration.py` scales updates by the empirical
dense-calibration CDF. This replaces generic unit-sized log steps, which badly
overshot risks clustered near1. It brackets/interpolates in rank coordinates or
uses a bounded rank residual, maps back to a strict-FP32 scalar threshold, and
leaves types within2points unchanged. Existing BLASST inverse-L fitting is not
changed. Completed good policies are reused; bad legacy new-risk policies and
all completed observations are archived/reused before refinement. `max-rounds`
counts additional verification points on resume, not discarded prior evidence.
Before final freeze, `policy_audit.py` recomputes selected local/global counts
from all completed raw calibration shards and verifies every policy's identity,
thresholds, operator, prompts and seeds at all four target levels.
The final contract also carries the underlying calibration/development source
hashes, not only the audit-file hashes, so report regeneration checks the raw
selection evidence. Repaired cap-one labels and selection-only sources are
explicit in the calibration table.

After complete calibration/development evidence, write `candidate_decision.json`
with selected methods, rationale, `heldout_used:false`, the scientific fingerprint,
and `development_audit_sha256`. Calibrate missing25/90 targets for the selected
methods plus both BLASST references, then explicitly freeze and run:

```bash
# Supply the actual evidence-selected method names; do not copy placeholder names.
python -m experiments.diffusion_gemma_value_aware.evaluate freeze --methods SELECTED_METHODS
python -m experiments.diffusion_gemma_value_aware.launch run-final
python -m experiments.diffusion_gemma_value_aware.report
# Optional additional paired seed, only after a complete primary audit:
python -m experiments.diffusion_gemma_value_aware.launch confirm --methods FROZEN_CONDITION_NAMES --seed-offset 10000
```

For an automatic handoff after an active calibration worker, queue the already
selected identities with `queue --after-pid WORKER_PID --stage finalize --methods
mass_exact mass mass_value`. The helper waits for that exact process and a free
GPU, resumes only missing calibration policies in one bounded pass, and invokes
the same raw-audit freeze gate before final inference. The final runner explicitly
processes dense first despite sorted JSON keys, reusing valid caches on resume.
The handoff also runs the CPU report and requires its complete audit; partial
results or a normal worker exit alone never constitute completion. Additional
paired-seed confirmation and empirical review remain separate later steps.

The final contract includes all80 examples per condition. Reporting reads completed
shards only, audits prompts/seeds/decoding and threshold sources, and produces
compressed exact routing buckets plus count-weighted marginal summaries. Physical
deletion, softmax omission, full-PV omission, and compensation have separate columns
and plots. AIME heldout24 comparisons use nearby **measured** BLASST sparsity within
three points; no unmatched target-based accuracy advantage is inferred. Additional
paired seeds retain frozen thresholds and all30 AIME /50 LongBench prompts.

## Information/overhead accounting

BLASST and value-risk variants still require token QK before deciding. They can
omit block softmax/PV after that decision. Value summaries add O(Kd) metadata
construction per changed V and small scalar loads per block; max/mean/RMS are
reductions, exact p95 requires selection/sorting, and token alignment adds one
norm/lognorm per KV token. Centering adds O(Qd) vector distance per candidate;
it needs the running output, not a future/current dense output. Mean compensation
requires block softmax sums and O(Qd) scaled-mean accumulation but avoids full
O(QKd) PV for replaced blocks. Ranking and true normalized mass/contribution
diagnostics require global/future information; they are not one-pass routers.

The streaming constraint follows [FlashAttention](https://arxiv.org/abs/2205.14135)
and the BLASST reference follows the repository's documented extension of
[BLASST](https://arxiv.org/abs/2512.12087). This experiment measures quality and
mask/work opportunities, never realized kernel speedup.
