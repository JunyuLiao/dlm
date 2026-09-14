# Value-aware follow-up: AIME26 and LongBench v2

## Current user revision: quick25 at50%, then pause

The side-chat update on2026-09-13 supersedes the broad60/full-sweep execution
below: **10 AIME +5/domain LongBench-v2**, all13 conditions at50%, then present
the full results and **pause for user review before later targets/full study**.
The old worker was intentionally interrupted; all saved outputs and frozen
contracts remain intact. The new quick-review contract does not weaken the
old60-example audit or change operators/policies/settings.

The fixed subset selects five non-calibration AIME problems per exam by seed42
hash; v2 easy/hard counts per domain are3/2,2/3,3/2 (eight easy,seven hard total).
Selection never reads scores or cache availability. Exact previous results are
reused and new compatible shards remain in the original final cache. Reports,
selection, hashes and the explicit review gate are in `quick50_review/`.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m pytest tests/test_value_aware_quick_review.py -q
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m experiments.diffusion_gemma_value_aware_followup.quick_review prepare
HF_HUB_OFFLINE=1 PYTHONPATH=src:. TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 python -m experiments.diffusion_gemma_value_aware_followup.quick_review launch
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m experiments.diffusion_gemma_value_aware_followup.quick_review report
```

The launcher reuses the shared H100 lock/idle check and900-second monitor. The
worker attempts every independent condition, audits cached dense without new
dense inference, regenerates the325-result report, and has **no later-stage
launch path**. One missing sample prevents a complete report. The provisional
keep flag (at most one extra wrong per benchmark versus the better BLASST)
is an agent-proposed tolerance, not specified by the user or an equivalence
guarantee. Both reference deltas and actual sparsity gaps are always shown.

The remainder of this README documents the preserved larger study, which is
**not authorized to continue beyond the quick review without user direction**.

Active user objective: continue the six proposed value-aware BLASST directions,
show their results against original BLASST, and investigate whether extra value
information improves accuracy at **matched actual physical sparsity**. Do not
substitute the previously selected exact-mass-only comparison for these methods.

The user explicitly confirmed LongBench **v2** on 2026-09-13. It has no generative
summarization/code-completion tasks. The confirmed mapping is single-document QA,
multi-document QA and code-repository understanding, ten multiple-choice questions
each. AIME26 uses all30, with full30/heldout24/calibration6 separately labeled.

## Frozen data, scoring and limitations

- V2 data revision `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`, official code
  revision `2e00731f8d0bff23dc4325161044d0ed8af94c1e`.
- Each v2 domain: five easy/five hard final questions, one easy/one hard
  calibration and one easy/one hard development question. Selection is
  short-first then seed42 hash ordering, before observing scores. Code development
  uses medium contexts because all12 short code items are needed for final/calibration.
- Model-input cap32,768 tokens, official-style head/tail truncation, official
  zero-shot prompt and128-token answer budget. This is a short-focused,
  32k-context v2 subset, **not full-context v2 performance**. Fourteen final
  v2 questions are truncated:3 single-doc,2 multi-doc,9 code.
- The exact official answer extractor is loaded without starting pred.py's
  server client. It searches the full response; a leading `thought` line does
  not cause the old TREC first-line failure. Invalid/unparsed answers score zero.
  Report parsing failures separately; never silently change scorers after scores.
- AIME prompts/seeds/budgets and native model settings are unchanged. AIME24
  non-calibration questions were evaluated previously, so new-design results on
  them are exploratory, not fresh untouched heldout evidence. V2 finals never
  select methods or thresholds.
- BF16, model revision from the previous experiment,128×64 physical tiles,
  prefix+canvas eligible, native scaling/GQA/structural masks. Whole sparsity
  covers decoder denoising attention; encoder/prefill stays dense and excluded.
  Requested temperature0 remains the native0.4–0.8 sentinel, not greedy.

## Required methods and staged execution

All six families must be represented, not silently replaced by a finalist:
value-gap, mass×value, denominator-aware output-risk, token-aligned norms,
output-centered deletion, and mean compensation. Include mass-only and zero-PV
controls, plus the exact-mass refinement. Original capped BLASST and the explicitly
labeled aggressive extension are references. Retain QK/mass/contribution rankings
and plain/value Sol Gaussian/ranking as same-state diagnostics; the BLASST
investigation remains primary.

1. Read/audit the prior proposed-method evidence (history/report.md). These are
   AIME6 and **LongBench-v1** calibration results, not new benchmark finals.
2. Cache dense final60, then shared calibration/development dense states. Reuse
   unchanged AIME dense and screening sources with exact provenance; no old v1
   sample can stand in for a v2 question. Run a fresh AIME/near32k CUDA smoke.
3. Screen all original pooling choices on shared states, reuse calibration code
   and sparse verification, then freeze local/global policies. Evaluate a
   same-state/no-value control before attributing a gain specifically to V;
   legacy BLASST and new risks have different state/threshold conventions.
4. Run a broad50% comparison first, showing all six proposed methods on all30
   AIME and all30 v2 questions. This is an interim stage, not the full goal.
   Complete requested25/50/75/90 curves for frozen final candidates; preserve
   negative/pruned development evidence and never label partial conditions final.
5. Diagnose failures and test justified refinements only on calibration/development;
   disclose any design influenced by exposed final results. No blind Cartesian
   sweep over every pooling/threshold combination and no guaranteed-value-win claim.
6. Complete final report/audit: per-domain and aggregate accuracy, dense-relative
   deltas/paired intervals, actual overall/local/global count-weighted sparsity,
   retained mass, relative attention-output error, token/sequence agreement,
   thresholds, work accounting, layer/head/step and prefix/canvas diagnostics,
   FlashAttention information/metadata/arithmetic analysis, plots and conclusions.

Compensation and zero-PV target **PV omission**, not physical deletion. Exact-mass
routing needs block softmax before deciding. No custom kernels or speedup claims.
More information is an opportunity, not a theorem about the quality of a chosen
heuristic; V RMS normalization makes several magnitude features nearly constant.

## Supplemental diagnostics and calibration

The reused `JointScreen` records refinements and signal-guarded rankings only;
it does **not** include the original value-pooling probes. The follow-up adds a
separately versioned supplemental observer on calibration/development examples,
checking exact generated-token parity against the cached joint dense run.
Analysis joins the observers while counting dense execution records once.
This is a required statistics replay, not a second final baseline.

The additional `no_value_control` uses the existing value-routing operator with
the pooled-magnitude/reference ratio set to exactly one. Its masks therefore
depend only on QK and retained pre-block state, not V. Actual retained outputs
still use unmodified V. It has the same scalar-threshold family as the proposed
value rule and is separately distinguished from original/aggressive BLASST.

Calibration reuses the existing inverse-valid-KV-length BLASST fit/search and
empirical-risk-rank refinement for scalar criteria. It fits local/global
separately using only the designated six calibration examples per benchmark.
Identical AIME policies may reuse their audited original calibration shards;
changed pooling cannot. All policies are verified from raw physical tile counts.
Original BLASST's unattainable types deploy constant lambda1 at **all** final
lengths, not merely a calibration-length-saturated inverse-L threshold. An
algebraically identical exact1 representation may reuse an already saturated
calibration run, with the raw threshold representation and length-bound proof
retained in its provenance. Other attention types are not silently changed.

The first sparse calibration batch includes all six families, original and
aggressive BLASST, mass-only, exact-mass, zero-PV and the no-value control at50%.
Each family gets unpruned native parity and sparse finite-output CUDA smoke on
AIME and near32k v2 inputs. Independent calibration groups continue after a
failure; incomplete groups are never published as verified policies.

## Commands

From `/home/exouser/ljy/dlm`, use the `ljy_dlm` Python environment:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m experiments.diffusion_gemma_value_aware_followup.protocol
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m experiments.diffusion_gemma_value_aware_followup.history
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m pytest tests/test_value_aware_followup_*.py -q
HF_HUB_OFFLINE=1 PYTHONPATH=src:. TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 python -m experiments.diffusion_gemma_value_aware_followup.run launch
HF_HUB_OFFLINE=1 PYTHONPATH=src:. TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 python -m experiments.diffusion_gemma_value_aware_followup.jobs launch --stage supplement
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m experiments.diffusion_gemma_value_aware_followup.analysis
HF_HUB_OFFLINE=1 PYTHONPATH=src:. TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 python -m experiments.diffusion_gemma_value_aware_followup.jobs launch --stage calibrate -- --targets 0.5
```

`launch` starts only the initial dense/screen stage. The supervisor holds an
exclusive lock, refuses an occupied GPU, and checks its actual child every900
seconds. Results/checkpoints are resumable; independent samples continue after
failure. Worker exit alone does not imply completion: inspect initial_audit.json.
Do not edit execution_contract.json-pinned sources while the worker is running.
The original experiment and its superseded full-sweep contract remain untouched.
Run stages sequentially: supplemental audit must pass before analysis; complete
analysis/proposals are required before calibration. Both launchers share the
same exclusive GPU-worker lock and refuse an occupied GPU. The calibration CLI
also accepts `--methods`, `--targets` and `--max-rounds` for explicit resumes.

## Development, final freezing, and reports

`development run` executes the six disjoint v2 development examples under each
verified policy, then joins their results with both benchmarks' raw calibration
evidence. It never reads final-generation files. Every family must pass audited
native-parity/finite-output CUDA smoke; missing policies or failed families are
recorded while independent groups continue.

`secondary` adds fresh guarded Sol/ranking smoke on AIME and near32k v2 inputs,
reusing the existing native short dense reference. The `workflow` wrapper can
queue development and secondary smoke behind the current supervisor without
editing its running code or occupying H100. Dependencies use PID plus process
start time to avoid PID-reuse errors; checks occur every900 seconds. Read the
actual supervisor PID from `job.json` when using `--after-supervisor`.

```bash
# Use the current supervisor PID, not a historical PID copied from a log.
HF_HUB_OFFLINE=1 PYTHONPATH=src:. TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 python -m experiments.diffusion_gemma_value_aware_followup.workflow launch --after-supervisor 424315 --stages development secondary
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m experiments.diffusion_gemma_value_aware_followup.development report
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m experiments.diffusion_gemma_value_aware_followup.final freeze --phase broad50 --rationale 'Document the actual complete calibration/development findings here before freezing.'
HF_HUB_OFFLINE=1 PYTHONPATH=src:. TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 python -m experiments.diffusion_gemma_value_aware_followup.workflow launch --stages final --final-phase broad50
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m experiments.diffusion_gemma_value_aware_followup.report --phase broad50
```

Do not run the freeze command until the complete development comparison has
been inspected and the rationale accurately records that evidence. The freeze
gate refuses incomplete comparisons/policies and verifies raw smoke outputs,
selected calibration thresholds, pooled configurations and source hashes.

- `broad50`: dense plus all12 methods/controls at50%,13 conditions ×60 samples
  =780 results. Explicitly interim, not completion of the requested study.
- `full`: all12 methods/controls at25/50/75/90;16 diagnostic ranking settings;
 10 plain/value Sol settings; dense.75 conditions ×60 samples =4,500 results.
  It requires all four-target policies and the secondary smoke. The broad50
  outputs are reused verbatim when their full configuration matches.

Final execution verifies all60 cached dense baselines first and does not rerun
dense inference. Identical sparse configurations/thresholds may reuse another
completed final source when only a target label differs, e.g. capped lambda1
settings. Such aliases retain the raw source hash and are audited against every
requested prompt/seed/budget/config/threshold; approximate equivalence is not
accepted. Every final condition still covers all30 AIME and all30 v2 examples.

Reports live in `reports/broad50/` or `reports/full/`. They regenerate solely
from completed final outputs, cached dense outputs, and the frozen contract;
no generation or calibration runs during reporting. The audit stays incomplete
until all expected raw sample-condition pairs and derived artifacts exist.
It rejects even one missing final sample. Tables include per-task accuracy,
full30/calibration6/previously-exposed24 AIME views, physical counts, exact-PV
versus denominator mass, output error, agreement, thresholds and paired CIs.
Compressed per-layer/head/step output is deterministic across regeneration.
Final empirical synthesis, calibration-miss analysis, streaming-compatibility
assessment and any additional paired-seed confirmation still require the
completed experiment; passing unit tests alone is not evidence of those results.

## Read-only mechanism checks while GPU stages run

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. OMP_NUM_THREADS=2 python -m experiments.diffusion_gemma_value_aware_followup.diagnostics
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. OMP_NUM_THREADS=2 python -m experiments.diffusion_gemma_value_aware_followup.distribution_diagnostics
```

These commands use only completed calibration/development shared-state sources;
they do not read final scores, generate outputs, or change scientific policies.
`diagnostics/` exports same-budget risk comparisons, candidate-count-weighted
value-signal differences and streaming information/cost requirements. The
RMS-normalization explanation is checked against the installed model source.
`distribution_diagnostics/` exports count-weighted Gaussian moments, corrected
Sol target-versus-actual masks, and layer/step mass-bound/worst-row summaries.
Saved per-bucket quantiles are not mislabeled as pooled distribution quantiles.
Neither set of diagnostics establishes end-to-end accuracy gains.

If a transient failure leaves calibration incomplete while the original queue
continues, `recover launch --after-queue ACTUAL_QUEUE_PID` can schedule one bounded
retry behind that queue. It checks the real PID/start-time dependency every900s,
then calls the unchanged calibration stage only for methods with missing policies
and resumes development with all completed matching shards cached. It uses the
same H100 lock and idle check. It does not delete failures, modify scientific
settings, automatically freeze final candidates, or retry indefinitely.

After a canonical final report passes its raw-shard and derived-artifact audit,
run the independent synthesis without inference or policy selection:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. OMP_NUM_THREADS=2 python -m experiments.diffusion_gemma_value_aware_followup.synthesis --phase full
```

`synthesis/full/` adds calibration-to-final shifts and cap-one failures,
value/no-value and mass-only controls matched on actual budgets with paired
intervals, explicit local/global budget mismatches, and point-score-preserving
observations that are not mislabeled as statistical equivalence. It verifies
raw source and artifact hashes before reading scores. `--phase broad50` is
allowed only as an explicitly interim report. Actual empirical conclusions,
any paired-seed follow-up, and the requirement-by-requirement completion audit
still require inspection of the completed full experiment.
