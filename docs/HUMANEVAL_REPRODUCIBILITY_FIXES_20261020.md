# HumanEval reproducibility: observed failures, corrections and verification

Updated 2026-10-20 (Hong Kong). Reviewed repository state: `09877bcb`.
The corrected K256 and K128 experiments both use scientific deploy
`435511e43cf186a836761667f36fbf4f0f6061ec`. This document explains the existing
fixes and completed evidence; it introduces no new runtime or experiment.

The corrected panels reproduce outputs and native denoising trajectories across
audit and clean passes for all 164 problems in both thinking modes. The original
inconsistency required **two distinct numerical corrections**. A separate job
ownership repair made interrupted execution resumable.

## 1. What was inconsistent, and why a seed was insufficient

Earlier identical-question, identical-seed trials could produce different
generated text, denoising counts, canvas counts and occasionally correctness.
Some repeats matched exactly within a process while independent processes
diverged. A later operator replay also captured intermittent variation from
identical Q/K/V within the mass-observing attention path. It was never established
that every repeat would differ.

A seed controls random-number generation. It does not, by itself, fix compiler
kernel choices or ensure that a GPU kernel returns the same output every time.
Finite-precision addition depends on grouping: two parallel reductions can
round differently even when their mathematical formulas and inputs agree.

DiffusionGemma feeds continuous predictions into later self-conditioning and
uses thresholded acceptance and stability/entropy stopping. A small initial
difference can therefore alter a later decision. Once trajectories diverge,
they can consume different amounts of random noise and stop at different calls.
The amplification explains how small operator differences can lead to large
changes in total calls or final answers; it does not make every observed method
difference a numerical artifact.

The investigation compared loaded model/input identities, initial noise and
native trajectories, and qualified diagnostic observers against unobserved
repeats. Audit synchronization alone was not established as the cause. Original
unqualified rankings remain historical and cannot be combined with corrected
results as if they shared one runtime identity.

## 2. Correction A: explicitly bind model and helper compilation

### Reproduced mechanism

The same generated fused normalization kernel, replayed on identical saved
inputs, produced different rounded outputs when only its reduction block size
changed from `R0_BLOCK=512` to `1024`; `XBLOCK=1` and eight warps stayed fixed.
Each individual configuration repeated exactly.

| Valid output | Changed elements / 481,536 | Maximum absolute change |
|---|---:|---:|
| Layer output, BF16 | 11 | 0.0078125 |
| Internal normalized state, FP32 | 86,596 | 0.0001220703125 |
| Input to next Q/K/V projection, BF16 | 10 | 0.03125 |

This establishes one concrete numerical mechanism. The first-segment observer
was diagnostic, and a failed pointer-binding attempt was excluded from native
trajectory claims. Reduction replay alone did not establish complete repair.
See the [mechanism record](HUMANEVAL_REDUCTION_MECHANISM_20261019.md) and
[sanitized operator evidence](../results/humaneval_reproducibility_20261019/attempt001/operator_diagnostics/attempt002/summary.json).

### Implemented change

The named compiler policy explicitly sets deterministic Inductor compilation
for the model and separately recompiles these original native Python helpers
with the same numerical options:

- `_softcap_logits`
- `_compute_num_rejected`
- `_compiled_sample_step`

The helpers keep their original Python callable identity and formulas. The
qualified policy disables horizontal combo fusion and combo benchmarking,
retains native FULL/PIECEWISE CUDA graph modes, and records actual generated
kernel metadata and effective compilation settings.

The environment-only `TORCHINDUCTOR_DETERMINISTIC=1` attempt still reported 40
non-deterministic kernels. Applying options only to the model left separately
compiled helper kernels uncontrolled. These failed attempts are preserved; the
successful baseline qualification checked 53/53 recorded kernels rather than
accepting an environment variable as proof.

Implementation: [compiler policy and native helper binding](../scripts/v31_humaneval_repro_policy_worker.py).
Receipts are checked by the [corrected campaign coordinator](../scripts/v31_humaneval_cfgdet_study.py).
The successful compiler baseline did not eliminate every mixed-method mismatch;
the next correction was also required.

## 3. Correction B: disable overlap only in mass-observing FA4

### Reproduced mechanism

An 8,192-call operator screen used identical captured Q/K/V at two context
lengths, two independent processes and reversed factor order. Each factor cell
had 512 calls and 510 nontrivial comparisons against its process-local baseline.
For context length 3,532:

| Attention path | Splits | Overlap | Changed comparisons |
|---|---:|---|---:|
| Mass-observing FA4 | 16 | enabled | 4/510 |
| Mass-observing FA4 | 1 | enabled | 4/510 |
| Mass-observing FA4 | 16 | disabled | 0/510 |
| Mass-observing FA4 | 1 | disabled | 0/510 |

All tested paths at length 2,252 were stable. Across both lengths, all 2,048
mass-observing calls with overlap disabled were stable, as were all 4,096 plain
calls. Attention-mass auxiliary bytes matched for all 4,096 observing calls;
the variation was in the attention output. The first automatic-split discrepancy
changed 2,322 BF16 elements, with maximum absolute difference 0.0625. No model
or sampler forwards were run in this screen.

This isolates an observed failure to the customized mass-observing overlap
path. The exact failing instruction/register/shared-memory mechanism remains
unproved. The counts are conditional replay observations, not independent
estimates of a universal failure probability.

### Implemented change

The [observation-only wrapper](../scripts/v31_humaneval_observe_no_overlap.py)
sets `intra_wg_overlap=False` only for recognized mass-observing FA4 calls.
It rejects an unknown auxiliary layout or a conflicting explicit overlap
setting, preserves plain FA4 arguments, and records the policy hash, observed
call count, disabled-overlap setting and zero extra model forwards.

The qualified wrapper SHA-256 is
`f13970b24c6f8f27c09ba38ea882368aba9db8be9abf40833f914c44cc1a9e63`.
See the [complete operator screen](../results/humaneval_fa4_overlap_screen_20261019/attempt001/publication/attempt001/summary.md).
Afterward, two independent 152-request native diagnostic engines passed all
72 complete sampler/cache trace comparisons and observed/pure output and
trajectory comparisons before the corrected panels were released.

## 4. What the corrections preserve, and what latency includes

The corrected experiment retains the model, attention mask fix corresponding
to upstream PR #51994, native proposal/acceptance/re-noising, self-conditioning,
temperature schedule, stopping and output limits. It adds no model forwards to
force agreement. Fixed-budget methods keep the same per-query-head/Q128-tile
prefix budget; V2 remains an explicitly variable-threshold method.

Actual arithmetic execution changes, so the correction has a new source/runtime
identity and newly matched controls. FULL, PIECEWISE and all-kept FA4 remain
distinct consumers; reproducibility within a method does not require different
consumers to generate identical outputs.

Clean request latency includes the compiler policy, corrected observing FA4,
selector work and full-value correction work. Accounting/oracle instrumentation
runs separately in the audit pass. No isolated before/after correction speed
comparison is established by this panel, and reducing model calls alone does
not establish speedup.

## 5. Completed verification on both budgets

The [combined publication](../results/humaneval_corrected_k256_k128_20261020/attempt001/summary.md)
contains all 164 tasks, seed 42, thinking on/off, at K256 and K128. For **each**
budget:

- All 22 full audit/clean comparisons pass: 18 method-label/mode comparisons
  plus four V31 before/after comparisons across the two passes and modes.
- Output hashes, private completion text, denoising forwards, canvases, output
  lengths, sampler-call counts and native emission intervals agree.
- Independent engine-prefix replication passes. This is bounded independent
  replication, not a second independent full-164 accuracy trial.
- Independent official rescoring agrees on 5,904 main and 432 qualification
  records, with zero mismatches and zero timeouts.
- Clean measured CUDA capture and monitored JIT deltas pass their required
  zero checks. Actual source, scope, budget and numerical-policy receipts are
  verified before accepting the results.

The eight scientific methods plus the extra V31 technical bookend explain the
nine labels. Audit/clean passes and shared dense controls are technical repeats,
not independent accuracy samples. The three budget-independent dense controls
are shared across K256/K128; selector arms are regenerated at each budget.

Certificates: [K256 equivalence](../results/humaneval_corrected_k256_k128_20261020/attempt001/k256/full164_audit_clean_equivalence.json),
[K128 equivalence](../results/humaneval_corrected_k256_k128_20261020/attempt001/k128/full164_audit_clean_equivalence.json),
[K256 rescoring](../results/humaneval_corrected_k256_k128_20261020/attempt001/k256/independent_rescoring.json),
[K128 rescoring](../results/humaneval_corrected_k256_k128_20261020/attempt001/k128/independent_rescoring.json),
and [independent combined readback](../results/humaneval_corrected_validation_20261020/attempt001/receipt.json).

## 6. Separate operational repair

The interrupted K256 calibration had a detached monitor that did not own its
coordinator. The coordinator exited; its exact exit cause was not recorded.
A persistent user service now owns the queue, both coordinators and workers,
with bounded jobs, 15-minute process/GPU snapshots and fail-closed completion
gates. Successful leaves were preserved, the interrupted attempt was retained,
and continuation created a fresh leaf attempt. This repairs job ownership and
resumability; it is separate from the numerical corrections.

The completed continuation recorded 32 snapshots, at most one observed GPU
worker and a maximum snapshot interval of 901 seconds. Its completion is
recorded in the combined publication; this documentation does not resume work.
See the [durable continuation design](HUMANEVAL_DURABLE_RESUME_20261020.md).

## 7. Boundaries on conclusions and future comparisons

The earlier LongBench-v2 investigations also recorded unexplained repeat/audit
differences. The HumanEval mechanisms are plausible contributors, but no
corrected-runtime LongBench replay here establishes that they caused those
historical discrepancies. Keep the older results and their interpretation
limits intact; a future attribution study needs identical-case old/corrected
runtime contrasts and source-matched controls.

The verified HumanEval equality is empirical evidence for these pinned runs,
not a proof of determinism for every shape, seed, GPU or future software version.
Seed 43 covers only six previously examined problems per method/mode.

For future studies, bind both correction policies and runtime/source identities;
qualify identical-question/seed independent repeats and audit/clean equivalence;
check actual kernel/graph/budget/scope receipts; and retain source-matched dense,
all-kept and V31 controls. Do not substitute operator stability, equal final
correct counts or a single successful repeat for those gates.

Detailed accuracy, calls, clean latency, physical work, uncertainty and tails are
in the [HumanEval results report](../results/humaneval_documentation_20261020/attempt001/README.md).
