# HumanEval BLASST versus value-aware audit

## Scope

This audit examines `results/humaneval_query_sensitivity_v1`, the seed-42
HumanEval pilot. It contains 24 fixed HumanEval tasks: six calibration tasks
and 18 disjoint evaluation tasks. Every sparse method uses one calibrated
local and one global threshold pair for all denoising calls. The scorer runs
the official HumanEval tests in an isolated Python subprocess, with a five
second timeout. Raw predictions and execution results are retained beside the
summary.

## What the run establishes

The observed pilot scores are:

| method | pass@1 | overall physical sparsity | local | global | mean canvas steps |
|---|---:|---:|---:|---:|---:|
| dense | 17/18 (94.4%) | 0.0% | 0.0% | 0.0% | 6.04 |
| BLASST | 16/18 (88.9%) | 55.0% | 55.2% | 53.9% | 14.86 |
| T_prior | 13/18 (72.2%) | 55.2% | 56.7% | 47.8% | 18.16 |
| C_gate | 13/18 (72.2%) | 57.9% | 59.1% | 51.5% | 21.13 |
| Gaussian32 | 7/18 (38.9%) | 57.3% | 57.4% | 57.2% | 29.29 |

BLASST is close to T_prior in overall sparsity. C_gate and Gaussian32 are
2.9 and 2.3 percentage points sparser than the requested 50% target in the
actual evaluation. Therefore the BLASST-versus-C_gate comparison is not a
matched-budget comparison; BLASST-versus-T_prior is the closest comparison in
this pilot.

The paired question outcomes are particularly simple:

* BLASST passes three questions that C_gate misses (HumanEval/70, /108, and
  /163), with no reverse wins.
* BLASST passes three questions that T_prior misses (/108, /151, and /163),
  with no reverse wins.
* Both sparse query methods fail /50 and /129, so those failures are not a
  BLASST-versus-query separation.

Thus the reported 16.7 percentage-point gap is a real property of this exact
18-question, one-seed run. It is not a reliable estimate of a population
accuracy gap. The paired difference is only three binary outcomes; an exact
sign test has one-sided probability 1/8 after ignoring the 15 ties. A paired
bootstrap percentile interval is [0, 33.3] percentage points because the
sample contains only three nonzero differences.

## Failure mechanism visible in the raw outputs

The failure statuses explain most of the gap:

| method | passed | failed tests | syntax errors |
|---|---:|---:|---:|
| BLASST | 16 | 2 (/50, /129) | 0 |
| C_gate | 13 | 3 (/50, /70, /108) | 2 (/129, /163) |
| T_prior | 13 | 2 (/50, /129) | 3 (/108, /151, /163) |
| Gaussian32 | 7 | 1 (/50) | 10 |
| dense | 17 | 1 (/50) | 0 |

The BLASST failures are executable programs that fail an official test. The
extra C_gate and T_prior failures are mostly malformed or incomplete Python:
an unclosed loop/body, an unclosed parenthesis, or an invalid expression. For
example, C_gate's /70 completion ends with a malformed function body, C_gate's
/163 completion is not parseable, and T_prior's /108, /151, and /163 outputs
are parse errors. BLASST supplies parseable code on all of those questions.

The trajectory statistics support the same diagnosis. BLASST hit no canvas cap
and averaged 14.86 denoising calls per canvas. C_gate hit four caps and
averaged 21.13 calls; T_prior hit three caps and averaged 18.16 calls. Their
failed completions are also much longer than their successful completions on
average (C_gate: about 1,037 versus 509 characters; T_prior: about 1,305
versus 513). The long failed outputs contain model self-corrections and
unfinished drafts. This is evidence of a sparse-trajectory stability problem,
not evidence that the sensitivity coefficient is accurately identifying more
important code queries and simply making a better tradeoff.

This is a plausible HumanEval-specific mechanism. Pass@1 is discontinuous:
one missing colon, bracket, indentation level, or final return gives score
zero even when the semantic part of the function is mostly correct. The code
prompts also have strong lexical and indentation continuity. A router that
preserves the high-scoring query-key scaffold for punctuation, delimiters,
and function structure can therefore win pass@1 without having lower general
attention error. The value-aware and query-adaptive routers use projected
value-direction scores and causal confidence/trajectory weights. Those scores
can remove a tile whose projected contribution is modest while it still
supports a low-frequency syntax token. More denoising calls do not repair such
an irreversible token mistake; C_gate and T_prior actually use more calls on
average than BLASST in this run.

The opposite pattern was observed on AIME26 and RULER, where semantic answer
tokens and long-context retrieval make value-direction selection more useful.
This does not imply that either router is universally better. It indicates
that the current value score is not measuring the same thing as a code-output
syntax-preservation score.

## Protocol limitations

The result is auditable but not yet a trustworthy general comparison:

1. There are only 18 evaluation tasks and one decoding seed.
2. Calibration uses six tasks and selects by overall sparsity only; local and
   global calibration errors are reported but not constrained.
3. The evaluation sparsities are not identical, especially for C_gate.
4. The subset is a random pilot rather than the full 164-task HumanEval set or
   a task-balanced repeated sample.
5. The dense completions are reused from the archived native run, so a fresh
   same-process dense replay would be useful as a determinism check.

The official executor and raw completions make the numerical result itself
credible for this run. The limitations prevent the result from proving that
BLASST is better than value-aware routing on HumanEval in general.

## Decisive follow-up

The next experiment should separate score mechanism from budget and sampling
noise:

1. Use at least 60--100 fresh HumanEval tasks, or the full set if the run is
   acceptable, with seeds 42, 43, and 44. Keep calibration disjoint and freeze
   one local/global pair per method.
2. Select and report operating points using actual overall, local, and global
   sparsity, with all three within the same tolerance. Compare BLASST and
   T_prior at the same measured tile budget before interpreting accuracy.
3. Add code-quality diagnostics: parse success, function extraction success,
   output length, first syntax-error location, failed-test category, and token
   agreement with the dense completion. This exposes whether a method loses on
   syntax before semantic tests are reached.
4. Run a same-state teacher-forced replay on the same prompts. At a fixed tile
   count, compare QK/BLASST and projected-value/query-adaptive masks and record
   corruption of punctuation, indentation, delimiters, and the final return
   token. This tests the proposed mechanism without changing denoising call
   count or future state.
5. Ablate BLASST's state convention and score separately: retained-history
   state versus recomputed dense-prefix state, and QK selection versus value
   selection. If BLASST's HumanEval advantage disappears under the same state,
   the issue is trajectory state rather than query sensitivity.

No formula change should be selected from the current 18-question result
alone. The immediate target is to establish whether syntax preservation is a
real signal that should be added to C_gate/T_prior, or merely a small-sample
artifact.
