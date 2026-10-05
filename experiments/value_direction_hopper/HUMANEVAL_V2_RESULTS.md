# HumanEval v2: matched-sparsity investigation

This report supersedes the earlier 18-question pilot interpretation. It uses
100 fixed HumanEval prompts, seed 42, six balanced task strata, and one frozen
local/global log-threshold pair per sparse method. The thresholds were selected
from tile counts only and then applied unchanged to every denoising call.

## Protocol and calibration

- Calibration: 12 prompts, two per stratum; the calibration manifest is saved
  in `results/humaneval_query_sensitivity_v2/calibration_complete.json`.
- Final evaluation: 100 prompts, with 8 parsing/format, 18 string/text, 18
  search/optimization, 18 collections/sequences, 12 predicate/logic, and 26
  numeric/logic tasks.
- The final budget-match pass uses the 100 final inputs to choose a threshold
  pair using sparsity only. This is transductive budget matching, not a fully
  held-out calibration. The earlier disjoint calibration results remain
  archived separately.
- All sparse methods match whole, local, and global physical sparsity within
  1 percentage point; no accuracy labels were used to select thresholds.

| Method | Local log threshold | Global log threshold | Whole | Local | Global | Max calibration error |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian32 | -0.621274 | -0.704153 | 49.50% | 49.48% | 49.60% | 0.53 pp |
| BLASST | 1.251236 | 1.362464 | 50.31% | 50.24% | 50.64% | 0.65 pp |
| C_gate | 0.578833 | 0.488666 | 49.47% | 49.46% | 49.53% | 0.54 pp |
| T_prior | 0.403323 | 0.315919 | 50.52% | 50.42% | 50.99% | 0.99 pp |

## Main result

| Method | Pass@1 | Mean calls/canvas | Median calls | P90 calls | Capped canvases | Syntax errors | Failed tests |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 92/100 (92%) | 5.40 | 5 | 12 | 0 | 1 | 7 |
| Gaussian32 | 88/100 (88%) | 9.63 | 8 | 26.6 | 1 | 4 | 8 |
| BLASST | 90/100 (90%) | 10.25 | 10 | 21.2 | 0 | 1 | 9 |
| C_gate | 87/100 (87%) | 8.45 | 6.5 | 20.1 | 1 | 5 | 8 |
| T_prior | 84/100 (84%) | 12.47 | 8 | 43.5 | 7 | 8 | 8 |

BLASST beats C_gate on 6 prompts and loses on 3 (paired delta +3 pp). It
beats T_prior on 7 and loses on 1 (+6 pp), and beats Gaussian32 on 4 and loses
on 2 (+2 pp). With 100 prompts and one seed, these are useful diagnostics but
not population-level significance claims. The exact paired lists are in
`results/humaneval_query_sensitivity_v2/paired_diagnostics.json`.

## Phase sparsity and trajectories

| Method | Call 1 whole | Call 2 whole | Later whole | Mean renoised positions/call | Mean flips/call |
|---|---:|---:|---:|---:|---:|
| Gaussian32 | 54.41% | 50.27% | 48.81% | 104.42 | 38.52 |
| BLASST | 56.51% | 52.59% | 49.28% | 114.17 | 42.54 |
| C_gate | 48.51% | 45.51% | 50.16% | 94.87 | 35.91 |
| T_prior | 43.97% | 43.67% | 51.73% | 83.77 | 24.95 |

The call-by-call renoising counts are the clearest mechanism signal:

- Dense: 183, 127, 67, 40, 31.
- BLASST: 211, 189, 171, 156, 122.
- C_gate: 195, 145, 104, 74, 65.
- T_prior: 194, 141, 101, 73, 78.

Thus the earlier statement that query-adaptive methods uniformly need more
denoising calls was incorrect. C_gate is faster than BLASST in this run
(8.45 vs 10.25 mean calls). T_prior is the method with a clear long-tail
problem: it relaxes many positions early, then reopens a tail late, producing
7 capped canvases and a P90 of 43.5 calls.

## Why T_prior is unstable

T_prior uses `h = q + (1-q)z`, where `q` is an EMA of the previous renoising
mask and `z` is the EMA of top-1 flips. The aggregate statistics look good
(higher confidence, lower entropy, and fewer average renoised positions), but
they conceal a delayed-feedback failure mode:

1. The trajectory term lowers sensitivity after a short apparently stable
   run.
2. Physical tile closure still routes a whole tile when any row remains
   sensitive, so a few missed code-critical rows can keep reopening tiles.
3. The delayed flip EMA then raises sensitivity after the sparse update has
   already perturbed the next logits. This is visible in the increase from 73
   to 78 renoised positions at calls 4 to 5 and in the cap-heavy tail.

This is a query-selectivity/feedback problem, not a calibration mismatch.
C_gate's confidence gate keeps initial sensitivity high until an accepted
stable run is observed. Its higher average sensitivity (3.19 versus 2.43 for
T_prior) costs more work per call but lets the trajectory settle sooner.

## Why the 18-question BLASST pilot had zero syntax errors

That observation was real for the pilot but not strong evidence of a syntax
preservation property. In the 100-question run BLASST has one syntax error,
the same count as dense; Gaussian32 has four, C_gate five, and T_prior eight.
The zero count in 18 prompts is compatible with a roughly 1% syntax-error rate
by ordinary sampling variation. The larger run also includes a difficult
stratified set and exposes a shared hard task (HumanEval/132) where BLASST and
C_gate both produce syntax errors. BLASST has no capped canvases and fewer
syntax failures than C_gate/T_prior here, which is a useful association, but
the data do not establish that BLASST's QK routing preserves syntax causally.

The safer explanation is that BLASST has a more stable **trajectory tail** on
this sample: its renoising count declines monotonically and it never hits the
call cap. That can reduce the opportunity for malformed late rewrites, while
remaining distinct from a guarantee of syntactic correctness.

## Corrected conclusion and next investigation

The v2 evidence supports three narrower conclusions:

1. Matched 50% sparsity removes the simple calibration explanation for the
   BLASST/C_gate gap, but the final threshold fit is still transductive.
2. C_gate is currently the best query-adaptive latency point: it is faster than
   BLASST and Gaussian32, while trailing them by only 3 and 1 percentage
   points respectively in this one-seed sample.
3. T_prior needs refinement before further accuracy comparison. The priority
   is to prevent late reopening without removing the causal first-call
   protection. A sensible next ablation is a bounded monotone trajectory term
   or a one-way decay of the flip EMA after a stable run, with the same frozen
   uniform threshold protocol and per-task cap diagnostics.

Raw outputs, manifests, thresholds, trajectories, and the completion marker are
under `results/humaneval_query_sensitivity_v2/`; the runner and report scripts
are `humaneval_query_sensitivity_v2.py` and
`humaneval_query_sensitivity_v2_report.py`.
