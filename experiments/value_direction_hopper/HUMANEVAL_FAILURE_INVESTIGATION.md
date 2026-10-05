# Why 50% sparse attention loses on HumanEval

This investigation uses the completed 100-question HumanEval v2 run at seed
42, with separate tile-count calibration and one frozen local/global threshold
pair per method. The scorer is the same isolated HumanEval execution path for
all methods: each completion is stripped, statically checked, and run in a
fresh Python subprocess against the task tests. The run has a valid
`final_complete.json` marker.

## Observed result

| Method | Pass@1 | Whole sparsity | Mean calls/canvas | P90 calls | Syntax errors | Capped canvases |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 92% | 0% | 5.40 | 12.0 | 1 | 0 |
| Gaussian32 | 88% | 49.50% | 9.63 | 26.6 | 4 | 1 |
| C_gate | 87% | 49.47% | 8.45 | 20.1 | 5 | 1 |

The whole/local/global sparsities are all within one percentage point of the
50% target. The paired 100-task bootstrap intervals for sparse minus dense
accuracy are Gaussian32 -4 pp [-10, +1] and C_gate -5 pp [-10, 0]. This is
directional evidence, not a final population estimate, but it is not explained
by a gross sparsity mismatch.

## The extra calls are a symptom, not a correctness mechanism

The C_gate and Gaussian32 trajectories use more calls, but those calls do not
repair the wrong programs. The dense-pass/sparse-fail groups show this
directly:

| Method | Dense-pass/sparse-fail tasks | Dense calls in group | Sparse calls in group | Mean sparse output prefix agreement with dense |
|---|---:|---:|---:|---:|
| Gaussian32 | 6 | 9.83 | 39.50 | 42.7% |
| C_gate | 6 | 10.50 | 25.67 | 35.8% |

For dense-pass/sparse-pass tasks, prefix agreement is about 82% for both
methods. The failures therefore correspond to large trajectory changes rather
than harmless alternative implementations. Several examples have a very long
tail:

- Gaussian32: `strange_sort_list` 52 calls with an unclosed parenthesis,
  `minPath` 81 calls with an unmatched parenthesis, and `generate_integers`
  46 calls with a missing colon.
- C_gate: `sort_array` 38 calls with an unclosed parenthesis and `minPath`
  54 calls with an invalid branch.
- Other C_gate failures finish quickly but are semantically wrong: `words_string`
  takes 5 calls and uses an incorrect splitting rule; `max_fill` takes 6 calls
  and sums all rows instead of counting buckets; `count_nums` takes 8 calls
  and mishandles negative values.

The stopping diagnostic is not detecting these failures. Failed C_gate programs
end with mean entropy about 0.002 and failed Gaussian32 programs about 0.002;
most have zero final top-1 flips. The sampler is stable relative to its own
trajectory, while the trajectory has already settled on the wrong code.

### Tail concentration and the meaning of a stable ending

The completed per-question traces quantify both observations. Here `steps` is
the total number of denoising calls across all canvases for one completion.

| Method | Failure rate | Mean steps, passes | Mean steps, failures | Failure share of all steps | Top five failures' share of all steps |
|---|---:|---:|---:|---:|---:|
| Gaussian32 | 12/100 | 9.10 | 33.58 | 33.5% | 24.2% |
| C_gate | 13/100 | 8.00 | 25.15 | 32.0% | 23.4% |
| T_prior | 16/100 | 10.38 | 45.25 | 45.4% | 24.6% |

The tail is therefore real: roughly one eighth to one sixth of the questions
produce about one third to one half of all denoising work. It is not the whole
explanation, because passing sparse cases also take more calls than passing
dense cases. For C_gate, the ten largest completions account for 30.6% of all
calls and 95.7% of calls spent on failed questions; for Gaussian32 the
corresponding numbers are 32.5% and 97.0%.

The final stopping state is also not a correctness signal. Among failed
questions, C_gate has 12/13 with zero final top-1 flips, 12/13 with native
stopping on the last canvas, and 12/13 with final entropy below 0.01. The
corresponding Gaussian32 counts are 11/12, 11/12, and 11/12. The exceptions
are the capped malformed tails (`H132` for C_gate and `H129` for Gaussian32),
which remain visibly unstable. For comparison, all ten BLASST failures and all
eight dense failures also finish with zero flips and native stopping. Stability
means that the sampler has converged to its own trajectory; it does not mean
that the program passes its tests.

There are two distinct mechanisms:

1. **Early semantic commitment.** A sparse attention perturbation changes an
   identifier, operator, or branch choice. Later confidence becomes high and
   native stopping accepts the wrong implementation. Additional calls polish
   the wrong branch instead of providing a correctness signal.
2. **Continuation tail corruption.** Long functions often use two 256-token
   canvases. Once the first canvas has diverged, later sparse calls can produce
   verbose repairs, malformed indentation, or an unfinished delimiter. This
   creates extra calls and sometimes a cap hit, but the extra work is downstream
   of the original error.

This explains why mean call count alone is misleading. C_gate is faster than
Gaussian32 (8.45 versus 9.63), but it still loses accuracy because its errors
are irreversible semantic changes, not merely insufficient convergence.

## Why denoising helps AIME but not code

The AIME comparison provides a useful counterfactual with the same native
denoiser and the same uniform-threshold idea:

| Method | Accuracy | Mean calls |
|---|---:|---:|
| Dense | 46.67% | 14.97 |
| Gaussian32 | 51.11% | 19.12 |
| C_gate | 54.44% | 18.63 |

On AIME, C_gate converts 14 dense-fail seed/question cases into correct
answers and loses 7 dense-pass cases. The dense-fail cases are genuinely hard
and use more calls even in dense decoding; C_gate's extra calls can therefore
change the reasoning path and discover a better final number.

On HumanEval, C_gate converts only one dense failure into a pass and changes
six dense passes into failures. The dense reference is already at 92%, so most
available cases have no headroom. The evaluator is exact pass@1: one malformed
delimiter or one wrong edge-case branch makes the whole completion fail. There
is no verifier in the denoising loop that can tell the sampler which code token
is wrong. Extra calls respond to instability, but they do not supply program
tests or a syntax constraint.

The difference is therefore mechanistic. AIME has long reasoning with many
redundant textual paths and a single final numeric decision; code generation
has coupled discrete constraints across indentation, delimiters, names, and
edge-case branches. Once a sparse update changes one of these, later
confidence is not evidence of correctness.

The archived RULER decomposition is consistent with the same mechanism. At the
reported 70% Gaussian32 point, retrieval-heavy tasks remain mostly 90--100%
accurate, while CWE falls to 77% and the QA tasks to 70--80%. Repeated keys and
answer spans provide redundancy; code has little redundancy for exact syntax
and control flow. The aggregate RULER run itself is below dense, so this is a
task-level diagnostic rather than a claim of a universal RULER win.

Changing the value projection alone is unlikely to be sufficient. The AIME
rank sweep at 50% gives dense 56.67%, full-centered 50.00%, and Gaussian32
50.00% on the same 30 prompts. A full-dimensional value route is still useful
as a HumanEval isolation control, but the AIME result shows that trajectory
and discrete generation sensitivity remain separate from projection rank.

## Reassessment after the full-cohort confidence experiments

The two proposed explanations are supported, with an important qualification.

### Long tails dominate sparse-call inflation, but do not explain the accuracy loss alone

On the 100-question seed-42 cohort, C_gate has 13 failed completions. They
consume 326 of 1,023 calls, or 32.0% of all calls, and average 25.15 calls
versus 8.00 for the passing completions. The five longest failures consume
23.4% of all calls. Relative to the matched dense run, four examples
(`H132`, `H106`, `H116`, and `H129`) account for 149 of the 179 additional
calls on C_gate failures. The tail is therefore the main cause of the mean
inflation for the sparse failures.

It is not the complete gap to dense. Passing C_gate examples average 8.00
calls versus 6.20 on dense passing examples. The sparse method therefore has a
smaller, broad convergence cost in addition to the concentrated failure tail.
The mean is not evidence that extra calls improve correctness: the longest
trajectories often end in code that still fails.

The same pattern is stronger for Gaussian32 and T_prior: failed completions
consume 33.5% and 45.4% of all calls, respectively. This confirms the first
hypothesis as a robust property of the sparse trajectories rather than an
artifact of one query formula.

### Stable endings are common among wrong outputs

The second hypothesis is also supported. Among failed C_gate completions,
12/13 end with zero top-1 flips, native stopping, and final processed entropy
below 0.01. The corresponding counts are 11/12 for Gaussian32, 14/16 for
T_prior, and 12/12 for the full-cohort `C_gate_rank` run. Dense failures are
also stable at the end (8/8), so this is not a sparse-only stopping bug.

The correct interpretation is that the denoiser has converged to its own
trajectory. It is not a correctness certificate. For example, C_gate's
`H106`, `H116`, and `H129` failures are stable at the end, but their sparse
outputs contain a wrong arithmetic branch or malformed syntax. A later
confidence threshold cannot recover information that was removed by an earlier
attention update.

The early signal does contain useful but limited information. C_gate failures
that are dense-pass/sparse-fail have mean first-call confidence 0.485 versus
0.620 for dense-pass/sparse-pass tasks; the question-level low-confidence
AUROC is about 0.68. At the second call the means are 0.674 versus 0.812.
This is enough to rank risk, but not enough to justify a persistent debt for
all low-confidence rows.

### What the confidence experiments ruled out

The full 100-question evaluations are matched to approximately 50% physical
sparsity with one frozen local/global threshold pair:

| Method | Pass@1 | Whole sparsity | Mean canvas calls | P90 total calls | Capped canvases |
|---|---:|---:|---:|---:|---:|
| Dense | 92% | 0% | 5.40 | 12.0 | 0 |
| C_gate | 87% | 49.47% | 8.45 | 20.1 | 1 |
| C_selective (fixed memory update) | 84% | 50.33% | 11.93 | 48.0 | 8 |
| C_soft, alpha=0.25 | 80% | 50.20% | 15.17 | 49.1 | 15 |
| C_gate_rank, lambda=.05 | 88% | 49.41% | 9.38 | 26.5 | 0 |

`C_selective` confirms that an EMA initialized from the first observed
uncertainty is still too persistent when combined with a max hazard. `C_soft`
confirms that removing the gate and exposing confidence ranking immediately
after the first call destabilizes the trajectory. `C_gate_rank` with a weak
rank residual is close to C_gate but still has a larger tail. None meets the
near-dense call target (<7 mean calls) or dense accuracy.

A bounded first-call tail pulse was then tested. On the balanced 30-question
development cohort it reached 93.33% with 6.50 mean total calls and no caps,
but on all 100 questions it returned to 87% and increased the mean to 12.09
total calls (9.45 calls per canvas) with 11 failures. It gained three C_gate
failure cases and lost three C_gate successes; the three losses added long
tails of 58, 60, and 61 calls. Early uncertainty alone therefore does not
identify which trajectory needs protection.

### General next direction

The evidence now favors a confidence-*progress* gate, rather than an
uncertainty debt or a first-call-only pulse. A causal state should lift the
C_gate hazard only when uncertainty fails to improve after a completed call;
queries whose confidence is recovering should immediately return to the
ordinary C_gate trajectory rule. This avoids the pulse's false positives while
still targeting the long tails whose first two calls remain uncertain.

One implementation is a bounded negative-progress signal:


\[
u_t=\sqrt{1-p_t},\qquad
r_t=\operatorname{clip}\left(\frac{u_t-b_t}{d_t+\epsilon},0,1\right),
\]

where `b_t` and `d_t` are causal robust location and scale estimates from the
completed calls in the same canvas. Let `p_t` be a short EMA of positive
uncertainty change, for example

\[
p_t=\rho p_{t-1}+(1-\rho)\operatorname{clip}
  \left(\frac{u_t-u_{t-1}}{d_t+\epsilon},0,1\right),
\]

and add only a bounded lift:

\[
h_t= h^{C\_gate}_t + \lambda p_t(1-h^{C\_gate}_t).
\]

This is causal and benchmark-agnostic. It responds to stalled or worsening
confidence, while confidence recovery removes the extra sensitivity without a
stored debt. It uses no token names, syntax tests, benchmark labels, or
separate thresholds. Acceptance requires fewer dense-pass/sparse-fail cases,
mean calls below seven, and no new cap tail on the full 100-question cohort.

The important experimental distinction is to separate an early trajectory
intervention from a stopping intervention. Confidence at the final call is
not a valid recovery signal, so changing native stopping criteria would not
address this failure mode.

The raw traces and complete-cohort reports are in
`results/humaneval_query_sensitivity_v2/` and
`results/humaneval_query_sensitivity_confidence_v10/`; the corrected memory
and confidence-blend controls are in `results/humaneval_query_sensitivity_confidence_v11/`
and `results/humaneval_query_sensitivity_confidence_v12/`.

## Next iteration: causal kind-conditioned progress guard

The failure split gives a sharper target than an uncertainty-only formula.
Among C_gate outputs that fail, the first two completed calls are both less
settled and more sparsely routed than for successful outputs:

| C_gate subset | Call 1 confidence / entropy / sparsity | Call 2 confidence / entropy / sparsity | Late sparsity |
|---|---:|---:|---:|
| Pass (87) | .619 / 1.367 / 45.5% | .812 / .661 / 41.6% | 42.8% |
| Fail (13) | .510 / 1.946 / 49.2% | .675 / 1.323 / 45.4% | 50.5% |

Thus the long tail is not caused only by a stable final wrong answer. Hard
canvases enter call 2 with a persistent confidence deficit, receive a less
effective attention mask, and remain expensive. The higher sensitivity values
on these canvases do not guarantee denser physical routing: the difficulty is
not being represented by the current query score at the tile decision.

The available attention audit identifies a second, orthogonal mismatch. For
Gaussian32 on RULER4K, sampled local-attention relative L2 error rises from
0.257 at call 1 to 0.424/0.442 at calls 2/3 (layer-0 cosine falls to
0.899/0.894), while global error is 0.232/0.191/0.197. The BLASST control has
the opposite allocation (global error about 0.29--0.31, local 0.08--0.12),
so this is evidence for a method/attention-kind interaction, not a universal
claim that local layers always need more tiles. These measurements are sampled
RULER diagnostics; a C_gate-specific probe is required before setting any
fixed local/global preference.

The next method should therefore retain C_gate's causal base and add a bounded
progress signal that is separately normalized for local and global attention:

\[
u_t(q)=\sqrt{1-p_t(q)},\qquad
g_t(q)=\operatorname{EMA}\!\left[\operatorname{clip}
  \left(\frac{u_t-u_{t-1}}{\sigma_{t,k}+\epsilon},0,1\right)\right],
\]

where `k` is the attention kind and the update uses only the completed prior
call. The router should expose a causal row-level operator-risk proxy for that
call, initially the fraction of rejected KV tiles (and, when available, the
near-threshold rejected contribution). Let `a_{t,k}(q)` be its within-kind
robust percentile/EMA. The sensitivity used on the *next* call is

\[
h_{t,k}(q)=h^{C\_gate}_t(q)+\lambda\,g_t(q)a_{t,k}(q)
  \bigl(1-h^{C\_gate}_t(q)\bigr),\qquad
s_{t,k}=1+\beta h_{t,k}.
\]

Call 1 remains `s=1+beta`; call 2 is driven by call-1 observations. There is
still exactly one frozen local and one frozen global log threshold for all
calls. The kind-specific state changes query ranking, not the threshold
schedule. A robust percentile is preferable to a raw skip count because it
prevents one difficult layer or one unusually long canvas from creating a
permanent debt. If the probe shows no predictive value, remove `a` and test
the scalar confidence-progress guard alone.

### Required ablation order

1. Instrument dense, C_gate, and Gaussian32 on a balanced 30-question
   HumanEval slice and a small RULER slice. Record per-call confidence,
   entropy, accepted/renoised counts, row skip fraction by attention kind,
   near-threshold mass when available, and subsequent-call confidence change.
2. Test whether call-1/2 local/global row statistics predict the next-call
   confidence deficit and the dense-pass/sparse-fail split. This is the gate
   for adding a kind-conditioned term.
3. If predictive, evaluate `C_kind_progress` at matched 50% sparsity with
   seed 42 first. Compare C_gate, Gaussian32, and dense on accuracy, mean and
   P90 calls, caps, and phase sparsity. Promote to the 100-question and
   multi-seed runs only if mean calls move toward 7 without creating a new
   long tail.
4. In parallel, run the retained-state BLASST replay already specified in
   `BLASST_VS_VALUE_AWARE_INVESTIGATION.md`; it is a separate router-semantic
   control and must not be folded into the query-score result.

The success criterion is paired: recover at least the dense accuracy on the
sample without increasing the C_gate long-tail rate, while keeping mean calls
below seven. A score that merely increases sensitivity everywhere is not a
successful iteration because the current failures already have higher
physical sparsity despite higher sensitivity.
