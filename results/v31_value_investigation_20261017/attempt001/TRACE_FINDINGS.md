# Bounded native first-divergence trace

The single exploratory trace completes with zero structural errors and494.47s
GPU process reservation. It uses question0/seed1 because that cell exhibited
two audit backend discrepancies in the crossed experiment. V31 before/after,
reference residual, and device residual are measured after the same four
explicit seed42 warmups used by the cross. This is post-hoc localization, not
an accuracy estimate. All native stopping/output budgets remain unchanged.

Evidence: [trace_analysis/attempt001/summary.json](trace_analysis/attempt001/summary.json)
and its binding marker. Private attention/state/cache tensors and generations
are retained in native_trace/attempt001/private/. Public outputs contain only
numeric comparisons and hashes.

## Observed mechanism

Call indices below are zero-based denoising calls within the request. Native
commit forwards are not denoising calls; they still consume sampler randomness.
All218 V31 calls and321 calls of each residual implementation were captured.

| Call | First observed change versus V31 |
|---:|---|
| 0 | No difference: initial Q, attention, logits, canvas, RNG and self-conditioning match. |
| 1 | Dense observation creates the correction cache; selected support and native output still match. |
| 2 | First held correction changes layer29 output, logits and self-conditioning. Q and support still match, all33 accepted rows match, and sampled canvas/RNG still match. Argmax differs on20 rows. |
| 3 | Q now differs despite the sampled canvas still matching. This is consistent with the captured self-conditioning divergence at call2. |
| 6 | Reconstructed acceptance first differs on5 rows (63 versus58 accepted); actual post-sampler canvas first differs on5 tokens. RNG still matches. |
| 8 | Re-observation produces the first different selected support. It follows earlier output/state divergence. |
| 18 | V31 becomes confident and schedules commit; residual does not. Mean entropy is0.000681 versus0.011837. |
| 19 | V31 has consumed a commit sampler call and begun the next canvas; residual is still denoising its first. CUDA RNG states first differ at the aligned denoising-call index. |
| 20 | Residual schedules its first commit:21 denoising calls for the first canvas versus19 for V31. |

The entropy-bound acceptance mask is reconstructed using the existing sampler
formula. Canvas, argmax, confidence, commit flags, self-conditioning fingerprints
and CUDA RNG states are captured from the actual native execution. Full-logit
hashes are captured for the first six calls only; later full-logit equality is
unknown. No model forward or random draw is added by the observer.

This is direct evidence that a value correction can change the trajectory
through continuous self-conditioning before accepted tokens or masks change.
It is also a counterexample to treating an initially unchanged acceptance mask
or slightly lower mean entropy as evidence of faster convergence: at call2 the
residual entropy is slightly lower (3.345 versus3.354), but its first commit is
later. The initial divergence is not caused by different RNG consumption; the
RNG offset changes after the stopping/commit schedules diverge.

| Method | Correct | N | C | N/C | T |
|---|---:|---:|---:|---:|---:|
| V31 before | Yes | 218 | 9 | 24.22 | 2215 |
| Reference residual | No | 321 | 13 | 24.69 | 3298 |
| Device residual | No | 321 | 13 | 24.69 | 3298 |
| V31 after | Yes | 218 | 9 | 24.22 | 2215 |

The extra103 calls include four additional canvases and somewhat slower
within-canvas convergence. This case does not establish a population accuracy
effect. Current dense attention was not separately recomputed on these traced
held calls, so the trace does not claim that current dense-output error was
reduced at the moment the answer trajectory worsened.

## What the trace resolves and leaves unresolved

Reference and device residual agree throughout all321 captured calls in Q,
attention-output hashes, support, sampler canvas/argmax, acceptance, commit
flags, self-conditioning and RNG. All26 paired correction-cache observations
match exactly in anchor-selected output, residual, dual and squared norm;
maximum tensor discrepancy is zero. Observation K/V fingerprints also match.
Both final output hashes match three of the four corresponding untraced trials.
The V31 bookends match each other and all corresponding untraced controls.

The intermittent cross-run residual discrepancy was not reproduced. Its cause
therefore remains unresolved; the trace does not prove general backend
equivalence or identify a deterministic cache-math defect. It does show that
the unfavorable residual trajectory occurs with the unchanged reference as
well as the optimized implementation, under identical captured states.

The full-swap native first-token divergence was not separately traced within
this budget. Source inspection places its first possible direct output effect
on a held FA4 call after map replacement; its observation returns the inherited
dense output. That is a source-derived boundary, not a measured first-divergence
call for every swap trajectory. Its16/16 crossed backend matches, complete
partial-cohort results and saved-state operator diagnostics remain the direct
evidence for the swap assessment.
