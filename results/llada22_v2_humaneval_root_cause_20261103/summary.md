# LLaDA2.2 V2 HumanEval failure analysis

## Disposition

The EOS-confidence-zero intervention is retired. It changes the sampler
stopping trajectory rather than improving V2 attention support, and raises
denoising work per canvas from 4.634 to 10.322. Its completed run is preserved
as a negative ablation only; it is not an active V2 method.

## The decisive trajectory contrast

The original LLaDA2.2 V2 HumanEval failure is primarily a fewer-canvases
failure, not an excessive-steps-per-canvas failure:

| LLaDA2.2 HumanEval arm | Accuracy | Mean T | N/request | C/request | N/C | Sparsity |
|---|---:|---:|---:|---:|---:|---:|
| Dense native | 85.98% | 243.1 | 28.8 | 8.57 | 3.357 | 0% |
| V31 mass ranking | 50.61% | 218.9 | 36.8 | 7.84 | 4.692 | 49.98% |
| V2 Gaussian32 | 6.10% | 62.9 | 14.2 | 3.07 | 4.634 | 49.22% |
| V3a singleton value | 62.80% | 245.3 | 42.3 | 8.67 | 4.881 | 49.99% |

V2's N/C is nearly identical to V31, while its canvases/request are 61%
lower. All 164 V2 requests end in EOS, none reaches the 1,024-token cap, and
98 have syntax errors. The sparse map is moving the LLaDA denoising state into
an early-stopping basin; this is not explained by 50% tile sparsity or by too
few steps within each canvas.

## Same LLaDA model, different benchmarks

| Benchmark | Arm | Accuracy | Mean T | C/request | N/C | Sparsity |
|---|---|---:|---:|---:|---:|---:|
| AIME26 | Dense | 40.00% | 7018.6 | 220.14 | 6.991 | 0% |
| AIME26 | V31 | 23.33% | 4691.7 | 147.46 | 8.972 | 50.00% |
| AIME26 | V2 | 31.11% | 4695.9 | 147.67 | 9.255 | 51.72% |
| RULER128K | Dense | 85.44% | 28.2 | 1.82 | 4.283 | 0% |
| RULER128K | V31 | 86.01% | 28.4 | 1.82 | 4.291 | 50.00% |
| RULER128K | V2 | 86.23% | 28.1 | 1.82 | 4.339 | 49.19% |
| HumanEval | Dense | 85.98% | 243.1 | 8.57 | 3.357 | 0% |
| HumanEval | V31 | 50.61% | 218.9 | 7.84 | 4.692 | 49.98% |
| HumanEval | V2 | 6.10% | 62.9 | 3.07 | 4.634 | 49.22% |

V2 does not generally terminate early: it runs about 148 canvases/request on
AIME and matches V31 on RULER canvases. HumanEval is the isolated collapse.

## Same HumanEval benchmark, different models

| Model / arm | Accuracy | Mean T | C/request | N/C | Sparsity |
|---|---:|---:|---:|---:|---:|
| LLaDA2.2 dense | 85.98% | 243.1 | 8.57 | 3.357 | 0% |
| LLaDA2.2 V31 | 50.61% | 218.9 | 7.84 | 4.692 | 49.98% |
| LLaDA2.2 V2 | 6.10% | 62.9 | 3.07 | 4.634 | 49.22% |
| LLaDA2.2 V3a | 62.80% | 245.3 | 8.67 | 4.881 | 49.99% |
| TraDo dense | 79.27% | 124.3 | 31.86 | 1.840 | 0% |
| TraDo V31 | 42.68% | 121.5 | 31.13 | 2.275 | 49.55% |
| TraDo V2 | 76.22% | 127.6 | 32.70 | 1.876 | 48.84% |
| TraDo V3a | 47.56% | 133.5 | 34.15 | 2.006 | 49.67% |

TraDo V2 retains near-dense code length and canvas count and beats V31 and V3a
by large margins. Gaussian32 online value selection is therefore not
inherently incompatible with code generation. This is a LLaDA2.2-specific
interaction between the support map and the native model/sampler state.

LLaDA2.2-mini is a 20-layer 16Q/4KV MoE with 256 experts and 8 active experts,
128K context, and Levenshtein block editing. TraDo-8B is a 36-layer SDAR model
with 32Q/8KV heads and 32K context. Both use head dimension 128 and a 4:1 GQA
ratio, so head dimension or GQA ratio alone does not explain the contrast. The
MoE block routing and edit-aware sampler are plausible sensitivity amplifiers,
but this is not a causal architecture ablation.

## Attention-support evidence

The multi-model receipts contain tile counts and denoising counters, but do not
record per-request full-attention output error or retained attention mass. Those
quantities cannot be claimed as directly measured on LLaDA2.2 HumanEval.

The existing 111-snapshot offline selector diagnostic is a separate
long-context/model study, so it is mechanistic evidence for the selector family,
not a direct LLaDA2.2 measurement:

| Selector, age | Full-dimensional mean-row error | Retained attention mass | IoU with V31 mass map |
|---|---:|---:|---:|
| V31, 0 | 0.250 | 0.369 | 1.000 |
| V2, 0 | 0.404 | 0.224 | 0.139 |
| V31, 1 | 0.243 | 0.354 | 1.000 |
| V2, 1 | 0.315 | 0.272 | 0.360 |
| V31, 7 | 0.240 | 0.369 | 1.000 |
| V2, 7 | 0.260 | 0.344 | 0.413 |

The label `preserve_mass` describes V2's online denominator/state update; it
does not mean the selected support preserves native attention mass. A separate
six-snapshot full-V check found projected-V/full-V map IoU 0.858, but full-V/V31
IoU only 0.156 and mean full-dimensional error 0.330 versus 0.247 for V31.
This points to the value-deletion/support objective and reuse, rather than
rank-32 projection alone, as the important mismatch.

## Conclusion and next diagnostic

The strongest current explanation is that V2 selects support with low native
mass retention and larger full-dimensional attention error; LLaDA2.2's
MoE/block-editing denoising state is unusually sensitive to that perturbation on
code continuation; and V31/V3a avoid enough of the basin shift to preserve a
complete code trajectory. This is a supported mechanism hypothesis, not a
proven per-layer causal chain.

The next experiment should be observation-only on the original sampler: log
per-canvas entropy, accepted/renoised-token fraction, argmax flips, EOS score,
stopping predicate, and support-map IoU for LLaDA2.2 V2/V31/V3a. On identical
LLaDA Q/K/V snapshots, compare equal-budget masks using full-dimensional
attention error, retained mass, post-attention/logit change, and argmax changes.
Then test fresh versus four-call-held selection and a mass-anchored value hybrid
before any further full HumanEval generation.

Sources: `results/multi_model_value_report_20261102/attempt002/summary.json`,
`results/v31_value_selectors_20261008/diagnostics/offline_accuracy_analysis_20261009/aggregate.json`,
`results/v31_value_v2_refinement_20261010/reports/refinement_analysis_20261010/summary.md`,
and the retired ablation at
`results/llada22_v2_eos_conf0_humaneval_20261102/attempt003/summary.md`.
