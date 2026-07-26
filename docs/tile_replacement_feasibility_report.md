# Compact replacement of low-veto BLASST tiles

Date: 2026-07-21

## Executive verdict

**No-go for landmark-token training.** A compact eight-slot summary does not meet the required coverage and trajectory-robustness gate.

The experiment establishes six main facts:

1. The sufficient-statistic implementation is correct. All-row replacement with exact unnormalized mass and exact conditional value reconstructs ordinary sparse BLASST to `7.93e-8` mean relative output error on held-out QKV groups.
2. Veto-only replacement is not the right target. Even exact mass plus exact value on veto rows has `0.0699` mean relative output error and a `1.541` worst-row relative error because the physical tile's contribution to non-veto rows is discarded. Accurate replacement must cover all rows.
3. Value reconstruction is the larger compact bottleneck, although mass is also material. With all-row replacement, calibrated maximum-logit mass plus exact value has `0.0345` mean output error; exact mass plus the deployable mean-query R8 value summary has `0.0444`. Combining them raises error to `0.0552`.
4. Increasing summary slots helps but does not reach a robust regime. Mean-query value error falls from `0.784` at R1 to `0.581` at R8 on veto rows. Even an oracle row-centroid R8 summary leaves `0.0262` mean output error and `0.885` maximum row error.
5. A gate-scale policy replacing **10.282% of all physical tiles** (14.136% of ordinary-BLASST-kept tiles) fails full generation: 7/8 exact final sequences, 95.742% trajectory-mean masked-token top-1, 0.614 mean reveal Jaccard, and minimum hidden cosine 0.9283.
6. A much stricter policy produces 8/8 exact final sequences, but replaces only **0.2726% of physical tiles** (0.3753% of kept tiles), reaches only **99.4397%** masked-token top-1, and has 92/128 exact reveal-position sets. It therefore fails both the coverage and near-perfect reveal requirements.

No tested nonzero compact policy satisfies the complete gate. The maximum tested coverage with exact final sequences is 0.2726%, and final equality hides intermediate reveal and state divergence. The measured representational headroom does not justify HiLS-style landmark training yet.

## Direct answers

| Question | Answer |
|---|---|
| Can a few vetoing rows be represented by a compact tile summary? | Not by veto-only replacement, and not robustly by the tested R≤8 all-row summaries. Exact veto-only replacement already fails structurally. |
| Is mass or value the main bottleneck? | Value is slightly larger at R8: 0.0444 mean output error for exact mass + mean-Q R8 versus 0.0345 for predicted mass + exact U. Their errors compound to 0.0552. |
| How many summary slots are required? | Error improves through R8, but no tested R is sufficient. The experiment cannot name a safe finite R; R8 remains far from the trajectory gate. |
| Is veto-only replacement sufficient? | No. All-row replacement is necessary. Exact/exact veto-only error is 0.0699, versus `7.93e-8` for exact/exact all-row replacement. |
| Are confidence ordering and reveal decisions preserved? | Not at meaningful coverage. The gate-scale trajectory has 0.922 mean confidence-rank correlation, 0.614 reveal Jaccard, and only 30/128 exact reveal sets. The strict policy improves these to 0.996, 0.893, and 92/128, still short of near-perfect reveal agreement. |
| What coverage remains robust over complete trajectories? | The largest exact-final tested policy replaces 0.2726% of physical tiles, but still misses the 99.5% masked-token and reveal gates. No nonzero tested coverage satisfies the full robustness definition. |
| Is landmark-token training justified? | No. The compact upper bounds and analytic summaries do not demonstrate the required representational headroom. |

## Implementation and deliverables

### Exact collector and composer

[`blasst/tile_replacement_reference.py`](../blasst/tile_replacement_reference.py) implements:

- exact per-tile FP32 `log Z`, conditional value `U`, and unnormalized numerator `N = ZU`;
- ordinary BLASST keep/veto/new-maximum decisions;
- simultaneous composition of remaining exact tiles and replacement components;
- veto-only and all-row replacement scopes;
- exact physical-tile deletion and recomposition.

The composer operates on unnormalized softmax sufficient statistics. It does not subtract an already normalized tile output and does not apply replacements sequentially.

### Mass and value predictors

[`blasst/tile_replacement_predictors.py`](../blasst/tile_replacement_predictors.py) contains:

- exact, existing-online, mean-logit, maximum-logit, maximum-plus-count, and mean-plus-variance mass estimators;
- scalar calibration learned only from calibration contexts;
- uniform, K-space K-means, joint K/V K-means, mean-query, and mean-veto-query summaries at R=1,2,4,8;
- zero, mean-V, norm-weighted mean, top-m, oracle SVD, oracle row-centroid, and per-instance entropy-capacity value baselines.

Mean-veto-query, oracle centroid, oracle SVD, and per-instance capacity results are diagnostic upper bounds, not deployable mechanisms.

### Full-model runtime and evaluation

[`blasst/tile_replacement_runtime.py`](../blasst/tile_replacement_runtime.py) is a slow correctness-oriented runtime that applies replacement at every attention invocation. It supports:

- veto-only or all-row replacement;
- exact or calibrated-maximum mass;
- exact or mean-query value reconstruction;
- all-candidate, exact-mass, predicted-mass, predicted-error, and cumulative qualification policies;
- no, veto-row, or all-row new-maximum protection;
- optional protected layers and per-layer/per-step statistics.

[`tools/run_tile_replacement_predictor_benchmark.py`](../tools/run_tile_replacement_predictor_benchmark.py) runs the QKV predictor matrix, joint replacement matrix, policy sweep, calibration, and per-layer/head/noise diagnostics.

[`eval/eval_tile_replacement_llada.py`](../eval/eval_tile_replacement_llada.py) runs calibration screening and independent baseline/candidate denoising trajectories, recording hidden, logit, confidence-order, reveal, generated-state, edit-distance, and coverage metrics.

[`tools/analyze_tile_replacement_experiment.py`](../tools/analyze_tile_replacement_experiment.py) produces flattened CSVs and all figures in the analysis artifact directory.

### Correctness tests

The focused tests in:

- [`tests/test_tile_replacement_reference.py`](../tests/test_tile_replacement_reference.py),
- [`tests/test_tile_replacement_predictors.py`](../tests/test_tile_replacement_predictors.py), and
- [`tests/test_tile_replacement_runtime.py`](../tests/test_tile_replacement_runtime.py)

verify exact reconstruction, identity, explicit-mask equivalence, simultaneous multiple replacement, row-scope targeting, predictor shapes, capacity identities, and runtime equivalence. All 10 focused tests pass. The full repository suite passes 114 tests, with seven unrelated CUDA-kernel tests skipped by the sandboxed test process.

## Experimental protocol

### Hardware and model

- GPU: one NVIDIA H100 80GB HBM3.
- PyTorch: 2.5.1+cu124.
- Model: `GSAI-ML/LLaDA-8B-Instruct`, BF16 weights.
- Attention geometry: 128 query rows × 64 KV positions, head dimension 128.
- Context length: 4096.
- Reference accumulation: FP32 sufficient statistics.
- Purpose: quality only; no latency or production-kernel claims.

### Prompt-disjoint QKV benchmark

- Twelve unique cyclic contexts constructed from the repository evaluation corpus.
- Contexts 0–3: calibration; contexts 4–11: held-out test.
- Mask ratios: 0.90, 0.50, 0.15.
- Sampled layers: 0, 15, 31.
- Sampled query heads: 0, 7, 15, 31.
- K values: 1, 2, 4, 8, 16.
- Final artifact: 108 layer/context/ratio groups and 432 head-level failure rows.
- Calibrated maximum-logit bias learned on calibration data: `+0.946381`.

### Full denoising trajectories

- Eight unique held-out contexts, indices 4–11; none used for scalar calibration or policy screening.
- A 3968-token visible prefix and a 128-token generated suffix.
- Sixteen denoising steps with the same fixed reveal rule for reference and candidate.
- Ordinary sparse BLASST and replacement states evolve independently after divergence.
- All transformer layers and heads use the replacement runtime.

The contexts are disjoint corpus rotations rather than eight independent natural-instruction datasets. This is broader than the earlier three-prompt pilot but remains a small deterministic quality study.

## Candidate headroom

Before qualification, low-veto tiles provide substantial structural coverage:

| K | Physical candidate tiles | Fraction of ordinary-BLASST-kept tiles |
|---:|---:|---:|
| 1 | 5.887% | 7.891% |
| 2 | 9.348% | 12.532% |
| 4 | 13.992% | 18.758% |
| 8 | 19.884% | 26.656% |
| 16 | 27.121% | 36.359% |

This is candidate headroom, not safe replacement coverage. The primary K=8 set is large enough to pass the work gate only if the compact representation is accurate.

## Mass prediction

Metrics below are held-out veto-row measurements for K≤8 candidates.

| Mass estimator | logZ corr. | Normalized-mass corr. | Mean relative mass error | Pair ranking | Top-1 candidate ranking |
|---|---:|---:|---:|---:|---:|
| Exact | 1.000 | 1.000 | `2.12e-7` | 1.000 | 1.000 |
| Existing online proxy | 0.273 | 0.574 | 2054.1 | 0.697 | 0.486 |
| Maximum logit | **0.970** | **0.965** | **0.456** | **0.856** | **0.672** |
| Calibrated maximum logit | 0.970 | 0.965 | 0.456 | 0.856 | 0.672 |
| Mean plus variance | 0.904 | 0.604 | 1.834 | 0.780 | 0.632 |
| K-means K, R8 | 0.755 | 0.643 | 2.744 | 0.773 | 0.603 |

Calibration lowers maximum-logit's absolute logZ error from 1.024 to 0.662, but a global additive logZ bias cancels during within-row normalization and therefore does not improve normalized-mass correlation or relative mass error.

The old online proxy is decisively worse. Exact final-denominator information solves much of the ranking problem, but the remaining maximum-logit mass error still causes meaningful output error.

## Value-summary capacity

Held-out veto-row results with exact mass fixed:

| Value method | Mean U cosine | Mean relative U error | Mean relative contribution error | Mean worst-veto-row U error |
|---|---:|---:|---:|---:|
| Zero | 0.000 | 1.000 | 0.0556 | 1.000 |
| Mean V | 0.527 | 0.930 | 0.0553 | 1.057 |
| Uniform R8 | 0.628 | 0.845 | 0.0509 | 0.992 |
| K-space K-means R8 | 0.710 | 0.730 | 0.0459 | 0.885 |
| Joint K/V R8 | 0.697 | 0.776 | 0.0478 | 0.956 |
| Mean-query R8 | **0.770** | **0.581** | **0.0350** | **0.715** |
| Mean-veto-query R8, diagnostic | 0.902 | 0.325 | 0.0174 | 0.380 |
| Oracle SVD R8 | 0.798 | 0.569 | 0.0350 | 0.691 |
| Oracle row centroids R8 | 0.886 | 0.406 | 0.0242 | 0.519 |
| Per-instance entropy identity | 1.000 | `1.98e-7` | `1.03e-8` | `2.31e-7` |

R-slot scaling is monotonic but still far from exact:

| Method | R1 | R2 | R4 | R8 |
|---|---:|---:|---:|---:|
| Uniform | 0.930 | 0.911 | 0.884 | 0.845 |
| K-space K-means | 0.930 | 0.885 | 0.812 | 0.730 |
| Joint K/V | 0.930 | 0.892 | 0.843 | 0.776 |
| Mean query | 0.784 | 0.730 | 0.663 | **0.581** |
| Mean veto query, diagnostic | 0.501 | 0.447 | 0.388 | **0.325** |

Values are mean relative U error. No curve reaches an acceptable plateau by R8. The exact per-instance identity confirms that the composition and prediction interfaces can express the target when the full tile distribution is retained; it is not evidence of compression.

## Joint replacement matrix and bottleneck isolation

All K≤8 candidates are replaced in this held-out matrix.

| Scope / combination | Mean relative output error | Maximum row error | Minimum row cosine | Mean denominator error |
|---|---:|---:|---:|---:|
| All rows, exact mass + exact U | `7.93e-8` | `3.39e-7` | 1.0000 | `4.50e-8` |
| Veto only, exact mass + exact U | **0.0699** | **1.5410** | 0.485 | 0.0932 |
| All rows, predicted mass + exact U | 0.0345 | 0.5679 | 0.880 | 0.0586 |
| All rows, exact mass + oracle centroid R8 | 0.0262 | 0.8846 | 0.472 | `4.50e-8` |
| All rows, exact mass + mean-query R8 | **0.0444** | 1.1449 | 0.092 | `4.50e-8` |
| All rows, predicted mass + mean-query R8 | **0.0552** | 1.3686 | 0.088 | 0.0586 |

The predicted-mass/exact-U versus exact-mass/predicted-U comparison identifies value reconstruction as the larger R8 compact error source. The oracle centroid's lower mean error but large tail shows that better summary fitting alone does not eliminate unsafe rows.

The veto-only exact result is more important than the compact comparison: replacing only veto rows cannot reconstruct a physically retained tile. The non-veto rows collectively contain consequential mass and numerator even though they did not block BLASST's online skip test.

## Policy sweep and new-maximum protection

The deployable compact mechanism in this sweep uses calibrated maximum-logit mass and mean-query R8 value summaries over all rows.

| K8 policy | New-max protection | Physical replacement | Kept-tile replacement | Mean output error | Maximum row error |
|---|---|---:|---:|---:|---:|
| All candidates | None | 19.884% | 26.656% | 0.0552 | 1.3686 |
| Exact veto-max < 0.03 | None | 10.775% | 14.445% | 0.0176 | 0.5667 |
| Exact all-max < 0.03 | None | 9.587% | 12.852% | 0.0144 | 0.2886 |
| Predicted contribution error < 0.03 | None | 6.717% | 9.004% | 0.00852 | 0.3501 |
| Cumulative exact mass B=0.03 | None | 3.440% | 4.611% | 0.00473 | 0.1702 |
| Predicted max mass < 0.01 | All rows | 1.204% | 1.615% | 0.000549 | 0.0869 |
| Predicted max mass < 0.003 | All rows | **0.2387%** | **0.3200%** | `4.65e-5` | 0.0239 |

All-row new-maximum protection reduces the exact-veto-max `<0.03` sampled policy from 10.775% to 4.980% physical coverage. It is not mathematically necessary for exact all-row sufficient-statistic replacement, but it is useful protection against compact prediction tails. At low noise every sampled K≤8 candidate introduces a new maximum, compared with 43.6% at mid noise and 3.8% at high noise; blanket protection consequently removes almost all low-noise opportunity.

## Full-forward screening

Calibration screening explains the selected trajectory policies:

| Policy | Noise ratio | Physical replacement | Masked top-1 | Confidence rank | Reveal agreement | Exact reveal sets | Minimum hidden cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact veto-max <0.03, no protection | 0.90 | 9.480% | 100.0% | 0.9960 | 1.000 | 4/4 | 0.9921 |
| Same | 0.50 | 10.306% | 100.0% | 0.9967 | 0.875 | 2/4 | 0.9907 |
| Same | 0.15 | 11.511% | 100.0% | 0.9853 | 1.000 | 4/4 | 0.9836 |
| Predicted max <0.003, all-new-max protection | 0.90 | 0.457% | 100.0% | 0.9987 | 1.000 | 4/4 | 0.9990 |
| Same | 0.50 | 0.191% | 100.0% | 0.9983 | 1.000 | 4/4 | 0.9981 |
| Same | 0.15 | 0% | 100.0% | 1.0000 | 1.000 | 4/4 | 1.0000 |

Single-forward screening made both policies look substantially safer than their complete trajectories. It cannot be used as the success criterion.

## Held-out complete trajectories

| Metric | Gate-scale policy | Strict compact policy |
|---|---:|---:|
| Qualification | Exact veto-row max <0.03 | Predicted all-row max <0.003 + all-new-max protection |
| Physical replacement | **10.2818%** | **0.2726%** |
| Kept-tile replacement | 14.1361% | 0.3753% |
| Replaced tile calls | 27,599,991 | 731,724 |
| Exact final sequences | **7/8** | **8/8** |
| Differing final tokens / edit distance | 7 / 7 | 0 / 0 |
| Trajectory-mean masked top-1 | **95.7424%** | **99.4397%** |
| Mean masked-logit cosine | 0.99179 | 0.99935 |
| Maximum masked KL | 8.1375 | 0.03156 |
| Mean confidence-rank correlation | 0.92182 | 0.99609 |
| Mean reveal-position Jaccard | **0.61407** | **0.89297** |
| Exact reveal-position sets | 30/128 | 92/128 |
| Minimum hidden cosine | 0.92825 | 0.96870 |
| Minimum generated-state agreement | 0.6875 | 0.9375 |

The gate-scale policy passes the 10-percentage-point physical coverage requirement but fails every trajectory-quality requirement. Its one failing context diverges by seven final tokens, while other contexts can diverge and later reconverge.

The strict policy is the strongest result: all final sequences match, yet its intermediate trajectories are not exact. Every context except one has an intermediate divergence, exact reveal sets occur in only 72% of steps, and masked top-1 is 99.4397%—just below the explicit 99.5% threshold. More importantly, its physical coverage is 37× below the 10-point gate and its kept coverage is 53× below the 20% gate.

## Layer, head, row-state, and noise failures

For all-candidate compact replacement on held-out QKV:

| Breakdown | Candidate tiles | Mean compact output error | Mean veto mass error | Mean veto value error | Exact veto-only output error |
|---|---:|---:|---:|---:|---:|
| Layer 0 | 600 | 0.0310 | 0.260 | 0.307 | 0.0342 |
| Layer 15 | 1,317 | 0.0573 | 0.518 | **0.721** | **0.1024** |
| Layer 31 | 1,748 | **0.0772** | 0.444 | 0.431 | 0.0732 |
| High noise | 862 | 0.0177 | 0.341 | 0.370 | 0.0253 |
| Mid noise | 1,157 | 0.0517 | **0.451** | 0.492 | 0.0619 |
| Low noise | 1,646 | **0.0961** | 0.431 | **0.597** | **0.1224** |

Head 7 has the largest mean compact output error among sampled heads (0.0728), while head 0 has the largest value error (0.627). Visible rows are slightly worse than masked rows under all-candidate replacement: 0.0605 versus 0.0514 mean relative output error. This supports all-row protection and explains why a veto-only or masked-only mechanism can propagate errors through later denoising steps.

The dominant failure region is low noise and later layers, where candidate prevalence is high and compact value error is worst. Protecting all new maxima removes that region rather than making it compressible.

## Iterative refinement decisions

The experiment followed the requested refinement logic:

1. Exact mass/exact U validated the composer.
2. Veto-only exact replacement exposed a structural all-row error, so subsequent compact trajectories used all-row replacement.
3. Mass calibration selected maximum logit with a calibration-only bias; the existing online proxy was rejected.
4. Slot count was increased through R8; mean-query summaries were selected as the best non-oracle value mechanism.
5. All-candidate screening exposed low-noise confidence and reveal failures.
6. Exact-veto-mass qualification recovered gate-scale coverage but failed held-out trajectories.
7. All-new-maximum protection and a predicted maximum-mass threshold of 0.003 recovered exact final sequences, but collapsed coverage and still missed the top-1/reveal gate.

The optional offline probe was not run. A probe could improve the chosen analytic mapping, but the diagnostic evidence already shows that veto-only replacement is invalid and that even oracle-fitted R8 value summaries retain large row-error tails. Training a feature probe before a compact oracle demonstrates gate-scale trajectory robustness would not establish the representational prerequisite for landmark-token training.

## Landmark-training gate

| Requirement | Best evidence | Status |
|---|---|---|
| ≥10 percentage points physical replacement **or** ≥20% kept replacement | Gate-scale: 10.282% physical; strict: 0.273% physical / 0.375% kept | Gate-scale passes coverage; strict fails |
| Near-perfect reveal agreement | Gate-scale 0.614 Jaccard, 30/128 exact; strict 0.893, 92/128 exact | **Fail** |
| ≥99.5% trajectory-mean masked-token agreement | Gate-scale 95.742%; strict 99.440% | **Fail** |
| Strong hidden-state cosine | Minima 0.928 and 0.969 | Insufficient with other failures |
| High exact final-sequence agreement | 7/8 and 8/8 | Strict passes final equality only |

No row satisfies the entire gate. The correct decision is **do not proceed to landmark-token training**.

## Reproduction

Focused tests:

```bash
python -m unittest -v \
  tests.test_tile_replacement_reference \
  tests.test_tile_replacement_predictors \
  tests.test_tile_replacement_runtime
```

H100 predictor benchmark:

```bash
python tools/run_tile_replacement_predictor_benchmark.py \
  --output artifacts/tile_replacement_experiment/predictor-benchmark-h100.json
```

Gate-scale trajectory:

```bash
python eval/eval_tile_replacement_llada.py \
  --mode trajectory --split test \
  --k 8 --scope all_rows \
  --mass-method calibrated_maximum \
  --value-method proxy_mean_q --summary-slots 8 \
  --policy exact_vetomax --qualification 0.03 \
  --new-max-protection none \
  --output artifacts/tile_replacement_experiment/trajectory-compact-exact-vetomax-0p03-test.json
```

Strict exact-final trajectory:

```bash
python eval/eval_tile_replacement_llada.py \
  --mode trajectory --split test \
  --k 8 --scope all_rows \
  --mass-method calibrated_maximum \
  --value-method proxy_mean_q --summary-slots 8 \
  --policy predicted_maxmass --qualification 0.003 \
  --new-max-protection all_rows \
  --output artifacts/tile_replacement_experiment/trajectory-compact-predicted-maxmass-0p003-newmax-all-test.json
```

Tables and plots:

```bash
MPLCONFIGDIR=/tmp/matplotlib-tile-replacement \
python tools/analyze_tile_replacement_experiment.py \
  --predictor artifacts/tile_replacement_experiment/predictor-benchmark-h100.json \
  --trajectory artifacts/tile_replacement_experiment/trajectory-compact-exact-vetomax-0p03-test.json \
  --trajectory artifacts/tile_replacement_experiment/trajectory-compact-predicted-maxmass-0p003-newmax-all-test.json \
  --output-dir artifacts/tile_replacement_experiment/analysis
```

## Artifacts

- [`predictor-benchmark-h100.json`](../artifacts/tile_replacement_experiment/predictor-benchmark-h100.json): complete held-out predictor, replacement, policy, and failure matrix.
- [`analysis/coverage.csv`](../artifacts/tile_replacement_experiment/analysis/coverage.csv): K-dependent candidate coverage.
- [`analysis/mass_predictors.csv`](../artifacts/tile_replacement_experiment/analysis/mass_predictors.csv): flattened mass metrics.
- [`analysis/value_predictors.csv`](../artifacts/tile_replacement_experiment/analysis/value_predictors.csv): flattened value metrics.
- [`analysis/replacement_matrix.csv`](../artifacts/tile_replacement_experiment/analysis/replacement_matrix.csv): oracle and compact combinations.
- [`analysis/policy_sweep.csv`](../artifacts/tile_replacement_experiment/analysis/policy_sweep.csv): qualification and new-maximum sweep.
- [`analysis/failure_breakdown.csv`](../artifacts/tile_replacement_experiment/analysis/failure_breakdown.csv): per-context/layer/head/noise diagnostics.
- [`analysis/trajectory_summary.csv`](../artifacts/tile_replacement_experiment/analysis/trajectory_summary.csv): aggregate complete-generation metrics.
- [`analysis/predictor_slots.png`](../artifacts/tile_replacement_experiment/analysis/predictor_slots.png): R-slot scaling.
- [`analysis/attention_accuracy_coverage.png`](../artifacts/tile_replacement_experiment/analysis/attention_accuracy_coverage.png): held-out attention Pareto sweep.
- [`analysis/replacement_matrix.png`](../artifacts/tile_replacement_experiment/analysis/replacement_matrix.png): mass/value bottleneck comparison.
- [`analysis/layer_phase_failures.png`](../artifacts/tile_replacement_experiment/analysis/layer_phase_failures.png): layer and noise-phase failures.
- [`analysis/trajectory_frontier.png`](../artifacts/tile_replacement_experiment/analysis/trajectory_frontier.png): mandatory trajectory gate.
- [`trajectory-compact-exact-vetomax-0p03-test.json`](../artifacts/tile_replacement_experiment/trajectory-compact-exact-vetomax-0p03-test.json): gate-scale eight-context trajectory.
- [`trajectory-compact-predicted-maxmass-0p003-newmax-all-test.json`](../artifacts/tile_replacement_experiment/trajectory-compact-predicted-maxmass-0p003-newmax-all-test.json): strict eight-context trajectory.

## Limitations

- The broad QKV benchmark samples three layers and four heads, not every layer/head pair. Full trajectories do exercise all layers and heads.
- The eight trajectory contexts are deterministic rotations of a local corpus, not a statistically representative prompt benchmark.
- Only one fixed decoding schedule and seed-derived context construction were tested.
- The exact-veto-max gate-scale qualifier is an oracle diagnostic and is not deployable; the strict predicted-mass policy is the closer non-learned mechanism.
- Mean-query summary construction and post-pass replacement are quality references, not costed kernel designs.
- No offline learned probe was trained, so the report rejects proceeding now rather than claiming that every learned summary must fail.
- Exact final agreement is observed on eight contexts and is not a population-level guarantee.
