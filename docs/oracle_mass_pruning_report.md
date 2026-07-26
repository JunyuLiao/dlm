# Exact final-mass oracle pruning for BLASST

Date: 2026-07-21

## Executive conclusion

The experiment implementation and H100 trajectory evaluation are complete. The evidence supports this conclusion:

1. Exact dense-final normalized mass fixes most of the *ranking* problem. On 212,651 held-out `K <= 16` candidate tiles, all-row maximum exact mass has a log-Pearson correlation of **0.972** with the measured mass×value effect; the corresponding online mass proxy has correlation **0.238**.
2. Exact mass does **not** make all low-veto tiles safe. On held-out prompts, raw `K <= 16` candidates represent 27.04 percentage points of all physical tiles, but only **1.13 points** have all-row maximum exact mass below 0.01 and **6.28 points** are below 0.03. The median candidate all-row maximum mass is 0.98 for `K=16`, so many tiles that only a few rows veto are nevertheless dominant for at least one row.
3. The most robust physical aggregation is **maximum exact mass over all rows**. It has the best held-out correlation and directly protects the physical decision's worst affected row. Q99 is close (0.968) but deliberately ignores an extreme row; veto-max (0.957), masked-max (0.953), and total masked mass (0.910) are less conservative.
4. Cumulative all-row mass budgets reduce worst-case error tails. In the exact renormalized attention replay, a representative `K=8`, all-max, all-row budget of 0.01 adds **0.586 physical-sparsity points**, with mean relative output error **0.00174**, maximum absolute error **0.00438**, and worst row cosine **0.99998** versus ordinary sparse BLASST. More aggressive policies reach 2–4 points on this sample, but their worst-step and maximum errors rise sharply.
5. Full generation is much more sensitive than the attention replay suggests. In the policy-development matrix, the largest tested exact policy was `K=8`, cumulative all-row budget 0.03, current-step masks, and protection of every online new-maximum tile. It removed **40 of 589,824 physical tile calls (0.00678%)** and matched ordinary sparse BLASST on all six final 128-token sequences. With that policy frozen, a prompt-disjoint H100 final set removed **32 of 786,432 calls (0.00407%)** and matched all **8/8** final sequences on four new prompts × two seeds.
6. A single extra removal can bifurcate a diffusion trajectory. With no new-maximum protection, `K=8`, budget 0.01 removed one tile in four executions; the proof prompt diverged at step 0 and ended with 120/128 different tokens in both repeated executions. A direct rerun localized the tile to transformer layer 31, head 30. Step-0 token argmaxes still agreed everywhere and hidden cosine was 0.99999976, but one confidence-ordering change selected a different reveal position.
7. New-maximum protection and current-step recomputation are essential in this matrix. Protecting all new maxima changed `K=8`, budget 0.03 from 4/6 to 6/6 exact trajectories. Reusing the step-0 mask increased extra sparsity to 0.08138%, but only 4/6 trajectories remained exact. A masked-row-only budget reached 1.992% extra sparsity but only 2/6 exact trajectories.
8. The prompt-disjoint frozen-policy result achieved **99.727%** trajectory-mean masked-token top-1, 128/128 exact reveal-position sets, no intermediate generated-state divergence, and **0.00407%** extra physical sparsity. Thus the held-out frontier is **0% at 99.9%** and **0.00407% at 99.5%, 99%, 98%, and exact final sequence**. The earlier policy-development maximum was 0.00678%; neither small sample is a population-level safety guarantee.

The best interpretation is that the previous failure was substantially a predictor problem, but not *only* a predictor problem. Exact final mass is far better than the online proxy, yet generation is hypersensitive to reveal-order perturbations. The robust exact-generation headroom observed here is orders of magnitude smaller than the 0.5–1.2 percentage points suggested by attention-output replay alone.

## Implementation

### Correctness reference

[`blasst/oracle_mass_pruning.py`](../blasst/oracle_mass_pruning.py) implements:

- exact per-tile `(m, l, u)` collection from a complete QK pass;
- exact dense-final tile mass;
- ordinary reverse-traversal BLASST replay and veto counts;
- `K = 1, 2, 4, 8, 16` candidate selection;
- all requested maximum, veto-row, quantile, sum, mean, masked, and mixed-weight aggregations;
- independent threshold selection;
- cumulative all-row, masked-only, and mixed row budgets;
- score, traversal, and maximum-masked-mass tile ordering;
- all, masked-row, and no new-maximum protection;
- optional protected transformer layers;
- current-step and cached oracle-mask support;
- simultaneous complete-tile removal and exact renormalization;
- attention output error and masked/visible row metrics;
- layer/head removal accounting for full-trajectory failure localization.

The composition path sums unnormalized softmax sufficient statistics for active tiles and divides by the new denominator. It never approximates pruning by subtracting an already normalized tile contribution.

### Sweep and replay tools

[`tools/run_oracle_mass_experiment.py`](../tools/run_oracle_mass_experiment.py) performs the log threshold sweep, empirical candidate-score percentile sweep, cumulative budget sweep, ordering/scope/new-maximum ablations, current/fixed/previous-step oracle comparisons, per-step error analysis, Pareto extraction, and plotting.

[`tools/analyze_exact_oracle_mass_traces.py`](../tools/analyze_exact_oracle_mass_traces.py) streams the 6.7 GB veto trace corpus. It reconstructs the dense denominator from all 64 tile-local log-normalizers for each group instead of using the stored sparse-BLASST denominator.

[`eval/eval_oracle_mass_llada.py`](../eval/eval_oracle_mass_llada.py) is the CUDA full-model evaluator. It runs dense attention as a third reference, while ordinary sparse BLASST and oracle-pruned BLASST share the same exact post-pass composition arithmetic. This isolates the selected tile mask from floating-point reduction-order differences. It records:

- current-step or fixed oracle masks;
- masked-logit cosine and KL;
- masked and all-token top-1 agreement;
- hidden-state cosine;
- per-step generated-state agreement;
- first divergent step;
- differing generated tokens and fraction;
- final edit distance;
- exact generated-sequence agreement;
- extra physical sparsity and per-layer removal counts.

### Tests

[`tests/test_oracle_mass_pruning.py`](../tests/test_oracle_mass_pruning.py) verifies:

- tile masses against full softmax;
- mass normalization across tiles;
- simultaneous composition against explicit logit masking followed by a fresh softmax;
- physical candidate and aggregation shapes;
- cumulative row-budget enforcement;
- new-maximum protection relationships;
- identity metrics for no removal;
- end-to-end oracle selection and replay.

All 12 focused oracle and existing veto tests pass, including a protected-layer regression test. The complete suite also passes: 114 tests run, with 7 CUDA-only kernel tests skipped in the sandboxed test process.

## Data and evaluation scope

Three LLaDA evaluation sources were used locally.

### Prompt-disjoint exact-mass trace analysis

- Model recorded by the source traces: `GSAI-ML/LLaDA-8B-Instruct`.
- Sequence length: 4096.
- Physical shape: 128 query rows × 64 KV positions.
- Noise phases: approximately 0.9, 0.5, and 0.15 masked.
- Prompts: 12, split 4 calibration / 4 held-out / 4 final benchmark.
- Analyzed records: 2,359,296 tiles from target steps 1, 3, and 5.
- Exact-mass groups: 25,344.
- Maximum dense mass-sum error: `5.96e-7`.

This dataset supports candidate headroom, mass calibration, layer/phase diagnostics, and predictor correlations. It does not store full value vectors, so it cannot replay multiple simultaneous removals.

### Exact attention-output replay

- One development prompt.
- 16 actual denoising steps.
- One selected query tile, layer 0, head 0.
- 64 KV tiles, 128 query rows, value dimension 128 per step.
- Stored exact `(m, l, u)` statistics and dense attention output.
- Evaluated configurations: 1,480.

This dataset supports exact simultaneous removal and renormalized attention-output comparison. It is not prompt-disjoint and cannot reconstruct model hidden states, logits, or generated sequences.

The reconstructed dense output differs from the stored BF16 dense output by at most 0.00144 because the saved `u` statistics are BF16. Oracle-versus-ordinary comparisons use the same stored sufficient statistics on both sides and therefore do not inherit a mismatched reference path.

### Complete H100 generation trajectories

- GPU: one NVIDIA H100 80GB HBM3.
- Runtime: PyTorch 2.5.1+cu124, BF16 model weights.
- Model: `GSAI-ML/LLaDA-8B-Instruct` from the local cache.
- Policy-development prompts: three fixed prompts covering explanation, proof, and experiment design.
- Prompt-disjoint final prompts: four new prompts covering classification metrics, a number-theory proof, controlled experimental design, and graph search.
- Executions: two seeds per prompt, 128 generated tokens, 16 denoising steps; six development executions and eight frozen-policy final executions.
- Attention calls per execution: 1,024; physical tile calls per execution: 98,304.
- Comparison: oracle-pruned BLASST versus ordinary sparse BLASST with identical post-pass arithmetic.

Generation is deterministic under this evaluator, so the two seeds reproduce the same output for each prompt. Repeated seeds verify run reproducibility; they do not double the effective prompt sample size. The policy matrix has three effective development prompts, while the frozen final result has four prompt-disjoint effective prompts. The final artifact records `NVIDIA H100 80GB HBM3`, CUDA capability 9.0, and PyTorch 2.5.1+cu124.

## Broad exact-mass results

### Raw candidate headroom

On the held-out split, before applying a mass threshold:

| K | Candidate fraction of kept BLASST tiles | Candidate fraction of all physical tiles |
|---:|---:|---:|
| 1 | 8.42% | 6.07% |
| 2 | 13.50% | 9.72% |
| 4 | 20.20% | 14.54% |
| 8 | 28.24% | 20.33% |
| 16 | 37.55% | 27.04% |

This is only structural headroom. Removing these candidates based on veto count alone is unsafe.

### Exact all-row maximum mass qualification

Held-out additional physical sparsity after requiring exact all-row maximum mass below the threshold:

| K | `< 0.001` | `< 0.003` | `< 0.01` | `< 0.03` |
|---:|---:|---:|---:|---:|
| 1 | 0.003% | 0.035% | 0.338% | 1.655% |
| 2 | 0.005% | 0.051% | 0.506% | 2.553% |
| 4 | 0.008% | 0.070% | 0.700% | 3.676% |
| 8 | 0.011% | 0.094% | 0.906% | 4.900% |
| 16 | 0.014% | 0.117% | 1.127% | 6.278% |

The conservative frontier is therefore much smaller than the raw 6–27 point veto-count frontier.

Candidate mass itself explains why. Held-out all-row maximum mass percentiles are:

| K | p1 | p5 | p25 | p50 |
|---:|---:|---:|---:|---:|
| 1 | 0.0040 | 0.0094 | 0.0273 | 0.1314 |
| 2 | 0.0041 | 0.0098 | 0.0285 | 0.1594 |
| 4 | 0.0044 | 0.0102 | 0.0296 | 0.2153 |
| 8 | 0.0045 | 0.0106 | 0.0313 | 0.4000 |
| 16 | 0.0046 | 0.0110 | 0.0327 | 0.9826 |

For every K, p75 and above are 1.0 to stored precision. A physical tile can be critical to one row even when nearly every other row wants to skip it.

### Predictor comparison

Held-out correlation with `max_i(exact dense mass × local value norm / sparse output norm)` for `K <= 16` candidates:

| Aggregation | Exact dense-final mass | Online mass |
|---|---:|---:|
| all-row maximum | **0.972** | 0.238 |
| q99 | 0.968 | 0.226 |
| veto-row maximum | 0.957 | 0.234 |
| masked-row maximum | 0.953 | 0.218 |
| total masked mass | 0.910 | 0.104 |

Exact mass is a large improvement, and all-row maximum is the safest primary ranking. The small gap between all-max and q99 does not justify ignoring the worst physical row when the objective is robust generation.

### Phase and new-maximum structure

Candidate prevalence rises as noise falls. For held-out `K <= 16` candidates, the fraction of kept tiles is 19.14% at high noise, 41.30% at mid noise, and 64.37% at low noise.

New-maximum status changes even more sharply: 4.1% of high-noise candidates, 46.9% of mid-noise candidates, and 100% of low-noise candidates introduce a new running maximum. Protecting all new maxima would consequently erase most low-noise headroom. Exact final mass is the right diagnostic for testing whether that protection is unnecessarily conservative.

## Exact renormalized attention replay

All errors below compare oracle-pruned BLASST with ordinary sparse BLASST, not with dense attention.

### Independent all-max threshold examples

| K | Threshold | Extra physical sparsity | Mean relative error | Maximum absolute error | Minimum row cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.01 | 0.488% | 0.00091 | 0.00680 | 0.99964 |
| 8 | 0.01 | 1.172% | 0.00316 | 0.00680 | 0.99964 |
| 16 | 0.01 | 1.953% | 0.00589 | 0.00918 | 0.99964 |
| 1 | 0.03 | 1.270% | 0.00264 | 0.02410 | 0.99509 |
| 8 | 0.03 | 2.734% | 0.00868 | 0.02410 | 0.99509 |
| 16 | 0.03 | 4.297% | 0.01507 | 0.02465 | 0.99509 |

Raising K adds work opportunity at the same threshold, but the errors show why K is not a safety certificate once exact mass is known.

### Cumulative all-row budget examples

| Policy | Extra physical sparsity | Mean relative error | Maximum absolute error | Minimum row cosine |
|---|---:|---:|---:|---:|
| K=8, B=0.01, all-max order | 0.586% | 0.00174 | 0.00438 | 0.99998 |
| K=8, B=0.03, all-max order | 1.758% | 0.00723 | 0.01159 | 0.99987 |
| K=16, B=0.03, all-max order | 2.344% | 0.00779 | 0.00918 | 0.99989 |

At roughly 0.5–0.7 extra points, the cumulative policy limits maximum error to about 0.0039–0.0044, while matched independent points reach roughly 0.0068–0.0082. Independent thresholds can have lower average error at some sparsity points, but their unconstrained cumulative row mass produces larger tails. This supports cumulative mass as the robust refinement.

Masked-only budgets and masked-only aggregations sometimes produce more apparent sparsity at the same mean error, especially at low noise. That is not adopted as the primary result: when few or no masked rows remain, those rules can become vacuous and allow visible-row errors that may propagate later.

### Failure localization

For `K=8`, all-max, all-row budget 0.01:

- high noise: +0.391% sparsity, mean relative error 0.00071;
- mid noise: +0.868% sparsity, mean relative error 0.00278;
- low noise: no removals in this sampled query tile;
- worst individual step: step 5, mean relative error 0.01017.

At budget 0.03, the worst individual step is again step 5, with mean relative error 0.02591. The first refinement should therefore use a stricter mid-noise budget or protect the first layers during the mid-noise phase. The available exact-output replay only covers layer 0/head 0, so a per-layer threshold is not selected from this sample.

Broad candidate counts are lowest in early layers and highest in later layers: held-out `K <= 16` candidates range from 3,537 in layer 1 to 9,172 in layer 31. This is useful for work planning but is not an error measurement.

### New-maximum ablation

At `K=8`, all-row budget 0.03:

| Protection | Extra physical sparsity | Removed new-max tiles | Mean relative error |
|---|---:|---:|---:|
| none | 1.758% | 3 | 0.00723 |
| masked rows only | 1.758% | 3 | 0.00723 |
| all rows | 1.465% | 0 | 0.00721 |

The three extra removals occurred at low noise; their phase-average mean error was 0.00013. Exact final mass can override the online new-maximum signal in this attention sample. The full-generation result below shows that this local conclusion does not transfer: all-new-maximum protection is required by the exact pilot policy.

### Current versus stale oracle masks

One-step-stale masks were similar to current-step masks in the single attention trace: at budget 0.03 they produced 1.660% extra sparsity and 0.00692 mean error versus 1.758% and 0.00723. A mask fixed from step 0 removed nothing because the conservative step-0 selection was empty. The full trajectory experiment below reaches the opposite practical conclusion: a fixed mask can remove many more tiles after the state changes and break generation.

## Full-generation trajectory results

### Policy-development matrix

All table entries below aggregate six executions (three prompts × two deterministic repeats). Extra sparsity is relative to every physical attention-tile call, including tiles already skipped by ordinary sparse BLASST.

| Policy | Extra physical sparsity | Exact final sequences | Final differing tokens | Mean masked top-1 |
|---|---:|---:|---:|---:|
| K8, all-row B=0.003, no new-max protection | 0% | 6/6 | 0 | 100.000% |
| K8, all-row B=0.01, no new-max protection | 0.00068% | 4/6 | 240 | 74.473% |
| K8, all-row B=0.03, no new-max protection | 0.00949% | 4/6 | 78 | 92.330% |
| K8, all-row B=0.03, masked new-max protection | 0.00949% | 4/6 | 78 | 92.330% |
| **K8, all-row B=0.03, all new-max protection** | **0.00678%** | **6/6** | **0** | **99.708%** |
| K8, q99, all-row B=0.03, all new-max protection | 0.00678% | 6/6 | 0 | 99.708% |
| K8, independent tau=0.01, all new-max protection | 0.00068% | 4/6 | 240 | 74.473% |
| K8, fixed step-0 mask, B=0.03, all new-max protection | 0.08138% | 4/6 | 14 | 95.510% |
| K8, masked-row B=0.03, all new-max protection | 1.99212% | 2/6 | 128 | 73.307% |

### Prompt-disjoint frozen-policy validation

After choosing the bolded K8 policy above, it was frozen and evaluated on four new prompts × two seeds:

| Metric | Held-out result |
|---|---:|
| Additional physical sparsity | **0.004069%** (32/786,432 calls) |
| Fraction of ordinary-BLASST-kept tiles removed | 0.004180% |
| Exact final sequences | **8/8** |
| Differing final tokens / total edit distance | **0 / 0** |
| Trajectory-mean masked-token top-1 | **99.7265%** |
| Minimum single-step masked-token top-1 | 94.1667% |
| Minimum hidden-state cosine | 0.998168 |
| Maximum masked-token KL | 0.0005633 |
| Exact reveal-position sets | **128/128 steps** |
| Minimum intermediate generated-state agreement | **100%** |

Each unique prompt removed 3–5 physical tiles per execution. The two seeds produced identical outcomes, so the effective final sample size is four, not eight.

### Best exact policy

The largest tested exact policy was:

```text
K = 8
aggregation = all_max
selection = cumulative
all-row budget = 0.03
new-maximum protection = all
oracle refresh = every denoising step
```

In the policy-development matrix it removed 40 of 589,824 physical tile calls (0.00678%), or 0.00707% of tiles kept by ordinary BLASST. All six final sequences and edit distances matched exactly. Across its 96 development denoising-step records:

- trajectory-mean masked-token top-1 agreement: 99.708%;
- minimum single-step masked-token top-1 agreement: 88.333%;
- minimum hidden-state cosine: 0.97028;
- maximum masked-token KL: 0.02473;
- minimum intermediate generated-state agreement: 98.438%.

Exact final generation despite non-exact intermediate states means the trajectories can reconverge. Final equality alone therefore hides real transient perturbations.

On the prompt-disjoint final set, the same frozen policy removed 32 of 786,432 calls (0.00407%) and all 8/8 sequences matched. Unlike the development runs, all 128 final-set reveal-position sets and intermediate generated states matched exactly; small logit and hidden-state differences did not change the denoising path.

### Agreement frontier

Using trajectory-mean masked-token top-1 agreement as the near-exact criterion, the policy-development frontier is:

| Agreement requirement | Maximum tested extra physical sparsity | Exact final sequences |
|---:|---:|---:|
| 99.9% | 0% | 6/6 |
| 99.5% | 0.00678% | 6/6 |
| 99% | 0.00678% | 6/6 |
| 98% | 0.00678% | 6/6 |

These are maxima over the development policy grid, not interpolated thresholds or confidence bounds. On the prompt-disjoint final set, the frozen policy achieved 0.00407% at the 99.5%, 99%, 98%, and exact-sequence targets. It did not meet 99.9%; the zero-removal reference is the only held-out point at that target.

### Veto-count ablation

With all-new-maximum protection and all-row budget 0.03, K was non-monotonic:

| K | Extra physical sparsity | Exact final sequences | Mean masked top-1 |
|---:|---:|---:|---:|
| 1 | 0.00271% | 4/6 | 74.115% |
| 2 | 0.00407% | 6/6 | 99.642% |
| 4 | 0.00441% | 6/6 | 99.659% |
| 8 | 0.00678% | 6/6 | 99.708% |
| 16 | 0.01289% | 4/6 | 73.771% |

Exact mass does not make K redundant: the candidate set and ascending-score ordering determine which tiles consume the cumulative row budget. Both K1 and K16 admitted the sensitive proof-path removal, while K2, K4, and K8 did not.

### Failure mechanism and ablations

The smallest observed failure is especially informative. `K=8`, all-row budget 0.01, without new-maximum protection removed one new-maximum tile in transformer layer 31, head 30, at step 0 for the proof prompt. Sparse and oracle token argmaxes still agreed at every masked position; hidden cosine was 0.99999976, logit cosine was approximately 1, and KL was tiny. Nevertheless, one of eight reveal positions changed because the confidence ranking changed. Generated-state agreement immediately fell to 98.4375%, and the trajectories then compounded to 120/128 different final tokens in both deterministic repeats.

Protecting new maxima only on currently masked rows did not repair the budget-0.03 failure; its result was identical to no protection. Protecting new maxima on all rows did repair it, showing that visible-row effects matter for later denoising steps.

The fixed-mask and masked-budget ablations show how quickly nominal headroom becomes unsafe:

- reusing step-0 masks removed 480 tiles (0.08138%) but reduced exact agreement to 4/6;
- budgeting only masked rows removed 11,750 tiles (1.99212%) but reduced exact agreement to 2/6;
- q99 and all-max made identical decisions in this geometry and therefore do not distinguish their robustness here.

## Answers to the experimental questions

### If exact final normalized attention mass were available, how much additional physical sparsity could be obtained safely?

At the attention-output level, the sampled replay suggested roughly **0.5–1.2 additional percentage points** with small local errors. The complete-generation result is far stricter: the development maximum was **0.00678%**, while the same frozen policy produced **0.00407%** on four prompt-disjoint final prompts with 8/8 exact sequences. The held-out 0.00407% is the more defensible result, but it is still not a general production safety bound.

### Does exact mass solve the predictor problem, or are diffusion outputs inherently sensitive?

It solves much of the predictor ranking problem: 0.972 correlation versus 0.238 for online all-max. It does not remove inherent diffusion sensitivity. A step-0 perturbation small enough to preserve every token argmax still changed the confidence ordering, reveal position, and ultimately 120/128 output tokens.

### Which tile-level aggregation rule is most robust?

**Maximum exact mass across all valid rows.** It is the best held-out predictor and aligns with the physical all-row removal. Q99 is a useful density-oriented ablation, not the robust default. Masked-only and sum rules can conceal an important visible or extreme row.

### How important is cumulative removed mass?

Important, but not sufficient. At matched moderate sparsity, cumulative all-row budgets reduce attention-error tails relative to independent thresholds. In generation, the independent tau=0.01 policy failed after four total tile removals, while the cumulative B=0.03 K8 policy with all-new-maximum protection removed 40 and remained exact. Candidate-set ordering and new-maximum protection still determine which tile consumes the budget.

### Can low-final-mass tiles that introduce online running maxima be removed safely?

Not safely in this generation matrix. Allowing new-maximum removals yielded 4/6 exact sequences at budget 0.03; protecting all of them yielded 6/6. Masked-row-only protection did not help. The attention replay's benign result did not transfer to complete generation.

### What is the maximum extra sparsity compatible with exact or near-exact full generation?

The maximum in the exploratory policy grid was **0.00678%** with 6/6 exact final sequences and 99.708% trajectory-mean masked-token top-1. The frozen prompt-disjoint result was **0.00407%** with 8/8 exact sequences and 99.727% mean top-1. Use **0.00407%** as the held-out answer. At a 99.9% mean-agreement requirement, the maximum observed result remains 0% because every nonzero held-out policy point fell below that threshold.

## Recommended policy for expanded validation

The best pilot policy was frozen for the prompt-disjoint final validation and should remain the candidate for a larger validation:

```text
K = 8
aggregation = all_max
selection = cumulative
all-row budget = 0.03
new-maximum protection = all
ordering = ascending all_max score
oracle refresh = every denoising step
```

The next evaluation should use substantially more held-out prompts and prompt classes, not more seeds under deterministic decoding, report confidence intervals over unique prompts, and explicitly test reveal-order ties. Because the held-out gain is only 0.00407%, kernel overhead would dominate unless the policy is greatly improved; this oracle study makes no production-kernel claim.

## Reproduction

Focused tests:

```bash
python -m unittest -v \
  tests.test_oracle_mass_pruning \
  tests.test_veto_pruning \
  tests.test_veto_policy_reference
```

Exact attention replay and plots:

```bash
MPLCONFIGDIR=/tmp/matplotlib-oracle \
python tools/run_oracle_mass_experiment.py \
  --output-dir artifacts/oracle_mass_experiment/attention_replay
```

Prompt-disjoint exact dense-final mass analysis:

```bash
python tools/analyze_exact_oracle_mass_traces.py \
  artifacts/veto_traces_h100_v1 \
  --output artifacts/oracle_mass_experiment/exact_mass_trace_analysis.json
```

Full CUDA trajectory run:

```bash
python eval/eval_oracle_mass_llada.py \
  --k 8 \
  --aggregation all_max \
  --selection cumulative \
  --budget 0.03 \
  --budget-scope all \
  --order score \
  --new-max-protection all \
  --output artifacts/oracle_mass_experiment/trajectory-k8-b0p03-protect-newmax.json
```

Frozen prompt-disjoint final run:

```bash
python eval/eval_oracle_mass_llada.py \
  --prompts configs/oracle_mass_final_prompts.txt \
  --seeds 20260723,20260724 \
  --k 8 \
  --aggregation all_max \
  --selection cumulative \
  --budget 0.03 \
  --budget-scope all \
  --order score \
  --new-max-protection all \
  --output artifacts/oracle_mass_experiment/trajectory-k8-b0p03-protect-newmax-final.json
```

Trajectory summary and plot:

```bash
MPLCONFIGDIR=/tmp/matplotlib-oracle-summary \
python tools/summarize_oracle_trajectories.py \
  artifacts/oracle_mass_experiment/trajectory-k*.json \
  --csv artifacts/oracle_mass_experiment/trajectory_summary.csv \
  --plot artifacts/oracle_mass_experiment/trajectory_frontier.png
```

## Artifacts

- [`attention_replay/summary.csv`](../artifacts/oracle_mass_experiment/attention_replay/summary.csv): aggregate 16-step sweep.
- [`attention_replay/per_step.csv`](../artifacts/oracle_mass_experiment/attention_replay/per_step.csv): layer/head/step and phase rows for every configuration.
- [`attention_replay/pareto.csv`](../artifacts/oracle_mass_experiment/attention_replay/pareto.csv): extracted attention-output Pareto points.
- [`attention_replay/accuracy_sparsity.png`](../artifacts/oracle_mass_experiment/attention_replay/accuracy_sparsity.png): independent versus cumulative curve.
- [`attention_replay/aggregation_pareto.png`](../artifacts/oracle_mass_experiment/attention_replay/aggregation_pareto.png): aggregation comparison.
- [`attention_replay/k_pareto.png`](../artifacts/oracle_mass_experiment/attention_replay/k_pareto.png): veto-count ablation.
- [`attention_replay/phase_pareto.png`](../artifacts/oracle_mass_experiment/attention_replay/phase_pareto.png): high/mid/low-noise comparison.
- [`exact_mass_trace_analysis.json`](../artifacts/oracle_mass_experiment/exact_mass_trace_analysis.json): prompt-disjoint exact dense-final mass analysis.
- [`trajectory_summary.csv`](../artifacts/oracle_mass_experiment/trajectory_summary.csv): aggregate complete-generation results for the selected matrix.
- [`trajectory_frontier.png`](../artifacts/oracle_mass_experiment/trajectory_frontier.png): exact-generation agreement versus additional physical sparsity.
- [`trajectory-k8-b0p03-protect-newmax.json`](../artifacts/oracle_mass_experiment/trajectory-k8-b0p03-protect-newmax.json): full step-level record for the best exact tested policy.
- [`trajectory-k8-b0p03-protect-newmax-final.json`](../artifacts/oracle_mass_experiment/trajectory-k8-b0p03-protect-newmax-final.json): frozen-policy H100 record for four prompt-disjoint final prompts × two seeds.
- [`trajectory-failure-localization.json`](../artifacts/oracle_mass_experiment/trajectory-failure-localization.json): direct H100 layer/head/step record for the single-tile failure.
- [`oracle_mass_final_prompts.txt`](../configs/oracle_mass_final_prompts.txt): the frozen final prompt set.
- Other `trajectory-*.json` files in the same directory contain the threshold, K, new-maximum, fixed-mask, and budget-scope ablations.

## Limitations

- The policy-development matrix has only three unique prompts, and the prompt-disjoint final set has only four. Two deterministic seeds per prompt verify repeatability but do not increase either effective sample size; 8/8 is therefore not a useful statistical safety bound.
- Exact simultaneous output replay covers one development prompt, one query tile, layer 0, and head 0.
- The prompt-disjoint trace has broad layer/head coverage but lacks full value vectors, hidden states, and logits.
- The broad generation matrix was selected on the same three development prompts and is exploratory. The subsequent four-prompt final run froze the selected policy and is prompt-disjoint, but remains too small for a population estimate.
- The oracle requires a complete QK pass and post-pass selection. The experiment measures approximation headroom, not an implementable latency gain.
- No kernel speedup or feasibility claim is made; this is an oracle approximation study only.
