# Denoising-aware veto-row pruning: H100 analysis and verdict

## Verdict

**No-go for production integration.** The idea has substantial *raw* physical-tile
headroom, but no training-free, online-computable policy converts that headroom into
safe pruning. The cheapest policy with enough modeled work reduction, critical-state
protected `K=8` veto pruning, materially changes one-step masked predictions and full
denoising output. Consequently, the staged plan correctly stops before implementing
or timing a production Triton specialization.

Ordinary BLASST remains the recommended implementation. Post-QK veto pruning should
stay disabled; no new kernel was added.

## 1. What was examined

The physical attention tile is 128 query rows by 64 KV positions, with head dimension
128. For query row `i` and KV tile `j`, ordinary BLASST computes

```text
r[i,j] = exp(local_max[i,j] - running_max_before[i])
```

`local_max` is the maximum QK logit inside the current 64-column tile.
`running_max_before` is the online-softmax maximum from earlier KV tiles. A row casts a
**keep vote**, or vetoes skipping, when `r[i,j] >= lambda`. Ordinary BLASST skips the
physical tile only if every valid row votes to skip, equivalently
`max_i r[i,j] < lambda`.

This study changes the decision *after QK*. It therefore cannot avoid QK work. It asks
whether a tile kept by only a few rows can still skip the softmax update, V load, and
PV matrix product with tolerable error.

### Current reference K-veto logic

For each tile:

1. Compute all 128-row QK scores and the ordinary BLASST decision.
2. Count `veto_count = number of valid rows with log(r[i,j]) >= log(lambda)`.
3. Consider only tiles ordinary BLASST kept and for which `1 <= veto_count <= K`.
4. Optionally require all veto rows to be visible, non-masked, or stable.
5. Apply a new-maximum protection rule.
6. If the tile remains eligible, skip its downstream softmax/PV update.

The quality candidate used **critical new-maximum protection**: never remove a tile
that introduces a new running maximum for a masked, newly revealed, or prefix row.
New maxima belonging only to previously revealed or stable-visible rows remain
eligible. This is more permissive than protecting every new maximum and more
conservative than protecting masked rows alone.

All pruning is opt-in through a callback in the PyTorch/reference attention path. With
the callback absent, behavior is unchanged. The reference policy and any future
kernel implementation are separate switches; there is currently no enabled production
kernel implementation.

### Other policy families

- **Quantile score:** replace the maximum row score by q90--q99.
- **Oracle normalized mass:** use final attention normalization,
  `alpha[i,j] = exp(tile_logsumexp[i,j] - final_logsumexp[i])`.
- **Online normalized-mass proxy:** normalize against only past tiles plus the current
  tile. This is legal at decision time but does not know future KV tiles.
- **Mass x V magnitude:** multiply mass by maximum, mean, or normalized Frobenius V
  norm.
- **Variance/centroid score:** multiply mass by the maximum residual norm around the
  V-tile centroid; the offline centroid variant measures the residual if the tile were
  approximated by its mean V vector.
- **Token-state policies:** use masked, newly revealed, previously revealed,
  stable-visible, and prefix labels to restrict rows or relax thresholds.
- **Cumulative per-row budget:** accumulate estimated row error within a layer using
  sum, root-sum-square, or maximum rules and state-dependent budgets.
- **Global masked budget:** aggregate estimated error only over masked/prefix rows.

## 2. Data validity and experimental design

The H100 trace contains 4,718,592 physical-tile records from 12 contexts and six
denoising states. The main analysis uses the three target current states: 89%, 49%,
and 14% masked, totaling 2,359,296 records. Contexts—not individual tiles—were split
4/4/4 into calibration, held-out, and final-benchmark sets. Thresholds were grouped by
noise phase and 8-layer bucket.

Trace integrity passed every structural check:

| Check | Result | Definition |
|---|---:|---|
| Unique records | 4,718,592 / 4,718,592 | No duplicate layer/head/step/query/KV identity. |
| Complete query groups | 73,728 | Every sampled query group contains all 64 KV tiles. |
| Vote/skip mismatches | 0 | Stored row votes exactly reconstruct ordinary BLASST's physical skip. |
| Padding violations | 0 | Invalid rows contain no active decision data. |
| Final-LSE mismatches | 0 | Every tile for one row group carries the same final log-normalizer. |
| Normalized-mass failures | 0 | Tile masses sum to one within the FP16 trace tolerance. |
| Maximum mass-sum error | 0.01550 | Maximum absolute deviation from one after compact FP16 storage; tolerance was 0.02. |
| Finite query-change rate | 100% after step 0 | Previous-step query data exists wherever it is mathematically defined. |

Token states account for 69.52% masked, 0.69% newly revealed, 10.29% previously
revealed, and 19.50% stable-visible veto rows. Prefix rows produced no vetoes in this
sample.

## 3. Metric definitions

### Work and sparsity metrics

- **Total tiles:** physical 128x64 tiles evaluated across sampled contexts, layers, and
  heads.
- **Baseline skipped tiles:** tiles ordinary BLASST skips using the all-row maximum.
- **Baseline physical sparsity:** baseline skipped tiles / total tiles.
- **Additional skipped tiles:** baseline-kept tiles removed only by the candidate.
- **Additional physical sparsity:** additional skipped tiles / total tiles. This is a
  percentage-point increase, not a relative percentage.
- **New physical sparsity:** baseline plus additional physical sparsity.
- **Downstream work reduction:** additional skipped tiles / baseline-kept tiles. It is
  the fraction of remaining softmax/PV tile work eliminated; QK is unchanged.
- **PV tiles eliminated:** equal to additional skipped tiles.
- **V bytes avoided:** eliminated tiles x 64 KV rows x 128 dimensions x 2 BF16 bytes.
- **Softmax elements avoided:** eliminated tiles x 128 x 64.
- **PV FLOPs avoided:** eliminated tiles x 2 x 128 x 64 x 128.
- **QK FLOPs avoided:** always zero for post-QK pruning.
- **Half-downstream speedup:** `1 / (1 - 0.5 * downstream_reduction)`. This is an
  analytical scenario assuming downstream work is exactly half the kernel and the
  decision itself is free. It is not a measured speedup.
- **Work-gate pass:** at least +20 percentage points physical sparsity or 20% downstream
  reduction, plus at least 5% modeled kernel speedup. It says nothing about quality.

### Offline safety metrics

For a row, the conservative trace screening estimate is

```text
relative_tile_error = oracle_final_mass * exact_local_value_norm
                      / final_attention_output_norm
```

The per-state limits are 0.5% for masked/prefix, 0.75% for newly revealed, 2% for
previously revealed, and 5% for stable-visible rows. Depending on the protection
ablation, selected new maxima are also declared unsafe.

- **Unsafe removal:** a removed tile violates at least one applicable per-row screening
  limit or protected-new-maximum rule.
- **Unsafe removal rate:** unsafe removals / additional removals.
- **Wilson 95% upper bound:** upper endpoint of a binomial 95% Wilson confidence
  interval for the unsafe rate. Calibration required this to be <=1% with at least 256
  observations per group.
- **New-max removal:** a removed tile where at least one row's local maximum exceeds its
  previous running maximum. Such a removal also changes online-softmax scaling.
- **Affected masked rows:** count of masked row/tile effects above `1e-8`; a row can be
  counted for more than one removed tile.
- **Maximum/p99/mean masked relative error:** summaries of the screening estimate over
  removed tiles. These are attention-output perturbation estimates, not token error
  rates.
- **Log-Pearson correlation:** Pearson correlation after taking logs of the candidate
  score and exact relative output effect. One means perfect ranking on a multiplicative
  scale; zero means no useful linear ranking.

### Model-quality metrics

- **Masked top-1 agreement vs sparse:** fraction of currently masked positions whose
  argmax token matches ordinary BLASST. This is the primary one-step fidelity metric.
- **Masked top-1 agreement vs dense:** fraction matching compiled dense attention.
- **Sparse masked top-1 agreement vs dense:** ordinary BLASST's own agreement with
  dense, providing the baseline approximation gap.
- **All-token top-1 agreement vs sparse:** argmax agreement over masked and visible
  sequence positions.
- **Masked logit cosine:** cosine similarity between flattened candidate and sparse
  masked-position logits; one is identical direction, but can hide individual argmax
  changes.
- **Masked logit KL:** mean KL divergence from sparse masked-token probability
  distributions to candidate distributions; zero is identical.
- **Hidden-state cosine:** cosine similarity of final hidden states. This detects
  internal drift that may propagate to later denoising steps.
- **Final generated-token agreement:** fraction of generated positions matching after
  a complete trajectory.
- **Exact generated sequence:** true only if every generated token matches.

## 4. Veto-row headroom

On the full three-phase analysis set, ordinary BLASST kept 1,525,032 tiles. The
distribution of why those tiles were kept is:

| Veto rows | Fraction of kept tiles | Additional sparsity if all removed | New-max fraction | Mean veto attention mass |
|---:|---:|---:|---:|---:|
| 1 | 9.19% | 5.94 pp | 45.43% | 0.0580 |
| 2--4 | 12.22% | 7.90 pp in this bin | 50.44% | 0.0747 |
| 5--8 | 8.12% | 5.25 pp in this bin | 57.35% | 0.0895 |
| 9--16 | 9.28% | 6.00 pp in this bin | 61.02% | 0.0992 |
| 17--32 | 10.45% | 6.76 pp in this bin | 63.08% | 0.1134 |
| 33--64 | 13.09% | 8.46 pp in this bin | 67.81% | 0.1599 |
| 65--128 | 37.64% | 24.33 pp in this bin | 76.37% | 0.1245 |

Cumulative few-veto headroom is +5.94 pp for K=1, +9.39 pp for K=2, +13.84 pp
for K=4, +19.09 pp for K=8, and +25.09 pp for K=16. Thus the hardware granularity
really does strand work: 29.54% of kept tiles are blocked by at most eight rows.
However, 50.78% of the K<=8 headroom introduces a new running maximum.

The failure is strongest at low noise: **100% of kept tiles in every veto-count bin
introduce a new maximum**. This occurs because the low-noise BLASST threshold is
`lambda=1`; a keep vote then means a row has at least tied or exceeded the previous
maximum. Few vetoes at low noise are therefore not harmless exceptions.

## 5. Which scores predict real output effect?

Held-out log-Pearson correlations with exact relative tile-output effect:

| Predictor | Correlation | Online at decision time? |
|---|---:|---|
| Oracle final-normalized mass, max row | 0.9356 | No |
| Oracle centroid-residual effect | 0.9312 | No |
| Oracle variance bound | 0.8532 | No |
| Oracle mass x max V norm | 0.8278 | No |
| Online mass, max row | 0.4734 | Yes |
| Online mass x max V norm | 0.4637 | Yes |
| Online mass x normalized Frobenius V norm | 0.4568 | Yes |
| Online mass x mean V norm | 0.4551 | Yes |
| Online variance bound | 0.4177 | Yes |
| q99 row score | 0.0338 | Yes |

Final-normalized attention mass is the best practical *description* of effect, but it
requires future tiles or a second pass. The legal online estimate has less than half
the oracle ranking strength. V magnitude and variance consistently reduce correlation
rather than improve it, so their additional loads are unjustified. Raw score quantiles
are nearly uncorrelated with actual output effect.

With all new maxima protected, the calibrated oracle exact-effect policy safely adds
only 0.114 percentage points of sparsity, removes 0.158% of downstream work, and has an
optimistic half-downstream speedup of 1.00079x. Calibrated online mass, quantile, V, and
masked-budget policies select effectively no useful work. No policy passes the combined
safety/work gate.

## 6. State protection and cumulative-budget results

Allowing new maxima from previously revealed and stable-visible rows creates enough
raw work for two policies on held-out traces:

| Policy | Baseline sparsity | Additional sparsity | Downstream reduction | Unsafe rate | New-max removals | Modeled half-downstream speedup |
|---|---:|---:|---:|---:|---:|---:|
| Critical-protected K=8 | 28.00% | +16.08 pp | 22.33% | 97.79% | 58,696 | 1.1257x |
| Critical-protected K=16 | 28.00% | +19.46 pp | 27.02% | 97.82% | 67,684 | 1.1562x |

These are **work-only passes** and analytical-safety failures. Stable-only K=16 with
all new maxima protected adds just 1.11 pp, so stable rows alone do not provide enough
coverage. Relaxing protection yields work only by admitting consequential new maxima.

The cumulative sweep tested 432 combinations. No online proxy passes the work gate.
The only held-out work passes use the impossible oracle exact relative effect, a loose
5% budget, and maximum rather than additive accumulation. Critical protection then
gives +18.44 pp and 25.61% downstream reduction, but 96.82% of removals violate the
screening label. Sum and root-sum-square accumulation are more conservative without
recovering useful safe coverage. Therefore cumulative budgets do not outperform
independent thresholds.

## 7. Real one-step quality on one H100

Native 4096-token, BF16 LLaDA-8B-Instruct forwards used paired dense, ordinary sparse,
and candidate evaluations. Each row below is one fixed context/seed; this is a quality
falsification test, not a claim about population means.

| Noise / policy | Baseline sparsity | Additional sparsity | New sparsity | Downstream reduction | Masked agreement vs sparse | Masked agreement vs dense | Sparse vs dense | All-token agreement vs sparse | Logit cosine | KL | Hidden cosine vs sparse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| High, K=8 critical | 8.78% | +12.53 pp | 21.31% | 13.74% | 93.95% | 92.22% | 95.57% | 94.53% | 0.99911 | 0.01010 | 0.99649 |
| Mid, K=8 critical | 29.38% | +18.09 pp | 47.47% | 25.61% | 96.58% | 95.19% | 97.12% | 96.95% | 0.99689 | 0.01701 | 0.98470 |
| Low, K=8 critical | 47.15% | +20.47 pp | 67.62% | 38.73% | 97.51% | 96.98% | 97.34% | 95.53% | 0.99483 | 0.01171 | 0.95196 |
| Mid, K=16 critical | 29.27% | +21.87 pp | 51.13% | 30.92% | 95.88% | 94.45% | 97.12% | 96.58% | 0.99583 | 0.02307 | 0.98042 |

K=8 changes 6.05%, 3.42%, and 2.49% of masked top-1 predictions at high, mid,
and low noise. K=16 is worse than K=8 on the paired mid-noise context. High cosine
scores do not rescue the policy: argmax decisions change materially, and hidden-state
drift becomes especially large late in denoising.

At low noise, all 429,290 additional K=8 removals introduce a new maximum. At mid
noise, 88,866 do. This directly explains why token-state-only protection fails: visible
and stable-row attention still shapes future hidden states and later predictions.

## 8. Full denoising trajectory

For a 128-token generation over 16 denoising steps:

| Metric | Result |
|---|---:|
| Candidate final-token agreement vs ordinary sparse | 87.50% |
| Candidate final-token agreement vs dense | 86.72% |
| Ordinary sparse final-token agreement vs dense | 92.19% |
| Exact candidate sequence vs sparse | False |
| Baseline physical sparsity | 2.65% |
| Additional physical sparsity | +7.29 pp |
| Downstream work reduction | 7.49% |
| Additional new-max tiles | 4,195 / 7,169 |

Sixteen of 128 candidate tokens differ from ordinary sparse BLASST. Errors therefore
do compound across denoising, while the shorter sequence supplies less work headroom
than the native-length one-step test.

## 9. Runtime interpretation

There is no measured latency for an enabled veto kernel because no policy passed the
offline safety and real-quality gates. Implementing a production specialization would
violate the experimental plan and could turn an already-rejected algorithm into a
misleading kernel-overhead benchmark.

The 1.1257x and 1.1562x figures above are work-model upper scenarios for K=8/K=16,
assuming QK and all fixed costs consume half the kernel and the new decision costs
nothing. They are not H100 measurements. Their corresponding quality regressions
already reject the policies.

For context, the previously measured native 4096 batch-1 baseline is 151--153 ms for
ordinary sparse BLASST versus about 159 ms for dense, a 3.7--4.7% sparse advantage.
Those are full-forward measurements from the earlier pre-QK study, not measurements of
the veto callback. The callback path is intentionally a correctness evaluator and is
not used for latency claims.

## 10. Answers to the research questions

1. **How much sparsity is blocked by a few rows?** K<=8 blocks +19.09 pp in the
   unrestricted trace and accounts for 29.54% of baseline-kept tiles; K<=16 blocks
   +25.09 pp.
2. **Which estimator is best?** Oracle final-normalized attention mass, correlation
   0.9356. It is not available in a one-pass online kernel. The best legal online mass
   proxy reaches only 0.4734.
3. **Does V magnitude or variance help?** No. Every tested V norm and the variance
   residual reduce correlation relative to mass alone.
4. **Can visible/stable rows be relaxed?** They can be relaxed only at material quality
   cost. Stable-only safe headroom is too small; critical protection admits dangerous
   visible/stable new maxima.
5. **Do cumulative budgets help?** No. Online cumulative policies miss the work gate;
   oracle loose-budget passes are 96--97% analytically unsafe.
6. **What is the cheapest policy with real H100 speedup?** None. K-veto is cheapest and
   has modeled work headroom, but fails quality before kernel benchmarking is justified.
7. **Is physical-tile pruning viable?** Ordinary exact BLASST pruning is viable and
   already faster than dense. This additional veto-row approximation direction is not
   viable under the training-free, one-pass, 128x64 constraints tested here.

## 11. Reproducibility and artifacts

- Trace integrity: `artifacts/veto_traces_h100_v1/integrity.json`
- Conservative all-new-max sweep: `outputs/veto_pruning/all_newmax/`
- Critical-state sweep: `outputs/veto_pruning/critical_newmax/`
- Cumulative budgets: `outputs/veto_pruning/cumulative_budgets.csv`
- Native quality JSON: `outputs/veto_pruning/quality/`
- Offline analysis: `tools/analyze_veto_pruning.py`
- Cumulative analysis: `tools/analyze_veto_cumulative.py`
- One-step quality evaluator: `eval/eval_veto_llada_one_step.py`
- Full-trajectory evaluator: `eval/eval_veto_llada_generation.py`
- Reference policies: `blasst/veto_pruning.py`

Validation status: 70 reference tests passed with 6 CUDA-only skips in the sandboxed
run. The separate H100 suite ran 10 tests and all passed. `git diff --check` passed.
