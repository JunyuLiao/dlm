# Hardware-aligned 2D attention-result caching for LLaDA

**Final verdict: no-go for production integration.** Exact physical-tile contributions do have a useful oracle signal at a loose 1% single-tile attention-output tolerance, but no causal policy passed all safety, memory, bandwidth, and H100 latency gates. In particular, the best causal policy reused 14.10% of held-out tiles, fell to 13.57% on the untouched final split, violated the final safety bound, and loaded cached BF16 statistics 1.51× more slowly than recomputing fresh tiles in the H100 diagnostic. A production kernel and full-model enabled trajectory run were therefore deliberately gated off.

![Decision summary](./tile_reuse_decision.png)

## Decision in one table

| Question | Measured answer | Verdict |
|---|---:|---|
| Are exact tiles stable? | 40.28% held-out oracle reuse at ≤1% one-tile full-output error; 11.55% at ≤0.1% | Only at a loose local tolerance |
| Does joint Q/K/V drift help? | 2.22% safe held-out reuse vs 1.08% Q-only | Better, but far below useful coverage |
| Does previous attention mass help? | Best causal policy: 14.10% held-out reuse, 99.22% precision | Material signal, not robust enough |
| Does it generalize to the final split? | 13.57% reuse, 98.50% precision, 1.63% Wilson unsafe upper bound | Fails the 1% safety gate |
| Is dynamic 2D better than fixed regions? | Yes on precision; fixed-region precision was only 12.81–46.06% | Dynamic wins statistically, still loses overall |
| Is full query-tile output reuse viable? | 0.026% oracle reuse at ≤1% | No |
| Is the full BF16 cache practical? | 65 GiB at batch 1; 260 GiB at batch 4 | No |
| Is compact INT8 sufficient? | 33.5 GiB at batch 1 and max row error reached 1.076% | Still too large and numerically borderline |
| Is cached loading cheaper on H100? | Fresh 0.1024 ms vs cached BF16 0.1545 ms for 256 tiles | No; 0.663× speedup |
| Is an optimized cache kernel justified? | Candidate fails online robustness, memory, and latency gates | No |

The complete machine-readable master table is in `master_experiment_table.csv`.

## Experimental setup and scope

### Geometry and model

| Item | Value |
|---|---:|
| Model | `GSAI-ML/LLaDA-8B-Instruct` |
| GPU | NVIDIA H100 80GB HBM3 |
| Sequence length | 4096 |
| Denoising steps | 16 |
| Physical query tile | 128 rows |
| Physical KV tile | 64 rows |
| Head dimension | 128 |
| Layers | all 32 |
| Sampled heads | 0, 7, 15, 31 |
| Contexts | 8 disjoint prompts |
| Context splits | 2 development, 2 calibration, 2 held-out, 2 final benchmark |
| Exact paired tile records | 983,040 |

One deterministic complete query tile was traced per context. Every trace group contains all 64 KV tiles, all 32 layers, and four heads. The 983,040 count is:

```text
8 contexts × 15 step transitions × 32 layers × 4 heads × 64 KV tiles
```

Integrity validation found 983,040 unique identities, no duplicates, no incomplete 64-tile groups, no non-finite values, 128/128 expected raw tensor dumps, and 8/8 complete trajectories. Raw `(m,l,u)` tensors were retained for layer 0/head 0; summary counterfactuals were retained for every sampled layer/head.

This breadth is enough to reject the current design, but not to claim a universal property of every query tile or workload. The head and query-tile subsampling is an explicit limitation.

## Definitions: what every reported metric means

For one query tile `g`, KV tile `j`, and query row `i`, the cached sufficient statistics are:

```text
m[g,j,i] = maximum QK score within KV tile j
l[g,j,i] = sum exp(score - m) within KV tile j
u[g,j,i] = sum exp(score - m) V within KV tile j
```

Two components `a` and `b` compose by using `m=max(m_a,m_b)`, rescaling both `l` and `u` into that shared exponential frame, and adding them. The final output is `u/l`. This composition is algebraically exact.

| Metric | Definition and interpretation |
|---|---|
| **Reuse fraction / coverage** | Reused physical tiles divided by all eligible physical tiles. This is the fraction of QK, softmax, V load, and PV work the policy attempts to replace. |
| **Local tile-output error** | Relative vector error between current and previous `u/l` for one tile in isolation. This was used diagnostically but is not the primary oracle because a low-mass tile can change locally without affecting full attention. |
| **One-tile full-output error** | Maximum over query rows of the relative error in the *complete* current attention output after replacing exactly one current tile with its previous-step `(m,l,u)`, while all other tiles remain current. This is the primary reuse effect. |
| **Oracle reusable** | A tile whose measured one-tile full-output error is no larger than the stated threshold. The oracle uses future/current attention information and is only an upper bound, not an implementable policy. |
| **Prediction precision** | Among tiles the causal policy reused, the fraction that the oracle labeled safe at the 1% threshold. `1 - precision` is the empirical unsafe rate. |
| **Wilson 95% unsafe upper bound** | A binomial 95% upper confidence bound on the unsafe rate. Policies were admitted only when this was ≤1% on calibration/validation. This penalizes both observed failures and small sample sizes. |
| **Oracle-safe recall** | Safe tiles selected by the policy divided by all oracle-safe tiles. It measures how much of the available oracle opportunity is recovered. |
| **Maximum reused effect** | Largest one-tile full-output error among selected tiles. This exposes rare severe mistakes that averages hide. |
| **Q/K/V drift** | Relative change between current and previous representations, summarized per tile using Frobenius, mean row, maximum row, q90, or q99 row-relative norms. |
| **Previous tile mass** | The previous step's fraction of the full softmax denominator assigned to the physical tile, summarized over query rows. It is legal at decision time because it comes from cached history. |
| **Log-Pearson correlation** | Pearson correlation after applying `log1p` to predictor score and tile effect. Higher is better, but correlation alone does not establish a safe decision boundary. |
| **Final query-tile output error** | Relative error between complete 128-row attention outputs at adjacent denoising steps. This tests coarser whole-query reuse. |
| **Masked top-1 agreement** | Fraction of masked positions whose argmax token matches the dense baseline. |
| **Hidden/logit cosine** | Cosine similarity between reference and candidate tensors; one is ideal. |
| **Speedup** | Baseline latency divided by candidate latency. Values above one are faster; below one are slower. |

Calibration used context-disjoint thresholds in phase × eight-layer buckets. Each selected prefix required at least 256 calibration examples and a ≤1% Wilson upper unsafe bound. The final-benchmark prompts were not used for policy selection.

## Exact composition validation

The reference implementation in `blasst/tile_reuse.py` independently forms every 128×64 component and composes fresh or cached statistics. Unit tests verify exact agreement with direct dense attention in FP32 and verify mixed fresh/cached composition.

At model level, a 256-token BF16 run compared the ordinary dense FlashAttention path with the all-fresh tiled reference:

| Metric | Result |
|---|---:|
| Masked-logit relative error | 0.547% |
| Masked-logit cosine | 0.999985 |
| Masked top-1 agreement | 98.425% |
| Hidden relative error | 1.385% |
| Hidden cosine | 0.999904 |

The formula is exact; the nonzero model-level difference is BF16 accumulation/order divergence between implementations, compounded through 32 layers. Because top-1 agreement is below the desired 99%, even an optimized all-fresh tiled implementation would need closer numerical matching before becoming a production baseline.

## Track-by-track results

### Track A — oracle physical-tile reuse: conditional pass

Held-out oracle coverage by one-tile full-output error threshold:

| Maximum relative error | Reusable tiles |
|---:|---:|
| 0.001% (`1e-5`) | 2.14% |
| 0.01% (`1e-4`) | 2.99% |
| 0.05% (`5e-4`) | 6.73% |
| 0.1% (`1e-3`) | 11.55% |
| 0.5% (`5e-3`) | 30.15% |
| 1.0% (`1e-2`) | 40.28% |

The median single-tile effect was 1.816%, p90 was 27.74%, and p99 was 169.95%. The 30–40% oracle gate passes only at the loosest 1% local perturbation threshold. Moreover, one-tile effects are not additive: simultaneously reusing many individually safe tiles can shift normalization and compound their errors.

An earlier local-`u/l` oracle reported only 2.15% safe tiles at 1%. That metric was too pessimistic because it ignored tile mass. The schema-v2 exact counterfactual corrected this: low-mass tiles can change substantially in isolation while barely affecting the full attention output.

### Track B — representation-drift prediction: no-go

Best held-out policies satisfying the ≤1% Wilson safety constraint:

| Predictor | Reuse | Precision | Oracle-safe recall |
|---|---:|---:|---:|
| Q only | 1.08% | 99.44% | 2.68% |
| K/V only | 1.04% | 99.80% | 2.57% |
| Joint Q+K | 1.88% | 99.68% | 4.66% |
| Joint Q+K+V | 2.22% | 99.74% | 5.51% |

Joint 2D drift is genuinely better than Q-only reuse, but the absolute gain is about 1.14 percentage points. Plain drift correlation peaked around 0.43 and could not separate rare high-impact changes well enough.

The strongest refinement multiplied current drift by cached previous tile mass. Correlation rose to 0.70 because the predictor now represents both “how much the tile mattered” and “how much its inputs moved.” In a causal cache simulation with a sum staleness budget:

| Split | Reuse | Precision | Unsafe rate | Wilson unsafe upper | Oracle-safe recall |
|---|---:|---:|---:|---:|---:|
| Calibration | 12.86% | 99.21% | 0.79% | ~1.0% | — |
| Held-out | 14.10% | 99.22% | 0.78% | 0.88% | 34.73% |
| Final benchmark | 13.57% | 98.50% | 1.50% | 1.63% | 32.58% |

It therefore passes held-out safety but fails the untouched final split. Maximum selected one-tile errors were also extreme (155.1% held-out and 96.7% final), showing that a small number of false decisions can be catastrophic.

Reuse was concentrated in layers 0–8 (roughly 36–52% by layer), fell to 15.4% at layer 9 and 7.9% at layer 10, and was zero in layers 16–31. High-noise held-out reuse was only 5.28%; mid-noise was 13.00%; low-noise was 25.44%, with the low-noise bucket itself failing the 1% Wilson bound. This narrow concentration violates the requirement that the optimization not work only in a few layers or one noise phase.

### Track C — token-state-aware reuse: no-go

| Direct policy, held-out | Reuse | Precision | Wilson unsafe upper |
|---|---:|---:|---:|
| Stable query × stable KV | 3.51% | 97.43% | 2.93% |
| Stable query × any KV | 8.62% | 93.14% | 7.21% |
| Any query × stable KV | 4.93% | 96.38% | 3.97% |
| Protect masked/new tokens | 21.41% | 64.81% | 35.60% |

Token visibility is not a sufficient proxy for attention stability. State protection can be an extra guard, but it cannot rescue the predictor by itself. Applying critical-state protection to the mass-based causal policy roughly halved coverage and still failed final-split safety.

### Track D — fixed refresh: no-go

| Schedule | Nominal tile reuse | Reused steps with ≤1% complete-query error | Median complete-query error |
|---|---:|---:|---:|
| Reuse 1, then refresh | 50.0% | 4.69% | 5.40% |
| Reuse 2, then refresh | 62.5% | 3.75% | 12.36% |
| Reuse 3, then refresh | 75.0% | 2.08% | 11.12% |

Periodic reuse is simple but unsafe. Adjacent denoising steps are not uniformly smooth, so age alone cannot identify dangerous transitions.

### Track E — cumulative staleness: signal exists, candidate still no-go

Sum, root-sum-square, maximum, independent, and age-limited variants were simulated causally. The best broad policy was cached previous maximum mass × `(Q_max + K_max + 0.5 V_max)` with sum accumulation and refresh on budget failure. Limiting maximum age to one or two reduced reuse to approximately 9.4% and 12.6% without solving the final robustness issue.

Raw multi-tile composition tests showed apparently excellent cumulative results for layer 0, including 46.2% representation-drift reuse with zero measured mixed-output error. This is not a general result: the raw data cover only layer 0/head 0 and many early-layer statistics were bitwise unchanged. It cannot override the all-layer counterfactual and split-generalization failures.

Cumulative control reduces long stale chains; it does not fix a predictor that occasionally assigns a tiny score to a high-impact change. Those rare underestimates dominate the safety failure.

### Track F — whole query-tile output reuse: decisive no-go

On held-out data, only 0.026% of complete query-tile outputs changed by ≤1% from the previous step. Median change was 78.51%, p99 was 610.80%, and a calibrated Q-drift policy selected nothing. The 1 GiB batch-1 cache is attractive, but the reusable object is not temporally stable.

### Track G — fixed region versus dynamic 2D: dynamic wins, but not enough

| Fixed region, held-out | Coverage | Precision |
|---|---:|---:|
| First KV tile / prefix | 1.56% | 12.81% |
| Outside local radius 4 | 85.94% | 39.34% |
| Outside local radius 8 | 73.44% | 38.71% |
| Local radius 4 only | 14.06% | 46.06% |

There is no safe fixed prefix/local split in these traces. Dynamic prior-mass × drift is meaningfully more precise, but its 13–14% coverage and robustness failure are insufficient for hardware deployment.

## Cache memory, precision, and traffic

There are 2,097,152 physical tiles at batch 1:

```text
32 query tiles × 64 KV tiles × 32 layers × 32 heads
```

| Cache format | Bytes/tile | Batch 1 | Batch 4 | Numerical result |
|---|---:|---:|---:|---|
| FP32 `m/l` + BF16 `u` | 33,792 | 66.0 GiB | 264.0 GiB | Reference-quality cache |
| BF16 `m/l/u` | 33,280 | 65.0 GiB | 260.0 GiB | Mean max-row error 0.229%; p99 0.322% |
| FP32 `m/l` + global FP8 `u` | 17,408 | 34.0 GiB | 136.0 GiB | Mean max-row error 1.60%; fails 1% |
| BF16 `m/l` + rowwise INT8 `u` + scales | 17,152 | 33.5 GiB | 134.0 GiB | Mean 0.520%; p99 0.931%; max 1.076% |
| Final BF16 query output only | 32,768/query tile | 1.0 GiB | 4.0 GiB | Storage good; stability fails |

Restricting rowwise INT8 to layers 0–8 would still occupy 9.42 GiB at batch 1 and 37.69 GiB at batch 4. It also adds dequantization and scale traffic. Admission-only caching would reduce residency but requires irregular indices and writes; with short reuse runs it may write an entry only to read it once.

For each reused BF16 tile:

| Quantity | Amount |
|---|---:|
| QK FLOPs avoided | 2,097,152 |
| PV FLOPs avoided | 2,097,152 |
| K bytes avoided | 16,384 |
| V bytes avoided | 16,384 |
| Total K+V bytes avoided | 32,768 |
| BF16 `(m,l,u)` cache bytes read | 33,280 |
| Rowwise INT8 cache bytes read | 17,152 |

Thus BF16 result reuse reads slightly *more* data than the K+V loads it replaces, before cache metadata and writes. H100 can recompute dense 128×64 work efficiently with tensor cores, while the cached `u` path is a large vector load plus FP32 online-softmax composition.

## H100 microbenchmark and layout feasibility

The framework-level diagnostic batched 256 independent physical tiles:

| Path | Latency | Speedup vs fresh |
|---|---:|---:|
| Fresh QK + softmax + PV | 0.1024 ms | 1.000× |
| Cached BF16 load + compose | 0.1545 ms | 0.663× |
| Cached rowwise INT8 + dequantize + compose | 0.1711 ms | 0.599× |
| Mixed, 14% contiguous reuse | 0.2690 ms | 0.381× vs fresh-all diagnostic |
| Mixed, 14% random reuse | 0.2658 ms | 0.385× |

This is not a fused persistent kernel and therefore is not an absolute lower bound on a custom implementation. It is, however, a valid gate: there is no intrinsic cached-path advantage to justify a much larger kernel project. A fused kernel cannot remove the `u` bytes, cache capacity, update writes, dequantization, or online composition.

The best held-out dirty map averaged 3.42 reuse/dirty transitions per 64 KV tiles, with p95 18. Reused runs averaged 5.03 tiles but had median length 2; dirty runs averaged 20.95 with median 4. Coarse range merging would recover regularity by recomputing many of the already scarce reusable tiles. A work queue would retain coverage but introduce index construction and load imbalance.

No production kernel, gate-disabled full-model overhead measurement, or candidate full-forward benchmark was implemented because the staged protocol required a candidate to pass offline robustness, memory, and cached-load break-even first.

## Why this design fails fundamentally

This is not mainly a threshold-tuning failure. Three structural facts collide:

1. **The reusable result is too large.** The `u` tensor contains 128×128 values per tile. Loading it costs approximately the K+V traffic saved, and storing every tile consumes 65–66 GiB per batch element.
2. **The useful event is not “representations moved little.”** A tile is safe when its *current normalized contribution* is small or stable. Small Q/K/V norm drift can still reorder dot products and softmax mass; large local changes can be harmless when tile mass is tiny. Previous mass improves prediction because it measures importance, but it becomes stale exactly when routing changes.
3. **Rare errors and simultaneous decisions dominate.** Average correlation and 99% precision look strong, yet a few selected tiles have 97–155% effects. Multiple individually small substitutions also share the same normalization, so their errors can compound. Conservative calibration then pushes coverage below the level needed to amortize metadata and irregular execution.

This explains why further scalar-threshold sweeps are unlikely to change the verdict.

## Recommended next research direction: certified omission, not result reuse

The next serious experiment should remove the cached `u` tensor entirely. Instead of predicting that a previous result is reusable, construct a **one-sided temporal certificate** proving that the current tile cannot receive enough softmax mass to matter. Certified tiles are omitted; uncertified tiles are computed as ordinary 128×64 blocks.

Let `q=q0+Δq` and `k=k0+Δk`, where `0` denotes the previous denoising step. For query row `i` and KV tile `j`, every current score perturbation satisfies:

```text
|q·k - q0·k0| / sqrt(d)
  <= (||Δq_i|| max_j||k0||
      + ||q0_i|| max_j||Δk||
      + ||Δq_i|| max_j||Δk||) / sqrt(d)
  = ε[i,j]
```

If the previous tile log-normalizer is `z0[i,j] = m0[i,j] + log(l0[i,j])`, then:

```text
current log Z[i,j] <= z0[i,j] + ε[i,j]
current global log Z[i] >= max_a (z0[i,a] - ε[i,a])
```

Therefore a conservative current mass bound is:

```text
mass_upper[i,j]
  = exp(z0[i,j] + ε[i,j] - max_a(z0[i,a] - ε[i,a]))
```

This decision uses no current 128×64 QK matrix. It requires previous Q/K (or conservatively quantized versions), row norms, one `z0` value per query-row/tile, and current Q/K deltas. Crucially, it does **not** load or store `u` and it never asserts that a stale contribution is correct.

### Why this is materially different

- It replaces a fallible two-sided regressor with a one-sided mathematical bound.
- It targets negligible *current mass*, the causal factor revealed by the trace analysis.
- It skips QK, exponentiation, V load, and PV without reading an equally large cached result.
- Safety margins can explicitly include BF16/FP8 quantization error.
- Bounds for many tiles can be summed into a query-row error budget, addressing simultaneous-skip accumulation before execution rather than after an error occurs.

The likely difficulty is bound looseness, not silent unsafe skips. If the exact temporal bound is too loose, the only worthwhile refinement is a hierarchical residual certificate: project Q/K to a small rank `r` (for example 8 or 16), evaluate the cheap projected interaction, and bound the omitted residual with cached residual norms. That computes a small screening product, not the full QK matrix, while retaining a one-sided guarantee.

### Practical falsification experiment

Do not start with a kernel. Extend traces with exact per-row `||ΔQ||`, per-KV-tile `max||K0||`, `max||ΔK||`, previous `z0`, V-norm bounds, and the exact combined-output error after omitting all certified tiles. Then:

1. Evaluate certificate tightness with no fitted threshold: coverage at mathematically bounded absolute error budgets.
2. Process one anchor KV range first and use its current normalizer as a stronger denominator lower bound.
3. Aggregate candidate bounds per query row before skipping so the total budget, not each isolated tile, is controlled.
4. Test 4-, 8-, and 16-tile contiguous super-regions; accept some recomputation to keep Hopper scheduling regular.
5. Stop if certified omission is below 25–30%, if metadata/QK-delta traffic exceeds 10% of dense attention time, or if the bound misses any unaccounted numerical error.
6. Only then build a persistent query-tile kernel in which an anchor range is fresh, certified ranges are omitted, and all remaining work stays in complete 128×64 tiles.

This experiment is higher value than another learned threshold because it directly attacks both fundamental blockers: unsafe prediction and `u` bandwidth. If the certificate is too loose, the evidence would indicate that no robust no-QK pre-skip can be extracted from temporal information without changing or fine-tuning the model to expose an explicit routing signal.

## Go/no-go audit

| Required gate | Result |
|---|---|
| Oracle physical-tile reuse ≥30–40% | **Pass only at 1% single-tile error:** 40.28% held-out |
| Online policy recovers substantial oracle coverage | **Weak:** 34.73% oracle-safe recall, 14.10% absolute reuse |
| Full-trajectory masked agreement ≥99% | **Not run:** candidate failed earlier mandatory gates |
| Hidden stability across trajectory | **Not run:** same gate |
| Exact generated-sequence agreement | **Not run:** same gate |
| Cache memory practical | **Fail:** 65 GiB BF16 batch 1; compact format 33.5 GiB |
| Cached load cheaper than recomputation | **Fail:** 0.663× BF16, 0.599× INT8 |
| Gate-disabled overhead small | **Not integrated:** no qualifying candidate |
| H100 attention speedup >5% | **Fail at microbenchmark gate** |
| Full-forward improvement | **Not run:** no qualifying candidate |

The absence of full-trajectory candidate metrics is a gate outcome, not evidence of quality. It would be misleading to spend a full integration run on a policy that is already unsafe on the final split and slower at its core operation.

## Reproduction map

```bash
conda activate ljy_dlm

# Validate the exact trace dataset
python tracing/validate_tile_reuse_trace.py \
  --trace-dir artifacts/tile_reuse_h100_v2

# Reproduce offline oracle, drift, state, region, refresh, and cumulative analyses
python tools/analyze_tile_reuse.py \
  --trace-dir artifacts/tile_reuse_h100_v2 \
  --output-dir outputs/tile_reuse_h100_v2

# Reproduce the causal mass-aware cache simulation
python tools/simulate_causal_mass_reuse.py \
  --trace-dir artifacts/tile_reuse_h100_v2 \
  --output outputs/tile_reuse_h100_v2/causal_mass_policy.json

# Reproduce cache-precision analysis from retained raw tensors
python tools/analyze_tile_reuse_precision.py \
  --trace-dir artifacts/tile_reuse_h100_v1 \
  --output outputs/tile_reuse_h100_v2/cache_precision.json

# H100 framework-level diagnostic
python scripts/benchmark_tile_reuse_h100.py \
  --tiles 256 --warmup 10 --repeats 50 \
  --output outputs/tile_reuse_h100_v2/h100_microbenchmark.json

# Reference/unit checks
python -m unittest tests.test_tile_reuse tests.test_tile_reuse_trace
```

Primary artifacts:

- `summary.json`: oracle, correlations, memory/traffic model, and aggregate results
- `calibrated_thresholds.json`: frozen context-disjoint predictor thresholds
- `causal_mass_policy.json`: causal simulations, split results, layer/phase breakdown, and fragmentation
- `cache_precision.json`: BF16/FP8/INT8 numerical ablation
- `h100_microbenchmark.json`: fresh, cached, compact, mixed-ratio, and fragmentation diagnostics
- `master_experiment_table.csv`: compact cross-track decision table
- `tile_reuse_decision.png`: report figure

