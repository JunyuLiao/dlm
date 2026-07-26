# Certified pre-QK physical-tile omission

**Verdict: mathematically valid, practically no-go.** The causal certificate never violated its score interval or tile-mass upper bound, including after repeated omissions. Its useful coverage is nevertheless two orders of magnitude below the 25–30% minimum needed to justify hardware integration.

![Certified omission result](./certified_omission.png)

## What was implemented

For each query row `i` and physical KV tile `j`, the implementation maintains a causal interval for the tile log-normalizer:

```text
lower[i,j] <= log(sum_k exp(q_i · k / sqrt(d))) <= upper[i,j]
```

For adjacent Q/K representations, every score change is bounded without computing the current 128×64 QK matrix:

```text
epsilon[i,j] = (
    ||delta_q_i|| max_j ||k_previous||
  + ||q_previous_i|| max_j ||delta_k||
  + ||delta_q_i|| max_j ||delta_k||
) / sqrt(d)
```

The interval propagates as:

```text
candidate_lower = previous_lower - epsilon
candidate_upper = previous_upper + epsilon
```

A denominator lower bound and tile-mass upper bound then follow:

```text
log_denominator_lower[i] = logsumexp_j(candidate_lower[i,j])
mass_upper[i,j] = exp(candidate_upper[i,j] - log_denominator_lower[i])
```

Tiles are greedily omitted only while the **sum** of their mass upper bounds remains below the budget for every one of the 128 query rows. Omitted tiles keep the propagated interval. Recomputed tiles reset both bounds to the exact current log-normalizer. This makes consecutive decisions causal; the tracer never substitutes current dense statistics into the decision state.

The decision path requires no current full QK matrix and no cached `u` value tensor. Dense QK is still computed by the research tracer after the decision to validate the bounds and exact simultaneous-omission error.

## All-layer H100 pilot

| Item | Value |
|---|---:|
| Model | LLaDA-8B-Instruct |
| GPU | H100 80GB HBM3 |
| Context | 4096 tokens |
| Denoising trajectory | 16 steps |
| Layers | all 32 |
| Heads | head 0 |
| Physical geometry | 128 query × 64 KV × 128 dimension |
| Paired tile records | 30,720 |
| Complete query groups | 480 |
| Interval violations | 0 |
| Mass-bound violations | 0 |

This is a gate experiment, not a general quality benchmark: it uses one context and one head. The gap to the required coverage is large enough that expanding the sample cannot make the current design hardware-viable.

## Main results

| Aggregate mass budget | Causal omission | Causal max query error | Exact-denominator diagnostic | Exact-mass oracle | Oracle max query error |
|---:|---:|---:|---:|---:|---:|
| 0.25% | 0.160% | 0.002% | 0.456% | 4.271% | 1.372% |
| 0.50% | 0.267% | 0.374% | 0.700% | 6.468% | 2.850% |
| 1.00% | 0.446% | 0.714% | 1.035% | 9.697% | 5.503% |

Definitions:

- **Causal omission** uses only propagated history and current Q/K norm changes.
- **Exact-denominator diagnostic** gives the temporal numerator bound an impossible perfect current denominator. It measures the maximum benefit a fresh anchor strategy could provide without fixing the numerator bound.
- **Exact-mass oracle** uses the current QK-derived softmax mass. It is an upper bound on any mass-only omission method.
- **Max query error** is the maximum, over all 480 layer/head/step query groups, of the maximum-row relative attention-output error after omitting all selected tiles simultaneously.

The 0.5% causal budget is the most defensible measured operating point: every query group remains below 1%, but only 0.267% of physical tiles are omitted. Even with zero decision overhead, the theoretical attention-only speedup is:

```text
1 / (1 - 0.002669) = 1.00268x
```

That is approximately 0.27%, far below the required 5% measured attention speedup and before paying any certificate cost.

## Why it fails

### 1. The temporal numerator bound is extremely loose

Maximum propagated interval width reached 367.6 score units. Cauchy–Schwarz discards the direction of 128-dimensional Q/K changes and uses the largest K change in an entire 64-row tile. A perfect current denominator raises 0.5%-budget coverage only from 0.267% to 0.700%. Therefore computing fresh anchor tiles cannot rescue the design.

### 2. Exact mass itself offers little safe omission

At a 0.5% mass budget, even the exact-current-mass oracle omits only 6.47% of tiles and already reaches 2.85% maximum query error. Reducing the budget restores quality but pushes oracle coverage lower still. This is the fundamental gate: a perfect score predictor cannot create enough negligible total attention mass.

This differs from the earlier 40.28% single-tile *reuse* oracle. Replacing a tile with a similar stale contribution preserves its mass and value contribution; omission removes the contribution and renormalizes everything else. Many tiles are individually replaceable, but very few can be removed together under a strict aggregate budget.

### 3. Metadata work overwhelms the opportunity

At native batch 1, a full implementation would require approximately:

| State/work | Cost |
|---|---:|
| Previous BF16 Q+K | 2.0 GiB |
| FP32 log-normalizer lower+upper intervals | 2.0 GiB |
| BF16 intervals, requiring outward-error inflation | 1.0 GiB |
| Row–tile bound evaluations per denoising step | 268,435,456 |

These costs are much smaller than caching `u`, but are not small compared with a 0.267% work reduction.

### 4. Opportunity is narrow

At the 0.5% causal budget, only layers 0, 3, 7, and 8 omitted anything; layer 7 reached 6.77%, while every other sampled layer was at or below 0.94%. Optimizing one layer with 6.77% omission changes total 32-layer attention work by only about 0.21%.

## Refinements considered and rejected

- **More/fresher denominator anchors:** ruled out by the exact-denominator diagnostic.
- **Tighter low-rank or signed Q/K sketches:** could reduce numerator looseness, but the exact-current-mass oracle is already far below the hardware gate. It cannot overcome the lack of removable aggregate mass.
- **Larger mass budget:** raises coverage slowly and violates the 1% output target; the mathematical `Vmax`-normalized bound is `2 × mass_budget`, not a 1% relative-output guarantee.
- **Layer-specific deployment:** the few useful layers contribute too little to full-forward latency.
- **Learned mass prediction:** cannot beat the exact-current-mass omission oracle and would remove the certification property.

## Final decision

Do not implement a Triton/Hopper omission kernel and do not collect a large multi-context dataset for this candidate. The failure is at the oracle opportunity and cost/coverage levels, not a calibration issue.

The reusable implementation remains valuable as a correctness/reference tool:

- `blasst/certified_omission.py`: interval propagation, mass bounds, aggregate selection, and exact omission composition
- `tracing/tile_reuse_collector.py`: causal state simulation and noncausal bound decomposition
- `tools/analyze_certified_omission.py`: coverage, safety, layer/phase, and memory analysis
- `tests/test_certified_omission.py`: mathematical invariant and composition tests

## Reproduction

```bash
conda activate ljy_dlm

python -m unittest tests.test_certified_omission tests.test_tile_reuse_trace

python tracing/validate_tile_reuse_trace.py \
  --trace-dir artifacts/certified_omission_all_layers_h100_v1

python tools/analyze_certified_omission.py \
  --trace-dir artifacts/certified_omission_all_layers_h100_v1 \
  --output outputs/certified_omission_all_layers_h100_v1/summary.json
```

