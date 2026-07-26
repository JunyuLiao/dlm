# Training-free BLASST query-row regrouping

## Decision

**No-go for optimized-kernel integration at 4096 tokens on LLaDA-8B.**

There is measurable row-grouping headroom, but the practical previous-step
method does not meet the experiment plan's gate.  Exact per-head bitmap sorting
from the adjacent preceding denoising step improves physical tile sparsity by
3.56 percentage points and reduces required P@V/V tiles by 5.63% on average.
The requested thresholds were at least 15 percentage points or 20% work
reduction over a substantial fraction of layers/steps.

The H100 prototype costs 0.305 ms per layer for signature packing, 32 per-head
sorts, Q gather, and output scatter.  The complete sparse attention kernel is
0.723 ms per layer at the same batch-1, 4096-token, 32-head shape.  Even the
impossible assumption that all kernel time scaled with required P@V tiles gives
only 0.041 ms of potential saving, before accounting for unchanged QK work and
irregular grouped accesses.  Optimized integration is therefore not justified.

## What was implemented

- Opt-in exact Triton row-mask instrumentation.  The normal kernel allocates
  and stores nothing when disabled.  Each stored byte is the keep vote for one
  original query row and KV tile; a unit test reconstructs the kernel's parent
  tile counter from these votes.
- `--dump-blasst-row-masks [PATH]` and `--dump-only` in
  `scripts/llada_blasst_kernel_benchmark.py`.  Dumps are request-isolated,
  include layer/step/noise/tile geometry, token state and original row order,
  and pack the KV dimension by eight.
- Offline identity, bitmap, Top-K, SimHash, bucket, MinHash and plan-specified
  union-greedy grouping, with per-head and shared-head permutations, state
  classes, spatial windows, physical sparsity/work estimates, and temporal
  Jaccard.
- A semantics-preserving reference gather/attention/scatter path already in
  `blasst.runtime`; tests cover arbitrary permutations, inverse scatter,
  original-position attention bias, request boundaries and lambda-zero dense
  equivalence.
- H100 overhead benchmark and result plot.

The implementation stores the minimum required Boolean mask rather than full
row scores.  Current-step oracle masks are used only in the simulator; the
practical results use step `t-1` to group step `t`.

## Experimental setup

| Item | Value |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3 |
| Model | `GSAI-ML/LLaDA-8B-Instruct` |
| Shape | batch 1 per request, sequence 4096, 32 heads, head dim 128 |
| Physical tile | 128 query rows x 64 KV rows |
| Adjacent trajectories | 90→89%, 50→49%, 15→14% remaining masks |
| Threshold schedule | existing calibrated high/mid/low schedule |
| Trace size | 192 records, 195.9 MiB packed |
| Layers | all 32 |

The original 90→50→15 trace is retained as a coarse-step stress test.  Its
mean temporal Jaccard is only 0.361.  The decision uses adjacent pairs, whose
mean row/head Jaccard is 0.877 and better represents a one-step reuse policy.

## Sparsity results

The best oracle-like practical-signature upper bound uses the current mask and
one bitmap permutation per head.  The realizable method uses the preceding
adjacent-step mask.

| Target mask ratio | Original physical sparsity | Current-step per-head | Gain | Previous-step per-head | Gain | Previous work reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.89 | 7.97% | 17.12% | +9.15 pp | 8.46% | +0.49 pp | 0.50% |
| 0.49 | 28.45% | 40.23% | +11.78 pp | 33.86% | +5.41 pp | 7.42% |
| 0.14 | 47.96% | 57.47% | +9.51 pp | 52.74% | +4.78 pp | 8.96% |
| **Mean** | **28.61%** | **38.75%** | **+10.15 pp** | **32.17%** | **+3.56 pp** | **5.63%** |

No previous-step record reaches the 15-point gate or 20% work-reduction gate.
The previous-step result recovers about 35% of the current-step bitmap gain.
All layer means are positive, but individual layer/state records range from
-1.87 to +11.64 percentage points.

Sharing defeats the opportunity: two heads with one permutation produce only
+0.63 points in a locality-preserving 512-row window and essentially zero in a
1024-row window; four or more heads regress.  LLaDA uses MHA, so a KV-head
permutation is the same as the costly per-head case.

Spatial windows trade benefit for locality.  With previous-step per-head
bitmaps, global/1024/512/256-row windows yield +3.56/+3.14/+2.83/+1.90 points.
The existing coarse implementation only permuted inside a 128-row parent tile,
which mathematically cannot alter that tile's KV union and therefore could not
test the central hypothesis.

## Pattern-rule refinement

Exact bitmap sorting is the only positive previous-step rule:

| Signature | Mean physical-sparsity gain |
| --- | ---: |
| Exact bitmap | +3.56 pp |
| Top-K (best K=2) | -3.72 pp |
| MinHash (best 2 hashes) | -4.51 pp |
| SimHash bucket (best 4 bits) | -7.33 pp |
| SimHash (best 8 bits) | -13.80 pp |

These robust hashes lose the strong original positional structure faster than
they capture row-mask similarity.  State-aware sorting also regresses because
the input is already spatially organized and state partition boundaries leave
poorly filled physical tiles.

The sampled plan-specified greedy union rule does not beat bitmap sorting.  Its
seed-by-original-order heuristic greedily fills one head or an aggregated
surrogate, while the metric sums unions across all heads; optimizing a shared
thresholded surrogate can therefore make the real multi-head union worse.

## Runtime breakdown

H100 medians for one layer, batch 1, 4096 rows, 32 heads:

| Component | Time (ms) |
| --- | ---: |
| Pack 64-bit per-head signatures | 0.130 |
| Stable per-head sort | 0.069 |
| Gather Q | 0.065 |
| Scatter output | 0.065 |
| Combined measured path | 0.305 |
| Entire baseline sparse attention kernel | 0.723 |

This prototype uses explicit gather/scatter.  Indirect kernel reads could
remove those two 0.065 ms operations, but signature+sort alone is 0.199 ms,
almost five times the maximally optimistic 0.041 ms tile-work saving.  It also
cannot improve QK because the grouping decision is learned only after the
preceding step.

No end-to-end enabled latency is reported: the plan explicitly says to proceed
to the execution prototype only after the offline 15-point/20%-work gate.  Both
offline benefit and the isolated H100 cost test reject that stage.  Reporting
an enabled end-to-end number would require implementing a path already shown
incapable of net speedup.

## Reproduction

```bash
conda run -n ljy_dlm python scripts/llada_blasst_kernel_benchmark.py \
  --context-length 4096 --num-contexts 1 \
  --dump-blasst-row-masks outputs/query_regrouping/row_masks_adjacent_4096.pt \
  --dump-temporal-pairs 0.90:0.89,0.50:0.49,0.15:0.14 --dump-only

conda run -n ljy_dlm python tools/analyze_query_regrouping_refined.py \
  outputs/query_regrouping/row_masks_adjacent_4096.pt \
  --output-dir outputs/query_regrouping/refined_adjacent

conda run -n ljy_dlm python tools/analyze_query_regrouping_signatures.py \
  outputs/query_regrouping/row_masks_adjacent_4096.pt \
  --output outputs/query_regrouping/signature_ablation.json

conda run -n ljy_dlm python scripts/query_regrouping_overhead_benchmark.py \
  outputs/query_regrouping/row_masks_adjacent_4096.pt \
  --layer 15 --mask-ratio 0.49 \
  --output outputs/query_regrouping/overhead_h100.json
```

The main plot is
`outputs/query_regrouping/refined_adjacent/regrouping_results.png`; tabular
per-layer results and temporal stability are in the same directory.
