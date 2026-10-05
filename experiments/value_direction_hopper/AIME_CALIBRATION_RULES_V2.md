# AIME26 trajectory-aware threshold calibration (v2)

This protocol calibrates the actual DiffusionGemma decoder trajectory, not an
offline distribution of routing scores. It is used by
`aime_temporal_sweep.py` in the versioned `query_adaptive_aime_temporal_v7`
results root.

## Quantity being matched

For each complete calibration generation, count physical 128-query by 64-KV
tiles. For each attention stratum, pool counts before dividing:

`sparsity = sum(skipped eligible tiles) / sum(eligible tiles)`.

The selected policy must match all three pooled rates—whole, local, and
global—to within two percentage points of the requested target. A method is
not considered matched if it reaches the whole-model percentage by trading
local sparsity against global sparsity.

## Two-step protection

Calls 1 and 2 of every 256-token canvas use a fixed early local/global pair
seeded from the previous completed trajectory study. The pair is never raised
by the late search. Calibration rejects a candidate if either early call
the pooled calls-1--2 stratum rate exceeds the following ceilings (pooled
across calibration examples):

| target | early whole | early local/global |
|---:|---:|---:|
| 30% | 45% | 47% |
| 40% | 55% | 57% |
| 50% | 55% | 57% |
| 60% | 70% | 72% |
| 70% | 75% | 77% |

These are safety ceilings, not target sparsities. Calls 3 and later use the
method's normal previous-step state (for temporal sensitivity, beta=3 and
gamma=0.5). Dense calls count as zero skipped tiles if a fallback is added;
none is implicitly added here.

## Independent local/global search

The old AIME sweep applied one shared additive offset to the local and global
log thresholds. That is insufficient because the two strata have different
score distributions and different trajectory feedback. The v2 search is:

1. Load the previous v6 policy only as a starting point.
2. Evaluate a small predeclared set of independent local/global late offsets
   (including cross-offset pairs) using complete generations.
3. Starting from the best measured point, coordinate-refine the late local
   and global log thresholds independently. If a stratum is below target, its
   threshold is increased by 0.45 log units; if above target, it is decreased
   by 0.45. At most six coordinate rounds are evaluated.
4. Rank candidates by maximum absolute whole/local/global error, then by
   passing the early ceilings, denoising steps, and executed tiles.

Generation-level nonmonotonicity is allowed: every candidate is run rather
than interpolated from a score histogram. The final policy is frozen before
any final-seed generation and records every candidate trace, pooled counts,
phase rates, and the exact calibration IDs.

## Limitations

The current AIME manifest contains six calibration problems that are also in
the historical 30-problem evaluation cohort. Therefore this is a reproducible
development calibration, not fresh held-out confirmation. The final report
must show actual O/G/L sparsity and must label a target unattainable if no
candidate passes all constraints.
