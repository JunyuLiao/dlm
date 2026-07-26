# Pre-QK H100 experiment analysis

## Setup

- GPU: NVIDIA H100 80GB HBM3
- Model: GSAI-ML/LLaDA-8B-Instruct, BF16
- Sequence length: 4096
- Dense control: compiled FlashAttention
- Sparse control: existing 128x64 BLASST kernel with no proxy metadata
- Pre-QK kernel: separate metadata-aware/deferred-V 128x64 specialization
- Safety: two warmup tiles, refresh every eight tiles, local and sink anchors,
  and non-cascading `+inf` metadata after a pre-skip

All 45 tests, including the five CUDA kernel tests, passed on the H100.

## Trace and calibration result

The v3 trace contains 9,944,064 records, no duplicate identities, 3,237
spatial coordinates with complete sharing, and 6,629,376 previous-step joins.
The prompt-disjoint split has 20 calibration and 12 held-out samples.

No threshold among 6,342 candidates was certified at false-skip budgets of
1e-2, 1e-3, or 1e-4. Consequently, the only deployable configuration has zero
pre-skips.

Held-out trace endpoints after applying the same kernel safety anchors:

| Transition | Threshold | QK avoidance | False-skip rate | False new-max rate |
|---|---:|---:|---:|---:|
| 0.9 -> 0.5 | 0.001 | 0.066% | 3.030% | 0.758% |
| 0.9 -> 0.5 | 0.048586 | 8.589% | 12.079% | 4.489% |
| 0.5 -> 0.15 | 0.003 | 0.169% | 0 observed | 0 observed |
| 0.5 -> 0.15 | 0.048586 | 5.904% | 1.677% | 1.677% |

The zero-observed-error 0.5 -> 0.15 threshold is not statistically certified
and covers too little work to offset kernel overhead.

## End-to-end batch-1 measurements

| Transition / policy | Dense ms | Sparse ms | Pre-QK ms | QK avoided | Pre-QK vs sparse | Masked agreement vs sparse |
|---|---:|---:|---:|---:|---:|---:|
| 0.9 -> 0.5, safe disabled | 158.81 | 153.20 | 164.11 | 0% | 0.934x | 100% |
| 0.9 -> 0.5, tau=0.001 | 158.84 | 151.42 | 162.82 | 0.038% | 0.930x | 99.561% |
| 0.9 -> 0.5, tau=0.048586 | 158.52 | 152.28 | 163.37 | 7.449% | 0.932x | 98.828% |
| 0.9 -> 0.5, tau=0.3 diagnostic | 157.20 | 152.54 | 160.05 | 23.961% | 0.953x | 95.068% |
| 0.5 -> 0.15, safe disabled | 158.85 | 151.78 | 161.40 | 0% | 0.940x | 100% |
| 0.5 -> 0.15, tau=0.003 | 157.32 | 150.62 | 161.42 | 0.193% | 0.933x | 99.186% |
| 0.5 -> 0.15, tau=0.048586 | 158.14 | 151.33 | 160.69 | 5.729% | 0.942x | 98.860% |

The existing sparse kernel is 3.7--4.7% faster than dense in the two safe
batch-1 runs. The metadata-aware kernel is 5.9--6.7% slower than sparse when
the gate is safely disabled.

## Batch-4 validation

| Transition / policy | Dense ms | Sparse ms | Pre-QK ms | QK avoided | Pre-QK vs sparse | Masked agreement vs sparse |
|---|---:|---:|---:|---:|---:|---:|
| 0.9 -> 0.5, tau=0.048586 | 631.32 | 606.45 | 633.70 | 8.427% | 0.957x | 98.401% |
| 0.5 -> 0.15, tau=0.048586 | 630.88 | 602.23 | 620.28 | 5.856% | 0.971x | 99.104% |

Batch 4 confirms that the apparent lack of speedup and agreement loss are not
single-context artifacts. Pre-skipping recovers only 0.75--0.84% relative to
the same specialized kernel with its gate disabled, while remaining 2.9--4.3%
slower than the original sparse kernel.

## Alternative previous-information sources

The remaining source families were screened on the same prompt split and
safety-filtered physical tiles.

### Previous layer

Formal calibration evaluated 9,216 previous-layer threshold candidates and
certified zero at 1e-2, 1e-3, and 1e-4 false-skip budgets. At thresholds giving
about 1% held-out QK coverage, false-skip rates were approximately 61% at high
noise, 29% at mid noise, and 12% at low noise. Same-step previous-layer scores
are therefore substantially less reliable than previous-step scores.

### Earlier and leader heads

Adjacent-head and fixed-head-0 scores had large false-skip rates throughout
the useful coverage range. A deliberately favorable calibration selected the
best legal source head and threshold independently for every layer/target-head,
requiring at least 128 predictions and zero observed calibration errors. Frozen
held-out results were:

| Noise phase | Selected layer/head pairs | QK avoidance | False-skip rate | False new-max rate |
|---|---:|---:|---:|---:|
| High | 1 | 0.006% | 2.817% | 0% |
| Mid | 152 | 1.914% | 0.494% | 0.087% |
| Low | 401 | 5.922% | 0.410% | 0.410% |

These results are not statistically certified. They also require serial or
multi-launch head dependencies, conflicting with the current parallel-head
kernel and adding overhead beyond the already-unprofitable previous-step path.

### Conservative combinations

Requiring both a previous-step score and the calibrated leader-head condition
reduced held-out false skips, but coverage remained too small:

- mid noise: 1.20% QK avoidance at 0.097% false skips;
- low noise: 4.94% QK avoidance at 0.100% false skips.

The implementable previous-step + previous-layer intersection was much weaker:
the best zero-observed-error low-noise setting avoided only 0.140% of QK. Using
two prior denoising steps also produced zero observed held-out errors but only
0.082% QK avoidance. None approaches the measured 50--70% hardware break-even,
and none passes a family-wise statistical certificate.

## Conclusion

The investigated pre-QK approaches are not a working optimization:

1. no previous-step or previous-layer threshold passes the conservative
   certification rule;
2. thresholds with tolerable empirical error avoid substantially less than 1%
   of QK work;
3. 5.7--8.4% QK avoidance does not offset metadata loads, score stores, and
   control-flow overhead;
4. even 23.96% avoidance remains 4.7% slower than sparse and reduces agreement
   with sparse to 95.1%.

Measured savings imply an approximate 50--70% QK-avoidance break-even for this
end-to-end design. Previous-step, previous-layer, earlier-head, leader-head,
two-step temporal, and conservative intersection policies do not provide that
coverage at an acceptable false-skip rate. The production recommendation is
therefore to keep both `use_pre_qk_kernel` and `enable_pre_skipping` disabled.

If this direction is revisited, it needs a materially different metadata
design: a compact precomputed bitmask amortized across multiple tiles, no FP16
score store on every target tile, and a predictor that demonstrates greater
than roughly 50% physical QK avoidance with near-perfect held-out precision.
