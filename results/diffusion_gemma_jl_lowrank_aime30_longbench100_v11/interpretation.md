# Low-rank centered routing: interpretation

All 1,260 outputs passed the final audit and independent raw-only report regeneration. No execution failures or audit violations were recorded in v11. The earlier v10 CPU preflight compatibility failure is preserved separately; it performed no new GPU inference.

## AIME26

Gaussian1 scored **14/30 (46.7%)** at **50.04%** physical sparsity (global 51.53%, local 49.68%). Dense scored 17/30; Gaussian2 and Gaussian24 scored 16/30; Gaussian8, Gaussian32 and full-dimensional centered each scored 15/30. Gaussian1's paired difference from full-dimensional was −3.33 percentage points, with a 95% interval of [−20.00, +13.33]. This does not establish a meaningful accuracy difference.

## LongBench v2, all 100 cached questions

All sparse conditions below requested 50% sparsity; the columns report achieved physical sparsity. Scores are the pinned NeMo MCQ evaluation with the existing 4,096-token output budget and 32K input cap.

| Method | Correct/100 | Whole sparsity | Global | Local | Retained mass | Token agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 34 | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| Original BLASST | 40 | 43.19% | 47.01% | 23.20% | 93.89% | 7.16% |
| Aggressive BLASST | 42 | 46.53% | 46.19% | 48.30% | 75.88% | 4.77% |
| Mass-only bound | 37 | 48.62% | 48.22% | 50.69% | 74.44% | 4.75% |
| Full-dimensional centered | 42 | 49.31% | 49.19% | 49.91% | 72.43% | 5.27% |
| Gaussian1 centered | 39 | 48.42% | 47.56% | 52.91% | 68.37% | 4.72% |
| Gaussian2 centered | 43 | 47.78% | 46.90% | 52.38% | 69.37% | 3.97% |
| Gaussian8 centered | 40 | 46.32% | 45.51% | 50.56% | 71.51% | 6.34% |
| Gaussian32 centered | 38 | 46.57% | 45.87% | 50.19% | 72.06% | 5.26% |

Gaussian2 has the highest observed score, but it is only one question ahead of full-dimensional centered and aggressive BLASST. Its paired accuracy differences are +1 point [−7, +9] and +1 point [−6, +8], respectively. Local sparsity also differs by 2.47 and 4.07 points. There is no supported winner from these comparisons.

Gaussian2 versus mass-only is +6 points [−2, +14], with overall/global/local sparsities within two points. Gaussian8 versus Gaussian32 is +2 points [−4, +8], with all three sparsities within 0.37 points. These are useful approximately matched comparisons, but remain inconclusive.

## What the diagnostics do and do not show

Small projections can preserve downstream performance without preserving every important vector difference. On the fixed early shared-QKV samples, severe risk-underestimation events for ranks 1/2/8/32 were 6,723/3,683/169/0 out of 20,480 supported AIME row-block events, and 16,274/9,800/1,737/0 out of 473,280 LongBench events. These count projected centered risk below half the full-dimensional risk when true risk is at least 0.05. They are **not** counts of dangerously skipped physical tiles; many events are protected by other queries in the same tile.

The extra distortion does not produce a monotonic accuracy ranking. In particular, Gaussian2 scored highest on LongBench despite less retained attention mass, lower token agreement and higher trajectory-local output error than several competitors. These diagnostics alone are not reliable surrogates for benchmark accuracy.

Calibration transfer is imperfect: Gaussian1/2/8 passed the same 48–52% calibration checks, but final LongBench whole-model sparsity was 48.42/47.78/46.32%, with local layers consistently sparser than global layers. The final set never caused threshold retuning. Original BLASST's attainable point is not a matched 50% result.

## Limits

The headline sets include six historical calibration questions each; noncalibration24/94 and domain results are in the canonical report. All examples were previously examined, not fresh held-out confirmation. This sweep uses one generation seed and one projection seed; different ranks also use different random directions. Bootstrap intervals are paired by prompt, exploratory, and not multiplicity-corrected.

Physical sparsity uses summed skipped/eligible tile counts. Attention mass and execution-local error use dense attention on each run's own QKV; after generation divergence these are not the cached dense trajectory. Token agreement includes all positions after divergence, with missing/extra positions counted as disagreements. Shared-state diagnostics cover 32 early AIME and 34 early LongBench snapshots per method, not full generation.

These results motivate the separately queued generation-seed43 replication, but do not establish broad seed robustness, a dimension optimum, or hardware speedup. Numerical kernels, policies, and the canonical audited report remain unchanged.
