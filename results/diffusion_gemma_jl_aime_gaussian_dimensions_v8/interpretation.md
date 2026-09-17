# AIME26 projection-dimension results

Completed and independently verified: 270/270 result slots, with no failures. New Gaussian centered dimensions 8/16/24 are compared with unchanged cached dimension-32/full-dimensional and dense results. Target physical sparsity is 50%; the same 30 questions, prompts, 2048-token budget, decoding settings, seed42 and projection seed1729 were used. Physical sparsity is summed skipped tiles divided by summed eligible tiles.

| Method | Correct/30 | Accuracy | Whole sparsity | Global sparsity | Local sparsity | Retained mass | Token agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 17 | 56.67% | 0% | 0% | 0% | 100% | 100% |
| Gaussian8 centered | 15 | 50.00% | 50.78% | 49.32% | 51.14% | 72.35% | 4.33% |
| Gaussian16 centered | 11 | 36.67% | 51.04% | 50.68% | 51.12% | 72.54% | 4.73% |
| Gaussian24 centered | 16 | 53.33% | 50.67% | 50.35% | 50.75% | 73.19% | 4.78% |
| Gaussian32 centered, cached | 15 | 50.00% | 50.66% | 50.67% | 50.66% | 73.31% | 5.20% |
| Full-dimensional centered, cached | 15 | 50.00% | 51.31% | 51.43% | 51.28% | 72.68% | 4.88% |

Gaussian24 has the highest observed sparse accuracy here, one question above Gaussian32/full-dimensional and one below dense. Its +3.33percentage-point difference against Gaussian32 has a paired95% interval of [-16.67,+23.33]pp: this does not establish superiority. Gaussian8 matches both references in correct count despite using a much smaller sketch. Gaussian16 is four questions below both references in this fixed-seed run. Accuracy does not improve monotonically with dimension; no projection-dimension winner is established by these 30 questions.

All projected/full-dimensional methods achieve final overall/global/local sparsity within48–52%. Rank8 versus full-dimensional nevertheless differs by2.10pp in global sparsity; the two methods do not meet a strict2pp pairwise match on every layer type. Gaussian24 versus Gaussian32 is closely matched in sparsity (0.01pp whole-model gap).

The six calibration questions remain included in the headline30. On the additional24, correct counts are dense14, Gaussian8=13, Gaussian16=9, Gaussian24=14, Gaussian32=12, full-dimensional12. These questions were previously examined, not fresh held-out confirmation. Only sparsity selected calibration thresholds. Dimension8 required three verified calibration points; dimensions16/24 accepted the first freshly verified point. No final-score threshold adjustment occurred.

Shared-QKV diagnostics cover32 early calibration states,192 eligible physical tiles and20480 supported query-row/block pairs per method. Severe risk underestimates (projected centered risk below half the full risk when full risk is at least0.05) number169/5/0/0 for dimensions8/16/24/32. Shared-state relative output errors are0.2615/0.2261/0.2084/0.2050, versus0.2088 full-dimensional. Their sampled sparsities differ (38.54/40.63/37.50/37.50/38.02%); these are limited diagnostics, not matched-sparsity full-generation results. Dimension8's equal headline accuracy does not imply equally faithful or robust routing. Conversely, dimension16's better diagnostic error than dimension8 did not yield better downstream accuracy.

The unchanged cached mass-only baseline scores14/30 at51.39% sparsity, aggressive BLASST11/30 at51.36%, and original BLASST16/30 at28.82%. Original BLASST is not a matched-sparsity comparison. Rank/seed sensitivity and small-sample uncertainty remain limitations; changing rank changes the deterministic matrix. No hardware-speedup claim is made.

Sources: [canonical report](report.md), [direct paired comparisons](dimension_pairwise.md), [all metrics](summary.csv), [shared-state diagnostics](shared_operator.csv), [audit](audit.json), [independent verification](regeneration_verification.json).
