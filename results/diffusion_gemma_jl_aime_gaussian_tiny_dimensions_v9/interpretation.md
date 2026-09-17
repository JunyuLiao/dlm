# Gaussian2/4 on AIME26: completed comparison

All330 result slots passed audit and independent raw-only report regeneration:60 new final slots plus270 cached comparisons. No failed configurations. The same30 AIME26 questions,2048-token budget, decoding settings, seed42 and projection seed1729 were used at50% target physical sparsity. Previous results remain unchanged.

| Method | Correct/30 | Accuracy | Whole sparsity | Global sparsity | Local sparsity | Retained mass | Token agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 17 | 56.67% | 0% | 0% | 0% | 100% | 100% |
| Gaussian2 centered | 16 | 53.33% | 50.21% | 51.63% | 49.85% | 73.03% | 4.90% |
| Gaussian4 centered | 13 | 43.33% | 51.02% | 50.24% | 51.21% | 72.22% | 4.47% |
| Gaussian8 centered | 15 | 50.00% | 50.78% | 49.32% | 51.14% | 72.35% | 4.33% |
| Gaussian16 centered | 11 | 36.67% | 51.04% | 50.68% | 51.12% | 72.54% | 4.73% |
| Gaussian24 centered | 16 | 53.33% | 50.67% | 50.35% | 50.75% | 73.19% | 4.78% |
| Gaussian32 centered | 15 | 50.00% | 50.66% | 50.67% | 50.66% | 73.31% | 5.20% |
| Full-dimensional centered | 15 | 50.00% | 51.31% | 51.43% | 51.28% | 72.68% | 4.88% |

Gaussian2 ties Gaussian24 in observed correct count and exceeds Gaussian32/full-dimensional by one question. Gaussian4 scores two questions below Gaussian32/full-dimensional. These small-sample results still show no monotonic relation between projection dimension and accuracy.

The paired95% accuracy-delta interval for Gaussian2 minus Gaussian32 is[-16.67,+23.33]percentage points around+3.33pp; versus full-dimensional it is[-10.00,+16.67]pp. Gaussian2 minus Gaussian4 is+10.00pp with interval[-6.67,+26.67]pp. These exploratory, unadjusted intervals do not establish a winner. Dimension2 also has0.81pp less overall sparsity than dimension4, and different global/local allocation, although their gaps are within2pp for all three groupings.

## Calibration and interpretation limits

Dimension2 required four full-budget calibration trials; dimension4 accepted the first freshly verified trial. The deployed natural-log thresholds (global,local) are(-1.2668640614,-0.8551128507) for dimension2 and(-1.2361350060,-0.7595282197) for dimension4. Each passed48–52% calibration sparsity overall/globally/locally; each also remains within that band on final evaluation. Thresholds were frozen before final generation and never selected using accuracy.

The six calibration questions remain included in headline30. On the remaining24, dimension2 scores13/24 and dimension4 scores11/24, versus Gaussian32/full-dimensional12/24 and dense14/24. These previously examined questions are not fresh held-out confirmation. Changing dimension also changes the deterministic Gaussian matrix; this is a one-seed study, not seed-averaged evidence.

Very small sketches distort risk more strongly despite dimension2's competitive observed accuracy. On32 shared early calibration QKV states, severe underestimates number3683/20480 supported query-row/block pairs for dimension2 and1434/20480 for dimension4, versus169/20480 for dimension8 and0/20480 for dimensions24/32. Severe means projected centered risk below half the full-dimensional risk when full risk is at least0.05. These are diagnostic row-level events, not a count of physically skipped dangerous tiles.

Shared-state relative output error is0.2847 for dimension2,0.2366 for dimension4,0.2050 for Gaussian32 and0.2088 for full-dimensional. Sampled physical sparsities differ (40.10%,41.67%,37.50%,38.02%), so these are not matched-sparsity operator comparisons. Limited early-state coverage cannot establish safety. On the full generated trajectories, corresponding-QKV local errors are0.2808/0.2722/0.2527/0.2574 respectively. Lower operator error did not consistently translate into higher benchmark accuracy.

Physical sparsity uses summed skipped/eligible tiles, not mean percentages. Retained mass is measured against dense attention on each execution's corresponding QKV; token agreement includes all positions after divergence and counts missing/extra tokens as disagreements. No hardware-speedup claim is made.

Sources: [canonical report](report.md), [summary CSV](summary.csv), [paired comparisons](dimension_pairwise.csv), [shared diagnostics](shared_operator.csv), [audit](audit.json), [independent verification](regeneration_verification.json).
