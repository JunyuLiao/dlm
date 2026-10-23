# V2 execution optimization and benchmark follow-up

Completed user-requested panels. Clean generation/request timing is separate from instrumented attention phase timing. Historical controls retain their original source and date; their timing is supplementary, not a matched execution-speed test. Native sampler, dense LOCAL, prefill and commits are preserved.

| Panel | Method | Official accuracy | N | N/C | Mean S (s) | Mean W (s) | GLOBAL all-step sparsity | Held sparsity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| aime26_k1920 | allkept_fa4 (historical) | 54.44% | 26547 | 13.254 | 5.8898 | 5.9087 | 0.00% | n/a |
| aime26_k1920 | dense_full_fix51994 (historical) | 56.67% | 25360 | 13.497 | 5.1517 | 5.1751 | 0.00% | n/a |
| aime26_k1920 | dense_piecewise (historical) | 56.67% | 25360 | 13.497 | 5.2206 | 5.2394 | 0.00% | n/a |
| aime26_k1920 | v2_online_optimized (historical) | 55.56% | 27570 | 13.541 | 6.3949 | 6.4140 | 44.94% | 45.58% |
| aime26_k1920 | v31_optimized | 60.00% | 25506 | 13.229 | 5.8822 | 5.9013 | 41.54% | 48.97% |
| math500_k384 | allkept_fa4 (historical) | 90.00% | 35163 | 10.327 | 1.4218 | 1.4394 | 0.00% | n/a |
| math500_k384 | dense_full_fix51994 (historical) | 89.80% | 35372 | 10.241 | 1.3093 | 1.3322 | 0.00% | n/a |
| math500_k384 | dense_piecewise (historical) | 89.80% | 35372 | 10.241 | 1.3264 | 1.3440 | 0.00% | n/a |
| math500_k384 | v2_online_optimized (historical) | 87.60% | 36604 | 10.434 | 1.5543 | 1.5721 | 47.60% | 49.36% |
| math500_k384 | v31_optimized | 88.20% | 35608 | 10.354 | 1.5043 | 1.5221 | 46.43% | 58.48% |
| ruler128k_s45 | allkept_fa4 (historical) | 87.12% | 615 | 5.913 | 0.3108 | 6.6066 | 0.00% | n/a |
| ruler128k_s45 | dense_full_fix51994 (historical) | 86.68% | 611 | 5.875 | 0.3198 | 6.6089 | 0.00% | n/a |
| ruler128k_s45 | dense_piecewise (historical) | 86.68% | 611 | 5.875 | 0.3112 | 6.6016 | 0.00% | n/a |
| ruler128k_s45 | v2_online_optimized | 87.45% | 624 | 6.000 | 0.3512 | 6.6401 | 51.86% | 61.99% |
| ruler128k_s45 | v31_optimized (historical) | 86.78% | 577 | 5.548 | 0.2941 | 6.5946 | 47.33% | 93.49% |

RULER uses its official fractional-credit score out of 100. V2 alone is recalibrated to 40-50% all-step GLOBAL physical sparsity on the examined 13-task calibration cohort. Full 104-problem work is measured independently in the table. LongBench-v2 was cancelled by the user before qualification or generation; prepared artifacts and original four-panel protocol remain preserved. This report completes only the three-panel authorized scope. Prior short-benchmark V2 generations remain intact. The frozen per-suite Markdown helper uses legacy held-target/four-question wording; the calibration receipt and this report specify the actual all-step target and 13 examined questions.

| Panel | V2 threshold | Calibration all-step sparsity | Full-panel all-step sparsity | Full-panel held sparsity |
|---|---:|---:|---:|---:|
| aime26_k1920 | 0.0594603557501 | preserved threshold | 44.94% | 45.58% |
| math500_k384 | 0.2 | preserved threshold | 47.60% | 49.36% |
| ruler128k_s45 | 0.00185615530061 | 48.75% | 51.86% | 61.99% |

| Budget comparison | Accuracy difference (pp) | Paired 95% interval (pp) |
|---|---:|---:|
| aime26_k1920_vs_k128 | +15.56 | [+5.56, +26.67] |
| aime26_k1920_vs_k256 | +13.33 | [+4.44, +22.22] |
| math500_k384_vs_k128 | +1.60 | [-0.60, +3.80] |

Intervals use 10,000 paired bootstrap draws, clustering AIME seeds by problem and RULER by task. Nonsignificance is not noninferiority. summary.json includes N,C,N/C,T,S/N,S,W, GLOBAL/LOCAL/overall physical counts, first-canvas and initial/refresh/held/carried work. attention_latency.json reports audit refresh/held timings; paired_uncertainty.json reports item-clustered accuracy and clean time/call ratios. Ratios above 1 indicate more time/calls. Sparsity and fewer calls alone are not speed evidence.

The output-preserving execution change reuses only unchanged prefix projections within a canvas and changes CTA placement. Shared-routing and direct-paged candidates were slower and remain preserved negatives. Held execution uses the shared resident consumer without selection, metadata rebuild or KV copy; workload/mask skew is reported rather than presented as a method-specific kernel difference.

Preserved HumanEval and prior AIME/MATH/RULER results: [completed earlier report](../../optimized_refresh_results_20261023/attempt003/summary.md).
