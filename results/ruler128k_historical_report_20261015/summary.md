# Existing RULER128K 104-problem panel

Historical panel, reaggregated on 2026-10-15. Eight problems per each of 13 tasks; panel seed 1. V1 and V2 use tau=0.027. No new GPU experiment was run.
Official RULER score awards fractional credit for recovered targets. Full-credit items below have official score 100; they are a separate statistic.

| Method | Official score /100 | Full-credit items | N/problem | C/problem | Tokens/problem | GLOBAL kept tiles | GLOBAL sparsity | Overall sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current_v31_control | 86.394 | 80/104 | 5.548 | 1.00 | 26.27 | 99,178,720 | 47.33% | 45.12% |
| value_v1_online_discard_mass | 86.154 | 78/104 | 5.692 | 1.00 | 26.05 | 99,030,419 | 48.75% | 46.47% |
| value_v2_online_preserve_mass | 85.673 | 78/104 | 5.885 | 1.00 | 25.89 | 96,830,509 | 51.53% | 49.12% |
| value_v3a_singleton_delete | 86.298 | 79/104 | 5.567 | 1.00 | 26.22 | 99,221,280 | 47.49% | 45.27% |
| value_v3b_approx_batch8 | 81.490 | 72/104 | 7.173 | 1.00 | 26.13 | 103,075,520 | 57.67% | 54.98% |

## Absolute physical attention work over all 104 problems

| Method | GLOBAL eligible | GLOBAL kept | GLOBAL skipped | Overall kept | Kept GLOBAL / V31 |
|---|---:|---:|---:|---:|---:|
| current_v31_control | 188,290,720 | 99,178,720 | 89,112,000 | 108,401,120 | 1.00000 |
| value_v1_online_discard_mass | 193,227,520 | 99,030,419 | 94,197,101 | 108,492,819 | 0.99850 |
| value_v2_online_preserve_mass | 199,770,080 | 96,830,509 | 102,939,571 | 106,612,109 | 0.97632 |
| value_v3a_singleton_delete | 188,950,880 | 99,221,280 | 89,729,600 | 108,475,680 | 1.00043 |
| value_v3b_approx_batch8 | 243,506,400 | 103,075,520 | 140,430,880 | 115,002,720 | 1.03929 |

## Interpretation and provenance

V2 versus V31: -0.721 official score points, +6.066% denoising calls, -2.368% retained GLOBAL tiles and -1.650% overall retained rectangles. Its calls are 612 versus 577. All 104 outputs fit one visible canvas, so N/C equals N/problem. The shorter denoising trajectory seen on LongBench does not occur in this historical RULER panel.

The same 92,898,560 retained GLOBAL tiles come from full-support dense/initial/refresh phases for both V31 and V2. This is 93.67% of V31 retained GLOBAL work and 95.94% of V2 work. Sparse held/carried work changes substantially but is a small part of the total retained work. These are measured phases, not a claim about clean latency.

Physical GLOBAL sparsity uses GLOBAL kept/eligible tiles; overall sparsity includes dense LOCAL attention. Neither includes extra value-selector discovery, prefill or commit work. Value-selector discovery is 58,991,520 rectangles for V2; its recorded discovery/selection times are 12.31/21.50 seconds. Zero V31 value-selector counters do not mean zero mass-selector overhead.

The available compute receipts count attention tiles and logical split-FA4 adapter calls. They do not count literal CUDA kernel launches. No dense/all-kept completed records were found for this panel, so no dense result is inferred from another campaign.

All 20 shards have completion markers and clean receipts; official scorer coverage is 104/104 per arm. Manifest, prompts, seeds, budgets and runtime geometry match across all 104 paired items. One recorded source hash differs: v31_value_kernels.py is f9b5dfdd... for V31 and 192f027f... for all value arms. Other recorded execution-source hashes agree. Therefore these are historical observations, not a newly verified byte-identical deployment comparison.

No tau=0.021 RULER result is present. The recent LongBench tau=0.021 results are a separate source/protocol family. The paused RULER/AIME matrix and all clean timing remain untouched.
