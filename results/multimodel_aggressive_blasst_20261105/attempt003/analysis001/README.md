# Aggressive BLASST on LLaDA2.2-mini and TraDo-8B-Instruct

Six complete new panels, 768 benchmark requests. Existing baseline, V31/V3a and V2-CTI results are preserved.

| Model | Benchmark | Method | Score | Full credit | Mean output | Steps/request | Canvases/request | Steps/canvas | Sparsity | EOS / caps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llada22 | humaneval | dense_native | 85.98% | 141/164 | 243.1 | 28.76 | 8.57 | 3.357 | 0.00% | 164 / 0 |
| llada22 | humaneval | allkept_fa4 | 84.15% | 138/164 | 249.2 | 31.42 | 8.77 | 3.581 | 0.00% | 163 / 1 |
| llada22 | humaneval | mass_topk_reuse_r4 | 50.61% | 83/164 | 218.9 | 36.76 | 7.84 | 4.692 | 49.98% | 164 / 0 |
| llada22 | humaneval | value_v3a_singleton_reuse_r4 | 62.80% | 103/164 | 245.3 | 42.32 | 8.67 | 4.881 | 49.99% | 163 / 1 |
| llada22 | humaneval | current_tile_v2 | 72.56% | 119/164 | 219.3 | 26.09 | 7.77 | 3.355 | 55.67% | 164 / 0 |
| llada22 | humaneval | blasst_aggressive_inverse_length_reuse_r4 | 1.22% | 2/164 | 32.5 | 9.24 | 2.48 | 3.734 | 46.29% | 164 / 0 |
| trado | humaneval | dense_native | 79.27% | 130/164 | 124.3 | 58.61 | 31.86 | 1.840 | 0.00% | 164 / 0 |
| trado | humaneval | allkept_fa4 | 78.05% | 128/164 | 120.7 | 57.47 | 30.95 | 1.857 | 0.00% | 164 / 0 |
| trado | humaneval | mass_topk_reuse_r4 | 42.68% | 70/164 | 121.5 | 70.83 | 31.13 | 2.275 | 49.55% | 163 / 1 |
| trado | humaneval | value_v3a_singleton_reuse_r4 | 47.56% | 78/164 | 133.5 | 68.49 | 34.15 | 2.006 | 49.67% | 162 / 2 |
| trado | humaneval | current_tile_v2 | 74.39% | 122/164 | 122.2 | 59.99 | 31.32 | 1.915 | 53.31% | 164 / 0 |
| trado | humaneval | blasst_aggressive_inverse_length_reuse_r4 | 74.39% | 122/164 | 114.8 | 58.34 | 29.46 | 1.980 | 51.96% | 164 / 0 |
| trado | ruler32k | dense_native | 64.64% | 70/130 | 31.4 | 21.82 | 8.65 | 2.523 | 0.00% | 128 / 2 |
| trado | ruler32k | allkept_fa4 | 64.85% | 71/130 | 31.3 | 21.57 | 8.63 | 2.499 | 0.00% | 127 / 3 |
| trado | ruler32k | mass_topk_reuse_r4 | 63.49% | 68/130 | 29.9 | 20.53 | 8.28 | 2.480 | 49.97% | 127 / 3 |
| trado | ruler32k | value_v3a_singleton_reuse_r4 | 64.96% | 71/130 | 30.0 | 20.33 | 8.30 | 2.449 | 49.97% | 129 / 1 |
| trado | ruler32k | current_tile_v2 | 64.72% | 71/130 | 31.0 | 21.28 | 8.56 | 2.486 | 49.45% | 126 / 4 |
| trado | ruler32k | blasst_aggressive_inverse_length_reuse_r4 | 63.79% | 70/130 | 29.9 | 20.40 | 8.29 | 2.460 | 48.03% | 127 / 3 |
| llada22 | ruler128k | dense_native | 85.44% | 97/130 | 28.2 | 7.81 | 1.82 | 4.283 | 0.00% | 130 / 0 |
| llada22 | ruler128k | allkept_fa4 | 84.82% | 97/130 | 28.2 | 8.03 | 1.82 | 4.405 | 0.00% | 130 / 0 |
| llada22 | ruler128k | mass_topk_reuse_r4 | 86.01% | 100/130 | 28.4 | 7.82 | 1.82 | 4.291 | 50.00% | 130 / 0 |
| llada22 | ruler128k | value_v3a_singleton_reuse_r4 | 85.03% | 98/130 | 28.2 | 8.00 | 1.82 | 4.388 | 50.00% | 130 / 0 |
| llada22 | ruler128k | current_tile_v2 | 85.21% | 99/130 | 28.0 | 8.01 | 1.82 | 4.411 | 49.63% | 130 / 0 |
| llada22 | ruler128k | blasst_aggressive_inverse_length_reuse_r4 | 84.01% | 98/130 | 28.3 | 8.07 | 1.82 | 4.445 | 48.20% | 130 / 0 |
| trado | aime26 | dense_native | 8.89% | 8/90 | 1072.8 | 539.07 | 268.92 | 2.005 | 0.00% | 86 / 4 |
| trado | aime26 | allkept_fa4 | 4.44% | 4/90 | 1015.7 | 526.09 | 254.68 | 2.066 | 0.00% | 89 / 1 |
| trado | aime26 | mass_topk_reuse_r4 | 3.33% | 3/90 | 778.7 | 443.28 | 195.37 | 2.269 | 50.00% | 88 / 2 |
| trado | aime26 | value_v3a_singleton_reuse_r4 | 2.22% | 2/90 | 716.1 | 423.67 | 179.78 | 2.357 | 49.97% | 89 / 1 |
| trado | aime26 | current_tile_v2 | 8.89% | 8/90 | 1018.1 | 527.96 | 255.24 | 2.068 | 50.51% | 88 / 2 |
| trado | aime26 | blasst_aggressive_inverse_length_reuse_r4 | 10.00% | 9/90 | 1030.5 | 519.86 | 258.40 | 2.012 | 50.69% | 87 / 3 |
| llada22 | aime26 | dense_native | 40.00% | 36/90 | 7018.6 | 1539.02 | 220.14 | 6.991 | 0.00% | 85 / 5 |
| llada22 | aime26 | allkept_fa4 | 32.22% | 29/90 | 7356.3 | 1717.70 | 230.74 | 7.444 | 0.00% | 86 / 4 |
| llada22 | aime26 | mass_topk_reuse_r4 | 23.33% | 21/90 | 4691.7 | 1322.93 | 147.46 | 8.972 | 50.00% | 90 / 0 |
| llada22 | aime26 | value_v3a_singleton_reuse_r4 | 27.78% | 25/90 | 5259.1 | 1399.02 | 165.21 | 8.468 | 50.00% | 90 / 0 |
| llada22 | aime26 | current_tile_v2 | 30.00% | 27/90 | 5964.2 | 1445.84 | 187.31 | 7.719 | 51.95% | 87 / 3 |
| llada22 | aime26 | blasst_aggressive_inverse_length_reuse_r4 | 0.00% | 0/90 | 32.3 | 13.69 | 2.40 | 5.704 | 41.12% | 90 / 0 |

Calibration and achieved lambda ranges:

| Model/task | log kappa | Calibration sparsity | Full sparsity | Minimum lambda | Maximum lambda | Calibration exposure |
|---|---:|---:|---:|---:|---:|---|
| llada22/humaneval | 15 | 40.98% | 46.29% | 6810.4529 | 34052.264 | transductive count-only |
| trado/humaneval | 6.2121233 | 49.18% | 51.96% | 0.43904852 | 6.92721 | transductive count-only |
| trado/ruler32k | 2.8654441 | 48.70% | 48.03% | 0.00054014423 | 0.00058915598 | independent retrieval development |
| llada22/ruler128k | 2.3977134 | 48.14% | 48.20% | 8.407228e-05 | 9.5548368e-05 | independent retrieval development |
| trado/aime26 | 3.2698565 | 48.68% | 50.69% | 0.012179428 | 0.36538284 | transductive count-only |
| llada22/aime26 | 22.5 | 42.44% | 41.12% | 14207986 | 61567938 | transductive count-only |

AIME uses 30 problems × seeds 42/43/44; HumanEval uses all 164 and seed 42; RULER uses 130 balanced problems and seed 42, 128K on LLaDA and 32K on TraDo. RULER score is fractional official retrieval credit. Prompt, cap, seed and scorer identities match the prior experiments.

Aggressive BLASST uses the preceding maximum over all seen tiles, including skipped tiles; strict all-active-row skipping, first/current protections and lambda(L)=exp(log_kappa)/L. Lambda may exceed 1. This is the existing aggressive extension, not literal paper pseudocode above lambda=1. The two model ports use FP32 accumulated scaled QK scores consistent with their FA4 attention path. No V2 value projection, CTI initialization or V2 threshold is used.

All denoising layers, Q128/KV64 tiles, four-call map reuse and no cross-canvas carry; full-dimensional FA4 consumer; dense prefill and commits. Native samplers, EOS, editing and output caps are unchanged. The earlier BLASST rule and Yuhan inherited map reuse remain distinct provenance.

Fresh thresholds use only small count-only calibration trajectories: six prompt-length strata per short benchmark, or 13 independent retrieval development items. Target 48–52%, final acceptance 40–60%; all rejected attempts are retained. Exact calibration points, full-panel drift and lambda ranges are recorded, with no accuracy-driven threshold selection.

Sparsity is pooled skipped/eligible physical denoising tiles, including mandatory and held calls, excluding selector discovery and dense prefill/commit. Dense QK discovery and instrumented accounting remain separate; no clean latency or speedup claim.

Independent CPU/CUDA masks, FP32 attention consumer checks, historical all-kept replay, exact shard/source/private-token-hash joins, coverage and fairness audits pass. Paired intervals in paired.json use question clusters (including AIME seeds), not independent seed-level samples. Nonsignificance does not establish noninferiority.
