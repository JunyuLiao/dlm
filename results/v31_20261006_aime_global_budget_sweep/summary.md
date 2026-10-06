# AIME26 global-budget sweep (V31)

Complete 30-problem AIME26 panel at seeds 42/43/44 (90 cells per arm), run on one H100 with vLLM 0.30.0 / FA4 SM90. The native adaptive sampler is unchanged and thinking is enabled. Sparse arms use V31 MAGE GLOBAL selection plus the compact Triton LOCAL consumer (`local_kernel=compact_triton`, LOCAL budget 512).

## Accuracy and denoising calls

| arm | GLOBAL budget | seed 42 | seed 43 | seed 44 | pooled exact match | denoising forwards (total) | canvases (total) | forwards / canvas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_full_fix51994 | — | 53.33% | 56.67% | 46.67% | 52.22% (47/90) | 27,189 | 1,974 | 13.774 |
| mage_k1024_local_k512_compact | 1,024 | 60.00% | 56.67% | 56.67% | 57.78% (52/90) | 26,094 | 1,945 | 13.416 |
| mage_k2048_local_k512_compact | 2,048 | 56.67% | 46.67% | 53.33% | 52.22% (47/90) | 26,721 | 1,969 | 13.571 |
| mage_k4096_local_k512_compact | 4,096 | 50.00% | 56.67% | 60.00% | 55.56% (50/90) | 25,387 | 1,885 | 13.468 |

The official exact-match scorer is averaged over the three panel seeds per problem (`avg@k`). Finish/cap diagnostics remain in `score/`; capped cells are not removed from the official accuracy denominator.

## Physical tile sparsity (count-weighted over all decoder attention calls)

| arm | GLOBAL sparsity | LOCAL sparsity (all calls) | LOCAL sparse-only sparsity | overall GLOBAL+LOCAL sparsity |
|---|---:|---:|---:|---:|
| mage_k1024_local_k512_compact | 57.38% | 25.42% | 32.79% | 37.82% |
| mage_k2048_local_k512_compact | 41.77% | 25.59% | 32.89% | 31.99% |
| mage_k4096_local_k512_compact | 17.43% | 25.42% | 32.74% | 22.30% |

Dense is the 0%-sparsity reference. Physical sparsity is `sum(skipped eligible tiles) / sum(eligible tiles)`; the all-call LOCAL column includes mandatory dense observation/warm calls, while the sparse-only column excludes those calls.

## Timing (arithmetic mean over 90 cells)

| arm | end-to-end (prefill + decode) | dense / arm speedup | decode only | dense / arm speedup | prefill component |
|---|---:|---:|---:|---:|---:|
| dense_full_fix51994 | 5.9738 s | 1.000x | 5.9540 s | 1.000x | 0.0189 s |
| mage_k1024_local_k512_compact | 7.6265 s | 0.783x | 7.6070 s | 0.783x | 0.0185 s |
| mage_k2048_local_k512_compact | 7.8193 s | 0.764x | 7.7998 s | 0.763x | 0.0185 s |
| mage_k4096_local_k512_compact | 7.3819 s | 0.809x | 7.3625 s | 0.809x | 0.0184 s |

Speedup is dense mean divided by the arm mean; values below 1 indicate the sparse arm is slower. The sweep shows no end-to-end speedup at these budgets despite the measured tile skipping.

## Reproduction and audit artifacts

- Frozen protocol: `README.md`, `config.json`, and `input_sha256.txt`.
- Sanitized public records are under each arm’s `public/` directory; raw completions/logs remain in the user-private run bundle.
- `receipt.json` records source identity, scorer hashes, tests, and tile counters; `final_complete.json` marks all four arms complete.
- Paired official comparison: [`score/paired.md`](score/paired.md).
