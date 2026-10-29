# Kernel-native query sensitivity iteration report

Date: 2026-10-05
Model/runtime: DiffusionGemma, canvas 256, H100, seed 42, HumanEval v2 manifest.
All sparse methods used one shared local/global log-threshold pair across denoising calls. Threshold fitting matched physical tile sparsity on the evaluation cohort after disjoint calibration; the project results mount was read-only, so outputs are under [private_path_sha256_c49d22676bee5249]

## Full 100-question results

| Method | Whole sparsity | Pass@1 | Mean canvas calls | P90 calls | Caps |
|---|---:|---:|---:|---:|---:|
| Dense | 0.00% | 92% | 5.40 | 12.0 | 0 |
| C_gate | 49.47% | 87% | 8.45 | 20.1 | 1 |
| C_tile_max | 50.96% | 84% | 10.44 | 21.3 | 1 |
| C_tile_tail (12.5%) | 50.37% | 83% | 10.91 | 38.4 | 1 |
| C_tile_tail (50%) | 49.15% | 83% | 10.02 | 40.1 | 4 |
| C_tile_max_norm | 49.63% | 84% | 11.90 | 32.8 | 14 |
| C_tile_floor | 50.21% | 84% | 9.28 | 26.6 | 1 |
| C_tile_majority | 49.96% | 84% | 10.95 | 44.1 | 2 |

Dense is 92% with 5.40 calls/canvas. C_gate remains the best sparse method in the full audit.

## Screening results

| Method | Cohort | Whole sparsity | Pass@1 | Mean canvas calls | P90 calls | Caps |
|---|---:|---:|---:|---:|---:|---:|
| C_tile_frac | 30 | 50.50% | 86.7% | 11.03 | 22.8 | 4 |
| C_tile_floor | 30 | 49.54% | 90.0% | 6.94 | 14.1 | 0 |
| C_tile_majority | 30 | 49.86% | 90.0% | 7.41 | 9.1 | 0 |
| T_tile_q | 30 | 49.74% | 73.3% | 18.81 | 96.0 | 11 |
| T_tile_q_slow (gamma .85) | 30 | 49.92% | 76.7% | 12.41 | 48.0 | 6 |

The 30-question gains from C_tile_floor and C_tile_majority did not survive the 100-question cohort. The raw and slow tile-q methods failed already on the screen.

## Failure evidence

The full C_tile_floor run rescued three C_gate failures but lost six C_gate wins. It created 57 and 84 call trajectories and raised failure mean output length to 1,135 characters. C_tile_majority reduced minority-row underprotection on the 30-question screen but produced 14.24 mean calls and P90 44.1 on 100 questions, with six syntax failures and two capped canvases.

The common cause is fixed-budget reallocation. A tile scalar protects or relaxes an entire 128-row group. With one shared threshold pair, changing that scalar changes which other tiles receive the remaining skip budget. A tile floor overprotects broad regions; a tile mean/tail/active-fraction score underprotects a critical row or relaxes too early. C_gate preserves the kernel's existing worst-row reduction and therefore retains the most useful within-tile selectivity.

## Literature used

- [MAGE](https://arxiv.org/abs/2602.14209) explicitly reports that all tokens in a diffusion block must share one KV subset and that this constraint can reduce routing recall; its solution reuses the first all-mask block's block-level route.
- [Semantic-Geometric Decoupled Routing](https://arxiv.org/abs/2609.22884) argues for hardware-aligned query-block routing with pre-RoPE semantic aggregation and a structural geometric prior, avoiding token-level post-hoc search.
- [LoSA](https://arxiv.org/abs/2604.12056) separates stable tokens from a small active fraction and reuses stable attention results, avoiding KV inflation from different query choices.

These papers support moving beyond a scalar sensitivity coefficient toward a route/cache representation tied to the physical block. The present experiments show that simply aggregating C_gate's scalar into a tile score is insufficient.

## Code and validation

Added causal physical-tile controls to `experiments/value_direction_hopper/query_adaptive.py`: C_tile_mean, C_tile_tail, C_tile_max, C_tile_max_norm, C_tile_frac, C_tile_floor, C_tile_majority, T_tile_q, and T_tile_q_slow. Existing C_gate behavior remains available and was included in every comparison. The focused suite passes: 56 tests in 4.5 seconds. Python compilation passes.

Recommended next kernel-native branch: a true tile route/cache state, seeded from the first all-mask call and reused while the tile remains stable, with a small active-tile refresh set. This follows MAGE/LoSA and requires changing the route representation, rather than adding another tile reduction to the current row-risk coefficient.

