# V31 AIME26 half-context reproduction (patched FA4 consumer)

This is the complete 30-problem AIME26 run at panel seeds 42, 43, and 44
(90 matched cells per arm). The sparse arm is the current V31 MAGE path:
`qblock_max`, 4096-token GLOBAL budget, first-call selection with carry,
settledness trigger 0.15, sticky score 1.386, FA4 selection, fused statistics,
chunked build, and Triton copy/merge. The requested 512-token LOCAL budget is
recorded but the current adapter does not route LOCAL layers through a sparse
consumer; LOCAL is therefore dense.

## Accuracy

| arm | seed 42 | seed 43 | seed 44 | pooled exact match |
|---|---:|---:|---:|---:|
| dense | 53.33% | 56.67% | 46.67% | 52.22% (47/90) |
| V31 MAGE | 63.33% | 60.00% | 46.67% | 56.67% (51/90) |

The scorer uses the pinned exact-match AIME metric. The sparse arm had 33
capped cells and the dense arm 38; this is retained as a diagnostic rather than
silently treating capped generations as finished.

## Calls and physical tile accounting

| arm | denoising calls | canvases | calls/canvas |
|---|---:|---:|---:|
| dense | 27189 | 1974 | 13.774 |
| V31 MAGE | 24999 | 1915 | 13.054 |

The pooled V31 receipt totals are `sparse_only_kept_tiles=146859360`,
`sparse_only_prefix_tiles=187415520`, and `global_prefix_tiles=219091040`. Thus GLOBAL
prefix physical sparsity is **21.64%** and the all-GLOBAL-call
work fraction is **81.49%** (the difference is the selection/warm
calls that are intentionally dense). LOCAL physical sparsity is **0%**.
Using the pinned 5-GLOBAL/25-LOCAL architecture, 16 query heads, 128-row
query blocks, 64-key tiles, and the native 1024-token LOCAL window, including the always-kept current
GLOBAL canvas tiles, the count-weighted overall decoder-attention sparsity is
**7.31%**.

## Timing

Arithmetic means over the 90 cells are shown for absolute spans; paired ratios
are geometric means over the same cells and use dense as the denominator.

| metric | dense mean | V31 MAGE mean | V31/dense | dense÷V31 |
|---|---:|---:|---:|---:|
| end-to-end (prefill + decode) | 5.9784s | 6.0698s | 1.052x | 0.951x |
| decode only | 5.9585s | 6.0505s | 1.052x | 0.950x |

The resulting end-to-end and decode-only speedups are below 1x: the V31 arm is
about 5% slower on this H100 at this workload, despite reducing denoising
calls and GLOBAL prefix work. No speed gain is claimed.

## Per-seed calls and sparsity

| seed | dense calls | V31 calls | dense canvases | V31 canvases | V31 GLOBAL sparsity | V31 overall sparsity |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 9065 | 8511 | 661 | 649 | 20.89% | 7.07% |
| 43 | 8534 | 8386 | 640 | 645 | 21.58% | 7.26% |
| 44 | 9590 | 8102 | 673 | 621 | 22.48% | 7.61% |


The sparse public receipts and private completions remain in the authorized
run directory. Only these aggregate values are checked into the repository.
