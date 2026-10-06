# V31 AIME26 GLOBAL + LOCAL sparse run

This is the complete 30-problem AIME26 panel at seeds 42, 43, and 44
(90 cells per arm), run on one H100 with the pinned vLLM 0.30 / FA4 SM90
stack. The dense arm is native vLLM dense with the PR #51994 mask fix. The
sparse arm is V31 MAGE with a 1728-token GLOBAL budget plus the new LOCAL
window router with a 512-token budget. Cache pages and sparse tiles are 64
tokens. The native LOCAL mask is preserved as a bidirectional 1023/1023 window.

The sparse arm is a V31 MAGE plus LOCAL-selector evaluation. Gaussian32/value-aware
routing and the query-sensitivity/C-gate method are not active in this run.

## Accuracy

| arm | seed 42 | seed 43 | seed 44 | pooled exact match |
|---|---:|---:|---:|---:|
| dense | 53.33% | 56.67% | 60.00% | 56.67% (51/90) |
| GLOBAL + LOCAL sparse | 56.67% | 56.67% | 56.67% | 56.67% (51/90) |

The pinned scorer found 90/90 records for each arm. Both arms had 36 capped
cells; capped outputs are retained in the diagnostics and are not treated as
finished answers.

## Denoising calls

| arm | total denoising forwards | canvases | pooled forwards/canvas |
|---|---:|---:|---:|
| dense | 26,536 | 1,980 | 13.402 |
| GLOBAL + LOCAL sparse | 25,823 | 1,956 | 13.202 |

## Physical tile accounting

The count-weighted receipts use query heads × 128-row query blocks × 64-key
tiles. They include dense observation/warm calls and the mandatory current
canvas tiles.

| scope | eligible tiles | kept tiles | physical sparsity |
|---|---:|---:|---:|
| GLOBAL | 251,668,160 | 136,157,440 | 45.90% |
| LOCAL, all calls | 389,212,000 | 290,625,600 | 25.33% |
| LOCAL, sparse-only calls | 300,448,800 | 201,862,400 | 32.81% |
| GLOBAL + LOCAL | 640,880,160 | 426,783,040 | 33.41% |

The requested 512-token LOCAL budget does not produce 50% physical sparsity:
short prefixes and the always-kept in-window canvas/boundary tiles make the
realized rate lower. The receipts retain the exact per-call counts in the
private attempt directory.

## End-to-end and decode timing

Arithmetic means over the 90 matched cells are shown. Speedup is dense divided
by sparse; values below one are slower than dense.

| metric | dense mean | sparse mean | sparse/dense | speedup dense/sparse |
|---|---:|---:|---:|---:|
| end-to-end (prefill + decode) | 5.9326 s | 7.1250 s | 1.201x | 0.833x |
| decode only | 5.9127 s | 7.1055 s | 1.202x | 0.832x |

The geometric paired speedups are 0.823x end to end and 0.822x decode only.
The sparse arm is therefore slower despite fewer denoising forwards.

## LOCAL kernel qualification

`local_kernel_timing.json` contains 50 synchronized alternating-order samples
at prefixes 187, 3500, and 8192 with Q256, H16/Hkv8, head dimension 256, and
the native 1023/1023 window. The selected 512-token map was numerically checked
against an FP32 masked oracle (maximum error about 0.0013 BF16 units), and the
all-kept sparse consumer was bitwise equal to dense FA4.

At prefix 8192, dense FA4 was 0.0531 ms and the best tested sparse configuration
was 0.0701 ms, or 0.757x speedup. The other prefixes were similarly slower.
The pinned FA4 block-sparse consumer has fixed scheduling/TMA overhead that
outweighs the skipped LOCAL tiles at this budget. The run does not claim an
actual LOCAL speedup. An H100 FlashInfer block-sparse alternative was also
checked, but its JIT build is unavailable in this environment because the
installed nvcc rejects FlashInfer's `--compress-mode=size` option.

## Reproduction and receipts

- Private full records, completions, and logs: `attempt001/full/` (ignored by Git).
- Public per-cell records: `attempt001/full/public/` (ignored by Git).
- Pinned score summary: `score.summary.json`.
- Clean LOCAL kernel timing: `local_kernel_timing.json`.
- CPU qualification: 21 tests passed, including local geometry, zero-copy
  page aliasing, FA4 local-fix isolation, and existing V31 accounting/clock tests.
- The one-cell smoke passed both dense and sparse arms before the full launch.
