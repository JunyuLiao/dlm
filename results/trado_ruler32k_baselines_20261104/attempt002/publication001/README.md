# TraDo-8B-Instruct on RULER32K

Completed all four requested arms: native dense, all-kept FA4, V31 and V3a (520 new requests). The existing current-tile V2 result is included for reference and was re-scored on CPU; no V2 generation was run.

| Method | Official score | Full credit / 130 | Mean output tokens | Steps / request | Canvases / request | Steps / canvas | Achieved sparsity | Cap hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native dense | 64.64% | 70 | 31.4 | 21.82 | 8.65 | 2.523 | 0.00% | 2 |
| All-kept FA4 | 64.85% | 71 | 31.3 | 21.57 | 8.63 | 2.499 | 0.00% | 3 |
| V31 | 63.49% | 68 | 29.9 | 20.53 | 8.28 | 2.480 | 49.97% | 3 |
| V3a | 64.96% | 71 | 30.0 | 20.33 | 8.30 | 2.449 | 49.97% | 1 |
| V2 current-tile initialization (existing) | 64.72% | 71 | 31.0 | 21.28 | 8.56 | 2.486 | 49.45% | 4 |

The official score gives fractional credit for retrieving the requested targets. Full credit counts completely correct samples. Steps / canvas is pooled denoising forwards divided by canvases. Cap hits count outputs that reached 128 tokens without EOS.

The cohort contains 130 problems: 10 from each of 13 official RULER tasks, with seed 42. All arms use exactly the same frozen prompts (29,798–32,472 tokens), without truncation, a 32,768-token context limit and a 128-token output cap. The model is pinned to revision `2d37bd3e`; the numerical runtime comes from `8c0d14b0`. Runs use BF16, batch size 1 and one H100.

The native sampler uses 4-token blocks, at most 4 denoising steps per block, temperature 1, top-p 1, top-k 0, confidence threshold 0.9 and native EOS. All 36 denoising layers are eligible for sparsification; prefill and commit forwards stay dense. Native dense uses SDPA; the all-kept control uses the same full-dimensional FA4 consumer as the sparse methods.

V31 ranks tiles by attention mass. V3a uses Gaussian32 full-information singleton risk, with projection seed `1729 + layer`. Each selector keeps `max(round(J/2), mandatory_count)` KV tiles, including the current tile, where `J` follows the effective context length. Tiles have 64 KV tokens and 128 query rows; decisions are per query head/block. Both methods refresh every fourth call and use no cross-canvas carry. No threshold calibration is needed for this fixed budget.

Achieved sparsity is `1 - sum(kept tiles) / sum(eligible tiles)` over actual denoising attention calls, including mandatory tiles and calls that reuse a map. Selector discovery, prefill and commit work are outside this denominator. This measures attention tile sparsity; no end-to-end speedup is claimed.

V3a exceeds V31 by 1.47 percentage points, with paired problem-bootstrap 95% CI [−0.32, +3.78] and 13-task-cluster CI [−0.71, +4.42]. This exploratory, previously examined cohort does not establish a clear quality advantage or noninferiority.

Validation passed: independent CPU/CUDA selector checks, FP32 masked-attention comparisons, exact historical all-kept replay, all 16 new shards, source/record hash checks, and per-selection half-budget and mandatory-tile assertions. The existing 130 V2 scores re-scored exactly.

Per-task official retrieval scores:

| Task | Native dense | All-kept FA4 | V31 | V3a | Existing current-tile V2 |
|---|---:|---:|---:|---:|---:|
| cwe | 12.00% | 12.00% | 12.00% | 12.00% | 12.00% |
| fwe | 93.33% | 90.00% | 93.33% | 90.00% | 93.33% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% | 100.00% | 90.00% |
| niah_multikey_2 | 40.00% | 40.00% | 40.00% | 40.00% | 40.00% |
| niah_multikey_3 | 20.00% | 20.00% | 10.00% | 10.00% | 10.00% |
| niah_multiquery | 82.50% | 80.00% | 80.00% | 77.50% | 77.50% |
| niah_multivalue | 72.50% | 75.00% | 70.00% | 85.00% | 82.50% |
| niah_single_1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 90.00% | 90.00% | 90.00% | 90.00% | 90.00% |
| qa_1 | 20.00% | 30.00% | 20.00% | 30.00% | 40.00% |
| qa_2 | 30.00% | 30.00% | 30.00% | 30.00% | 30.00% |
| vt | 80.00% | 76.00% | 80.00% | 80.00% | 76.00% |

Canonical numerical evidence: [summary](../analysis002/summary.json), [scored records](../analysis002/scored_records.jsonl), [paired intervals](../analysis002/paired.json), [analysis receipt](../analysis002/receipt.json), [frozen protocol](../protocol.json), and [completion marker](../final_complete.json). Earlier attempts and reports remain preserved.
