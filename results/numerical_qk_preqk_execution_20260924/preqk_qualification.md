# Pre-QK consumer qualification (`historical_route_preqk_current_output`)

Same Q/K/V and same retained support feed three implementations:

- **preqk** -- the new consumer: forms current QK inside the output program
  and never touches a dropped tile.
- **incumbent** -- `routing_only_current_output`'s path: `observe_scores`
  materializes every current score, then PV consumes the bitmap.
- **fp32** -- an independent FP32 reference (FP32 QK, explicit mask, FP32
  softmax and PV) over exactly the same retained support.

The incumbent is *not* ground truth; it is the BF16 path being replaced. The
question is whether preqk is further from FP32 than the incumbent already is.
Geometry follows the executed model: sliding layers `head_dim=256`, GQA 16/8;
full layers `global_head_dim=512`, GQA 16/2 (`config.json` `text_config`).

| case | D | retained/eligible | QK tile dots | expected | preqk vs incumbent | preqk vs fp32 | incumbent vs fp32 |
|---|---|---|---|---|---|---|---|
| local_short_prefix | 256 | 192/192 | 1536 | 1536 | 1.71e-05 | 6.360e-03 | 6.360e-03 |
| local_saturated_window | 256 | 608/608 | 4864 | 4864 | 2.20e-06 | 7.031e-03 | 7.031e-03 |
| local_partial_tiles | 256 | 224/224 | 1456 | 1456 | 9.50e-06 | 6.566e-03 | 6.566e-03 |
| local_narrow_window_illegal_tiles | 256 | 128/128 | 1024 | 1024 | 2.27e-05 | 5.719e-03 | 5.719e-03 |
| local_all_kept | 256 | 192/192 | 1536 | 1536 | 1.47e-05 | 6.489e-03 | 6.489e-03 |
| global_short_prefix | 512 | 192/192 | 1536 | 1536 | 1.84e-05 | 7.402e-03 | 7.402e-03 |
| global_long_prefix | 512 | 1024/1024 | 8192 | 8192 | 1.05e-05 | 1.018e-02 | 1.018e-02 |
| global_partial_tiles | 512 | 224/224 | 1456 | 1456 | 1.88e-05 | 7.514e-03 | 7.514e-03 |
| global_all_kept | 512 | 192/192 | 1536 | 1536 | 1.23e-05 | 7.462e-03 | 7.462e-03 |
| small_d64_gqa1 | 64 | 12/12 | 96 | 96 | 0.00e+00 | 3.019e-03 | 3.019e-03 |
| d128_gqa4 | 128 | 96/96 | 768 | 768 | 6.04e-06 | 4.304e-03 | 4.304e-03 |
| local_dropped_40pct | 256 | 285/512 | 2280 | 2280 | 5.96e-06 | 6.751e-03 | 6.751e-03 |
| global_dropped_60pct | 512 | 211/512 | 1688 | 1688 | 1.41e-05 | 7.614e-03 | 7.614e-03 |
| global_dropped_90pct | 512 | 79/768 | 497 | 497 | 9.97e-06 | 6.416e-03 | 6.416e-03 |

## Result

- **Accuracy: preqk is not further from FP32 than the incumbent.** Across all
  14 cases the two distances to FP32 differ by at most
  **7.03e-08** (they agree to ~4 significant figures everywhere). Both
  sit at the BF16 floor (~3e-3 to 1e-2 relative), which is a property of the
  frozen BF16 score path, not of this change.
- **The two BF16 paths agree to 2.27e-05** relative. That residual is
  accumulation order inside the QK dot (cuBLAS vs Triton MMA); the *rounding
  path* is matched deliberately -- BF16 matmul output, BF16 scaling, then FP32
  for the softmax, exactly as `observe_scores` does. With contiguous inputs the
  two are bit-identical; the residual appears only for the transposed
  (non-contiguous) views the model actually hands us.
- **Skipping is physical, not bookkeeping.** In every case the in-kernel
  counters equal the exact expected count (retained tiles x 16-row programs in
  their query block, short final tiles accounted for): no mismatches. The
  dropped cases (local_dropped_40pct, global_dropped_60pct, global_dropped_90pct) issue exactly
  `expected` QK dots and `expected` K/V tile loads, i.e. **zero** for every
  dropped tile.

## Declared limits

- A *dropped* tile's scores are never computed, so malformed (NaN/+inf) values
  inside a dropped tile cannot be detected here. That is intended: those
  scores do not reach the output. The historical scores that *chose* the
  support are separately checked by `route_only`'s per-tile `invalid_tiles`
  flag, which the adapter asserts on before consuming the support.
- Legality is derived from geometry (window, tails) exactly as
  `_attention_validity` defines it. A hand-doctored score tensor that cannot
  arise from `(q, k, window)` is outside the contract and is deliberately not
  part of this suite.
- `is_causal=True` is rejected rather than silently ignored; the decoder this
  is qualified for is bidirectional.
- These are synthetic states at real geometries. Real captured-state and
  full-forward evidence is reported separately.

Regenerate: `PYTHONPATH=src:. python scripts/preqk_qualification.py --output
results/numerical_qk_preqk_execution_20260924/preqk_qualification.json`.
Tests: `tests/test_numerical_reuse_preqk.py` (12 cases).
