# v27 risk-budget selector calibration (negative result)

Diagnostic protocol `dev_fidelity_v7` (`../specs/v27_dev_fidelity_v7.json`; dllm, piecewise_v3): the dev items of
`dev_fidelity_v6` (two LongBench-v2 32K and two 64K dev items, AIME 1 and 5; none is a scored panel item), with
`scripts/v27_sparsity_fidelity.py`. The risk budget (M1-DP named variant, `v27_dense_prefix.budget_skip`) drops,
per (head, query block), the lowest-risk prefix tiles only while their summed risk stays below
exp(threshold + shift). Rows: `<dataset>.jsonl.gz` (numbers only); per-request summaries: `summary_rows.jsonl`.

## Medians over held calls (mean of the two requests)

| selector | 64K kept / mass / error | 32K kept / mass / error | AIME kept / mass / error |
|---|---|---|---|
| per-tile threshold, default | 0.07 / 0.62 / 38% | 0.12 / 0.68 / 29% | 0.91 / 0.97 / 4.2% |
| per-tile threshold, -2ln2 (fidelity_v6) | 0.22 / 0.78 / 22% | 0.35 / 0.87 / 14% | 0.98 / 1.00 / 0.7% |
| budget b0 | 0.94 / 0.98 / 3.5% | 0.93 / 0.98 / 3.0% | 0.95 / 0.99 / 2.9% |
| budget b2ln2 | 0.82 / 0.97 / 4.1% | 0.80 / 0.97 / 3.6% | 0.87 / 0.97 / 4.2% |
| budget b4ln2 | 0.61 / 0.91 / 8.9% | 0.55 / 0.91 / 9.3% | 0.71 / 0.91 / 9.7% |
| budget b6ln2 | 0.25 / 0.76 / 24% | 0.22 / 0.76 / 23% | 0.49 / 0.81 / 21% |
| budget b8ln2 | 0.03 / 0.50 / 56% | 0.03 / 0.48 / 51% | 0.34 / 0.73 / 30% |

## Finding

At a matched kept fraction the budget captures the same attention mass as the per-tile threshold (64K: budget b6ln2
keeps 25% of tiles for 76% of the mass; per-tile -2ln2 keeps 22% for 78%). The budget only moves along the same
kept-vs-mass curve, so it is not carried into the final panel. What limits fidelity at a given sparsity is the M1
risk ranking of tiles, not the per-tile vs summed thresholding rule. The oracle diagnostic (`--oracle-every`) asks
how much mass the best block-sparse map of the same size would keep.
