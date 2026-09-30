# v27 sparsity fidelity: attention mass and output error of held sparse GLOBAL calls

`scripts/v27_sparsity_fidelity.py` on the diagnostic protocol `dev_fidelity_v6` (`../specs/v27_dev_fidelity_v6.json`;
dllm, substrate piecewise_v2; two requests each of LongBench-v2 32K, 64K and AIME26 items 1 and 5, seed 101).
Every FA4 block-sparse GLOBAL call that skips tiles is shadowed on the same q/k/v by FA4 with every tile kept and by
HF SDPA. Generation is unchanged (`same_as_production` is true for every row). Numbers only, no text:

- `<dataset>.jsonl.gz`: one row per shadowed layer call: canvas, denoising step, GLOBAL layer position (0-4 =
  layers 5, 11, 17, 23, 29), map age in steps, keys, kept tile fraction, kept attention mass (mean, 1st percentile
  and minimum over heads x queries; mass = exp(lse_kept - lse_all)), relative output error
  ||o_sparse - o_dense|| / ||o_dense|| and the exact-kernel envelope ||o_sdpa - o_fa4|| / ||o_fa4||.
- `summary_rows.jsonl`: per request and arm, quantiles and mass by map age and by step.

## Medians over held calls (two requests per dataset)

| arm | 32K kept / mass / error | 64K kept / mass / error | AIME kept / mass / error |
|---|---|---|---|
| exact kernel swap (SDPA vs FA4) | 1 / 1 / 0.12% | 1 / 1 / 0.13-0.14% | 1 / 1 / 0.12% |
| Fan M1 R1 A8 | 0.22 / 0.82 / 19% | 0.15 / 0.73 / 26% | 0.95 / 0.99 / 2.4% |
| Fan M3 R3 A8 | 0.21 / 0.82 / 19% | 0.15 / 0.74 / 26% | 0.93 / 0.98 / 3.1% |
| B | 0.15 / 0.73 / 23% | 0.10 / 0.68 / 33% | 0.90 / 0.98 / 3.7% |
| B, -2ln2 | 0.35 / 0.87 / 13% | 0.25 / 0.79 / 20% | 0.97 / 1.00 / 0.9% |
| M3 R6 DP | 0.12 / 0.68 / 29% | 0.07 / 0.62 / 38% | 0.91 / 0.97 / 4.2% |
| M3 R12 DP | 0.10 / 0.69 / 31% | 0.06 / 0.61 / 39% | 0.89 / 0.97 / 4.6% |
| M3 R6 DP, -ln2 | 0.21 / 0.82 / 20% | 0.12 / 0.72 / 28% | 0.96 / 0.99 / 1.9% |
| M3 R6 DP, -2ln2 | 0.35 / 0.87 / 14% | 0.22 / 0.78 / 22% | 0.98 / 1.00 / 0.7% |
| M3 R12 DP, -2ln2 | 0.29 / 0.85 / 16% | 0.19 / 0.76 / 24% | 0.97 / 1.00 / 0.8% |

(Per cell: the mean of the two requests' medians, rounded.)

## Findings

- **The default thresholds drop a large share of the attention mass.** Skipping about 90% of tiles at 32K-64K keeps
  only 60-75% of the attention probability; the GLOBAL attention output error is 23-39%, 200-300x the exact
  dense-kernel difference.
- **Most of the loss is present right after a decision.** Held maps lose only a few points more with age: B at 32K
  stays flat over 12 steps (0.69 -> 0.69); at 64K B falls 0.66 -> 0.61 and M3 R12 0.59 -> 0.54 over 9-12 steps.
  Longer refresh intervals therefore cost a little, and the threshold is the main lever.
- **The loss is concentrated in the first GLOBAL layers.** At 64K, default M3 R6 DP keeps 27% of the mass in
  layer 5 and 43% in layer 11 but 65-71% in layers 17-29; -2ln2 raises layer 5 only to 47%. A layer-aware or
  mass-targeted selector (or `route_layers: no_first`) is the direct next variant.
- **AIME is barely sparse.** With short prefixes the selectors keep 88-98% of tiles (97-99.9% of the mass); there
  is no speed to gain and still a 1-5% perturbation, so a length gate is the right deployment. The AIME accuracy
  losses in the q1v3 panel do not track this fidelity (-2ln2 has the smallest error and lost the most), so part of
  them is sampling noise.
