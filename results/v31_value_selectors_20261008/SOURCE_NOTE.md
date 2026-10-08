# Source inspection, 2026-10-08

Inspection base: `cdc12221`, branch `value-aware_cross-step-reuse_v2`.
The inherited cross-step reuse machinery is Yuhan's work. Gaussian32 and
query-sensitivity protection are Junyu's earlier families. The selectors in
this directory are this study's integrations.

The active `arm= mage`, `qblock_max` control computes
`max_valid_rows(log Z_ij - logsumexp_prefix_tiles(log Z_il))` for each query
head and Q128 block. It keeps `min(prefix_tiles, max(1, token_budget // 64))`
prefix tiles, using Torch `topk` (ties have no specified stable index order).
Canvas and boundary tiles are always kept outside that budget; sink/recent
prefix tiles, if configured, are forced inside it. No projected V enters this
score. On refresh a held tile receives the existing additive log-score sticky
bonus. Row weights, if configured, enter before the row maximum only on refresh.

The latest completed dense-LOCAL sweep freezes three budgets rather than a
single headline budget. For the long-context study we retain the documented
headline `lean_k8192_progress_sticky`: budget 8192, initial selection at call 1,
first-call carry enabled, settledness trigger 0.15, sticky 1.386, no row weights,
no pool, no sink/recent additions, five GLOBAL layers, native dense LOCAL.

MAGE `kvhead` averages globally normalized tile mass across rows and GQA
query heads. `qblock_max` uses worst-row prefix mass share instead. The older
`arm=method` online oracle scans ascending physical KV64 tiles and uses
`eta * norm(weighted_projected_tile_mean - running_output) / valid_KV_RMS`,
maximized over valid query rows, optionally weighted by query sensitivity.
Strict threshold ties retain support. These are distinct selector formulas.

All new selector implementations are independently derived from the requested
formulas. Correctness oracles use direct masked-attention recomputation.
No peer branch is merged and no frozen evidence is modified.
