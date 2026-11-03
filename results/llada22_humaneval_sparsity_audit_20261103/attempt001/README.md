# Recheck of LLaDA2.2-mini HumanEval sparsity

The published tile-sparsity calculations are correct. The earlier explanation
that V2 keeps almost no optional tiles, unlike V31, was misleading: on the same
short contexts, V31 also keeps only one optional KV64 tile. V2 retains slightly
more tiles than V31 in all four sampled EOS-writing states. The direct evidence
points to **tile identity**, not a uniquely smaller local V2 tile budget.

## Definition and full164-request recount

The code records `E = H * ceil(Q/128) * ceil(K/64)` eligible tile units and the
actual number `A` retained by the output consumer on every denoising attention
call. Published sparsity is `1 - sum(A)/sum(E)` across all164 requests, all20
layers and16 query heads. This includes current-block mandatory tiles and sparse
consumer calls during both selection and reuse. It excludes additional selector
discovery, dense prefill and block-commit forwards. It is a geometric tile count,
not whole-model FLOP reduction, retained probability mass, or a simple mean of
request percentages. Native dense uses the same geometry as a dense-equivalent
denominator.

All20 saved shard-record hashes and completion markers were checked. Recounting
all820 requests across five methods produces zero differences from the published
summary:

| Method | Eligible tile units E | Kept tile units A | Sparsity 1-A/E | Average available tiles per head/call | Average kept optional tiles per head/call |
|---|---:|---:|---:|---:|---:|
| Native dense |9,733,760|9,733,760|0.0000%|6.449|5.449|
| All-kept FA4 |11,563,200|11,563,200|0.0000%|7.012|6.012|
| V31 |10,911,040|5,457,920|49.9780%|5.655|1.829|
| V2 |3,035,840|1,541,706|49.2165%|4.070|1.067|
| V3a |13,064,320|6,533,440|49.9902%|5.882|1.942|

Each32-token aligned LLaDA canvas lies in one mandatory KV64 tile. Hence there
are exactly `20*16*N` mandatory tile units per arm; all prompts here exceed64
tokens, so the first tile is distinct from that mandatory tile. Subtracting this
common mandatory count yields optional-tile sparsity60.71% V31,65.25% V2,
60.23% V3a. These percentages still use each arm's own context/call distribution;
they are not a same-state comparison.

The mean per-request sparsity is49.46% V31,46.53% V2 and49.51% V3a, different
from the published pooled percentages. The pooled calculation consistently
weights requests by their eligible tile work.

## Correct mandatory/optional distinction

The shared protection mask requires only the tile intersecting the current
canvas. When the canvas occupies the latter32 positions of that KV64 tile,
the same protected tile also includes the preceding32 keys. This is a tile-level
protection, not an exact current-token-only restriction.

V2 additionally forces the first scanned tile to initialize its online state.
That first tile is **optional under the shared mask**, although V2's algorithm
forces it. The earlier phrase “almost no optional tiles” meant almost no tiles
in addition to this first tile and the common mandatory tile. It should not have
been interpreted as V2 retaining zero optional tiles or as V31 having a larger
local budget.

V31 and V3a use `max(round(J/2), mandatory_count)` total tiles per query
head/block, with Torch ties-to-even rounding. Mandatory tiles are included in
the budget. For one current mandatory tile:

| Available KV64 tiles J | V31/V3a total kept | V31/V3a optional kept | V2 if later optional threshold tests all fail |
|---|---:|---:|---:|
|2|1|0|2 total: first + current|
|3|2|1|2 total: first + current|
|4|2|1|2 total: first + current|
|5|2|1|2 total: first + current|
|6|3|2|2 total: first + current|
|7|4|3|2 total: first + current|

Thus with3–5 tiles, both V2 and V31 commonly retain one optional tile. V31 can
choose whichever optional tile has the highest mass score; V2's only optional
tile is usually the first scanned tile.

## Identical-context checks at first EOS-writing states

These are fresh masks recomputed on the same saved V2 Q/K/V, not native V31
states from its later/different trajectory. Counts average over20 layers and16
heads per item.

| Previously examined item | Valid keys | J | V2 total kept/head | V31 total kept/head | V2 tile sparsity | V31 tile sparsity |
|---|---:|---:|---:|---:|---:|---:|
|HumanEval/0|288|5|2.056|2.000|58.88%|60.00%|
|HumanEval/1|224|4|2.041|2.000|48.98%|50.00%|
|HumanEval/2|160|3|2.034|2.000|32.19%|33.33%|
|HumanEval/3|288|5|2.075|2.000|58.50%|60.00%|

Every row has one mandatory tile/head. Thus V2 keeps1.034–1.075 optional
tiles/head, versus V31's1.000. V2 is slightly **denser** at these states.

The16 complete observed request traces also independently reconstruct the
eligible counters from their actual block positions and denoising iterations.
For every traced native V31/V3a request, summing its exact per-J rounded budgets
reconstructs its saved kept counter with zero discrepancy. Applying those same
budgets to V2's observed geometries gives fewer kept tiles than actual V2 in
all four traced V2 requests.

The full164-run average available tiles differ (V24.070, V315.655) because V2
stops earlier, while V31 reaches later canvases with longer prefixes and larger
rounded budgets. Similar50% sparsity averages therefore do not establish that
the masks see the same context geometry. The observed lower full-run optional
count for V2 partly reflects its failed short trajectory; it is not independent
proof of the cause.

## Tile count versus valid-token pair count

A partial final KV64 tile counts as one geometric tile. If it has32 valid keys,
it receives the same tile weight as a full64-key tile. All skipped tiles are
full64-key tiles because the last tile is mandatory. Accordingly, on the sampled
EOS states, valid-query/key-pair sparsity is65.42/55.98/38.63/65.00% for V2 and
66.67/57.14/40.00/66.67% for V31. This changes absolute percentages but does not
reverse the comparison: V2 remains slightly denser. Complete token-pair sparsity
for all164 trajectories cannot be recovered from aggregate counters alone; the
original published quantity is explicitly tile sparsity.

## What remains supported after this correction

Both methods retaining one optional tile does not imply that tile has the same
effect. The earlier diagnostic already included an exact-count control: keep
V2's exact tile count in every layer/head, but choose optional support by native
mass. Across the80 EOS-state layer cases, retained mass rises58.06%->67.05%
and attention-output error/value-RMS falls0.18254->0.12957. A same-state full-model
probe reduces29 EOS argmaxes to12 and28 high-confidence EOS predictions to4 at
those identical per-head budgets. This is evidence about support identity; it
does not rely on V31 consuming more tiles.

The correct conclusion is that the original V2 route has a harmful first-tile
selection bias on these sampled failures despite keeping a comparable number
of tiles. The previous implication that V31 avoids failure by generally keeping
more optional tiles on the same short state is withdrawn. No GPU run,
recalibration, generation, score change, or selector implementation change was
performed for this CPU-only audit.

Sources are fingerprinted in `summary.json`; `receipt.json` and
`final_complete.json` bind the recount. Source code:
`scripts/llada22_humaneval_sparsity_audit.py`. Original numerical runtime:
snapshot224f8f11, attention SHA619ec332. Benchmark source:
`results/multi_model_value_report_20261102/attempt002/summary.json`.
