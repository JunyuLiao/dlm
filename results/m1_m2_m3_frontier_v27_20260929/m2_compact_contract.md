# M2 compact contract (`mu_mode = pooled_compact`)

M2 is our reproducible implementation of Fan's "pooling V" (`M2_projected_mean_KV64`). The name is ours; Fan did not give an exact formula. Two execution versions exist, and both keep the same M1 machinery around the mean.

**Shared by both versions:**
- historical QK gives the row-wise block log-mass `block_z`;
- current reference, causal T, first-support rule, scan order and the sequential retained accumulator are unchanged;
- the final output always uses CURRENT QK and CURRENT original V.

| version | `mu` for a KV64 tile | where computed | prefix summary |
|---|---|---|---|
| `M2_reference` (`pooled`, v26) | per query row: `legal_w @ sketch`, with `legal_w` uniform over that row's finite-score keys | inside the selector, one Q128×KV64×32 dot per tile per head | per row z/mu/active/bad (mu = 128×32 FP32 per tile per head) |
| `M2_compact` (`pooled_compact`, v27) | one vector per (batch, KV head, tile): mean of CURRENT projected V over real legal keys | `tile_pool()` from the unpadded current projection, once per decision call | z/active/bad only; mu is read from the compact pool |

**What changes in execution:**
- No per-row uniform dot.
- No 16 KB/tile/head per-row mu is stored at a score anchor or read at each LOAD decision.
- All Q heads of a GQA group, and all rows, share the pool.
- The pool is recomputed from the CURRENT projected V at every decision, which is a small [B,8,K,32] torch reduction. The v27 contract allows reusing prefix pools under a verified source identity; that is not done here, because rebuilding is cheap and removes a staleness risk.

**Validity guard (explicit rejection, never silent):**
- The compact pool is the row-wise reference only when every real query row's legal key set equals the tile's pooled legal set.
- The selector compares each row's finite-key count with the pool count, over real rows, on every non-summarized tile.
- Any mismatch sets `pool_mismatch` and triggers an asynchronous request failure.
- Padding (aligned16 tail, keys beyond K) is never counted.
- The compact mode is implemented only in the generic Triton selector; the static variant refuses it.

**Qualification so far** (GPU tests on H100):
- The decisions equal the independent row-wise FP32 reference (`geometry_diagnostic._select(pool=True)`) at three thresholds.
- A single row losing one legal key flags exactly that tile.
- Summary STORE, then LOAD, and an aligned16-pitched copy are all bit-identical to the plain compact route.

**Numerics.** The compact mean is an FP32 sum ÷ count. The reference is a tf32x3 dot with row weights. These are different reduction orders, so the two are named separate versions. Their quality must be measured, not borrowed from M2_reference.

**Cost.** Measured directly in the v27 profile (arm `M2c_pool_R1_A8` vs `M2_pool_R1_A8` and `M1_R1_A8`, same states, same GPU). See `direct_cost_breakdown.csv` when published.
