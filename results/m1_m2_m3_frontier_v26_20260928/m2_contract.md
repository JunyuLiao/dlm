# M2 contract (v26): `mu_mode=pooled`, named M2_projected_mean_KV64

Fan's notes say only "pooling V"; this is the reproducible definition used here, not a verbatim formula.

**M1 (unchanged).** For each KV64 tile J and query row i, the historical score gives the block log-mass z_iJ and in-tile weights w_ij. The selector uses mu_iJ = Σ_j w_ij · Z_j(t), where Z_j(t) = V_j(t) · R is the current projected V (the same rank-32 projection).

**M2.** Replace only mu:
- mu_iJ^pool(t) = mean of Z_j(t) over the keys j in J whose historical score is finite for row i (real and legal);
- partial tail tiles, padding and −inf positions are excluded from both numerator and denominator.

Everything else is identical to M1: z, alpha, reference, T, first-support protection, the sequential retained-state update (fed by the pooled value), the Q128 decision, and the final output from **current QK and original current V**. Pooled V is never used for PV.

**Implementation.**
- A `POOL` constexpr in both the static and generic `_route` kernels computes `mu = dot(finite / count, sketch)` with the same tf32x3 path. `POOL = False` is the unchanged M1 path, and the textual static/generic identity test still passes.
- `route_only(..., pool=True)` exposes it. The fused route+PV path refuses pooled mode.
- Prefix summaries (`prefix_block_summary`) store the pooled mu for immutable prefix tiles. Pooled mu depends only on the frozen prefix Z, so reuse is valid under the same lease/anchor identity. The per-row copy is redundant (rows share it when there is no row mask); a per-tile cache is a possible saving that was not implemented.

**Tests** (`tests/test_v26_m2_pool.py`, all passing on H100):
- pooled equals exact when Z is constant inside a tile;
- a counterexample where attention-weighted opposite values make pooled risk lower;
- illegal/padding keys cannot affect pooled decisions;
- config guards;
- `reserve_physical` with shared storage;
- kernel vs independent Torch reference, for static and generic variants, with bitmaps equal at three thresholds (the exact path is also re-checked).

**Scope.** GLOBAL layers only, with no row-varying mask (a mask is honoured through `finite`). The first version is R1/A8 on the bootstrap mainline with `aligned16_odd` storage.
