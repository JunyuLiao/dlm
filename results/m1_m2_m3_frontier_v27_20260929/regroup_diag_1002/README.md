# Query-row regrouping diagnostic (idea from chw/value_aware), 2026-10-02

Question: chw's branch reorders query rows so rows with similar skip sets share a kernel query tile, because a KV tile
can only be skipped for a query tile when every row in it agrees. How much would that save in our pipeline?

Method (`scripts/v27_regroup_diag.py`, no timing claims): dllm, the E5 frozen configuration
`M3_R6_A64_fused_dp_async_m1ln2_c0_fa4`, LongBench-v2 64K and 32K items 0–3, seed 404. At every dense-prefix decision
the per-row need sets are recomputed from the stored risk table (worst-row rule, fixed threshold). Kept eligible prefix
tiles are counted per head for the natural row order (rows 0–127 / 128–255), rows sorted by need count, and rows sorted
by the leading singular vector of the need matrix, at 128-, 64- and 32-row query tiles (work in 128-row units), plus
the per-row lower bound. The natural 128-row count equals the router's own skip map in every decision (receipt).

Pooled kept fraction of eligible prefix tiles (4 requests each; 875 decisions at 64K, 650 at 32K):

| query tile | 64K natural → best regroup | 32K natural → best regroup |
|---|---|---|
| 128 rows (FA4 today) | 14.0% → 13.4% (−4.2%) | 22.8% → 21.8% (−4.4%) |
| 64 rows | 10.9% → 10.4% | 18.1% → 17.2% |
| 32 rows | 8.5% → 8.3% | 14.4% → 14.1% |
| per-row bound | 2.1% | 4.2% |

Reading: regrouping recovers only 3–5% of kept tiles at any tile size (rows' need sets are not nested, matching chw's
own synthetic estimate of 2–4 pp). Finer query tiles are the larger lever (−22% kept work at 64 rows, −39% at 32 rows),
but only if a head_dim-512 kernel with 64/32-row query tiles keeps FA4's per-tile efficiency. At 64K the sparse GLOBAL
execution is a few ms of a ~28 ms forward, so the ceiling is roughly 1–2% per step before any efficiency loss.
Regrouping is therefore not integrated; finer query tiles are a possible kernel direction.

Raw outputs: `regroup_raw_run_a.txt` (128-row only) and `regroup_raw_run_b.txt` (all tile sizes). In run b the 64/32-row
fields were written with the factor 128/size instead of size/128; multiply them by (size/128)² (1/4 for 64 rows, 1/16 for
32 rows) to get the table above. The committed script has the corrected factor.
