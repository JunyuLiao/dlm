# v8 corrections to the v7 record (records preserved, summaries fixed)

Raw receipts and published commits are unchanged. Inline `CORRECTED (v8)`
notes were added to `../numerical_qk_preqk_execution_20260924/panel_report.md`.

## A. The v7 panel counter line was one request, not the aggregate

`panel_report.md` printed "18990 attention calls, 2820 score refreshes, 16170
pre-QK consumer calls, 2.10e10 materialized current-QK elements" as the
four-request total. Those are **aime26/14 alone**. Summing
`panel/request_rows.json` over the four requests:

| counter | correct four-request sum |
|---|---|
| attention calls | 32310 |
| score refreshes | 5100 |
| pre-QK consumer calls | 27210 |
| materialized current-QK elements | 32710819840 |

"Materialized" counts only scores written by `observe_scores`; the retained
dots performed **inside** the fused pre-QK consumer are additional and are
tracked by the kernel's own tile counters. Both must be reported; neither is
"total QK" on its own.

## B. Blaming the slowdown on aime26/14 was wrong

The v7 report attributed the end-to-end gap largely to `aime26/14` (251.49 s of
462.93 s) plus trajectory divergence. That framing is withdrawn. Excluding /14
entirely:

| arm | other three requests, wall (s) | forwards |
|---|---|---|
| native dense | 99.178 | 648 |
| pre-QK M1 | 211.447 | 444 |

The optimized arm is **2.132x slower on the other three requests while running
31% fewer forwards.** The slowdown is a genuine per-forward cost, not an
artifact of one capped answer and not explained by higher call counts. /14
remains in every quality and performance denominator.

Decomposition of the full four-request slowdown, from the same rows:
`2.833925 = 0.991713 (calls factor) x 2.857607 (amortized request-cost
factor)`. Pooled request ms per forward: **150.419 dense vs 429.839 pre-QK M1**.
These are whole-request figures including setup and any JIT; they are not
clean GPU forward measurements.

## C. The scaling sweep's scope was overstated

`scaling_sweep.json` uses synthetic QKV, a random Z/reference unrelated to V,
uniform T, threshold -1, and an independently random drop bitmap fed to the
consumer. Its `preqk_plus_route` column is an **arithmetic sum of separately
timed pieces**, not a directly timed pipeline, and it excludes
projection/reference/anchor/adapter overhead while comparing different local
visibility conventions. It is legitimate component stress data that indicates
where to look. It is **not** a measured "complete method 3.51x", and its
global-layer 0.90-0.96x is **not** a confirmed net benefit.

The honest statement after v7 is therefore: the selector is a *plausible*
major cost whose share of the real request gap was **not** established.

## D. What the v7 consumer evidence does and does not cover

The v7 pre-QK consumer genuinely forms current attention only on retained
tiles, with no K/V load and no dot for dropped consumer tiles; its 14
qualification cases pass exact work counters and sit essentially the same
distance from FP32 as the incumbent. That evidence stands **within its scope**:
synthetic same-support cases. Current projected-V preparation can still read
values for routing, so consumer-side skipping is not whole-model byte
avoidance.
