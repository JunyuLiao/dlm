# Exact prefix-block-summary selector: qualification

## What is cached, and why it is exact

At a fixed request/canvas/layer/query-head/position/mask and one real score
anchor `a`, for a KV block `J` lying WHOLLY inside the immutable prefix:

```
z_a[i,J]  = logsumexp_{j in legal J} S_a[i,j]
mu_a[i,J] = sum_j w_a[i,j|J] * (V_prefix[j] R)
```

`S_a` is frozen until its next real observation and the encoder-owned prefix
V/projection is frozen until a legitimate invalidation, so `z_a` and `mu_a`
are constant across the reuse steps. The runtime was recomputing them on
every decision. They are now written once at the anchor -- by `_route`
itself, so the reused values are that kernel's OWN FP32 outputs -- and read
back afterwards.

**Not cached:** alpha, risk, the retained log mass `l`, the projected
accumulator `o`, and the bitmap. Those depend on earlier decisions in the
same scan and on live T / live full-V RMS, and are recomputed every M1 step
in the same scan order. This is conditional on the source/mask/position
ownership contract, never on top-1 token stability. It is not pooling and
not caching the model's attention output; `mu` is a small routing ingredient
and the final output still goes through the unchanged pre-QK consumer.

## Real-state qualification (remote GPU, production causal T)

Armed step 90, `aime26/14`, production T actually in effect
(spread 2.625, i.e. genuinely nonuniform).

| layer | kind | prefix | tiles summarized | T spread | bitmaps identical | disagreeing tiles | summary |
|---|---|---|---|---|---|---|---|
| 0 | local | 1023 | 15/20 (75%) | 2.625 | **True** | 0 | 8.23 MB |
| 5 | global | 1923 | 30/35 (86%) | 2.625 | **True** | 0 | 16.47 MB |

The local layer is at a **saturated** prefix (1023 = `sliding_window - 1`), which
the script asserts rather than assuming.

## Trajectory exactness (the decisive check)

The same bounded generation, run twice, once per selector, identical seed and
state. The output consumer is identical between the arms, so any divergence
could not have been blamed on consumer rounding.

- tokens identical: **True** (first divergence: None)
- token counts: [512, 512], terminations: ['length', 'length']
- attention calls: [750, 750]
- summary selector actually engaged: 100 builds, 525 hits, 0 misses

Extended to **full-length natural generation** (two complete-answer smoke
requests, 8192-token budget, `aime26/2` and `/8`): completion tokens are
**identical token-for-token** to the v7 legacy-selector receipts, with
identical decoder calls (102, 181) and canvases (12, 15), while the selector
was genuinely active (2075 and 3750 summary hits). Those smoke runs were
executed with `--diagnostic` ON, so their wall times include per-step
diagnostic quantile/entropy work and are **not** comparable timing.

## Space, disclosed

33 FP32 scalars per row per prefix tile (z + 32-d mu) plus flags. At the
saturated local geometry that is **8.23 MB per local layer**, and the
observed resident total across the model is **205.8 MB**. The existing FP32
score cache is retained (other output modes still need it), so **total
resident memory increases**; this buys avoiding the re-read of 64 FP32 scores
and the 64x32 sketch, and the Qx64 by 64x32 product, per prefix tile per
decision.
