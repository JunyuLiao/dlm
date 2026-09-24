# v7 bounded generation panel: optimized M1 vs contemporaneous dense

Eight new full request executions, the entire budget for this round: the same
four development IDs, seed 42, thinking ON, native adaptive, 8192-token
budget, EOS. Both arms ran in the SAME session on the same code
(`historical_route_preqk_current_output`, `support=legacy_junyu_mask`,
`score_refresh_period=8`, `decision_interval=1`), so unlike the v6 panel the
dense controls here are **contemporaneous**, not historical.

| id | arm | correct | termination | tokens | calls | canvases | request wall (s) |
|---|---|---|---|---|---|---|---|
| aime26/2 | dense | yes | eos | 4941 | 146 | 20 | 24.12 |
| aime26/2 | preqk M1 | yes | eos | 3026 | 102 | 12 | 64.94 |
| aime26/8 | dense | yes | eos | 3945 | 132 | 16 | 20.12 |
| aime26/8 | preqk M1 | yes | eos | 3710 | 181 | 15 | 75.47 |
| aime26/14 | dense | no (cap+unparsed) | length | 8192 | 438 | 32 | 64.18 |
| aime26/14 | preqk M1 | no (cap+unparsed) | length | 8192 | 633 | 32 | 251.49 |
| aime26/20 | dense | yes | eos | 8161 | 370 | 32 | 54.94 |
| aime26/20 | preqk M1 | yes | eos | 3202 | 161 | 13 | 71.03 |

| aggregate | correct | mean wall (s) | mean calls | mean canvases | pooled s/call |
|---|---|---|---|---|---|
| dense (contemporaneous) | 3/4 | 40.84 | 271.5 | 25.0 | 0.1504 |
| preqk M1 | 3/4 | 115.73 | 269.2 | 18.0 | 0.4298 |

## Quality

The optimized arm scores **3/4** with correctness vector `[True, True, False, True]`,
**identical element-wise to the contemporaneous dense control** `[True, True, False, True]`.
Both miss `aime26/14` the same way (capped at 8192, unparsed). This is the
same count and the same vector as the v6 `routing_only_current_output` arm, so
rebuilding the consumer to skip dropped-tile QK did not cost quality. Four
questions at one seed remain a development signal, not population evidence.

## The dense control reproduced itself across sessions

The contemporaneous dense run reproduced the frozen v5 dense receipts
**exactly** in decoder calls (146/132/438/370) and canvases (20/16/32/32),
with walls within ~5% (24.12 vs 25.30, 20.12 vs 20.11, 64.18 vs 64.46, 54.94
vs 55.35). That retrospectively supports reusing those frozen receipts as
quality references, and it means the dense baseline here is trustworthy.

## End to end: NOT faster. This is the headline negative result.

Mean request wall is **115.73 s against dense's 40.84 s
(2.83x)**, and pooled cost per decoder call is
**0.4298 s vs 0.1504 s (2.86x)**. It is also worse end to end than the v6
`routing_only_current_output` arm (97.92 s mean), even though every matched
per-step and per-kernel measurement says the new consumer is faster than that
arm. Two things are going on and they must not be conflated:

- **Trajectory divergence.** The pre-QK consumer differs from the materialized
  path at BF16 accumulation-order scale (~2e-5 relative, qualification table).
  Over thousands of steps that is enough to change argmax decisions, so the
  arms walk different trajectories. Call counts per question moved in both
  directions against v6 (aime26/2 164->102, /8 257->181, /20 129->161,
  /14 472->633). `aime26/14` alone -- a question BOTH arms get wrong at the
  cap -- accounts for 251.49 s of the 462.93 s total. Four questions cannot
  separate a real regression from this.
- **A measured structural cost that is real regardless of trajectory**, in the
  scaling sweep: `route_only` on sliding layers. See below.

## Where the remaining time actually goes (measured, not inferred)

From `scaling_sweep.json`, attention only, 50% of tiles dropped:

- Sliding geometry at its saturated length (nk=1279, where **25 of 30 layers**
  live): dense 0.435 ms, `preqk` 0.379 ms, but `route_only` **1.146 ms** --
  the selector alone costs 2.6x the entire dense attention for that layer,
  pushing `preqk+route` to **3.51x dense**.
- Global geometry at >=2048 keys: `preqk+route` reaches **0.90-0.96x dense**,
  i.e. at parity or slightly better -- but only 5 of 30 layers are global.

So the output consumer this round rebuilt is no longer the bottleneck; the
historical-score selector is, and it is most expensive exactly where the model
is thickest. That is the measured limiting cost.

## Receipts

Redacted quality/config in `panel/` (no completion text, no extracted answers;
M1 config fingerprint `c012aba6cc802f9b`). Raw private receipts stay on the
remote host under `results/panel_v7/`. Counters for the optimized arm across
the four requests: 18990 attention calls, 2820 score refreshes, 16170 pre-QK
consumer calls, 2.10e10 materialized current-QK elements.
