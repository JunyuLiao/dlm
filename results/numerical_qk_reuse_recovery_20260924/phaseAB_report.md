# Phase A2 + Phase B: real diagnosis of the M1/M3 negative result

> **CORRECTED.** (1) Cell B is ALL-KEPT legacy-legal attention, not the
> fresh-Junyu *selected* support, so `B vs C` measures the effect of pruning
> at all, not the incremental effect of historical selection. (2) The run
> used default uniform sensitivity; production causal T weights were not
> applied. (3) `rel_l2(x,y)` normalizes by the second argument, which for
> `a_vs_e`/`c_vs_d` is the approximate output. (4) The A2 check covers 12
> uniform-T samples at two layers and narrows, but does not exclude, an
> implementation defect. See
> `../numerical_qk_preqk_execution_20260924/corrections.md` items C and D.
> Measured values are unchanged.

Real capture: `scripts/native_reuse_phaseAB_diagnostic.py`, native-dense
trajectory (untouched, generation output is the true dense answer), ids
`aime26/2` and `aime26/8`, seed 42, thinking on, capped at 600 new tokens
(diagnostic only -- covers 3 real canvases per id, prefixes 0->621/608
tokens, well short of a full AIME answer). Instrumented layers 0 (local/
sliding) and 5 (global/full), 120 real instrumented calls total. Raw:
`phaseAB_diagnostic.json`.

## Phase A2: is it an implementation/kernel defect?

**No disagreement in the sampled scope.** `kernel_vs_oracle`: 12
real-activation samples (both instrumented layers, both ids, at natural
score-refresh calls), Triton `cached_executor` vs the Torch `reference.py`
oracle, identical scores/V/threshold: max relative L2 = 0.00135, and the
skip/retain bitmap agreed exactly (`skip_agree=True`) in all 12 samples.
This extends the existing synthetic-QKV GPU tests to real captured
activations. Scope limits: 12 samples, two layers, **default uniform
sensitivity (not production's nonuniform causal T)**. That narrows, but does
not exclude, an implementation defect, and it is not full-decoder or
complete A1 equivalence.

## Phase B: which of the five cells actually hurts?

Mean relative L2 by real score age (score_refresh_period=8, so age 0..7
recurs every canvas), pooled over both ids and all 3 captured canvases:

| kind | age | n | A vs E (stale scores, all-kept) | B vs C (pruning-at-all, current scores) | C vs D (stale weights, same support) |
|---|---|---|---|---|---|
| global | 0 | 11 | 0.000 | 0.019 | 0.000 |
| global | 1 | 10 | **0.686** | 0.048 | 0.685 |
| global | 4 | 6  | 0.963 | 0.119 | 0.958 |
| global | 7 | 5  | 1.072 | 0.197 | 1.058 |
| local  | 0 | 11 | 0.000 | 0.152 | 0.000 |
| local  | 1 | 10 | **0.332** | 0.152 | 0.331 |
| local  | 4 | 6  | 0.497 | 0.127 | 0.495 |
| local  | 7 | 5  | 0.543 | 0.125 | 0.551 |

(Full table for every age 0-7: `phaseAB_diagnostic.json`.)

**Interpretation, directly from the five-cell design:**
- **A vs E isolates harm from stale final scores with no pruning at all.**
  It is already large at age 1 (0.33-0.69) and exceeds 1.0 by age 5 on
  global layers. NOTE: `rel_l2(x,y)` divides by `||y||`, and for this cell
  `y` is the *approximate* (cached-score) output, so values above 1 do NOT
  mean 'error bigger than the fresh signal' -- that gloss is withdrawn. This is not a corner case -- it is
  the typical age reached every 8 calls, and M1/M3 canvases average
  40-45 calls (`calls/canvas pooled: M1 40.55, M3 44.10` from the frozen v5
  smoke), so most calls in a canvas run at ages 1-7.
- **C vs D is numerically close to A vs E at every age.** D is literally
  what production M1 computes (stale scores restricted to the routed
  support). The two cells use different supports and different denominators,
  so their closeness is an empirical observation, **not a mathematical
  identity**. Read together they indicate the retained-support choice adds
  little on top of the scores already being stale.
- **B vs C compares all-kept legacy-legal attention against the pruned
  support, holding scores current.** Cell B is ALL-KEPT, not the fresh-Junyu
  *selected* support, so this is the cost of pruning at all -- not the
  incremental cost of choosing that support from historical information. It stays an order of magnitude smaller (0.02-0.20) and
  does not blow up with age the way A vs E does, even though the *routing
  decision itself* is being made from increasingly stale scores (mean
  retained/eligible tile fraction only drifts mildly across ages: global
  0.92->0.86, local 0.61->0.64-0.68). The tile-level selection is comparatively
  stable; the numeric attention weights are not.

**Conclusion (within the sampled scope):** the 0/4 collapse is attributed to
**stale final attention scores**. No implementation defect appeared in the
12 sampled uniform-T cases (which narrows rather than excludes one), and the
pruning cell stayed an order of magnitude smaller than the stale-score cells
at every age. Global (full-attention) layers
degrade faster than local (sliding) layers, consistent with global attention
depending on the whole growing context while local only looks nearby.

## What this selects for Phase D

Per the v6 branching rule: age 1 is *not* viable (0.33-0.69 already, not a
mild bump), and C remains comparatively good at every age (never above 0.20)
while D does not. This is exactly the documented trigger for
`routing_only_current_output`: reuse the *stale-score-derived retained
support* (which stays reasonably stable) but always recompute *current* QK
for the final softmax/PV, rather than reusing stale scores as production M1
does today. This does **not** save QK compute -- it is a support-reuse-only,
score-freshness-always variant, and its real retained-QK cost and any net
E2E benefit must be measured (Phase C profile), not assumed.
