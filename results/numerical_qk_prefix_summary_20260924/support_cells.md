# Support choice vs stale weights, on one common mask with real causal T

`scripts/preqk_support_cells.py` (written in v7, unrun until now). One bounded
capture on `aime26/2`, layers 0 (local) and 5 (global), 52 cell rows. Every
cell shares ONE legal-mask support, one captured state, the same current
V/reference/threshold, and the REAL causal T taken from completed prior
logits. Raw: `support_cells.json`.

Two selectors on the same state: `K_f` runs the selector on CURRENT scores,
`K_h` on CACHED scores. Four outputs follow, plus unpruned references:

- `O_hf vs O_ff` -- the **incremental** cost of choosing the support from
  history, with weights held current.
- `O_hh vs O_hf` -- the cost of **stale final weights at the SAME support**.
- `full_cached vs full_current` -- stale weights with no pruning at all.

Each error is normalized by its own stated reference and both norms are
stored, so the denominator is never ambiguous. Support agreement is measured
directly (tile Jaccard, disagreeing tiles, retained current mass), not
inferred from an output norm.

| kind | age | n | Jaccard | disagreeing tiles | retained mass K_f | retained mass K_h | O_hf/O_ff | O_hh/O_hf |
|---|---|---|---|---|---|---|---|---|
| local | 0 | 10 | 1.000 | 0 | 0.799 | 0.799 | 0.0000 | 0.0000 |
| local | 1 | 6 | 0.616 | 109 | 0.593 | 0.728 | 0.1857 | 0.4052 |
| local | 2 | 6 | 0.690 | 111 | 0.739 | 0.831 | 0.1230 | 0.4387 |
| local | 7 | 4 | 0.784 | 79 | 0.861 | 0.913 | 0.0640 | 0.5066 |
| global | 0 | 10 | 1.000 | 0 | 0.976 | 0.976 | 0.0000 | 0.0000 |
| global | 1 | 6 | 0.813 | 96 | 0.979 | 0.927 | 0.1109 | 0.8099 |
| global | 2 | 6 | 0.875 | 68 | 0.992 | 0.953 | 0.0994 | 0.8490 |
| global | 7 | 4 | 0.925 | 32 | 0.999 | 0.956 | 0.1332 | 0.9520 |

## Age 0 assertion

At score age 0 the cached scores ARE the current scores, so the two selectors
must coincide. Across all 20 age-0 rows: Jaccard **1.000**, disagreeing
tiles **0**, and all three error cells exactly **0.0000** -- for both layer
kinds, with and without nonuniform T. The comparison is therefore calibrated,
not accidentally agreeing.

Observed T spreads across the capture: [0, 1.5, 2.625, 2.812, 2.953, 2.977] (0.000 rows are early steps
with no completed history yet; the rest are genuinely nonuniform).

## What it says

**Stale final weights dominate the historical support choice, by roughly an
order of magnitude.** At matched age, `O_hh/O_hf` runs 0.37-0.53 (local) and
0.77-1.00 (global), while `O_hf/O_ff` -- the incremental cost of having picked
the support from history -- runs 0.05-0.22 (local) and 0.03-0.24 (global).

This is the v6 conclusion re-established with the confound removed: v6's cell
B was ALL-KEPT, so its `B vs C` mixed 'pruning at all' with 'pruning chosen
from history', and it ran with uniform T. Here both selectors prune, both sit
under one common mask, and T is real.

**Support agreement degrades then partially recovers with age.** Local Jaccard
falls to ~0.53-0.66 at ages 1-2 and returns to ~0.66-0.91 by age 7; global
stays higher throughout (0.71-0.96). The disagreement is real and sizeable --
up to 205 tiles -- so 'the support is stable' is too strong a statement.

**A genuinely unexpected observation:** on local layers at ages 1-2 the
HISTORICAL selector often retains MORE current softmax mass than the fresh
selector (e.g. 0.846 vs 0.718, 0.755 vs 0.611). Selecting from older scores is
behaving more conservatively there, not worse. That is an observation from one
capture on one question, not a claim about quality.

## Scope

One id, two layers, 52 rows, one seed. This closes the specific unfinished
diagnosis and is not a benchmark rerun. It does not establish that numerical
current-V routing beats a frozen mass/bitmap or a fresh value-aware selector;
that comparison is a separate study and was not launched.
