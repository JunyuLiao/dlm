# V3 accuracy diagnosis — full-row value selectors under V31 reuse

Date: 2026-10-10 UTC. This is a focused analysis of `current_v31_control`,
`value_v3a_singleton_delete`, and `value_v3b_approx_batch8`. The generation
panel is the completed 503-item LongBench-v2 audit; the mask diagnostics are
111 sampled frozen Q/K/V snapshots from layer 5, with ages 0, 1, and 7 after
refresh. The snapshot analysis is descriptive and does not run model forwards.

## Direct answer

Having all attention rows available does not make the selectors equivalent.
The three methods use the same row-level inputs but optimize different set
functions:

| method | per-tile score | set construction |
|---|---|---|
| V31 control | worst-row normalized attention mass share, with held-map hysteresis | independent top-k |
| V3a | worst-row projected-output error if that tile alone is deleted | independent singleton ranking |
| V3b | projected-output error after the already-deleted set is updated | sequential greedy approximation, eight deletions per batch |

The control and V3 methods also apply hysteresis to different base scores. The
control adds the sticky term to log mass; V3 multiplies a value-deletion score
by the sticky factor. Therefore, “full information” does not imply equal maps.

## Why V3a can have high IoU and lower accuracy

V3a's age-0 IoU with the control is 0.703. Because both masks have nearly the
same fixed prefix budget, that corresponds to roughly 17–18% of the selected
tiles being replaced, about 22 of 128 tiles per query unit. IoU measures the
bulk overlap, not which tiles were replaced.

V3a's singleton score is approximately

`alpha_j * ||mu_j - full_output|| / ((1-alpha_j) * nu)`

per row, followed by a worst-row reduction. It can therefore remove a tile
with substantial attention mass when its projected value is close to the full
output, while retaining a lower-mass tile whose value points in an unusual
projected direction. That is reasonable for the one-tile PV surrogate, but it
can be harmful when several tiles are removed together, when the projected
value is not aligned with the model's output projection, or when a rare query
row is sensitive to that tile.

The sampled diagnostics show the difference is small on average but persistent
in the tail:

| fresh snapshot age | V3a minus control max-row error | snapshots where V3a is worse |
|---|---:|---:|
| 0 | +0.0019 | 50.0% |
| 1 | +0.0016 | 55.0% |
| 7 | +0.0030 | 64.5% |

At age 0, V3a's mean-row error is 0.2534 versus 0.2495 for the control and its
retained mass is 0.3609 versus 0.3693. These are small averages, but the
sampler is sensitive to rare row-level changes. V3a also changes the native
trajectory on 500/503 items and averages 6.9 fewer denoising calls than the
control. A high IoU therefore means “mostly the same support,” not “the same
semantic support.”

There is a second issue: singleton scores are not additive. For a deleted set
`D`, the actual projected-output error depends on the vector sum of the
centered contributions and on the retained-mass renormalization. Ranking tiles
by their individual deletion errors does not minimize that joint error. V3a
can make a locally sensible but jointly poor set while remaining close in IoU.

The held-map tail is slightly worse than the fresh-map comparison suggests. In
the sampled held snapshots at age 7, V3a's max-row error averages 0.661 versus
0.638 for the control, even though its fresh age-7 IoU with the control is
0.828. This is why fresh-mask IoU should not be treated as a proxy for the
quality of the map after it has been carried through later calls.

## Why V3b is much worse

V3b is not exact greedy. It selects eight candidates using the same current
residual, applies all eight, and only then recomputes scores. The projected
residual objective is non-submodular because of vector cancellation and
renormalization, so batch-8 selection has no greedy dominance guarantee. Once
an early batch removes a bad combination, the later cleanup can remove more
tiles to meet the budget but cannot restore those earlier choices.

The snapshot evidence shows the failure emerging with reuse age:

| fresh snapshot age | V3b max-row error vs control | snapshots where V3b is worse | mean IoU with control |
|---|---:|---:|---:|
| 0 | −0.0344 | 0.0% | 0.183 |
| 1 | +0.0414 | 62.5% | 0.642 |
| 7 | +0.0883 | 74.2% | 0.558 |

The mean-row error hides this: V3b is initially good on average, but its
age-7 max-row error has mean 0.612 and p90 0.851, compared with control mean
0.524 and p90 0.572. A few unstable rows are enough to alter entropy,
argmax stability, and the native stopping rule. Consistent with that, V3b
averages 61.6 more denoising calls than the control and scores only 242/503
(48.1%). Its GLOBAL sparsity is also higher, 84.97% versus 83.16%.

The retained exact-greedy diagnostic supports the same conclusion. On its
partial 128-cell panel, exact greedy scored 58 correct, V3a 62, and the
control 67. Exact greedy was not better even before the batch-8 approximation,
although that panel is diagnostic rather than a full-suite claim. Thus batch-8
is a major additional problem, but not the whole explanation: the projected
value objective itself is not aligned with task accuracy under reuse.

An additional six-snapshot full-dimensional-V check rules out rank-32
projection as the only explanation. For the V2 value objective, projected-V
and full-V masks had IoU 0.858, while full-V and the native-mass control had
IoU 0.156. At the same threshold, normalized full-dimensional error was 0.330
for the full-V mask versus 0.247 for the mass control. Giving the full-V
selector more work (about 138 tiles versus the control's 128) still gave 0.262
error. The evidence therefore points to the value-deletion objective and its
support choice, not merely insufficient projection rank. This is a separate
six-snapshot operator diagnostic, not a V3 generation score, but it directly
tests the proposed value-direction mechanism.

## Why all three full-information masks have different IoU

“Full row information” describes what is available to the selector, not how it
is compressed:

- V31 keeps tiles that matter to the worst row's mass share. It is a robust
  support rule and has a fixed top-k set per query block.
- V3a keeps tiles with the largest worst-row value-deletion score. Value
  centering, per-row normalization, and the projected direction change the
  order even when the mass order is similar.
- V3b changes the order after each deletion batch. Its residual state makes
  the result path-dependent, so it can diverge from both V3a and V31.
- Held-map hysteresis is applied to different score scales in all three paths,
  further changing the boundary tiles.

The control's high accuracy is therefore not contradicted by V3a's high IoU.
The changed tiles are exactly where the objectives disagree, and those tiles
can have disproportionate downstream effect. IoU is useful for measuring
support stability, but it is not an accuracy metric.

## Accuracy-first consequence

The next test should not be another full generation sweep. Use identical frozen
Q/K/V snapshots and equal budgets to compare V3a, exact greedy, and the control
with full-dimensional attention output, post-output-projection/logit change,
per-row max error, and argmax changes. Then test a mass-anchor hybrid that
allows value-based replacements only under a retained-mass floor and an explicit
IoU floor. Separately compare fresh V selection with held V selection. This
will distinguish three causes that the current generation panel cannot separate:
non-additive singleton scoring, batch-greedy approximation, and cross-step
staleness.

Sources:

- Official audit: `reports/audit_scoring_no_exact_attempt002/summary.md`.
- Paired trajectory totals: `diagnostics/complete_audit_summary_no_exact_attempt002.json`.
- Sampled frozen-snapshot diagnostics:
  `diagnostics/offline_accuracy_analysis_20261009/aggregate.json` and its
  per-shard summaries.
- Selector definitions: `experiments/numerical_qk_reuse/v31_value_selectors.py`,
  `v31_value_kernels.py`, `v31_value_summary.py`, and
  `experiments/numerical_qk_reuse/vllm_adapter.py`.
- Partial exact-greedy diagnostic:
  `diagnostics/exact_completed_shard01/attempt001/summary.json` and
  `paired_comparisons.json`.
