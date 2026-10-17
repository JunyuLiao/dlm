# 2026-10-17 publication: methods, setup, complete statistics and reproduction

The three-direction study is complete. Its detailed publication is split into:

1. [All evaluated methods and experiment setup](../../docs/research_update_20261017/01_METHODS_AND_SETUP.md).
2. [Results, uncertainty, accounting and diagnostic limits](../../docs/research_update_20261017/02_RESULTS_AND_STATISTICS.md).
3. [Implementation integration and reproduction](../../docs/research_update_20261017/03_INTEGRATION_AND_REPRODUCTION.md).

The completed native study has four candidate methods plus V31 and three fresh
dense/all-kept references, 358 input/seed cells and 5,370 officially scored trials.
All structurally valid outcomes are retained. No candidate passes the unchanged
all-workload screen; clean timing remains paused and no full503 run was launched.
Initial operator/decision-only variants have their own catalogue and results,
not invented native scores. Historical pending evidence retains original status.

## Public statistics

[`attempt001/native_statistics.md`](attempt001/native_statistics.md) gives every
native method's score, trial count, N median/mean/p95, N/C, C, T, cap rate,
score/N/C/T differences with both interval analyses and per-repeat differences.
It labels instrumented S/W/S-per-N as audit cost, not synchronized clean timing.

| File | Content |
|---|---|
| [native_results.csv](attempt001/native_results.csv) | Every flattened native-summary field, including physical-work and extra refinement counters. |
| [native_comparisons.csv](attempt001/native_comparisons.csv) | All paired score/N/C/T contrasts, both cluster intervals and local screening. |
| [native_repeats.csv](attempt001/native_repeats.csv) | All candidate repeats, control bookends and references. |
| [native_phases.csv](attempt001/native_phases.csv) | GLOBAL dense/initial/carried/refresh/held accounting and per-trial normalization. |
| [native_control_noise.csv](attempt001/native_control_noise.csv) | Within/across-engine drift and uncertainty. |
| [native_trajectory_exposure.csv](attempt001/native_trajectory_exposure.csv) | Intervention coverage, outcome groups, N decomposition and tail sensitivity. |
| [operator_screen.csv](attempt001/operator_screen.csv) | All ten initial operator methods, including cap-2/lifetime/tangent ablations. |
| [whole_layer.csv](attempt001/whole_layer.csv) | Full-V signed whole-layer replay, all 16 heads and both Q128 tiles. |
| [reuse_guard.csv](attempt001/reuse_guard.csv) | Raw/directional/norm correction error and harmful tails. |
| [reuse_guard_by_layer.csv](attempt001/reuse_guard_by_layer.csv) | Layer-specific guard evidence. |
| [same_state_decisions.csv](attempt001/same_state_decisions.csv) | Initial and coverage-panel decision metrics, row counts and request intervals. |
| [final_complete.json](attempt001/final_complete.json) | Public exporter source/input/output SHA256 bindings and validation scope. |

The [original synthesis](../v31_value_decision_reuse_20261015/synthesis_20261017/attempt001/summary.md)
has standalone PNG/SVG/PDF uncertainty figures and compact tables. The
[original conclusions](../v31_value_decision_reuse_20261015/conclusions_20261017.md)
explain tail concentration, surrogate mismatch and source/scope limitations.
All original frozen reports and completion markers remain unchanged.

## Reproduce without CUDA or private generations

From the repository root, choose a new output path:

```bash
PYTHONPATH=src:. python -m scripts.v31_value_publication_20261017 \
  --output results/reproduce_public_20261017/attempt001
```

This uses only committed numeric evidence. It checks public scoring-output
hashes, complete cohorts, unequal repeat arithmetic, N/C/T/work sums and zero
capture/LOCAL-skip/budget-error receipts. It cannot independently score private
answers or verify private input/model bytes. See the integration document for
the separate full local verifier and authorized generation/scoring workflow.

## Publication inventory and retained local state

`publication_inventory.json` accounts for public pending artifacts and the
explicitly excluded local files/checkouts. Source hashes, scalar scoring outputs,
numeric per-frame/item audit records, protocols, completion/failed/pause markers
and public reports are committed. Failed/held/partial records are not upgraded
to completed efficacy evidence by publication.

Private prompts/gold/completions/token IDs, Q/K/V snapshots, caches, runtime logs
and credentials stay ignored. Independent reference repositories/worktrees are
recorded by URL/commit and local-state fingerprints; they are not vendored or
modified. Compiler probes, runtime locks and absolute local links into older
control runs are ignored; the inventory records relative targets for those
links, whose public numeric evidence is included at its original location.
These exclusions
preserve the repository's existing privacy and collaboration rules.
