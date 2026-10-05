# Protection refinement: completed and audited

The user launched the CUDA study on 2026-09-23. Results are in
`results/protection_refinement_v1/`. The worker completed all 1,820
control/final shards, all timing shards, and the smoke checks. After completion,
the raw-only report was regenerated with explicit executed-work accounting,
global/local sparsity, stopping-failure diagnostics, and an accuracy/sparsity plot.
The audit verifies frozen policy settings, source snapshots, unchanged native
stopping, runtime/diagnostic output parity, and complete shards.

Main result: no evaluated refinement meets the predeclared joint criterion of
60–65% executed physical sparsity, mean <5 denoising calls, and >=89% RULER4K
accuracy. Existing protection at 65% gives 64.59%, 6.49 calls, 90.00% accuracy,
and 0.782 s/example (0.70x native dense speed). Hysteresis lowers reopening
from 0.30% to 0.006% but leaves steps at 6.48. Confidence-only exactly reproduces
the old policy at 60%; stricter margin slightly worsens steps. The reference
dense rescue reaches 4.70 calls at its nominal 60% setting, but only 37.27%
executed sparsity because two full attention passes are performed. It uses more
QK and retained PV work than matched-kernel dense. Unprotected answer-position
prediction churn is the dominant stability failure in long trajectories.

Canonical report: `results/protection_refinement_v1/report.md`.
Machine-readable outputs: `summary.json`, `summary.csv`, `per_sample.csv`,
`stopping_diagnostics.csv`, `comparisons.csv`, `timing.csv`, `audit.json`,
`frozen_policies.json`, and raw shards under `final/` and `controls/`.
Verification: `pytest -q tests/test_protection_refinement.py
tests/test_intervention_policies.py` passed (6 tests).

These 130 prompts have been examined repeatedly. Results are exploratory;
fresh prompts and seeds would be needed for a generalization claim.
