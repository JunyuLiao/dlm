# Trajectory-aware Gaussian-32 threshold calibration (v1)

## Question and scope

Can temporal/query sensitivity preserve a short native DiffusionGemma RULER4K
trajectory at roughly 50% and 70% pooled *physical* 128x64 tile sparsity?
The previous T70 calibration matched 70% overall while skipping 85% in the
first two decoder calls and taking 23.24 calls/canvas. This experiment changes
threshold *selection and scheduling*, not the value-aware routing score,
physical tile decision, decoder, stopping rule, or kernel.

The first two **denoising calls**, not transformer-depth layers, are protected.
This is a predefined breakpoint: T has no informative flip history until the
third call. All methods use the same unweighted early query sensitivity and
the same early local/global thresholds; later C/M/T/T-uniform/T-tile-mean (and
shuffled-T control) use their own causal prior-step statistics.

## Frozen data and controls

- Model, BF16, H100 binary, Gaussian-32 seed 1729, native 256-token canvas,
  reversible acceptance, 48-call cap, 0.8->0.4 temperature, prompts/budgets,
  physical counters, masks, and GQA match `query_adaptive_allocation_v1`.
- Evaluation: its exact 130-prompt final manifest, 13 tasks x 10, seed 42.
  This cohort has already informed research decisions and is exploratory.
- Threshold calibration and policy-validation: separate task-balanced sets of
  two prompts/task each, selected from the same pinned RULER4K source pool and
  audited against every earlier calibration/evaluation/development manifest.
  A third disjoint one-prompt/task set is smoke-only. The final 130 prompts
  never select thresholds or a policy. Record all IDs and hashes.
- Reuse the existing native and matched-kernel dense final shards only after
  exact manifest, seed, model and backend provenance checks. Reuse archived
  unweighted/old-T results solely as named baselines, not new policy outputs.

## Calibration metrics and feasibility

For each full-generation candidate, pool `skipped / eligible` physical tile
counts separately for call 1, call 2, calls >=3, and the complete trajectory,
each for whole/global/local attention. Do not average per-example or per-call
percentages. Record accepted and renoised positions, entropy, top-1 churn,
mean/median/p90/p95 calls, 48-cap rate, accuracy, and total executed tiles.

At target 70%, the predeclared primary constraints on the *calibration* set
are:

- whole/global/local pooled overall sparsity within 2 percentage points of
  70%; first and second call <=75% whole and <=77% global/local;
- mean calls <=8, p90 <=12, and at most one of 26 prompts reaches the cap.

The early ceilings bracket the earlier successful T70 profile (73.0/73.4%
whole and a 75.6% second-call global rate) but are provisional guardrails,
not claims of a universal safe threshold. At 50%, use <=55% whole, <=57%
global/local for the first two calls and require mean <=5.5, p90 <=8, no cap.
For both targets, report per-prompt early-sparsity tails and accuracy; a pooled
constraint is not a guarantee for each prompt.

Use a separate validation set to check the chosen policy before final:
target/stratum error <=3 points, the same early ceilings, no material
accuracy loss relative to matched-kernel dense, and no step-count collapse.
The small validation set cannot certify one-point accuracy differences; all
passes/failures and paired scores must be reported. Final benchmark accuracy
and final sparsity never trigger retuning.

## Search and selection

1. Evaluate a small, frozen single-pair candidate set spanning the earlier
   successful and fresh failed T70 thresholds. Each point uses complete
   native adaptive generation on the calibration prompts. Keep all traces.
   No secant/bisection assumption is made about full-trajectory sparsity,
   because candidate thresholds change the future state and number of calls.
2. Prefer a feasible **single** shared local/global threshold pair, selecting
   the one closest to target whole/global/local sparsity; ties prefer fewer
   calls and fewer executed tiles. Do not trade the early/call guardrails away
   to hit exactly 70%. The same procedure applies at 50%.
3. If no single pair is feasible, use two phase pairs: a common early pair for
   calls 1-2 selected from the safe candidates, then a late pair for calls >=3
   searched on complete trajectories. The early pair is frozen before late
   search. Report the achieved total sparsity and any infeasible target. Never
   hide extra calls in the denominator or force termination.
4. Validate the selected T70 policy on the disjoint validation set. Refine
   only on calibration/validation data with a new recorded policy version.
   Then freeze before one full 130-prompt evaluation and two predeclared
   generation-seed confirmations. Compare with the earlier 6.78-call/89.7%
   result and the fresh 23.24-call/86.2% result; do not claim reproduction if
   the quality or trajectory is materially worse.
5. Only after this T70 checkpoint, calibrate unweighted, C, M, T,
   T-uniform, T-tile-mean and shuffled-T at 50% and 70%. All use the same
   target-specific early policy. Calibrate their late local/global thresholds
   to the T calibration whole/global/local totals, subject to early and
   trajectory guardrails. Run all viable arms on the matched 130 prompts;
   report every miss rather than relabeling it as a target hit.

No parameter is selected on final accuracy. A target with no feasible policy
is labeled unattainable under these guardrails. A separate uninstrumented,
interleaved H100 timing study is required before any speed claim.

## Verification, output, and execution

Use a new versioned result root with frozen source hashes, manifests,
candidate grids, calibration traces, policy selection, validation, raw
per-example/per-step shards, audit, summary CSV/JSON, plots and `report.md`.
Tests cover phase boundary/reset, early unit-weight parity, tile-mean
broadcast, physical count aggregation, calibration feasibility/selection,
manifest disjointness, and resumable identities. Smoke two examples for
diagnostic-on/off output parity, dense parity, correct local/global counters,
and expected call-1/2 thresholds before H100 sweep. Preserve errors and
continue independent candidates. Run one GPU worker and log a lightweight
health check about every 15 minutes.

Interpretation must separate: method-specific query allocation, changed
temporal sparsity, residual global/local mismatches, backend numerics,
prompt-set reuse, trajectory length, and measured end-to-end time.
