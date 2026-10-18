The bounded factorial diagnostic is complete. Its implementation and recorded-prefix checks pass, but clean causal promotion remains unresolved under the frozen A/A rule. Current-output effects are heterogeneous; preserving future support with sparse current output is the stronger provisional mechanism signal.

This supplement qualifies the frozen summary: its prefix gate means matched measured prefix/first-GLOBAL fingerprints, not a cloned full-engine checkpoint. It does not override the A/A/no-op comparability condition.

**longbench_v2**: 32 paired groups, 8 questions, seeds [1, 42], engines [9, 11]. All-six recorded prefixes match in 32 groups; all-four exposed/complete scope in 32.

A/A absolute N difference: mean 0.40625, p95 1.8, max 9; 2 output-hash disagreements and 0 score disagreements. Mean absolute N by engine: 9: 0, 11: 0.8125. These observed excursions are descriptive, not bounds on unknown nondeterminism.

| Contrast (positive means more calls after removal) | Mean ΔN | Explicit reference N | Relative ΔN | Question-cluster 95% interval | LOO ΔN range |
|---|---:|---:|---:|---|---|
| current_sparse_minus_dense | 7.4531 | 134.5938 | +5.54% | [-15.312, 29.500] | [0.661, 14.875] |
| future_hold_minus_fresh | 16.7344 | 129.9531 | +12.88% | [-2.859, 36.969] | [10.214, 22.625] |
| current_sparse_minus_dense_given_fresh | 7.1562 | 126.3750 | +5.66% | [-17.000, 36.034] | [-4.857, 14.964] |
| current_sparse_minus_dense_given_hold | 7.7500 | 142.8125 | +5.43% | [-28.938, 48.375] | [-8.214, 21.286] |
| future_hold_minus_fresh_given_dense | 16.4375 | 126.3750 | +13.01% | [-26.067, 59.500] | [2.000, 31.500] |
| future_hold_minus_fresh_given_sparse | 17.0312 | 133.5312 | +12.75% | [6.031, 29.126] | [12.893, 20.179] |

The main current-output reference averages DF and DH; the main future-support reference averages DF and SF. Conditional references are the right-hand baseline named in the JSON. The intervals are descriptive and do not establish noninferiority.

| Repeated score change | Adverse question/seed cases (distinct questions) | Beneficial question/seed cases (distinct questions) |
|---|---:|---:|
| current_given_fresh | 1 (1) | 3 (2) |
| current_given_hold | 2 (2) | 1 (1) |
| future_given_dense | 1 (1) | 2 (2) |
| future_given_sparse | 4 (3) | 2 (2) |

A case counts once only when its score-change direction repeats in both engines. Engine repeats are technical repetitions, not independent accuracy observations; RULER changes can be partial credit. All case identities, per-seed/per-engine effects, repeat mismatches, and signed tail contributions remain in the JSON.

**ruler128k**: 16 paired groups, 4 questions, seeds [1, 42], engines [10, 12]. All-six recorded prefixes match in 16 groups; all-four exposed/complete scope in 16.

A/A absolute N difference: mean 0, p95 0, max 0; 0 output-hash disagreements and 0 score disagreements. Mean absolute N by engine: 10: 0, 12: 0. These observed excursions are descriptive, not bounds on unknown nondeterminism.

| Contrast (positive means more calls after removal) | Mean ΔN | Explicit reference N | Relative ΔN | Question-cluster 95% interval | LOO ΔN range |
|---|---:|---:|---:|---|---|
| current_sparse_minus_dense | 0.3750 | 5.6250 | +6.67% | [-0.188, 1.312] | [-0.083, 0.583] |
| future_hold_minus_fresh | 0.2500 | 5.6875 | +4.40% | [-0.000, 0.562] | [0.083, 0.333] |
| current_sparse_minus_dense_given_fresh | 0.3750 | 5.5000 | +6.82% | [0.000, 0.750] | [0.167, 0.500] |
| current_sparse_minus_dense_given_hold | 0.3750 | 5.7500 | +6.52% | [-0.750, 1.875] | [-0.333, 0.833] |
| future_hold_minus_fresh_given_dense | 0.2500 | 5.5000 | +4.55% | [-0.375, 1.125] | [-0.167, 0.500] |
| future_hold_minus_fresh_given_sparse | 0.2500 | 5.8750 | +4.26% | [-0.000, 0.750] | [-0.000, 0.333] |

The main current-output reference averages DF and DH; the main future-support reference averages DF and SF. Conditional references are the right-hand baseline named in the JSON. The intervals are descriptive and do not establish noninferiority.

| Repeated score change | Adverse question/seed cases (distinct questions) | Beneficial question/seed cases (distinct questions) |
|---|---:|---:|
| current_given_fresh | 0 (0) | 0 (0) |
| current_given_hold | 0 (0) | 1 (1) |
| future_given_dense | 0 (0) | 0 (0) |
| future_given_sparse | 0 (0) | 1 (1) |

A case counts once only when its score-change direction repeats in both engines. Engine repeats are technical repetitions, not independent accuracy observations; RULER changes can be partial credit. All case identities, per-seed/per-engine effects, repeat mismatches, and signed tail contributions remain in the JSON.

**Post-hoc score/call decomposition: longbench_v2**

| Contrast | Score change | Engine groups (question/seed cases) | Mean ΔN |
|---|---|---:|---:|
| current_given_fresh | improved | 6 (3) | +63.833 |
| current_given_fresh | unchanged | 24 (12) | -10.000 |
| current_given_fresh | worsened | 2 (1) | +43.000 |
| future_given_dense | improved | 4 (2) | -14.000 |
| future_given_dense | unchanged | 26 (13) | +32.231 |
| future_given_dense | worsened | 2 (1) | -128.000 |
| future_given_sparse | improved | 4 (2) | +12.000 |
| future_given_sparse | unchanged | 20 (10) | +14.900 |
| future_given_sparse | worsened | 8 (4) | +24.875 |

These are descriptive outcome strata, not filtered causal estimates. Technical repeats are not independent answer observations. Every group remains in the headline comparisons.

**Post-hoc score/call decomposition: ruler128k**

| Contrast | Score change | Engine groups (question/seed cases) | Mean ΔN |
|---|---|---:|---:|
| current_given_fresh | improved | 0 (0) | n/a |
| current_given_fresh | unchanged | 16 (8) | +0.375 |
| current_given_fresh | worsened | 0 (0) | n/a |
| future_given_dense | improved | 0 (0) | n/a |
| future_given_dense | unchanged | 16 (8) | +0.250 |
| future_given_dense | worsened | 0 (0) | n/a |
| future_given_sparse | improved | 2 (1) | +1.000 |
| future_given_sparse | unchanged | 14 (7) | +0.143 |
| future_given_sparse | worsened | 0 (0) | n/a |

These are descriptive outcome strata, not filtered causal estimates. Technical repeats are not independent answer observations. Every group remains in the headline comparisons.

For LongBench, SF−DF has opposite seed signs and changes sign when Q77 is omitted; SH−DH also has a leave-one-question-out sign reversal. Pooled current-output effects exceed average A/A variation but are smaller than the largest observed A/A excursion. Same-arm sparse-fresh engine-repeat drift adds a33-call difference with unchanged score. These facts prevent a clean output-dominant conclusion.

Holding future support has a more stable N penalty, especially with sparse current output, and repeats adverse as well as beneficial score changes. Nevertheless, the dense-current future effect is nearly zero in one seed, substantial tails remain, and A/A drift qualifies the overall causal claim. The defensible decision is provisional mechanism triage, not a gate pass or a new method claim.

All factorial timings include diagnostic work. No >=10% incremental request-latency benefit is established; profile-to-clean transfer remains unresolved in the separate headroom review. This single-event study does not test a cheap selector, replacement of every refresh, or initialization/commit removal.

More denoising calls are not automatically harmful at different quality. For example, Q77 sparse-fresh extra calls accompany beneficial answer changes. The post-hoc score-change/call-change decomposition retains every row in headline effects and must not be treated as a filtered causal estimate.

Full self-conditioning was recorded only at the first post-event state. Equality of later compact entries does not prove equality of full hidden states or locate the earliest divergence.
