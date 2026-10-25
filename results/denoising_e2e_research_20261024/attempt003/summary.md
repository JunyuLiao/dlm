# Complete native end-to-end selector screen

User-requested cohort:10 AIME26,10 HumanEval,5 original native-eligible long-band LongBench-v2 questions, each at seeds42/43. Nine arms:450 scored item/seed/arm cells. Full original prompts and output limits are unchanged; nine original over-limit LongBench IDs remain blocked. This is an exposed descriptive screen, not a fresh benchmark or noninferiority test.

Protocol SHA256: `88e46d4088a094968ebbeaed05b0ce4b9cf6e824a643675eba257ac9920d35dd`. Generation checkpoint:4ac2b53f. See e2e_protocol.json, manifest_identity.json, launch receipts and closeout001/verification.json. No paused campaign resumed; no other workload preempted.

Methods are training-free selector variants using current Q/K/V on retained tiles. New variants affect only layer29; V31 retains its inherited five-GLOBAL-layer scope and K128/K8192 budgets. V31 is a practical reference with different physical work, not a matched-work attribution control. Current-error, response, margin, certified-reuse and sham share the new state-specific candidate construction/guards; their later native trajectories can change future feasibility and work.

## aime26

| Arm | Correct/cells | Accuracy | N | C | N/C | T | Mean clean W(s) | GLOBAL consumer sparsity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_piecewise | 12/20 | 60.0% | 5426 | 425 | 12.767 | 107609 | 5.0533 | 0.00% |
| allkept_layer29 | 11/20 | 55.0% | 5381 | 414 | 12.998 | 104751 | 5.6511 | 0.00% |
| current_error_fixed4 | 9/20 | 45.0% | 5930 | 452 | 13.119 | 114413 | 20.5766 | 1.57% |
| response_guard_qdirection | 15/20 | 75.0% | 4778 | 385 | 12.410 | 96709 | 16.6819 | 1.54% |
| margin_guard_weighted_bias | 10/20 | 50.0% | 5635 | 431 | 13.074 | 108831 | 19.7439 | 1.57% |
| validity_reuse_mass_certificate | 12/20 | 60.0% | 5556 | 442 | 12.570 | 111785 | 55.0437 | 1.56% |
| sham_fixed4 | 10/20 | 50.0% | 5037 | 397 | 12.688 | 99830 | 17.4589 | 1.55% |
| v31_optimized | 6/20 | 30.0% | 7031 | 492 | 14.291 | 125180 | 7.2421 | 77.03% |
| dense_full_fix51994 | 12/20 | 60.0% | 5426 | 425 | 12.767 | 107609 | 4.9912 | 0.00% |

| Candidate vs corrected FULL | Accuracy difference pp [paired95% CI] | N/C geometric ratio [95% CI] | W geometric ratio [95% CI] | Correct→wrong / wrong→correct cells |
| --- | --- | --- | --- | --- |
| dense_piecewise | +0.0 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.013 [1.012, 1.013] | 0 / 0 |
| allkept_layer29 | -5.0 [-35.000, 25.000] | 0.989 [0.928, 1.056] | 1.078 [0.936, 1.235] | 3 / 2 |
| current_error_fixed4 | -15.0 [-40.000, 5.000] | 1.028 [0.941, 1.118] | 3.887 [3.129, 4.826] | 4 / 1 |
| response_guard_qdirection | +15.0 [0.000, 35.000] | 0.970 [0.923, 1.009] | 3.152 [2.779, 3.537] | 0 / 3 |
| margin_guard_weighted_bias | -10.0 [-25.000, 0.000] | 1.008 [0.922, 1.114] | 3.703 [2.951, 4.537] | 2 / 0 |
| validity_reuse_mass_certificate | +0.0 [0.000, 0.000] | 0.998 [0.949, 1.050] | 10.744 [9.761, 11.615] | 0 / 0 |
| sham_fixed4 | -10.0 [-30.000, 10.000] | 0.972 [0.901, 1.051] | 3.210 [2.709, 3.744] | 3 / 1 |
| v31_optimized | -30.0 [-55.000, -10.000] | 1.143 [1.054, 1.239] | 1.467 [1.169, 1.888] | 6 / 0 |

Intervals use10000 paired question-cluster bootstrap draws retaining both seeds. The two seeds are not two independent questions. Small-cluster intervals are descriptive; nonsignificance does not establish accuracy preservation.

## humaneval

| Arm | Correct/cells | Accuracy | N | C | N/C | T | Mean clean W(s) | GLOBAL consumer sparsity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_piecewise | 20/20 | 100.0% | 741 | 89 | 8.326 | 19335 | 0.7502 | 0.00% |
| allkept_layer29 | 19/20 | 95.0% | 722 | 91 | 7.934 | 19902 | 0.8208 | 0.00% |
| current_error_fixed4 | 19/20 | 95.0% | 719 | 91 | 7.901 | 20327 | 2.1283 | 1.07% |
| response_guard_qdirection | 19/20 | 95.0% | 696 | 87 | 8.000 | 19688 | 2.0702 | 1.08% |
| margin_guard_weighted_bias | 18/20 | 90.0% | 726 | 91 | 7.978 | 20465 | 2.2480 | 1.08% |
| validity_reuse_mass_certificate | 19/20 | 95.0% | 754 | 92 | 8.196 | 21203 | 5.5481 | 1.09% |
| sham_fixed4 | 19/20 | 95.0% | 708 | 85 | 8.329 | 19880 | 2.0834 | 1.08% |
| v31_optimized | 19/20 | 95.0% | 842 | 91 | 9.253 | 20626 | 0.9486 | 45.47% |
| dense_full_fix51994 | 20/20 | 100.0% | 741 | 89 | 8.326 | 19335 | 0.7388 | 0.00% |

| Candidate vs corrected FULL | Accuracy difference pp [paired95% CI] | N/C geometric ratio [95% CI] | W geometric ratio [95% CI] | Correct→wrong / wrong→correct cells |
| --- | --- | --- | --- | --- |
| dense_piecewise | +0.0 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.015 [1.012, 1.020] | 0 / 0 |
| allkept_layer29 | -5.0 [-15.000, 0.000] | 0.957 [0.826, 1.091] | 1.074 [0.995, 1.144] | 1 / 0 |
| current_error_fixed4 | -5.0 [-15.000, 0.000] | 0.961 [0.853, 1.093] | 2.439 [1.936, 2.972] | 1 / 0 |
| response_guard_qdirection | -5.0 [-15.000, 0.000] | 0.978 [0.837, 1.136] | 2.379 [1.893, 2.890] | 1 / 0 |
| margin_guard_weighted_bias | -10.0 [-30.000, 0.000] | 0.969 [0.859, 1.104] | 2.572 [2.053, 3.115] | 2 / 0 |
| validity_reuse_mass_certificate | -5.0 [-15.000, 0.000] | 0.981 [0.844, 1.136] | 5.503 [3.825, 7.571] | 1 / 0 |
| sham_fixed4 | -5.0 [-15.000, 0.000] | 1.026 [0.902, 1.178] | 2.401 [1.945, 2.879] | 1 / 0 |
| v31_optimized | -5.0 [-15.000, 0.000] | 1.057 [0.904, 1.206] | 1.177 [1.036, 1.316] | 1 / 0 |

Intervals use10000 paired question-cluster bootstrap draws retaining both seeds. The two seeds are not two independent questions. Small-cluster intervals are descriptive; nonsignificance does not establish accuracy preservation.

## longbench_v2

| Arm | Correct/cells | Accuracy | N | C | N/C | T | Mean clean W(s) | GLOBAL consumer sparsity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_piecewise | 2/10 | 20.0% | 12128 | 298 | 40.698 | 75608 | 63.9045 | 0.00% |
| allkept_layer29 | 2/10 | 20.0% | 13512 | 358 | 37.743 | 90996 | 77.0436 | 0.00% |
| current_error_fixed4 | 3/10 | 30.0% | 13109 | 336 | 39.015 | 85450 | 235.3553 | 1.94% |
| response_guard_qdirection | 2/10 | 20.0% | 16031 | 404 | 39.681 | 102904 | 287.1000 | 1.94% |
| margin_guard_weighted_bias | 3/10 | 30.0% | 12410 | 324 | 38.302 | 81955 | 220.6073 | 1.94% |
| validity_reuse_mass_certificate | 2/10 | 20.0% | 10082 | 253 | 39.850 | 63875 | 536.7774 | 1.94% |
| sham_fixed4 | 4/10 | 40.0% | 11781 | 298 | 39.534 | 75383 | 216.7562 | 1.94% |
| v31_optimized | 4/10 | 40.0% | 6833 | 196 | 34.862 | 49114 | 28.8568 | 90.83% |
| dense_full_fix51994 | 2/10 | 20.0% | 12128 | 298 | 40.698 | 75608 | 63.7645 | 0.00% |

| Candidate vs corrected FULL | Accuracy difference pp [paired95% CI] | N/C geometric ratio [95% CI] | W geometric ratio [95% CI] | Correct→wrong / wrong→correct cells |
| --- | --- | --- | --- | --- |
| dense_piecewise | +0.0 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.001 [0.999, 1.003] | 0 / 0 |
| allkept_layer29 | +0.0 [0.000, 0.000] | 1.067 [0.944, 1.189] | 1.331 [0.874, 1.992] | 0 / 0 |
| current_error_fixed4 | +10.0 [0.000, 30.000] | 1.111 [1.009, 1.223] | 3.286 [2.243, 5.363] | 0 / 1 |
| response_guard_qdirection | +0.0 [0.000, 0.000] | 1.138 [0.973, 1.330] | 4.216 [2.434, 7.305] | 0 / 0 |
| margin_guard_weighted_bias | +10.0 [0.000, 30.000] | 1.043 [0.970, 1.150] | 3.002 [1.880, 5.269] | 0 / 1 |
| validity_reuse_mass_certificate | +0.0 [0.000, 0.000] | 1.027 [0.887, 1.204] | 5.416 [2.493, 13.862] | 0 / 0 |
| sham_fixed4 | +20.0 [0.000, 40.000] | 1.017 [0.926, 1.145] | 2.761 [1.898, 3.949] | 0 / 2 |
| v31_optimized | +20.0 [0.000, 60.000] | 0.884 [0.770, 0.974] | 0.558 [0.301, 0.832] | 0 / 2 |

Intervals use10000 paired question-cluster bootstrap draws retaining both seeds. The two seeds are not two independent questions. Small-cluster intervals are descriptive; nonsignificance does not establish accuracy preservation.

## Implementation, controls and diagnostic interpretation

ResponseGuard minimizes current sparse attention query-direction response J_Su, with residual (J_S−J_A)u recorded separately. K/V are current but their future motion is not predicted. MarginGuard is inverse-prior-margin-weighted attention bias, not the proposed exact downstream influence oracle. Certified reuse bounds retained mass under query drift with fixed within-canvas prefix keys and recomputed current-canvas partition; reset at every canvas. See THEORY_AND_LIMITS.md and code/selector.py for assumptions and independent algebra checks.

This screen extends the prior non-identifying four-unit experiment. It does not retroactively pass that gate or separate pulse propagation from repeated intervention. The methods repeatedly select/hold support, so full-request differences establish effects of those schedules, not an isolated contraction mechanism. Native/all-kept numerical consumer differences remain visible.

First-refresh support readout: 200 cell/score comparisons; 200 have identical candidate current-error/feasibility banks at the common native-bootstrap boundary; 1164 distinct routing-unit choices. These are candidate-diversity diagnostics, not independent benchmark samples or predictive confirmation.

Per-request native flags/entropy and actual native commit-based canvas counts with an explicitly retained uncommitted tail (CPU emission intervals can lag native commits) are in audit records. trajectory_analysis001 separates confidence-only, stability-only, both and history-readiness bottlenecks, first/terminal canvases and48-call caps. paired_trajectory001 includes paired N,C,N/C,T,S/N and request-time intervals and both-seed item summaries. There is no exact compiled-acceptance count claim.

Consumer rectangles exclude selector discovery. Each audit selector receipt separately reports discovery QK/PV rectangle counts and CUDA-event time for support management. selector_discovery events are nested inside complete_support_management_refresh; do not add those overlapping event totals. Request W includes all necessary selector work. Skipped consumer rectangles or fewer calls alone are not speedup. LOCAL stays dense; rectangle counts are not CUDA CTA counts.

A complete first warm request is followed by per-cell audit/clean full requests. The audit serves as a shape-specific warm request for clean timing. All audit/clean output hashes,N,C,T must match. Any new clean capture/JIT makes timing unqualified; audit instrument time is never the clean latency.

The prior restoration qualification remains a preserved failure: HE124 after4 has identical85-call full native model/logit/RNG/count fingerprints, but one restored output list differs by256 tokens. End-to-end evaluation uses fresh from-prompt requests and does not depend on restored outputs.

Original10 AIME and10 HumanEval prompts have prior dense exposure; some also have sparse diagnostic exposure. Four additional LongBench items were chosen without outcome filtering from native-context-eligible long-band raw prompts; prior historical exposure is not certified absent. The resulting five long questions cover only two domains. Context eligibility prevents a domain-balanced long-band subset under the unchanged native limit; no truncation/replacement of the nine blocked original items occurred.

## Resources and verification

Expanded panel allocated-worker time: 49117.370s; counted model forwards including startup/profile: 405458. Campaign GPU/forward ceilings were removed by the user. Per-process4-hour hang watchdog remains. This allocation is not active GPU utilization or clean request latency.

All 19 closeout checks pass; 0 clean timings are unqualified.32 focused CPU tests passed before generation. All16 model/config files were rehashed and match. Source/runtime/model/scorer identities and public privacy fields are verified. Private inputs/generations/logs stay ignored. Individual worker launch/exit and complete markers preserve failures and runtime evidence.

Earlier qualification/smaller preflight resources are separate, preserved under attempts001/002, and are not folded into clean request time or silently called panel coverage. The expanded panel never reuses the smaller preflight scores. No push is requested.

## Decisions

No production selector, novelty, formal accuracy noninferiority or full denoising-contraction claim is promoted by this exposed small screen. Exact paired task regressions, calls per canvas, output changes and request costs above determine whether any variant merits another bounded confirmation. A lower-call result with worse task outcomes fails the objective; a slow selector cannot claim speedup from fewer consumer tiles. Norm-only ValidityReuse remains a separate historical negative; this current-canvas-aware certificate is a named exploratory variant, not erasure of that result.

Yuhan authored the inherited cross-step reuse machinery. Gaussian32/value-direction routing, C-gate/query sensitivity, V31/V2, and the earlier negative studies retain their own provenance.

Inspect first: this summary; final_complete.json; analysis001/summary.json; paired_trajectory001/summary.json; scoring001/official_scores.json; support_analysis001/summary.json; trajectory_analysis001/summary.json; closeout001/verification.json; frozen e2e_protocol.json and code/.
