# Optimized V2/V31 multi-benchmark results

Completed requested comparison on one H100. HumanEval and AIME K128 results are preserved; MATH V2 was recalibrated and rerun at0.2; AIME V31K256 is additional; RULER uses the existing104 balanced sampled problems. LongBench-v2 was stopped before generation as requested. The earlier four-workload campaign remains partial.

| Panel | V2 | V31 | Corrected FULL dense | PIECEWISE dense | All-kept FA4 |
|---|---:|---:|---:|---:|---:|
| humaneval | 96.95% (159/164) | 97.56% (160/164) | 98.17% (161/164) | 98.17% (161/164) | 98.17% (161/164) |
| aime26_k128 | 55.56% (50/90) | 44.44% (40/90) | 56.67% (51/90) | 56.67% (51/90) | 54.44% (49/90) |
| aime26_k256 | 55.56% (50/90) | 46.67% (42/90) | 56.67% (51/90) | 56.67% (51/90) | 54.44% (49/90) |
| math500 | 87.60% (438/500) | 86.60% (433/500) | 89.80% (449/500) | 89.80% (449/500) | 90.00% (450/500) |
| ruler128k | 86.346 (79/104 full-credit) | 86.779 (79/104 full-credit) | 86.683 (81/104 full-credit) | 86.683 (81/104 full-credit) | 87.115 (81/104 full-credit) |

RULER reports official fractional-credit score out of100; the other panels use official task accuracy. AIME covers30 questions × seeds42/43/44, MATH all500 seed42, HumanEval all164 seed42. RULER is13 tasks ×8 sampled prompts at seed42 with the original official prefilled/thinking-off format.

| Panel | V2 mean W(s) | V31 mean W(s) | FULL mean W(s) | V2 / FULL | V31 / FULL | V2 / V31 |
|---|---:|---:|---:|---:|---:|---:|
| humaneval | 1.1755 | 1.2589 | 0.9531 | 1.233 | 1.321 | 0.934 |
| aime26_k128 | 6.4140 | 7.0757 | 5.1751 | 1.239 | 1.367 | 0.906 |
| aime26_k256 | 6.4140 | 6.5365 | 5.1751 | 1.239 | 1.263 | 0.981 |
| math500 | 1.5721 | 1.6069 | 1.3322 | 1.180 | 1.206 | 0.978 |
| ruler128k | 6.6321 | 6.5946 | 6.6089 | 1.004 | 0.998 | 1.006 |

Ratios above1 are slower. These are arithmetic mean clean request times; paired geometric ratios and95% clustered intervals are in paired_uncertainty.json. S/N is an amortized generation cost, not a measured single-forward price. Every measured request has a separate untimed warm request; warm-up cost remains in reservation accounting.

| Panel | V2 threshold | V2 GLOBAL sparsity | V2 held | V31 GLOBAL sparsity | V31 held |
|---|---:|---:|---:|---:|---:|
| humaneval | 0.560071620607 | 66.25% | 68.95% | 59.18% | 73.51% |
| aime26_k128 | 0.0594603557501 | 44.94% | 45.58% | 77.35% | 89.40% |
| aime26_k256 | 0.0594603557501 | 44.94% | 45.58% | 73.81% | 86.14% |
| math500 | 0.2 | 47.60% | 49.36% | 59.37% | 73.05% |
| ruler128k | 0.021 | 79.94% | 94.64% | 47.33% | 93.49% |

Physical work uses audited eligible/kept Q128×KV64 GLOBAL tiles. LOCAL remains dense and has a separate denominator. Complete initial/refresh/held/carried, first-canvas and whole-request counts are in physical_work.json. Native denoising trajectories, output length, generation/request time and phase GPU latency are in latency_and_trajectories.json. Refresh/held event times are audit-only; clean timing excludes instrumentation.

| Panel | Method | N | N/C | T | S/N(ms) | S(s) | W(s) | Refresh(us) | Held(us) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| humaneval | v2_online_optimized | 8573 | 9.462 | 209446 | 22.113 | 189.576 | 192.780 | 145.95 | 38.11 |
| humaneval | v31_optimized | 9343 | 10.358 | 206459 | 21.756 | 203.265 | 206.454 | 103.04 | 30.20 |
| humaneval | dense_full_fix51994 | 7885 | 9.255 | 197211 | 19.332 | 152.430 | 156.301 | not instrumented | not instrumented |
| humaneval | dense_piecewise | 7885 | 9.255 | 197211 | 19.543 | 154.100 | 157.278 | not instrumented | not instrumented |
| humaneval | allkept_fa4 | 8036 | 9.226 | 202719 | 21.038 | 169.058 | 172.239 | not instrumented | not instrumented |
| aime26_k128 | v2_online_optimized | 27570 | 13.541 | 514344 | 20.876 | 575.538 | 577.259 | 277.09 | 100.61 |
| aime26_k128 | v31_optimized | 30980 | 14.436 | 543905 | 20.500 | 635.089 | 636.813 | 216.42 | 35.83 |
| aime26_k128 | dense_full_fix51994 | 25360 | 13.497 | 474280 | 18.283 | 463.650 | 465.758 | not instrumented | not instrumented |
| aime26_k128 | dense_piecewise | 25360 | 13.497 | 474280 | 18.527 | 469.853 | 471.542 | not instrumented | not instrumented |
| aime26_k128 | allkept_fa4 | 26547 | 13.254 | 505343 | 19.968 | 530.086 | 531.783 | not instrumented | not instrumented |
| aime26_k256 | v2_online_optimized | 27570 | 13.541 | 514344 | 20.876 | 575.538 | 577.259 | 277.09 | 100.61 |
| aime26_k256 | v31_optimized | 28480 | 13.673 | 527820 | 20.595 | 586.557 | 588.286 | 209.05 | 39.59 |
| aime26_k256 | dense_full_fix51994 | 25360 | 13.497 | 474280 | 18.283 | 463.650 | 465.758 | not instrumented | not instrumented |
| aime26_k256 | dense_piecewise | 25360 | 13.497 | 474280 | 18.527 | 469.853 | 471.542 | not instrumented | not instrumented |
| aime26_k256 | allkept_fa4 | 26547 | 13.254 | 505343 | 19.968 | 530.086 | 531.783 | not instrumented | not instrumented |
| math500 | v2_online_optimized | 36604 | 10.434 | 837132 | 21.232 | 777.172 | 786.054 | 138.29 | 52.85 |
| math500 | v31_optimized | 37883 | 11.058 | 818376 | 20.975 | 794.586 | 803.473 | 104.58 | 28.92 |
| math500 | dense_full_fix51994 | 35372 | 10.241 | 825790 | 18.508 | 654.648 | 666.103 | not instrumented | not instrumented |
| math500 | dense_piecewise | 35372 | 10.241 | 825790 | 18.749 | 663.194 | 672.013 | not instrumented | not instrumented |
| math500 | allkept_fa4 | 35163 | 10.327 | 812547 | 20.217 | 710.887 | 719.721 | not instrumented | not instrumented |
| ruler128k | v2_online_optimized | 645 | 6.202 | 2647 | 53.838 | 34.726 | 689.736 | 6293.07 | 323.97 |
| ruler128k | v31_optimized | 577 | 5.548 | 2721 | 53.002 | 30.582 | 685.843 | 3530.79 | 222.08 |
| ruler128k | dense_full_fix51994 | 611 | 5.875 | 2670 | 54.441 | 33.263 | 687.327 | not instrumented | not instrumented |
| ruler128k | dense_piecewise | 611 | 5.875 | 2670 | 52.975 | 32.368 | 686.566 | not instrumented | not instrumented |
| ruler128k | allkept_fa4 | 615 | 5.913 | 2732 | 52.564 | 32.327 | 687.083 | not instrumented | not instrumented |

The phase GPU figures are per GLOBAL attention call and include the integrated refresh dispatch, projection/packing and selection work inside its event boundary. They are not clean end-to-end or isolated PV measurements. Native dense per-attention event timing was not instrumented; no native single-step kernel speedup is inferred from these rows.

| Panel | V2 − V31 accuracy points | Paired95% interval | V2 − FULL accuracy points | Paired95% interval |
|---|---:|---:|---:|---:|
| humaneval | -0.61 | [-3.05, +1.83] | -1.22 | [-3.66, +1.22] |
| aime26_k128 | +11.11 | [+4.44, +18.89] | -1.11 | [-8.89, +6.67] |
| aime26_k256 | +8.89 | [+1.11, +17.78] | -1.11 | [-8.89, +6.67] |
| math500 | +1.00 | [-1.20, +3.20] | -2.20 | [-4.40, +0.00] |
| ruler128k | -0.43 | [-3.56, +2.45] | -0.34 | [-3.08, +2.50] |

Intervals cluster AIME seeds by question and RULER by task;10,000 paired bootstrap draws. Nonsignificance does not establish noninferiority. Calibration is computation-only and transductive: MATH uses32 examined work strata, AIME four examined prompts, RULER one prompt per task; HumanEval reuses its saved calibrated threshold. No new held-out claim.

The old MATH V2tau3.2 result is supplementary:392/500,77.58% GLOBAL and79.43% held sparsity. Its four short calibration requests were unrepresentative. The rerun at0.2 reaches the requested40–50% on the complete500 panel.

RULER matches approximate held kept fraction, not total FLOPs. The preserved first-canvas absolute-work matching attempt failed: V31 refresh is dense, V2 refresh is sparse, and trajectories differ. Reported absolute/phase work must accompany sparsity; work_mismatch.json records these ratios for every panel. Both controls and negative calibration evidence remain available.

Verification: complete paired coverage, audit/clean output-hash and denoising-call agreement, explicit zero clean capture/JIT counters, zero audited LOCAL skips, sparse-selector budget checks and no extra sparse model forwards all pass. Kernel/source/runtime/model identity and receipt field coverage are in verification.json and source_identity.json. Native dense has no sparse adapter receipt; absent fields are not reported as measured zero.

V2 remains genuinely online with current QK/softmax/Gaussian32 projected PV followed by conditional full PV. QK/routing repeats across four output groups in the selected kernel. Best preserved8K complete synthetic refresh427.90us remains slower than plain dense363.58us and native paged dense287.47us. No refresh-parity claim follows from sparsity. Yuhan authored the inherited cross-step reuse baseline; Junyu Gaussian32/value-direction routing and the optimized execution variants remain separately attributed.

All original partial attempts, raw generations, failed designs and calibration evidence remain immutable. The prior full campaign has no completion marker; this report covers only the explicitly completed requested panels.
