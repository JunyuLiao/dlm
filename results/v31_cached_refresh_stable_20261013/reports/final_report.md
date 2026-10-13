# Stable cheap refresh: final development-panel report

**Decision: retain V31. No tested candidate meets the frozen accuracy and trajectory criteria; do not expand to 503 cells or start clean timing.**

All 13 arms completed 154 cells (64 LongBench-v2 seed-1 problems and 30 AIME26 problems at seeds 42/43/44), for 2,002 native cells and 52 validated shards. These are previously examined development data. Source, model, manifest, seed, sampler binding, GLOBAL/LOCAL accounting and actual CUDA capture checks passed. Native generation source is unchanged throughout the panel. Official scoring and analysis reproduced exactly after two preserved reporting retries.

## 1. Operator diagnostics

620 real Q/K/V snapshots across both workloads and all five GLOBAL layers; all 670 files, including immutable prefix K/V, were fingerprinted. Each proposed map is evaluated at the next call on identical current Q/K/V. Dense QK/PV work occurs only offline. Maximum fresh retained log-mass discrepancy from dense FA4 observation: 1.907e-6.

| Selector | Mean true retained mass | Row error mean / p95 / max | Next-call change p95 / max | Replacements | Immediate reversals |
|---|---:|---:|---:|---:|---:|
| v31_progress_hold | 0.754724 | 2.899 / 8.301 / 19.818 | 0.00000 / 0.000 | 0 | 0 |
| raw_s2 | 0.757182 | 2.844 / 8.108 / 19.312 | 0.36680 / 15.981 | 166,215 | 74,778 |
| stable_l0_g2_s2 | 0.755321 | 2.887 / 8.260 / 19.823 | 0.04699 / 14.384 | 6,073 | 244 |

The frozen choice is hysteresis **0**, persistence 2, event cap 2, cumulative cap 8, removal tolerance 0.025, and no probes. Zero hysteresis means the tested stability comes from persistence, caps and the removal guard. Relative to raw s2, the selected stable setting reduces replacements by 96.35% and p95 immediate output change by 87.19%, but large outliers remain. It sacrifices some of raw re-ranking's dense-output improvement. The row guard protects removed mass; it does not certify the effect of unobserved incoming values or the denoising trajectory.

Probe4 remains offline. At its matched exploratory configuration (log(1.25), guard 0.01), false-admission count remains 1 and p95 perturbation does not improve, so the predeclared probe gate fails. Earlier exploratory frequency traces remain preserved; attempt004 uses the frozen production parameters at every frequency.

## 2. Official native accuracy

Accuracy counts and mean denoising calls are separated by workload. N changes are relative to the fresh progress V31 control; they are not latency claims.

| Arm | LongBench correct/64 | AIME correct/90 | LongBench mean N | AIME mean N | N change LB / AIME |
|---|---:|---:|---:|---:|---:|
| dense_full_fix51994 | 25 | 51 | 338.12 | 275.17 | +40.73% / -0.12% |
| dense_piecewise | 25 | 51 | 338.12 | 275.17 | +40.73% / -0.12% |
| allkept_fa4 | 29 | 51 | 323.56 | 278.57 | +34.67% / +1.11% |
| v31_progress_dense_control | 34 | 49 | 240.27 | 275.50 | +0.00% / +0.00% |
| raw_global_progress_dense_s2 | 33 | 50 | 238.89 | 272.01 | -0.57% / -1.27% |
| stable_global_progress_dense_s2 | 31 | 49 | 271.86 | 274.34 | +13.15% / -0.42% |
| v31_no_progress_dense_hold | 33 | 55 | 262.69 | 283.29 | +9.33% / +2.83% |
| raw_global_no_progress_dense_s1 | 28 | 55 | 304.38 | 283.30 | +26.68% / +2.83% |
| raw_global_no_progress_dense_s4 | 32 | 55 | 249.17 | 283.28 | +3.71% / +2.82% |
| raw_global_no_progress_dense_s8 | 27 | 55 | 324.14 | 283.14 | +34.91% / +2.77% |
| stable_global_no_progress_dense_s1 | 32 | 55 | 218.64 | 283.70 | -9.00% / +2.98% |
| stable_global_no_progress_dense_s4 | 32 | 55 | 273.38 | 282.70 | +13.78% / +2.61% |
| stable_global_no_progress_dense_s8 | 35 | 55 | 293.83 | 283.87 | +22.29% / +3.04% |

The highest candidate accuracy is stable no-progress s8: 35/64 and 55/90 versus V31's 34/64 and 49/90. The LongBench improvement is only one net cell (4 wins, 3 losses). Its paired accuracy difference has a 95% problem-clustered interval of -6.25 to +9.375 percentage points. AIME's +6.667 points (11 wins, 5 losses) has a problem-clustered interval of -2.222 to +17.778 points. These intervals do not establish noninferiority or a confirmed improvement.

All no-progress cheap arms have exactly the same AIME correctness vector as the no-cheap hold baseline. GLOBAL physical sparsity on AIME is below 0.004% at K=128, so this workload mostly exercises all-kept prefixes. The AIME gain is already present without cheap mask updates and therefore does not demonstrate a benefit from cheap repair. Full and PIECEWISE dense controls have identical output hashes on all 154 cells; all-kept FA4 remains a distinct consumer control.

## 3. Denoising trajectory and promotion gate

Frozen limits: aggregate N ratio <=1.02 on each workload, mean N/C ratio <=1.02, p95 N ratio <=1.05, cap-hit delta <=2 percentage points, plus promising paired accuracy. No arm passes the complete trajectory gate.

| Selected arm / workload | Mean N/C | p95 N | Mean output tokens | Capped / unfinished |
|---|---:|---:|---:|---:|
| v31_progress_dense_control / LB | 17.75 | 490.25 | 2939.4 | 2 / 2 |
| v31_progress_dense_control / AIME | 12.10 | 552.15 | 5276.6 | 31 / 30 |
| raw_global_progress_dense_s2 / LB | 18.22 | 544.90 | 2708.6 | 1 / 2 |
| raw_global_progress_dense_s2 / AIME | 12.02 | 552.15 | 5248.5 | 29 / 28 |
| stable_global_progress_dense_s2 / LB | 18.69 | 612.20 | 2917.3 | 2 / 3 |
| stable_global_progress_dense_s2 / AIME | 12.06 | 552.15 | 5277.3 | 31 / 30 |
| stable_global_no_progress_dense_s1 / LB | 18.40 | 456.80 | 2530.4 | 1 / 3 |
| stable_global_no_progress_dense_s1 / AIME | 12.29 | 554.60 | 5326.8 | 32 / 27 |
| stable_global_no_progress_dense_s8 / LB | 19.26 | 637.00 | 3118.4 | 2 / 3 |
| stable_global_no_progress_dense_s8 / AIME | 12.28 | 554.60 | 5337.8 | 32 / 27 |

Stable progress s2 lowers LongBench mask turnover substantially, but scores 31/64 and raises mean N by 13.15%; mean output length is approximately unchanged. High consecutive-mask IoU (0.986 versus raw's 0.629) does not protect quality or the trajectory. Stable no-progress s8 raises LongBench N by 22.29%, mean N/C by 8.51%, and p95 N by 29.93%; AIME N rises 3.04%. Its N-ratio intervals are broad (LB 0.775-1.940; AIME 0.981-1.083), so the rejection is failure of the predeclared development screen, not proof of a precise population effect.

Stable no-progress s1 reduces LongBench mean N by 9.00%, but loses two correct cells, raises mean N/C by 3.67%, and shortens mean output by 13.91%. It is not evidence of a cheaper forward. Raw progress s2 stays near mean N but loses one LongBench cell and misses the N/C and upper-tail limits. Disabling progress refresh removes that dense work, while initial/per-canvas discovery, carry, prefill and commits remain unchanged. It does not mean one dense forward per problem.

## 4. Selector work and overhead

No production selector PV or outside-probe work ran. Retained and mandatory-tail QK is recomputed in a bounded duplicate pass, per query head/Q128 block, without a KV-head union. Ranking stays on GPU. The audited implementation includes a host read in tile-ID extraction and a map-equality decision; audit overhead includes dispatch/synchronization and map rebuilding. These are not clean kernel or request latencies.

| Arm, LongBench | Observations / mask-update events | Selector QK rectangles | Audit ms/observation | Max final cache/workspace MiB | GLOBAL kept / eligible |
|---|---:|---:|---:|---:|---:|
| v31_progress_dense_control | 0 / 0 | 0 | 0.000 | 171.69 | 659,541,920 / 3,865,609,920 |
| raw_global_progress_dense_s2 | 31,260 / 31,260 | 133,042,560 | 12.097 | 348.25 | 612,088,160 / 3,804,773,760 |
| stable_global_progress_dense_s2 | 36,050 / 20,102 | 153,428,800 | 6.766 | 348.25 | 684,976,320 / 4,488,865,440 |
| stable_global_no_progress_dense_s8 | 8,740 / 5,135 | 37,197,440 | 7.079 | 348.25 | 584,682,240 / 4,785,786,400 |

Stable production maps obey the event/cumulative caps (2/8), fixed K=128 eligible prefix tiles per query head/block, and short-prefix capping. Protected/carry support follows inherited V31 handling. GLOBAL and LOCAL physical accounting is geometry-derived H x Q128 x KV64 rectangles, not CUDA CTA counts or FLOPs. LOCAL skipped work is zero; native LOCAL calls equal 25N. The common denoising denominator includes GLOBAL full support and native LOCAL window-intersecting rectangles; prefill and commits are dense outside this denoising denominator. Dense-reason counters are GLOBAL-layer calls, not whole-model forward counts.

State/workspace memory is reported at request end; transient peak selector memory is not isolated. Overall peak CUDA allocation is retained in the receipts. Age histograms in native records are cache-retirement/request-end distributions, while operator age strata measure outside prediction error against current dense statistics. Retained-support drift does not certify unseen outside tiles. All-kept shortcut events are reported separately from real QK observations.

## 5. Clean timing and recommendation

**Clean timing remains paused and intentionally absent. No end-to-end speedup is claimed.** Retain the unchanged V31 attention-mass method for this budget and panel. Do not promote any tested cheap-refresh arm to the 503-cell audit. Preserve stable s8 as an accuracy/trajectory diagnostic and stable s1 as a call-count diagnostic; this small development panel does not establish either as a production improvement. Future repair needs stronger evidence about incoming blocks or output-direction safety, and a new frozen study rather than target-answer retuning.

## Receipts and limits

Authoritative files: final_complete.json, native_generation_complete.json, reports/fairness_audit.json, reports/native_workload_results.json, reports/operator_results.json, specs/execution_identity.json, specs/parameters_attempt004.json and specs/scoring_identity_attempt002.json. All 52 completed native leaves are immutable. Failed trajectory attempt001 and scoring_attempt001/002 remain preserved. Official extraction/scorer source hashes and private input fingerprints match. Additional original-vs-study lifecycle parity tests pass for ordinary, all-kept and padded-row layouts.

The prior panel is not reused for matched claims: its frozen source differs and used GPU memory utilization 0.85, whereas this source-matched study freezes 0.80. The new control reuses the inherited selection/lifecycle rules; its historical denoising totals need not match older source/runtime runs. The legacy STATE.json ledger is not used by this study and its paused campaigns remain untouched. Prompts, token IDs, gold answers, raw completions and Q/K/V tensors stay private.
