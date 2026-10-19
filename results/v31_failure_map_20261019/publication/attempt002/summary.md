# V31 budget-versus-convergence failure map: completed diagnostic

**Decision:** the complete screen found zero cases that meet the predeclared same-type failure criterion on both seeds. Confirmation and same-budget repair were gated out. This is a completed negative screen, not evidence of universal trajectory preservation or noninferiority.

## Frozen design and execution

The [three-stage plan](../../attempt003/spec.md) establishes reproducible failures, diagnoses current-output versus future-support repair at the same K, then conditionally develops an algorithm. Stage1 qualification and screening completed. Its promotion gate prevented stage2; stage3 remains future work. Yuhan owns the inherited mass-selection/cross-step-reuse baseline. No new selector was tested.

16 metadata-selected, previously examined LongBench-v2 questions, rendered lengths18K–120K, seeds1/42. Native DiffusionGemma-26B-A4B sampler, canvas256, output limits, GLOBAL-only sparsity and dense LOCAL are fixed. V31 uses Q128/KV64, step1 selection, carry, settle0.15 and sticky1.386. Each K has two within-engine V31 bookends; K8192 also has fixed16. Three dense controls cover the same32 cells. These are development results, not held-out confirmation.

Native source `5dcade2c27bbc8092bdbc94b231f9e324b0bcaee`; protocol `0ca7a002319049af7f0a7b379647ddca576af676909aa702ea78c0d9c31cb7e4`. All six qualification families and all320 screening requests completed. Independent verification checked376 records, including36 explicit warmups, with340/340 measured official rescoring agreements. All source/input/completion, seed, native-call and coverage checks pass.

## Outcome map

V31 rows pool the two technical bookends within the same32 question–seed cells. Calls per canvas is pooled N/C, rather than the average of per-request ratios.

| Arm | Accuracy | Mean N | Pooled N/C | Mean output tokens | Caps / records |
|---|---:|---:|---:|---:|---:|
| Dense FULL | 50.00% | 332.66 | 23.09 | 3557 | 2/32 |
| Dense PIECEWISE | 56.25% | 288.38 | 22.84 | 3108 | 1/32 |
| All-kept FA4 | 59.38% | 203.31 | 19.36 | 2559 | 0/32 |
| V31 K8192 | 62.50% | 201.41 | 20.79 | 2348 | 0/64 |
| V31 K4096 | 60.94% | 183.16 | 20.25 | 2188 | 0/64 |
| V31 K2048 | 56.25% | 238.08 | 21.83 | 2667 | 1/64 |
| Fixed16 K8192 | 59.38% | 285.78 | 25.19 | 2771 | 1/32 |

All-kept already reduces mean N from332.66 under FULL to203.31, versus201.41 for V31 K8192. Therefore the FULL comparison alone cannot attribute the call-count reduction to sparse masking. Consumer path and output length are material confounders. Their numerical cause remains unlocalized.

K2048 gives an exploratory pressure signal against all-kept: accuracy−3.125pp (95% question-cluster CI[−21.875,+12.5]), N/C ratio1.1274 [0.9858,1.2905], and N ratio1.1710 [0.8047,1.7313]. Against FULL, all three V31 budgets have positive accuracy point differences and lower N/C point ratios; none meets the predeclared FULL population flag. The reference choice changes the apparent conclusion. All intervals use20,000 question-cluster draws, seed1729, conditional on this small cohort and these engine runs.

[Budget frontier with both references](budget_frontier.png). Full paired intervals against FULL, PIECEWISE and all-kept are in [independent verification](verification.json).

## Why there is no repair cohort

Within-seed candidate counts (accuracy/convergence) are:

| K | Seed1 | Seed42 | Same type on both seeds |
|---:|---:|---:|---:|
| 8192 | 0/1 | 0/2 | 0 |
| 4096 | 0/1 | 1/2 | 0 |
| 2048 | 0/1 | 1/1 | 0 |

Question462 at K4096 is an exploratory mixed-type case: convergence failure on seed1 and accuracy failure on seed42. The frozen gate requires the same failure type, so this case was not promoted. Several other failures occur on only one seed. These observations motivate independent follow-up; they are not confirmed failures or evidence of a successful repair. [Within-seed map](seed_failure_map.png) and [numeric case accounting](case_accounting.json) retain all such outcomes.

Across96 V31 bookend pairs, five change output/N and one changes official score. At K2048 a single pair changes N from1081 to2140, contributing substantially to its aggregate call-count instability. This reinforces the need for new-seed/process replication before treating an outlier as a mask defect.

## Timing, work and budget limits

All six screen engines have one post-warmup fused-MoE JIT warning. Recorded W and S remain available as raw measurements, but this campaign establishes no qualified warm speedup. The accuracy/N/C screen is retained; the figures intentionally omit latency claims. No recorded request captures a new CUDA graph.

Qualification checks59,360 head/query-block selections including warmups, with zero fixed-K/tail errors or LOCAL skips. Physical GLOBAL/LOCAL accounting is audited separately with an explicitly denoising-only denominator; it must not be attached to the different clean trajectories as matched total work. Four dense qualification records have no finite-output metric; absence is reported rather than counted as zero. See [physical work](physical_work.json).

GPU reservation5355.761s=89.263min of14400s=240min. Qualification774.110s, reserved recovery650.094s, screen3931.557s; confirmation/repair0s. Startup, model/compile/native warmups, and the preserved1.001s zero-record launch failure are included. All reservations are closed and [H100 idleness is verified](../../attempt003/receipts/gpu_idle.json). Attempts001/002 and their continuation receipts remain intact. Older paused studies remain paused.

## Research implication and next diagnostic

This screen does not justify implementing another selector. The specific unresolved lead is aggressive-budget behavior relative to the all-kept consumer, together with seed-dependent and mixed-type failures. First confirm the K2048 regime with additional seeds/questions and a predeclared aggregate quality/convergence endpoint, and independently replay the mixed-type case and divergent tail. A future protocol may predeclare a combined harm endpoint; the current same-type rule must not be changed retrospectively. Separate any next warm-timing schedule from first-use compilation.

K4096 is also a competitive fixed-budget point in this screen. Any future selector must retain optimized fixed-budget and fixed16 references, rather than comparing only against K8192. Its apparent accuracy/latency trade-off still needs confirmation.

Only reproduced harmful cases should enter common-prefix, same-K output/support repair. Fresh support helping would motivate a selector/lifetime predictor; dense output alone helping would motivate a different protection mechanism and stronger early-dense controls. Unmatched prefixes, sham drift, unchanged supports or disappearing audit failures cannot establish repair causality. A negative single early intervention would not rule out a later intervention.

Algorithmic novelty remains to be established. [MAGE](https://arxiv.org/html/2602.14209v1) already includes reused support and layer-adaptive allocation, and evaluates static unmasking; [SparseD](https://arxiv.org/abs/2509.24014) already protects early steps with dense attention. Native adaptive convergence remains a relevant evaluation distinction, but the current screen has not yet supplied a confirmed algorithmic failure mechanism.

Artifacts: [raw automated summary](../../attempt003/analysis/attempt001/summary.md), [verification and per-item data](verification.json), [frozen protocol](../../attempt003/protocol.json), [completion marker](../../attempt003/final_complete.json).
