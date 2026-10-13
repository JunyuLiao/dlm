# V31 refresh factor isolation

**Completed: all 16 operator settings, two held references, and paired analysis of existing native results. No new 503-problem generation has started.**

The 64-problem LongBench panel is exploratory evidence for selecting further tests. It has only 8 of the 16 factorial cells, so progress with cheap refresh has no matched native comparison. Operator replay fills the factorial on identical frozen Q/K/V; it does not supply missing native accuracy or denoising results.

## Operator diagnostics

620 real snapshots / 670 validated files, all five GLOBAL layers in both workloads, 2,550 real cheap events checked against an independent NumPy oracle. Fresh QK error <= 1.91e-06; hybrid score error <= 1.64e-06. Map mismatches, fixed-budget errors, stable cap errors and actual CUDA graph captures are zero. The 26 new tests and 38 CUDA/FA4 tests passed.

Each arm has the same 570 post-anchor frames; 50 pre-anchor bootstrap/carry frames are excluded equally. The table compares the same 480 non-selection frames for every arm. Errors are full-dimensional row-output L2 versus dense attention. The output-change p95 includes zero-change and scheduled dense calls over the common 570-frame denominator.

| Progress | Interval | Raw / stable row error | Raw / stable retained mass | Raw / stable swaps | Raw / stable change p95 |
|---|---:|---:|---:|---:|---:|
| On | 1 | 2.8350 / 2.8784 | 0.757707 / 0.755872 | 330,670 / 10,027 | 0.44782 / 0.06539 |
| On | 2 | 2.8438 / 2.8871 | 0.757182 / 0.755321 | 166,215 / 6,073 | 0.32414 / 0.04143 |
| On | 4 | 2.8572 / 2.8950 | 0.756440 / 0.754912 | 78,762 / 1,579 | 0.17712 / 0.00000 |
| On | 8 | 2.8812 / 2.8991 | 0.755402 / 0.754724 | 20,629 / 0 | 0.00000 / 0.00000 |
| Off | 1 | 2.8741 / 2.9242 | 0.754833 / 0.752387 | 330,975 / 5,385 | 0.39853 / 0.04294 |
| Off | 2 | 2.8815 / 2.9265 | 0.754376 / 0.752212 | 166,382 / 5,033 | 0.28535 / 0.03898 |
| Off | 4 | 2.8933 / 2.9332 | 0.753641 / 0.751814 | 84,752 / 3,075 | 0.17625 / 0.02182 |
| Off | 8 | 2.9070 / 2.9398 | 0.752867 / 0.751492 | 36,961 / 920 | 0.04923 / 0.00000 |

Progress dense lowers mean error on common sparse calls by 0.89-1.57% at matched mode and interval. This is conditional operator evidence; its native effect with cheap refresh remains unmeasured.

More frequent raw refresh improves the operator mass/error monotonically in this replay. Stable also improves with frequency, but keeps less mass and has slightly higher error than raw at every matched setting. Stable substantially suppresses churn and next-call perturbation; that property does not establish better native accuracy.

Stable + progress + interval 8 makes **zero swaps** and exactly matches the progress hold map/error. Each of its 15 nonempty anchor periods has only one cheap event, so the two-event persistence condition cannot be reached. Its zero p95 is an inactive update mechanism on these snapshots. At other settings, a zero p95 can coexist with large rare changes; see the maxima in the operator receipt.

## Native accuracy and trajectory: stability alone

| Progress | Interval | Raw / stable correct out of 64 | Stable minus raw | Stable N change | Stable N/C change | Stable output length change |
|---|---:|---:|---:|---:|---:|---:|
| On | 2 | 33 / 31 | -2 | +13.80% | +2.57% | +7.70% |
| Off | 1 | 28 / 32 | +4 | -28.17% | -2.25% | -17.30% |
| Off | 4 | 32 / 32 | +0 | +9.71% | +3.36% | -6.63% |
| Off | 8 | 27 / 35 | +8 | -9.35% | -0.61% | -4.17% |

Stability is an interval-dependent bundle effect, not a uniform win. At interval 8 without progress it rescues eight cells relative to raw, but only gains one cell over progress V31 and two over matching hold, while increasing N by 22.29% and 11.85% respectively. At interval 1, lower N accompanies shorter outputs. The paired intervals and all existing one-factor contrasts are in `native_factor_contrasts.json`; they are exploratory and unadjusted for multiple comparisons.

**Interval alone, progress off:** raw has counts 28/32/27 and mean N 304.38/249.17/324.14 at 1/4/8. Stable has counts 32/32/35 and N 218.64/273.38/293.83. Interval 2 is missing. The operator trend toward more frequent updates does not predict this native accuracy ordering.

**Progress alone, hold controls:** progress-on versus progress-off gives 34 versus 33 correct, N -8.54%, N/C -10.34%, and output length +6.69%. Accuracy has a broad paired 95% interval of [-10.94, +14.06] pp. This isolates progress only at zero cheap updates. All no-progress cheap arms share the AIME hold correctness vector; AIME is mostly all-kept at K=128.

## Candidate selection for 503 problems

A compact factor-preserving design starts with the four raw/stable × progress-on/off arms at interval 2. Add stable no-progress intervals 1/4/8 to test the full frequency sweep. This gives **seven method arms**, two matched held references and three native-consumer controls; a full execution would be 12 × 503 = 6,036 cells. It identifies stability and progress at interval 2 and frequency within stable no-progress. It does not estimate every interaction across all 16 settings.

Stable no-progress s1 is the lower-N candidate; s8 is the higher observed accuracy candidate with trajectory inflation. Raw progress s2 is the near-control-N comparator. Include the missing no-progress s2 pair to isolate progress. Raw no-progress s4 is an optional frequency challenger if a raw interval sweep is selected. Do not drop a counterpart solely because it scored poorly on the noisy subset; keep the comparison needed to answer each factor question.

Use fresh source/runtime-matched 503 controls. The old 268/503 V31 result is historical, not the matched reference for this follow-up. The user will choose arms after this report; `503_arm_proposal.json` is a proposal and no native job has been launched.

## Selector overhead and clean timing

The replay used one H100, with 434.2 seconds of replay wall time including CPU reference checks. Peak replay allocation was 1.870 GiB for the multi-arm diagnostic. Per-arm QK/ranking CUDA event totals and selector-state bytes are in the operator summary. Replay wall time and simultaneous-arm peak allocation are not native selector latency or memory. Production uses duplicate retained/tail QK, unchanged FA4 output, no selector PV/probes, and GPU ranking with existing host decisions.

**Clean timing remains paused.** Native N and output-length effects are trajectory metrics, not speedup. Existing native receipts establish dense LOCAL routing and zero graph captures; this offline replay changes no native output code. Stable uses zero hysteresis, persistence 2 cheap events, cap 2/event and 8/anchor epoch, and removal guard 0.025. Persistence therefore requires at least 2/4/8/16 reuse steps at intervals 1/2/4/8. Internal stabilizer components are not separately ablated.

Sources: native publication `a136b40d`; factor replay freeze `be9e23dc`; operator receipt and source identity stored with this study. Source study artifacts remain immutable.
