# Refresh factor isolation: existing native evidence

Complete source-validated analysis of the existing development panel. Operator results are reported separately after completion. No new native target scoring or clean timing is part of this step.

## What can be isolated

The existing panel has only 8 of the 16 factorial settings. Progress on/off with cheap refresh cannot be isolated because no interval matches between those groups. Stability can be compared at progress-on interval 2 and progress-off intervals 1/4/8. Interval effects can be compared at progress-off intervals 1/4/8. The held references isolate progress at interval zero.

| Interval | Raw, progress | Stable, progress | Raw, no progress | Stable, no progress |
|---:|---:|---:|---:|---:|
| 1 | Missing | Missing | 28/64; 304.38 | 32/64; 218.64 |
| 2 | 33/64; 238.89 | 31/64; 271.86 | Missing | Missing |
| 4 | Missing | Missing | 32/64; 249.17 | 32/64; 273.38 |
| 8 | Missing | Missing | 27/64; 324.14 | 35/64; 293.83 |

Entries: official correct/64 and mean denoising calls N. Held progress V31: 34/64, N=240.27; held no-progress: 33/64, N=262.69. These are fresh matched controls; historical 268/503 is not their matched full-dataset result.

## Stability alone: stable minus raw at the same interval and progress setting

| Progress | Interval | Correct delta | Wins / losses | Accuracy delta pp [95% CI] | N change | N/C change | Output length change |
|---|---:|---:|---:|---|---:|---:|---:|
| On | 2 | -2 | 3 / 5 | -3.125 [-12.500, 4.688] | +13.80% | +2.57% | +7.70% |
| Off | 1 | +4 | 6 / 2 | +6.250 [-1.562, 14.062] | -28.17% | -2.25% | -17.30% |
| Off | 4 | +0 | 6 / 6 | +0.000 [-10.938, 10.938] | +9.71% | +3.36% | -6.63% |
| Off | 8 | +8 | 10 / 2 | +12.500 [3.125, 23.438] | -9.35% | -0.61% | -4.17% |

These are paired, problem-clustered, exploratory and unadjusted 95% intervals (10,000 bootstrap draws, seed 51994). The interval-8 stability contrast is favorable relative to a weak raw arm. Stable no-progress s8 is only +1 correct versus progress V31 and +2 versus matching no-progress hold; its N is +22.29% and +11.85% respectively. Stability rescue relative to raw is not a demonstrated improvement over V31. At interval 1, much of the N reduction accompanies shorter outputs; this is not a forward-cost result.

## Interval alone: no progress, longer interval minus shorter interval

| Mode | Candidate interval / reference | Correct delta | N change | N/C change | Output length change |
|---|---:|---:|---:|---:|---:|
| raw | 4 / 1 | +4 | -18.14% | -1.56% | +0.82% |
| raw | 8 / 1 | -1 | +6.49% | +2.92% | +6.36% |
| raw | 8 / 4 | -5 | +30.09% | +4.55% | +5.49% |
| stable | 4 / 1 | +0 | +25.03% | +4.09% | +13.82% |
| stable | 8 / 1 | +3 | +34.39% | +4.65% | +23.24% |
| stable | 8 / 4 | +3 | +7.48% | +0.53% | +8.28% |

Raw is nonmonotonic: interval 4 has the best observed count and mean N; interval 8 is worse. Stable trades the lowest mean N at interval 1 for the highest count at interval 8, with output length +23.24%. Interval 2 is missing, and these differences do not justify selecting a population-optimal interval from 64 problems.

## Progress alone at zero cheap refresh

Progress-on hold versus progress-off hold changes LongBench by +1 correct (8 wins, 7 losses), N -8.54%, N/C -10.34%, and output length +6.69%. The accuracy CI is [-10.94, +14.06] pp and aggregate N-ratio CI is [0.569, 1.432]. This comparison says nothing directly about progress with cheap updates: reset frequency, anchor freshness and stability caps can interact with those updates.

## Effective definitions and limits

Raw means global hybrid top-K without the stable constraints. Stable uses zero hysteresis, persistence 2 cheap events, cap 2 replacements/event, cap 8 between dense selections, and row removal tolerance 0.025. At cheap intervals 1/2/4/8, two-observation persistence takes at least 2/4/8/16 reuse steps. Thus the stable-frequency sweep changes effective response delay by design. Progress dense also resets its cache, persistence and cumulative allowance.

All no-progress cheap arms have the same AIME correctness vector as the no-cheap hold. At K=128, AIME GLOBAL physical sparsity is below 0.004%; it does not isolate cheap repair benefits. Source publication, all 52 native shards and official score coverage were verified without rescoring or editing the source study. Full details and uncertainty for every available contrast are in `native_factor_contrasts.json`.
