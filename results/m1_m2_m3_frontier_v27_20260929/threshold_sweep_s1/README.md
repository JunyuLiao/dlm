# v27 threshold sweep s1: selection threshold vs denoising steps, speed and accuracy

Protocol `v27_long_lb_sweep_s1_322b24218397083a` (`../specs/v27_long_lb_sweep_s1.json`), frozen before generation:
the 12 + 12 LongBench-v2 32K (28-40K tokens) and 64K (56-76K) panel items of v27_long_lb, seeds 101/202/303
(36 cells per arm per bin), substrate `piecewise_v2`, two H100 hosts with every arm of a cell on one host.
Paired against D_fa4_allkept with a question-clustered bootstrap (`summary.md`, `summary.csv`, `cells.csv`).
The threshold shift is added to the selector's log-mass threshold: positive skips more tiles, negative keeps more.

## 64K (dense: 18 of 36 correct)

| arm | correct | +/- | request W [95% CI] | decode S (excl. prefill) | steps/canvas N/C [CI] | S/N |
|---|---:|---|---|---|---|---:|
| Fan M3 R3 A8 | 21 | +4/-1 | 1.107 [0.977, 1.231] | 1.165 | 0.999 [0.945, 1.049] | 1.164 |
| M3 R6 DP, +ln2 (sparser) | 20 | +5/-3 | 1.074 [0.915, 1.255] | 1.091 | **1.395 [1.232, 1.573]** | 0.774 |
| M3 R6 DP, 0 | 19 | +3/-2 | 0.970 [0.855, 1.086] | 0.939 | 1.101 [1.038, 1.166] | 0.810 |
| **M3 R6 DP, -ln2** | 23 | +5/-0 | **0.868 [0.807, 0.932]** | **0.795 [0.725, 0.881]** | 0.952 [0.918, 0.991] | 0.846 |
| M3 R6 DP, -2ln2 | 20 | +4/-2 | 0.875 [0.773, 0.970] | 0.816 | 0.960 [0.910, 1.006] | 0.877 |
| M3 R6 DP, -3ln2 | 22 | +7/-3 | 0.958 [0.857, 1.055] | 0.941 | 1.001 [0.957, 1.047] | 0.918 |
| B, 0 | 20 | +5/-3 | 0.866 [0.823, 0.913] | 0.808 [0.764, 0.858] | 1.044 [1.002, 1.090] | 0.816 |
| B, -2ln2 | 20 | +3/-1 | 0.941 [0.830, 1.054] | 0.904 | 0.982 [0.928, 1.037] | 0.864 |
| M1 R1 DP, 0 | 19 | +3/-2 | 0.942 [0.845, 1.032] | 0.910 | 1.084 [1.005, 1.178] | 0.844 |
| M1 R1 DP, -2ln2 | 23 | +6/-1 | 0.924 [0.786, 1.045] | 0.880 | 0.976 [0.913, 1.031] | 0.914 |

## 32K (dense: 28 of 36 correct)

| arm | correct | +/- | request W [95% CI] | decode S | steps/canvas N/C [CI] | S/N |
|---|---:|---|---|---|---|---:|
| Fan M3 R3 A8 | 29 | +6/-5 | 1.150 [0.992, 1.319] | 1.178 | 0.991 [0.932, 1.055] | 1.139 |
| M3 R6 DP, +ln2 | 23 | +3/-8 | 1.236 [1.055, 1.432] | 1.290 | **1.363 [1.234, 1.501]** | 0.927 |
| M3 R6 DP, 0 | 24 | +4/-8 | 1.003 [0.894, 1.110] | 0.984 | 1.072 [0.994, 1.152] | 0.907 |
| M3 R6 DP, -ln2 | 31 | +6/-3 | 0.951 [0.801, 1.100] | 0.918 | 0.996 [0.940, 1.051] | 0.993 |
| M3 R6 DP, -2ln2 | 27 | +5/-6 | 0.926 [0.761, 1.110] | 0.891 | 0.976 [0.888, 1.078] | 0.964 |
| M3 R6 DP, -3ln2 | 27 | +6/-7 | 1.008 [0.855, 1.175] | 0.994 | 0.985 [0.904, 1.077] | 1.047 |
| B, 0 | 31 | +5/-2 | 0.938 [0.861, 1.023] | 0.915 | 1.015 [0.969, 1.063] | 0.914 |
| B, -2ln2 | 29 | +5/-4 | 0.977 [0.850, 1.138] | 0.959 | 0.975 [0.900, 1.082] | 1.004 |
| M1 R1 DP, 0 | 21 | +4/-11 | 1.129 [0.952, 1.330] | 1.142 | 1.101 [1.002, 1.208] | 0.943 |
| M1 R1 DP, -2ln2 | 22 | +3/-9 | 1.063 [0.912, 1.248] | 1.068 | 1.029 [0.945, 1.132] | 0.998 |

## Findings

- **Sparsity costs denoising steps.** Under the native confidence/stability stopping, a sparser selection makes each
  call cheaper (at 64K, S/N falls from 0.92 at -3ln2 to 0.77 at +ln2) but adds steps per canvas (+40% at +ln2, +10%
  at the default threshold). Keeping more tiles removes the extra steps (-ln2 and -2ln2: 0.95-0.98 of dense).
- **The end-to-end optimum is not the sparsest setting.** For M3 R6 DP the request time is U-shaped in the threshold;
  -ln2 is best at 64K (0.868 [0.807, 0.932], decode 0.795) with no accuracy loss (23 vs 18 correct, +5/-0).
- B at the default threshold is equally fast at 64K (0.866 [0.823, 0.913]) with only +4% steps. Keeping more tiles
  does not help B.
- At 32K no setting has a request-time CI excluding 1 (best: M3 R6 DP -2ln2 0.926, B 0.938).
- M1 R1 DP loses accuracy at 32K (21 vs 28, +4/-11, p = 0.12) and takes +10% steps.
- Fan's plain M3 R3 A8 is 1.11-1.15x dense in request time with unchanged step counts: its cost is the per-8-step
  observation, not steps.
- Accuracy differences at this size are screening only (36 cells per arm per bin).
