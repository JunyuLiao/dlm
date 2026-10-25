# Stage A1 reporting correction — CPU only

Both Q-only and full-QKV past-motion forecasts lose to persistence at every available task/layer aggregate, at both anchors and under both frozen averaging definitions. Apparent Q-only gains against the joint-QKV dense/sham controls cannot isolate query-direction information; Q-only dense and Q-only sham controls are missing. The failed frozen A2 gates remain failed. The original Response quality/total-work observations remain separate and open.

| Task | Layer | Anchor | Q / persistence | QKV / persistence | Available questions / seed cells / units |
|---|---:|---|---:|---:|---|
| aime26 | 5 | middle | 2.254 | 6.500 | 4/4; 7/8; 14/32 |
| aime26 | 5 | late | 2.033 | 3.191 | 2/4; 4/8; 10/32 |
| aime26 | 17 | middle | 2.609 | 6.879 | 3/4; 6/8; 9/32 |
| aime26 | 17 | late | 1.915 | 3.086 | 2/4; 3/8; 5/32 |
| aime26 | 29 | middle | 1.368 | 2.801 | 4/4; 8/8; 26/32 |
| aime26 | 29 | late | 1.607 | 2.632 | 4/4; 8/8; 26/32 |
| humaneval | 5 | middle | 2.224 | 5.078 | 4/4; 8/8; 10/32 |
| humaneval | 5 | late | 1.784 | 3.233 | 2/4; 4/8; 4/32 |
| humaneval | 17 | middle | 2.858 | 5.835 | 1/4; 2/8; 4/32 |
| humaneval | 17 | late | 2.778 | 6.988 | 2/4; 2/8; 2/32 |
| humaneval | 29 | middle | 1.380 | 2.751 | 4/4; 8/8; 24/32 |
| humaneval | 29 | late | 1.613 | 2.695 | 4/4; 8/8; 25/32 |
| longbench_v2 | 5 | middle | 2.636 | 4.512 | 4/4; 8/8; 32/32 |
| longbench_v2 | 5 | late | 2.843 | 4.039 | 4/4; 8/8; 32/32 |
| longbench_v2 | 17 | middle | 2.487 | 3.894 | 4/4; 8/8; 32/32 |
| longbench_v2 | 17 | late | 2.804 | 4.426 | 4/4; 8/8; 32/32 |
| longbench_v2 | 29 | middle | 1.727 | 2.996 | 4/4; 8/8; 32/32 |
| longbench_v2 | 29 | late | 1.726 | 3.003 | 4/4; 8/8; 32/32 |

All individual cells and both definitions are in summary.json. Missing/zero-feasible units are not negative forecast observations. No state identifies a qualifying full-response contrast.

Five individual 10% threshold crossings occur between the executable ratio of arithmetic mean unit losses and README geometric averaging. These are not new promotion results: identification is zero and even perfect selection has less than 3% bank-specific next-motion headroom.

| ID | Seed | Layer | Score | Comparator | Executable | README |
|---|---:|---:|---|---|---:|---:|
| longbench_v2/6708a096bb02136c067d1789 | 43 | 29 | q_past | direction_sham | 0.9319 | 0.8814 |
| aime26/17 | 42 | 5 | qkv_past | direction_sham | 1.0030 | 0.8852 |
| aime26/15 | 42 | 5 | q_past | direction_sham | 0.9111 | 0.6892 |
| aime26/15 | 43 | 5 | q_past | direction_sham | 0.9059 | 0.6657 |
| humaneval/HumanEval/152 | 42 | 29 | q_past | direction_sham | 0.9192 | 0.8951 |

Bank-wide matching (24,865 pairs) differs from actually selected matched contrasts: only 19/185 distinct Q-only and 18/179 QKV choices match current-error in error/mass. First-canvas common controls have zero confidence-only delays; later-canvas confidence stalls remain untested by A1.

A1 removed no model consumer work. Its diagnostic work is additional, synchronized audit timing is not clean latency, and FULL has a complete marker but no recovered parent wait status. Say no recorded GPU-worker failure rather than a verified successful FULL exit code. The earlier 299-field correction preserves all prior aggregates and negatives. Frozen files are unchanged and no GPU rerun was used for these corrections.
