# v27 quality panel q1v3: AIME26 + LongBench-v2, 11 arms, FA4 + piecewise_v2

Protocol `v27_quality_fa4pw_q1v3_f4d0717aeff8b6af` (`../specs/v27_quality_fa4pw_q1v3.json`), frozen before
generation: all 30 AIME26 problems and all 12 LongBench-v2 pool items (10-19K tokens), seeds 101/202, first-only,
two H100 hosts (every arm of a cell on the same host), substrate `piecewise_v2`. Summary (paired against
D_fa4_allkept, question-clustered bootstrap): `summary.md`, `summary.csv`; per-cell scores without text: `cells.csv`.

**Rerun cells.** Five AIME (problem, seed) cells on mpk (aime26/1 s202, /2 s101, /3 s202, /4 s101, /5 s202) failed
in every arm of run 001 with CUDA OOM (another process took 28 GiB of that GPU right after the idle check). They were
rerun for all 11 arms in run 002 (same frozen protocol, mpk only, the first 55 scheduled executions) and merged:
every arm of a cell still comes from one run on one host. AIME below covers all 60 cells.

## AIME26 (60 cells; dense D_fa4_allkept 35 correct)

| arm | correct | +/- vs dense | sign test p | request W [95% CI] | steps/canvas |
|---|---:|---|---:|---|---:|
| D_native (HF SDPA, exact dense) | 36 | +7/-6 | 1.00 | 1.01 [0.87, 1.13] | 1.00 |
| Fan M1 (R1 A8) | 29 | +4/-10 | 0.18 | 1.09 [0.93, 1.27] | 1.03 |
| Fan M2c (R1 A8) | 33 | +5/-7 | 0.77 | 1.06 [0.88, 1.24] | 1.00 |
| Fan M3 (R3 A8) | 31 | +2/-6 | 0.29 | 1.06 [0.89, 1.24] | 1.02 |
| B (A64 hold) | 26 | +4/-13 | 0.049 | 1.07 [0.91, 1.24] | 1.04 |
| M3 R6 A64 rp | 30 | +4/-9 | 0.27 | 1.01 [0.86, 1.14] | 1.00 |
| M1 R1 A64 DP | 33 | +4/-6 | 0.75 | 1.01 [0.87, 1.13] | 1.02 |
| M3 R6 A64 DP | 31 | +8/-12 | 0.50 | 0.99 [0.84, 1.11] | 1.04 |
| M3 R6 A64 DP, shift -2ln2 | 27 | +5/-13 | 0.10 | 0.98 [0.84, 1.11] | 1.02 |
| M3 R6 A64 DP, gate 8K keys | 35 | +0/-0 | 1.00 | 0.95 [0.83, 1.02] | 1.00 |

- The exact dense kernel swap already moves 13 of 60 cells (+7/-6): AIME sampling is noisy at this sample size.
- Every sparse arm without a length gate is net negative. Only B is individually significant (p = 0.049).
  Keeping more tiles (-2ln2) did not recover AIME accuracy.
- AIME prefixes stay below about 9K keys, where sparse attention has no speed benefit; the 8K-key gate runs dense on
  AIME and reproduces dense tokens. Its request ratio 0.95 [0.83, 1.02] for identical computation shows that AIME
  request-time differences within about 5% are not resolvable in this panel.

## LongBench-v2 pool (24 cells, 10-19K tokens; dense 14 correct)

- Accuracy: every arm 12-16 correct, all within noise (largest swing B +2/-4).
- Speed: no sparse arm is faster at this length. Request W 1.08-1.29 (Fan A8 arms 1.18-1.29); S/N 0.96-0.97 for
  B and M3 R6 DP (cheaper calls) but more steps per canvas (M3 R6 DP 1.12 [1.02, 1.23]).
- Keeping more tiles removes the step inflation: M3 R6 DP -2ln2 steps/canvas 0.97 [0.92, 1.03].
- D_native (HF SDPA) is 1.64x the FA4 dense request wall.
