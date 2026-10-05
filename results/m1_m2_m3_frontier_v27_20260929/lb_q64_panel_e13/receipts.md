# E13 receipts: q64 (64-row FA4 keep maps) vs M3 + c0 vs dense

- 1,062/1,062 runs ok; one host per cell (dllm, mpk, dlm2: 354 each); no timed new graphs. mpk started only after the
  HumanEval scoring on mpk had finished.
- q64 arm: `q_block=64` in all 354 runs; 71,475 refined routes; 71,405 of 93,090 FA4 list builds used 64-row lists (the
  rest are carried call-0 maps and other unrefined maps, which fall back to 128-row lists by design).
- Dense and M3 + c0 reproduce E5 on the same cells (64K W 0.854 vs 0.853; 32K 0.908 vs 0.907).

Direct q64 / M3 + c0 paired ratios (item-clustered bootstrap, seed 7) and accuracy (exact McNemar):

| bin | W [CI] | S [CI] | per-step S/N [CI] | N [CI] | accuracy q64-only / M3c0-only (p) |
|---|---|---|---|---|---|
| 32K | 0.995 [0.949, 1.046] | 0.990 [0.927, 1.059] | 0.998 [0.993, 1.005] | 0.992 [0.927, 1.064] | 13 / 18 (0.47) |
| 64K | 1.011 [0.964, 1.074] | 1.012 [0.940, 1.105] | 0.993 [0.990, 0.996] | 1.019 [0.946, 1.113] | 11 / 10 (1.00) |
| 96K | 1.025 [0.950, 1.116] | 1.010 [0.879, 1.176] | 0.993 [0.980, 1.005] | 1.018 [0.878, 1.195] | 6 / 8 (0.79) |

Per-host 64K per-step ratio: dllm 0.995, mpk 0.992, dlm2 0.992.
