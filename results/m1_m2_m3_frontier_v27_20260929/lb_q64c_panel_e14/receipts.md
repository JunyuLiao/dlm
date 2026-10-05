# E14 receipts: q64 and q64c with six fresh seeds (1010–1515)

- Protocol `v27_lb_q64c_e14_46bf7b5a765b7204`, deploy `v27_e14_0fc74fc`.
- 1,416/1,416 runs ok (354 per arm), one host per cell (dllm, mpk, dlm2), no timed new graphs.
- Path receipts:
  - q64: 71,265 of 92,775 FA4 list builds used 64-row lists; the rest are the carried call-0 maps.
  - q64c: 98,070 of 98,070 list builds were 64-row, including 22,420 carried 64-row call-0 maps.
  - M3 + c0: 21,740 carried call-0 maps, all 128-row.

## Direct comparisons (paired, item-clustered bootstrap; exact McNemar)

| comparison | bin | cells | accuracy (+/-, p) | W [CI] | per-step S/N [CI] | N [CI] |
|---|---|---:|---|---|---|---|
| q64c / M3 + c0 (E14) | 32K | 144 | 85 / 85 (+17/-17, 1.0) | 1.006 [0.950, 1.063] | 0.996 [0.992, 1.001] | 1.014 [0.939, 1.092] |
| | 64K | 144 | 82 / 74 (+14/-6, 0.12) | 1.023 [0.983, 1.064] | **0.991 [0.989, 0.995]** | 1.049 [0.989, 1.115] |
| | 96K | 66 | 29 / 31 (+5/-7, 0.77) | 1.078 [1.003, 1.169] | 0.985 [0.969, 1.001] | 1.146 [1.004, 1.323] |
| q64c / q64 (E14) | 32K | 144 | 85 / 88 (+17/-20, 0.74) | 0.998 [0.948, 1.042] | 1.002 [0.999, 1.006] | 0.995 [0.932, 1.051] |
| | 64K | 144 | 82 / 81 (+6/-5, 1.0) | 1.026 [0.987, 1.067] | 0.998 [0.994, 1.002] | 1.042 [0.979, 1.110] |
| | 96K | 66 | 29 / 34 (+5/-10, 0.30) | 1.080 [0.998, 1.191] | 0.989 [0.972, 1.003] | 1.146 [0.988, 1.364] |
| q64 / M3 + c0 (E13 + E14, 12 seeds) | 32K | 288 | 175 / 177 (+30/-32, 0.90) | 1.002 [0.968, 1.034] | 0.996 [0.993, 1.000] | 1.006 [0.961, 1.052] |
| | 64K | 288 | 158 / 150 (+24/-16, 0.27) | 1.004 [0.969, 1.049] | **0.993 [0.991, 0.996]** | 1.013 [0.958, 1.082] |
| | 96K | 132 | 63 / 62 (+14/-13, 1.0) | 1.012 [0.963, 1.071] | 0.995 [0.988, 1.003] | 1.009 [0.920, 1.121] |
| M3 + c0 / dense (E13 + E14, 12 seeds) | 32K | 288 | 177 / 179 (+37/-39, 0.91) | **0.925 [0.892, 0.958]** | 0.924 [0.918, 0.930] | 0.978 [0.931, 1.024] |
| | 64K | 288 | 150 / 155 (+18/-23, 0.53) | **0.860 [0.826, 0.891]** | 0.814 [0.805, 0.824] | 0.954 [0.897, 1.011] |
| | 96K | 132 | 62 / 53 (+20/-11, 0.15) | **0.801 [0.705, 0.902]** | 0.751 [0.735, 0.765] | 0.943 [0.800, 1.116] |

Per-host q64c / M3 + c0 per-step ratio at 64K: dllm 0.993, dlm2 0.994, mpk 0.988 (consistent). Per-host W varies
0.95–1.11 because of step counts.

Generation-only S for M3 + c0 / dense (12 seeds): 0.903 (32K), 0.777 (64K), 0.708 (96K). Prefill P is about 1.00
everywhere, so the methods leave prefill unchanged.
Absolute steps: `steps.md` (E14).
- Dense steps per block are 14.1 (32K), 17.8 (64K) and 26.9 (96K).
- At 96K, 157 of 910 dense blocks hit the 48-step cap.
