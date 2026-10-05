# Native QK reuse: four-question smoke report

Quality scope: 4/4 questions, 12 unique attempt-0 receipts; one seed in the checkpoint where present. Missing attempts are incomplete data, not failures. Timing retries never enter the quality denominator.

| Arm | Coverage | Correct | Caps | Unparsed | Mean whole wall (s) | Mean device timeline (s) |
|---|---:|---:|---:|---:|---:|---:|
| native_dense | 4/4 | 3/4 | 1/4 | 1/4 | 41.3 | N/A |
| fresh_junyu_T | 4/4 | 3/4 | 0/4 | 0/4 | 39.9 | 39.7 |
| M1 | 4/4 | 0/4 | 3/4 | 4/4 | 254.2 | 254.0 |
| M3 | 0/4 | N/A | N/A | N/A | N/A | N/A |

Device timeline starts after initial encoder prefill and ends after generation; it includes host gaps and later encoder/commit work. Missing event measurements are N/A. CPU generation and time-between-tokens (including burst events) are N/A.

| Arm | Output tokens (mean) | Decoder calls (total; mean/request) | Canvases (total; mean/request) | True per-canvas decoder calls (median [min, max]) |
|---|---:|---:|---:|---:|
| native_dense | 6,310 | 1,086; 271.5 | 100; 25.0 | 10 [3, 27] |
| fresh_junyu_T | 5,307 | 939; 234.8 | 84; 21.0 | 10 [4, 24] |
| M1 | 7,808 | 4,988; 1,247.0 | 123; 30.8 | 48 [8, 48] |
| M3 | N/A | N/A; N/A | N/A; N/A | N/A |

QK work is counted in score elements. Physical PV skip is counted in eligible block decisions; neither is a measured DRAM-byte or instruction reduction.

| Arm | QK reused / (current + reused) | Physical PV skipped / eligible | GLOBAL PV skip | LOCAL PV skip | Cache refresh / decision refresh |
|---|---:|---:|---:|---:|---:|
| native_dense | N/A | N/A | N/A | N/A | N/A |
| fresh_junyu_T | N/A | 30.9% | N/A | N/A | N/A |
| M1 | 86.9% | 59.5% | 62.0% (4/4 audited) | 57.8% (4/4 audited) | 19,560 / 149,640 |
| M3 | N/A | N/A | N/A | N/A | N/A |

Separate dense timing controls: 4 retry receipts; the original dense quality rows above remain unchanged. All reported comparisons are exploratory because cold/JIT state is unknown.

| Matched comparison | Questions | Whole-wall dense retry / method (geomean [bootstrap 95% CI]) | Device-timeline dense retry / method (geomean [bootstrap 95% CI]) |
|---|---:|---:|---:|
| dense timing control vs fresh_junyu_T | 4/4 | 0.96× [0.58, 1.74] | 0.96× [0.58, 1.75] |
| dense timing control vs M1 | 4/4 | 0.14× [0.09, 0.23] | 0.14× [0.09, 0.23] |
| dense timing control vs M3 | 0/4 | N/A | N/A |

Dispatch audits matched exactly to quality attempts: 4; unmatched: 0; rejected on consistency: 0. GLOBAL/LOCAL percentages use only matched audits. The dispatch audit verifies calls and routing, not physical memory traffic.

Limits: four questions and one seed do not establish quality noninferiority or a stable speedup. Synthetic mask qualification does not settle any live-model mask semantic discrepancy. CUDA graph performance is unqualified. The report is numerical only; it does not infer method success.

Source summary SHA-256: `b4c42eaf50a8670f85acd5a0add1d97533926d64ef4af69d0be370491f5e5810`.
Dispatch audit SHA-256: `701293f3620ede77273116f7c74e4c0732abf3dd26f74db35b363b029543e82d`.
