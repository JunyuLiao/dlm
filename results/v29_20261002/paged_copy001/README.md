# Fused paged KV copy component qualification

Source3a3a3c52d, H100, Torch2.13.0+cu130. Fixed synthetic native strides,
random physical page order, BF16, Hkv2/D512/page64, seed2901.
All10 full/tail copy cases bit-exact against Torch, including untouched prefixes.
Warm8,100 alternating-order samples; destination preallocated for both arms.
Original gather allocations remain in baseline cost. Event span includes host
dispatch gaps. This is not isolated device bandwidth or request acceleration.

| Prefix | Mode | Torch ms | Fused ms | Fused/Torch |
|---|---|---:|---:|---:|
| 0 | full | 0.076368 | 0.052464 | 0.686989 |
| 0 | tail | 0.076432 | 0.052256 | 0.683693 |
| 65 | full | 0.091488 | 0.052224 | 0.570829 |
| 65 | tail | 0.116000 | 0.064384 | 0.555034 |
| 32768 | full | 0.726560 | 0.132416 | 0.182251 |
| 32768 | tail | 0.094128 | 0.052480 | 0.557539 |
| 65536 | full | 1.390288 | 0.223216 | 0.160554 |
| 65536 | tail | 0.093504 | 0.052112 | 0.557324 |
| 98304 | full | 2.052128 | 0.311984 | 0.152030 |
| 98304 | tail | 0.093232 | 0.051760 | 0.555174 |

Reserved GPU seconds3.867908; internal body2.227179 seconds. Real-model copy
oracle, timed-path receipts and request quality/performance remain pending.
This is a standard optimization for matched all-kept and main, not sparse-method
novelty. No end-to-end benefit or accuracy result is claimed.
