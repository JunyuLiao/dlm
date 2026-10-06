# V31 AIME26 GLOBAL budget sweep with dense LOCAL

Complete 30-problem AIME26 panel at seeds 42/43/44 (90 cells per arm), on one H100 with vLLM 0.30.0 / FA4 SM90. Thinking is enabled and the native adaptive sampler is unchanged. The dense reference and all three sparse arms use the same manifest, source, and timing protocol.

All 25 LOCAL layers are native dense: `LOCAL_KV_BUDGET` was unset, `local_router` was not constructed, and the effective LOCAL sparse kernel is `native_dense`. Only the five GLOBAL layers use MAGE with budgets 1024, 2048, and 4096.

## Accuracy and denoising calls

| arm | GLOBAL budget | seed 42 | seed 43 | seed 44 | pooled exact match | denoising forwards | canvases | forwards / canvas | capped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_full_fix51994 | — | 60.00% | 56.67% | 53.33% | 56.67% (51/90) | 26,163 | 1,931 | 13.549 | 33 |
| mage_k1024_local_dense | 1,024 | 60.00% | 50.00% | 60.00% | 56.67% (51/90) | 26,081 | 1,951 | 13.368 | 36 |
| mage_k2048_local_dense | 2,048 | 60.00% | 63.33% | 53.33% | 58.89% (53/90) | 25,593 | 1,908 | 13.414 | 34 |
| mage_k4096_local_dense | 4,096 | 56.67% | 66.67% | 56.67% | 60.00% (54/90) | 25,836 | 1,929 | 13.393 | 34 |

The official exact-match scorer is averaged over the three panel seeds per problem (`avg@k`). Capped cells remain in the denominator.

## Physical tile sparsity

| arm | GLOBAL sparsity | LOCAL sparsity | overall GLOBAL+LOCAL sparsity |
|---|---:|---:|---:|
| mage_k1024_local_dense | 58.05% | 0.00% | 22.40% |
| mage_k2048_local_dense | 41.33% | 0.00% | 15.89% |
| mage_k4096_local_dense | 17.81% | 0.00% | 6.86% |

GLOBAL physical sparsity is exact `sum(global eligible − global kept) / sum(global eligible)` from the adapter receipts. LOCAL is exactly 0% because the native dense consumer is used. The overall column uses the pinned five-GLOBAL / 25-LOCAL, Q128/K64, 1023-token-sided-window geometry; dense LOCAL does not emit a local tile counter, so its eligible denominator is reconstructed from the audited V31 geometry map and is labeled geometry-derived.

## Timing (arithmetic mean over 90 cells)

| arm | end-to-end (prefill + decode) | dense / arm speedup | decode only | dense / arm speedup | prefill |
|---|---:|---:|---:|---:|---:|
| dense_full_fix51994 | 5.7553s | 1.000x | 5.7357s | 1.000x | 0.0187s |
| mage_k1024_local_dense | 6.2557s | 0.920x | 6.2363s | 0.920x | 0.0184s |
| mage_k2048_local_dense | 6.1797s | 0.931x | 6.1603s | 0.931x | 0.0185s |
| mage_k4096_local_dense | 6.2762s | 0.917x | 6.2568s | 0.917x | 0.0185s |

Values below 1x mean the sparse arm is slower. This dense-LOCAL rerun does not show an end-to-end speedup at any tested GLOBAL budget.

## Paired scoring and audit artifacts

- Official scorer: `score/*.summary.json`; paired comparison: [`score/paired.md`](score/paired.md).
- Public records are sanitized; raw completions and logs remain in the user-private run bundle.
- Frozen protocol: `README.md`, `config.json`, and `input_sha256.txt`; `final_complete.json` is the completion marker.
