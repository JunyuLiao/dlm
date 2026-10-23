# Output-preserving V2 execution optimization screen

Synthetic Q256/H16/Hkv2/D512 BF16 on H100; masks and output unchanged for the selected CTA-ordering plus immutable-prefix projection reuse. Native qualification and benchmark follow-up remain pending.

| Keys | Actual sparsity | Previous full projection us | Cached projection us | Cached + CTA order us | Unique Q128 queue us | Unique kept queue us |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 44.87% | 424.59 | 412.38 | 408.84 | 585.79 | 460.51 |
| 8192 | 89.97% | 412.99 | 401.93 | 395.65 | 463.97 | 423.93 |
| 131072 | 48.63% | 6517.22 | 6271.74 | 6071.61 | 8574.45 | 6676.57 |
| 131072 | 89.99% | 6267.72 | 5850.44 | 5879.53 | 6847.26 | 6151.73 |

Selected library: attempt007/build/380b43b3784cdce8/v2_online_sm90.so. Same mathematical and floating-point operations as200354cb, CTA scheduling order changed. Reuse only immutable prefix projection within a native canvas; current canvas projected every refresh and full RMS recomputed. First refresh/canvas recomputes all V.

Direct-paged TMA plus paged projection passed six bitwise changing-input graph tests but was slower (attempt006); retained negative. Shared-routing designs remain slower and are not selected. Their diagnostic output parity is explicitly recorded, not assumed.

Selected library passes35 CUDA operator/resident/V31 tests (attempt009); cached native integration passes one exact output/mask/request-reset test (attempt010). Three campaign-policy tests pass (attempt011). Attempts001/002/003 preserve compile and harness failures. Actual CUDA capture counters are zero during all synthetic timing, with graph creation outside timing.

Held consumer arithmetic is unchanged and shared by V2/V31. Identical-mask timing is recorded; method-specific work and trajectory must not be mistaken for an implementation difference. No new native accuracy or request speedup is claimed.
