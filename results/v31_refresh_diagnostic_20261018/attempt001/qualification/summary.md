# Refresh intervention qualification

Source `6aab8632`; protocol
`1e7714be3fd4690b41e37727a7b8ae283800c24f579f90eec03d90e37bfb964e`.
One predeclared development item: LongBench-v2 index422, seed42, 47,973 prompt
tokens. Four explicit warmups are excluded from the following six measured rows.
This is a completed implementation/causal qualification, not a benchmark panel.

| Policy | Official correct | Denoising calls | Canvases | Output tokens |
|---|---:|---:|---:|---:|
| Original V31 before | 1 | 515 | 24 | 6035 |
| Dense output, fresh support | 1 | 515 | 24 | 6035 |
| Dense output, held support | 0 | 2064 | 64 | 16384 |
| Sparse old-support output, fresh support | 0 | 404 | 14 | 3477 |
| Sparse old-support output, held support | 0 | 351 | 12 | 3060 |
| Original V31 after | 1 | 515 | 24 | 6035 |

All six complete prefix fingerprints match before the first treated GLOBAL layer
at zero-based denoising call5. Adapter canvas ID3 reflects the preceding chunked
prefill; it is not evidence that treatment waited until the third generated block.
The dense/fresh output hash and native counts match both original controls.

For each current-output choice, fresh and held branches return identical event
outputs at all five GLOBAL layers and identical first next sampler state,
including self-conditioning. Their future maps differ as intended and are
consumed on the next call without an intervening refresh. Dense versus sparse
current output changes the immediately following state. The two factors are
therefore separated in this qualifying execution.

All receipts pass: fixed eligible-prefix budget, mandatory tiles, five GLOBAL
layers, dense LOCAL work, unchanged RNG, one original attention call per layer,
zero extra native forwards/sampler calls, and zero measured CUDA captures.
Each factorial arm separately receipts five additional sparse attention oracle
calls; original controls execute none. Instrumented wall times are not speed
evidence. The dense/held branch reaches its unchanged native output limit.

Independent review verified public file hashes against the engine completion
marker, the prefix metadata, factor separation, and official scored rows.
GPU reservation:471.962705s, including startup and warmups. Qualification is
`passed`/`qualified`; the coordinator stopped at the explicit phase boundary.

Interpretation: this one item needs both components for its observed correct
outcome. Fewer calls in the sparse branches accompany wrong answers. Neither
observation establishes a general necessity, accuracy effect, or usable speedup.
Proceed to the preregistered headroom pass; the full503 campaign remains paused.
