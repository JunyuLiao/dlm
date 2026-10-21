# Wide QK bundle, exact KV64 admission, 2026-10-21

Synthetic kernel optimization; previous attempts and benchmark pause remain.
Compute four adjacent KV64 QK blocks with one N256 WGMMA bundle, then perform
the original softmax/Gaussian32/ordered admission and conditional full PV for
each KV64 block. Query and key channel chunks stream through a96KB shared
workspace. The full-dimensional accumulator is temporarily parked in shared
memory while wide QK accumulators are live, then restored before the shared
workspace receives128KB of transient FP32 scores. Shared scores never escape
the CTA and are consumed once. QK work across output groups remains duplicated.

No dense PV on rejected tiles; no selector simulation, second QK traversal,
extra model forward or sampler change. Initial qualification requires finite
outputs, independent exact masks, same-mask FA4 relativeL2<0.01, and changing
Q/K/V/Z/held graph tests. Only then measure matched Q256/H16/Hkv2/D512 BF16 at
8K keys/50% actual sparsity, charging fresh projection, pack and compact masks.
Warm compilation/capture outside timing, count actual captures, preserve every
failed build/test/design. Native calibration and full evaluation remain gated.

qualification001 passes80changed-input graph replays, exact masks and relative
L2<0.003. refresh001 complete8K refresh1282.18us vs361.08us dense is negative;
attempt006 corrects shared-memory bank layout and preserves this first result.
