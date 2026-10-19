# Exact-input replay of the localized native FA4 observation calls

Attempt010 first changes layer29 output on full_swap8/q129/42 at sampler100 and
residual/q129/42 at154. Attempt013 first closed engine captures show that both
calls are observing_dense_fwd anchors, with contiguous complete Q/K/V, scale1,
causal kernel/dynamic_causal=False, pack_gqa=False, num_splits=0. No held residual
correction or selected-map sparse consumer produced these two anchor outputs.
The paired full/layer-observed tensors at these calls match in engine1, while
the pure swaps repeat changes N156/187,C13/16,T3307/3981. Do not claim that these
new captures reproduce the precise earlier operator change.

Replay these complete captured contiguous operands in a separate GPU process
after both native engines close. Restore original strides and pointer alignment
modulo256; keep every native scalar/tensor argument and the actual mask-fix
source. The sole auxiliary0 buffer is write-only; alternate zero/NaN initialization
and require its true prefix slots to be written. Compare repeated output bytes,
captured-native output bytes and a separately computed FP64 diagnostic oracle.
No model/sampler/generation call, altered native trajectory or timing claim.

Run16 repetitions for native automatic observing FA4; then freed FP32 partial
scratch initialized with zero,finite37,NaN; single-split observing; automatic
plain FA4; single-split plain FA4. The latter are distinct diagnostic kernel
paths, never silently promoted runtime repairs. Record actual split/tile/overlap
selection, capture counter and input preservation. Any same-input output change
requires further internal partial/combine capture before a mechanism claim.
An all-stable replay is negative evidence, not a repaired native evaluation.

The source-bound stride/alias/page utility passed four CPU tests, and the final
comparison utility passed four CPU tests. Bind both exact files and receipts
before GPU. Private raw payloads remain untracked and immutable. Use a new replay
attempt, single worker, at most240s charged to existing phasek256/allocation002.
Preserve startup/JIT/failure charges; keep all full accuracy matrices gated.
