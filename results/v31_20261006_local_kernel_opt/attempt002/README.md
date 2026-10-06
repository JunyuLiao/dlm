# V31 LOCAL compact consumer qualification

This is a held-map kernel qualification for the named `local_compact_triton_q64`
variant. It does not change the frozen AIME26 result in
`results/v31_20261006_aime_global_local/`.

The selector remains the V31 LOCAL selector: a 512-token prefix budget,
Q128/KV64 eligibility, mandatory current-canvas tiles, and the native
bidirectional `(1023, 1023)` window. The new consumer expands each selected
Q128 row block to two Q64 blocks, packs retained tile indices, and performs
paged K/V loads plus online softmax in a Triton kernel. `LOCAL_KERNEL=fa4`
keeps the previous direct FA4 consumer available as a control.

Timing uses one H100, native interleaved KV64 pages, Q256, H16/Hkv8, D256,
100 synchronized alternating samples, and 20 launches per sample. The dense
reference is FA4 with the same native window. `old_fa4_q128` is the previous
LOCAL implementation; `fa4_q64` is the same FA4 consumer with the expanded
Q64 map. These are kernel timings only and exclude selection and observation.

The compact output agrees with the old FA4 sparse output within 0.001953125
maximum absolute error for every tested prefix. Median compact speedups over
dense are 1.30x at prefix 187, 1.12x at prefix 3500, and 1.29x at prefix 8192.
The physical map sparsities in these synthetic held maps are 14.29%, 37.50%,
and 39.47%; the AIME26 end-to-end panel remains the earlier 25.33% LOCAL
sparsity because its native sampler has varying prefixes and call mix.

This qualification is not an accuracy or end-to-end claim. A fresh AIME26
run is required before replacing the previous end-to-end timing or accuracy
numbers.
