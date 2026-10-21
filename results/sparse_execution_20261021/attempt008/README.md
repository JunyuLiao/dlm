# Within-CTA QK/PZ lookahead, full output, 2026-10-21

Synthetic build-only candidate; numerical and performance qualification pending.
WG1 computes the next KV64 full QK while WG0 computes the current Gaussian32
PZ/risk. Current probabilities stay in16registers per thread; a16KB union holds
packed Z/low probabilities until PZ finishes, then transient next-QK FP32 scores.
Both WGs execute retained PV256. Initial QK is split over keys; later QK occurs
once in the lookahead WG. No extra QK traversal or full PV on rejected tiles.

The first full-D512 build spills164-176bytes. Preserve this resource-negative
candidate, while attempt009 reduces per-CTA output channels to256 to remove
spills and improve occupancy. No native benchmark, calibration or speed claim.
