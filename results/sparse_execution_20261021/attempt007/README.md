# Packed split-key unique-QK execution, 2026-10-21

Synthetic-only refresh optimization with no benchmark relaunch. Q64 per CTA,
two warpgroups split each KV64 QK into two32-key products. The full tile softmax
uses a shared maximum and sum. One WG computes full Gaussian32 PZ and the
preserve-mass router; both WGs own256output channels and read the complete
shared probability tile for retained full PV. Q128 admission uses paired-CTA
release/acquire votes. Current packed Gaussian32, 3D TMA, base2 SFU, folded
unnormalized risk, early V and next K are named execution changes. Every QK
tile is computed once; no dedicated global queue or extra model forward.

Build uses251registers, no spills. qualification001 passes80changed-Q/K/V/Z/held
CUDA-graph replays, exact ordered masks and maximum relativeL2<0.003.
Primary8Kkeys/50% actual sparsity timing is frozen in refresh001, with projection,
RMS, sketch packing, compact masks and actual capture counters. A numerical
qualification is not a native request speedup or completed benchmark result.

refresh001 complete8K refresh583.81us vs359.63us plain dense is a negative.
Actual timed capture count0; all current projection/packing/list costs charged.
