# v11 preselected-support Hopper consumer: contract

Separate ABI (`support_consumer.h`, ABI 1). ATen namespace `vd_support_v1` is disjoint from Junyu's
`value_direction_hopper`, whose ABI v4 and fresh-T kernel are untouched. Machine-readable: `support_contract.json`.

**Computes** O = softmax(current transformed QK restricted to KEPT (= eligible && !skipped) and LEGAL
entries) @ current V, per query head, on the Q128 x KV64 grid. The support map is a read-only input produced
by the caller on the same stream.

**Work avoidance.** Both roles read the support entry of tile j BEFORE any work for j. A dropped tile issues no
K cp.async, no QK WGMMA, no V cp.async and no PV WGMMA. There is no projected-V work and no dense QK. Evidence:
executed-path counters, NaN poisoning (bit-identical output) and linear time scaling
(`executed_work_counters.json`).

**Legality.** In-kernel query-relative window, or an optional packed row mask. A kept block does not make
illegal positions legal.

**Invalid scores.** NaN/+inf among legal retained scores flag the row and zero its output. Tiles dropped by the
map are not computed (same declared contract as the Triton preqk consumer).

**Arithmetic** is fresh T's retained path. Output is bit-identical to fresh T when fed fresh T's own support.
It is numerically different from the frozen Triton consumer (see `kernel_qualification.md`).

**Pipeline.** One producer warpgroup (K load, QK, softmax, P) and 1-2 PV warpgroups; no cluster (the decision is
an input). One non-aligned `barrier.sync` hand-off per kept tile. P/scale are double-buffered by kept ordinal.
This is a non-TMA bring-up: K/V use cp.async, while fresh T keeps TMA.

Build `e3c283b8cbb9b79c` (nvcc 13.1 sm_90a + g++ ATen bridge), with hashes in `support_contract.json`.
The first build `1ed1d85f3d1ab49b` failed synccheck and is superseded.
