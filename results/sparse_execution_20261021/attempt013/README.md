# Low-word runtime descriptor arithmetic, 2026-10-21

Synthetic-only optimization following attempt011; no static shared-base
assumption. The descriptor's high32bits are the canonical SW128 stride/layout
constant. Only its low32bits advance along QK channel tiles, avoiding repeated
64-bit carry arithmetic. Runtime Q/K shared addresses are preserved.
qualification001 passes80changed-input graph replays with exact masks and
relativeL2<0.003. Complete8K/50% timing under refresh001 charges fresh projection,
pack, masks and actual capture counts. No native evaluation or calibration.

refresh001 complete8K refresh433.80us vs359.28us dense: no improvement over
the best interior specialization429.32us. Actual timed capture count0.
