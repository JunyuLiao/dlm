# Balanced scalar reductions, 2026-10-21

Synthetic-only specialization of the best interior/tail kernel. Four independent
local max chains replace one16-element chain, then combine before the same
quad maximum. Warp risk loads use eight lanes and shuffle reductions, with the
same Q128 decision. QK/PZ/PV arithmetic and threshold semantics are unchanged.
Build uses229-230registers without spills. qualification001 passes80changed-
input graph replays, exact masks and maximum relativeL2<0.003. Complete8K/50%
timing under refresh001 charges fresh projection, pack, masks and actual capture
counts. No native evaluation or workload calibration is performed.

refresh001 complete8K refresh430.84us vs363.15us plain dense, effectively a
tie with interior specialization429.32us across separate screens. No promotion.
Actual timed capture count0; no other GPU worker during the screen.
