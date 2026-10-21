# Local Q128 interleaved routing, 2026-10-21

Synthetic-only follow-up to the preserved sparse_execution attempt003. This
directory is unrelated to the paused optimized_refresh_benchmarks attempt004.
Two interleaved QK producers, one Q128 router and four output-channel consumers
use256threads per CTA, with two query warpgroups. The Q128 vote is local to the
router CTA. Seven role CTAs per Q128 unit,224blocks at the primary geometry,
fit a cooperative grid with two CTAs per H100 SM. Device queues are bounded
and every QK tile is computed once; full PV consumes retained blocks only.

qualification001 passes80changing-input graph replays, exact masks and maximum
relativeL2<0.003. Primary8K/50% refresh timing is frozen under refresh001.
No native workload calibration, model speedup or completed benchmark claim.

refresh001 complete8K refresh770.74us vs361.80us plain dense is negative.
Actual timed CUDA capture count0; no other GPU worker during the screen.
