# Q32 packed split-key occupancy experiment, 2026-10-21

Synthetic-only follow-up to attempt007. Four Q32 CTAs implement one Q128
decision; WGMMA still computes M64 but inactive rows are masked, not written.
Two WGs split32key columns and own256output channels, so each valid QK tile is
computed once.128CTAs at Q256/H16 improve device coverage at the expense of
inactive tensor rows and four-way cluster voting. Packed current Gaussian32,
3D TMA, base2 softmax, online threshold and full-PV skipping remain unchanged.
Qualification and8K/50% timing must pass before any promotion; preserve every
negative. No native evaluation, workload calibration or full benchmark launch.

qualification001 passes80changed-input graph replays with exact masks and
relativeL2<0.003. refresh001 complete8K refresh1270.90us vs361.98us dense is a
negative. Actual timed capture count0. Inactive rows and four-way voting cost
more than the added scheduling parallelism saves in this implementation.
