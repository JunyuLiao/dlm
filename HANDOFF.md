# v25b (aligned16 exposure audit, guards, odd-K bridge) — complete

Authority: user v25b doc; started at `af4ddad3`. GPUs: user standing authorization.
Cumulative 18,216.6 GPU-process s; executions 510 campaign (cap 512 not reset) + 16 v25b bridge allocation.
No active jobs; GPUs 0 MiB.

## Results (R = results/m3_numeric_trajectory_bridge_20260927/v23_bootstrap6)
- `R/v25b_memory_and_gate_audit.md` (main report, corrections to v25 wording).
- CP0 `R/v25b_alignment_exposure.{csv,json}`: 90/90 phase + counter conservation; pilot LB inputs KDIV 4/8
  (never odd-K); AIME/RULER odd-K but tiny routes.
- CP1 guards (`cac0cbd4`): physical score-cache accounting + strict qualification gates; tests 48 pass.
- CP2 bridge `v25b_aligned_bridge_2dafaf7cd531bff3` 16/16: odd-K LB aligned/logical warm 0.9742 / 0.9739
  (tokens identical); KDIV-8 control 1.0011 / 1.0017. `R/v25b_bridge_scored.*`.

## Next (user decision)
Optionally make aligned16 the default for odd K (KDIV-2 class unmeasured). The method question (M3 vs T/B) is unchanged.
