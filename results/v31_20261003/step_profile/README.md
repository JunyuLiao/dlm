# v31 step profile (diagnostic), 2026-10-03

`scripts/v31_step_profile.py`, PIECEWISE + FIX_51994, overlay3 adapter, 3 cells (+1 warm-up, first line of each
file) per arm: mpk = LongBench-v2 64K, dlm2 = 32K. Every intercepted GLOBAL call, the FA4 sparse call, the K/V refresh
and the sampler hook are bracketed by `torch.cuda.synchronize()`, so CPU time is exposed and async overlap removed;
use these numbers for attribution, not as request latency. Print with `python scripts/v31_prof_show.py <dir>`.
Each line: per-kind GLOBAL-call aggregates (`calls_by_kind`) and `step_list` = [step ms, GLOBAL ms, hook ms, kinds].
