# Bounded runtime headroom and refresh diagnostic

Authorized 2026-10-18. The preregistered design is in
`docs/REFRESH_DIAGNOSTIC_PLAN_20261018.md`.

This campaign owns only its new attempts and GPU-budget ledger. The previous
full503 value-method campaign remains paused. Native sampler behavior and input
generation limits remain unchanged. The total GPU reservation cap is7200s,
including initialization, warmups, failed workers and successor attempts.

The four diagnostic policies cross current dense/sparse output (sparse uses the
common pre-refresh map) with future fresh/held support. These instrumented runs
are not speed measurements. Separate clean and profiled headroom controls use
inherited V31, fixed16, corrected FULL dense, PIECEWISE dense, and all-kept FA4.

Each attempt has its own immutable protocol, source snapshot, input bindings,
worker receipts, official scored outputs and completion markers. `state.json`
reports partial coverage and budget use. An analysis completion marker never
stands for campaign completion. No performance or accuracy gain is established
by the design or CPU tests alone.

Active execution is attempt003, source `db755c43`, after the preserved profiling
startup failure in attempt001 and CPU-only preparation failure in attempt002.
The recovery contract is `docs/REFRESH_DIAGNOSTIC_RECOVERY_20261018.md`.
Attempt003 explicitly references attempt001's completed qualification and clean
timing engines, retaining their original source/protocol hashes. New profiling
and factorial leaves use the successor source; native execution code is unchanged.

The recovery allocation is qualification 900 s, headroom 1,800 s, factorial
3,900 s and reserve 600 s. The shared 7,200 s cap includes the failed worker.
All 67 diagnostic CPU tests pass. Native qualification and 20 clean headroom
measurements are complete. Shared/FULL profiles completed, but all six profiled
requests have different trajectories from their clean counterparts; clean
critical-path attribution remains unresolved. The all-kept profile was skipped
unstarted under the phase budget. Factorial engines are running under the
unchanged 3,900-second cap. A one-shot external trace of the original clean worker
is planned within the remaining 600-second reserve after factorial completion.
These statements are a checkpoint, not a final result. Use the active attempt's
state and final report for coverage and completion.
