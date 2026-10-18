# Bounded runtime headroom and refresh diagnostic

Authorized and executed 2026-10-18. The frozen plan is
`docs/REFRESH_DIAGNOSTIC_PLAN_20261018.md`, with recovery and external-profile
addenda. Read [the consolidated report](analysis/attempt001/summary.md) for
coverage, findings, go/no-go decisions and evidence links.

Execution has finished: 288 factorial measurements (48 six-arm groups), 20 clean
runtime measurements, shared/FULL diagnostic profiles, and one external trace.
The four factorial arms cross dense/sparse current output with fresh/held future
support at one genuine progress refresh, retaining V31 bookends. Native sampler,
dense prefill/commit and input generation limits are preserved; sparse-method
arms retain fixed K and dense LOCAL routing.
These oracle interventions are diagnostic and do not measure deployable speed.

The external trace reproduces all eight measured original-clean requests and
qualifies request/prefill boundaries. Refresh-specific critical-path savings
remain unresolved. Future support has the stronger provisional mechanism signal,
but repeatability limitations prevent causal promotion and no new method passes
the quality/runtime gates. See the report for uncertainty and per-case evidence.

Total charged reservation is 5423.862923 seconds (90.40 minutes) of 7200 seconds,
including initialization, warmups, failures and external processing. All
reservations are closed; GPU idle was verified. The final allocations were
qualification 900 s, headroom 1800 s, factorial 3900 s and reserve 600 s.
No further GPU work is queued. The prior full503 campaign remains paused.

Attempt001 preserves the profiling startup failure; attempt002 preserves a
CPU-only preparation failure. Attempt003 source `db755c43` explicitly references
attempt001's completed qualification/clean engines under original `6aab8632`
source and protocol hashes; native/factorial code is unchanged. The external
runner `2f5687d4` executes the original clean worker with copied private caches.

The all-kept profile was skipped unstarted under its phase budget. Therefore
attempt003 remains `budget_limited_partial` despite complete factorial coverage,
and has no global `final_complete.json`. Each completed leaf retains its own
source, protocol, receipt, score and completion marker. The consolidated
`analysis_complete.json` marks report completion only. Main diagnostic CPU tests
(67), external runner tests (20), and independent artifact/analysis checks passed.
Raw generations, input pools, caches and profiler traces remain private/ignored.
