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
