# Immutable profiling recovery — 2026-10-18

This addendum applies to the bounded study in
`REFRESH_DIAGNOSTIC_PLAN_20261018.md`. It authorizes no expansion of the cohort,
native generation limits, factorial requests, or 7,200-second campaign cap.

## Observed failure and repair

Attempt001 completed its native qualification and all four clean timing engines.
The first profiled engine (engine006) then failed during model initialization,
before any request record, because an attention hook accessed `collector.phase`
before `begin_request`. Its failed leaf, source snapshot, state and charged GPU
reservation remain intact. No result is imputed for the failed profile.

The repaired collector initializes inert state and makes inactive hooks call the
original implementation directly. Model initialization, graph capture and the
excluded implicit warmup must neither inspect request counters nor record spans.
CPU regressions exercise installation and invocation before the first request.
The diagnostic attention implementation, native worker, sampler, selectors and
benchmark inputs remain byte-identical to the qualified source.

## Successor provenance

Create attempt002 after committing the repair, coordinator/report support and
this addendum. Freeze a new source snapshot and protocol before generation.
Reference the five completed parent engines explicitly; do not copy, relabel or
rewrite their artifacts. Each reference binds the parent attempt, leaf, protocol
hash, source commit and completion-marker hash. Validate every marker-bound file
and the matching cohort, schedules and native configuration before accepting it.
Report the original source identity for each referenced engine.

Only completed qualification and clean timing engines can be inherited. The
failed profile and every remaining engine run in fresh successor leaves. The
inheritance check permits changes only to profiling, coordinator/report tooling,
their tests and this addendum. Any change to active native execution or the
factorial worker/runtime invalidates inheritance. The earlier qualification is
therefore evidence for unchanged factorial code; it is not a new-source run.
Every factorial group still has its own original-V31 bookends and dense_fresh
no-op control, with the same strict native receipt and prefix-matching checks.

## Explicit reserve allocation

Allocate 300 seconds of the original 900-second reserve to headroom recovery:
qualification 900, headroom 1,800, factorial 3,900, reserve 600 seconds. The total
remains 7,200 seconds. All attempts use the same campaign reservation ledger,
including the failed worker and original completed work. This is a documented
reallocation before successor generation, not an automatic budget extension.
No phase borrows additional time if it runs out.

All original exposure, missingness, accuracy, timing and research-triage rules
remain in force. In particular, incomplete seed/engine coverage is unresolved;
the cohort is not shortened to fit a favorable outcome. Profiled timing is not
clean latency. The older full503 campaign remains paused.
