# User resource authorization amendment — 2026-09-26

The user explicitly said: “不必在乎预算 没事 gpu是自己的”. This is a later
instruction about resources, and takes precedence over the original v18 resource
ceilings. The scientific scope and native generation semantics remain frozen.

Automatic approval review rejected removing every finite ceiling because that
could authorize unbounded computation. The bounded alternative below was then
approved by the automation tool. We do not bypass the rejected unlimited mode.

- Existing fixed panels only: primary RULER/AIME, secondary70 and allocation
  controls; no CVM expansion and no revival/replacement of the stopped candidate.
- Wall start remains 2026-09-26T18:02Z; finite campaign limit becomes36h.
- GPU work cutoff becomes2026-09-28T05:32Z; final handoff by06:02Z.
- GPU-process allowance24h/host,48h combined; at most2 qualified GPU workers.
- Keep7000 generated attempts combined and3500/host, all prior usage charged.
- Keep at most5 independent density-only calibration policy pairs per arm.
- Native stopping, thresholds, output budgets, question/seed/arm assignments,
  per-request operational watchdogs, warm acceptance and scoring do not change.

The original campaign_budget.json and deployed workers are immutable. The new
campaign_budget_extension_20260926.json preserves original accounting baselines,
records the original budget SHA and explicitly marks deadline_epoch as the GPU
cutoff (the scoring reservation must not be subtracted a second time).

Do not interrupt active AIME requests to apply this amendment. A newly committed
controller may adopt those workers. If the old finite budget produces a verified
clean stop between complete blocks, it may launch a uniquely named continuation
segment using the new budget, the same frozen protocol, GPU assignment, private
receipt directory and append-only ledger. It must check that every outstanding
block is wholly unstarted, that all old writers are closed, and that no first
result is replaced. Failed/uncertain launches or partly executed blocks require review;
this amendment never authorizes automatic retries or repeated quality attempts.
A failed first-result row in an otherwise completely recorded block remains an
occupied cell; it is never rerun, and does not by itself block later new blocks.

Generation workers must remain gold-free. Complete primary scoring before CP3;
continue recording resource use and all missing/failed cells. End the heartbeat
when CP4 is complete, or stop at the finite amended limits with honest partials.
