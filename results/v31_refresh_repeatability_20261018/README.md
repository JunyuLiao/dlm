# Completed native repeatability and refresh attribution follow-up

Read [the final analysis](analysis/attempt001/summary.md) and its source-bound
completion receipt. Authorized 2026-10-18 under a 45-minute reservation cap.
Actual usage: 1334.421052 seconds (22.24 minutes), including the preserved partial
replay. All workers finished; GPU idle is verified. No selector was changed.

The numerical sample gate passes: six captures / three unique first-refresh
states, 192 measured operator replays across two processes, exact output and
active-intermediate agreement. This does not explain the old full-engine drift.
Runtime attribution passes: eight clean and eight annotated measurements match
the original clean trajectory, with complete phase scopes and independent cost
reproduction. Progress refresh has only 2.4–3.0% gross request headroom on dynamic
V31 here; no credible ≥10% net opportunity is established after replacement cost.
No new selector trial is queued. Older full503 remains paused.

Frozen plan: docs/REFRESH_REPEATABILITY_PLAN_20261018.md. Native source 6aab8632;
wrappers 62ae618b; bootstrap correction eaf69a67; CPU-only recovery 9d864986.
Attempt001 preserves capture/clean/NVTX, the two unstarted replay leaves and
protocol_description_correction.json (capture 329 is 36K; runtime 339 is 41K).
replay_corrected/attempt001 preserves the CPU-slow partial replay;
replay_corrected/attempt002 contains both complete replacements. Addenda specify
all changes and the explicit use of the existing recovery reserve. No frozen
source, protocol, result or original cache was overwritten.

Twenty-nine relevant CPU tests and independent source/snapshot/trace checks pass.
Private tensors, inputs, generations, raw profiler traces and caches are ignored.
The final completion marker describes this bounded diagnostic only, not either
older campaign, a full accuracy benchmark or a production method improvement.
