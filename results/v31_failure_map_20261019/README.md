# V31 budget-versus-convergence failure map

Three-stage plan: `../../docs/V31_FAILURE_MAP_PLAN_20261019.md`.
Native execution source: `5dcade2c27bbc8092bdbc94b231f9e324b0bcaee`.
Hard cap: 14,400 H100 reservation seconds, including startup and warmup.
The shared `gpu_budget.json` accounts for all attempts.

- `attempt001/`: frozen inputs/source and completed V31 qualification at
  K8192/4096/2048. Its qualification phase stopped at the predeclared remaining-
  time gate. Preserve this partial attempt.
- `attempt002/`: explicit continuation under the committed recovery addendum,
  completed dense-control qualification with the existing 900-second reserve.
  Completed parent receipts are verified and referenced, not rerun. A subsequent
  relative-path launch failed before model import (zero records; one second).
- `attempt003/`: current clean screen and gated continuation, using absolute
  paths and all verified inherited qualification receipts.

Consult each attempt's `state.json` and phase completion markers for live status.
The native protocol is unchanged across attempts. Screening is 16 development
LongBench questions, seeds1/42, three K budgets and matched dense controls.
Only independently confirmed cases advance to same-budget repair diagnostics.
A `final_complete.json` is required before treating the new campaign as complete.

Private pools, raw generations, logs and execution caches remain under ignored
`private/` directories. Public evidence is sanitized. Older paused campaigns
are not resumed by this study.
