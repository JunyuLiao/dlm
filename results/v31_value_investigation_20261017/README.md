# Bounded investigation of paused full503 value methods

The investigation is complete. Both current methods receive a **no-go for
full503 continuation**. The parent benchmark remains paused and incomplete;
no further worker is queued. Its completed shards, private generations,
frozen source and protocols remain intact.

Start with [the findings and recommendation](attempt001/summary.md).
The result has an independent [receipt](attempt001/receipt.json) and
[completion marker](attempt001/final_complete.json). That marker completes
this investigation only, not the parent benchmark.

- [Committed staged plan](attempt001/PLAN.md), including hypotheses, controls,
  decision gates and the60-minute diagnostic GPU target.
- [Partial-cohort analysis](attempt001/CPU_FINDINGS.md): official scores,
  native trajectories, tails, paired uncertainty and existing reuse-age evidence.
- [Crossed diagnostic](attempt001/CROSSED_FINDINGS.md): two inputs, two seeds,
  reference/device implementations, four clean/audit engines,96 measured requests.
- [Native first-divergence trace](attempt001/TRACE_FINDINGS.md): one exploratory
  four-arm residual trace, with its pre-run [addendum](attempt001/TRACE_GATE_ADDENDUM.md).
- [Cost breakdown](attempt001/cost_analysis/attempt001/summary.json) and
  [saved-state graph qualification](attempt001/cost_profile/attempt003/summary.json).
- [GPU accounting](attempt001/gpu_accounting.json):45.55 conservative minutes,
  including retained failed profiler attempts; no active GPU process at sealing.
- [Guarded parent resume procedure](../v31_value_e2e_longbench503_20261017/native_attempt001/PAUSE_AND_RESUME.md).
  Resume requires new explicit user authorization.

Inherited V31 cross-step reuse is Yuhan's baseline. Full-value swaps and
direction-guarded residual are separate study variants at GLOBAL layer29.
Fixed per-query-tile budgets, dense LOCAL routing and native decoding are
preserved. Private state tensors, prompts, completions and gold remain under
ignored private directories; public evidence contains numeric records and hashes.
