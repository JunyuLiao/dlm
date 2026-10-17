# Full503 LongBench-v2 seed42 value-method performance

User-authorized new campaign. Clean timing is enabled for this study; historical
paused/held campaigns remain unchanged. Source ec87d397 freezes the experiment
coordinator. All eight native qualification engines and both official smoke-scoring passes
are complete. Full503 generation is running on PTY25199. Full-panel accuracy
and end-to-end performance are not complete yet.

Methods: full-value bounded swaps (`full_swap8`, `device_exact_v1`) and
direction-guarded residual (`residual_direction_guard`, same backend) change only
GLOBAL layer29. The inherited Yuhan V31 progress/carry/sticky policy remains the
control. Every head/Q128 tile retains prefix K128 KV64 blocks plus common
mandatory/carry additions. LOCAL and the native sampler remain unchanged.

All503 input identities use panel seed42 and the established request-seed hash.
The source schedule had seed1; the new schedule explicitly sets42. Inputs are
otherwise identical:237 contexts were already capped in the inherited manifest,
with prompt lengths10,285–120,017 and output budget16,384. No extra truncation.
This is full question coverage, not an untruncated-context evaluation.

The detailed pre-generation contract is
`docs/V31_VALUE_E2E_LONGBENCH503_PLAN_20261017.md`. New native frozen attempt:
`native_attempt001/`. It contains16 index-strided shards, separate clean/audit
passes, rotating common-engine candidate order, two V31 control bookends, and
independent FULL dense, PIECEWISE dense and all-kept FA4 references. Both passes
receive official scoring. No latency conclusion is derived from audit timing.

## Optimization evidence

`profile_reference/attempt002` measured existing selection overhead:26 host scalar
reads and10 dynamic nonzero calls per head/query tile. Its predecessor attempt001
failed at bootstrap state selection and remains preserved.

`qualification_fast/attempt002/tests.xml`:86 H100 tests pass, no skips/failures.
`replay_fast/attempt002`:256 real head/query-tile units from eight layer29
selection states. Exact device backend:zero support, swap-count, budget, moment,
or dense-reduction mismatches. It removes selection host reads and dynamic
nonzero. Per-unit host medians are4.083 vs5.015ms and5.406 vs6.626ms (about19%
less). Fast moments with shared preconverted K/V are1.766/0.671ms for the two
recorded geometries. These are isolated diagnostic measurements, not latency
speedups. The factored Gram objective is rejected:slower with one mask mismatch.

The optimized native runtime shares FP32 K/V across query heads/tiles, omits
unused RMS/tangent/correction arrays, uses fixed-shape device disjoint ranking
and joint admission, and removes redundant all-true gathers. The existing fused
held-step correction is retained. No asynchronous stale update or new sampler
rule is introduced. Native qualification compares reference and optimized masks
and correction caches at the same actual observation states.

## Reproduction and resumability

Use the pinned interpreter and frozen execution, not the evolving worktree:

```
PYTHONPATH=<attempt>/private/execution/src:<attempt>/private/execution \
/home/exouser/dyh/v31_env_20261005/bin/python \
<attempt>/private/execution/scripts/v31_value_e2e_study.py \
--stage run --study <attempt>
```

`--stage qualification` stops after native clean/audit qualification and scoring.
`--stage run` verifies/reuses completed jobs and completes the full panel.
Completed engine attempts have hash-bound `attempt_complete.json`; failed jobs
remain intact and a resume creates the next attempt number. Source/input/runtime
identities must match. Only a final completion marker permits a complete claim.
Generation records, raw completions, input manifests and runtime caches are
private. Public numeric receipts and aggregates are selected explicitly for Git.
