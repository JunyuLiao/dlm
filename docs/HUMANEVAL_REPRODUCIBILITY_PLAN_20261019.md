# HumanEval execution reproducibility and K256/K128 evaluation

The user requests investigation and repair before another full matrix. Preserve
the original K256 study, older paused studies, native sampler, consumers and
selector definitions. Starting checkpoint: 63ccf1e6. This is a new study under
results/humaneval_reproducibility_20261019; all raw inputs/tensors stay private.

## Staged reservation and gates

Self-imposed maximum: 43,200 H100 reservation seconds including startup, warmup,
failed attempts and verification. Initially release only 7,200 seconds for
diagnosis/qualification. Corrected K256 and K128 each have 18,000 seconds. No
phase borrows from another; unused time is not an instruction to sweep. One
worker, idle-device check, immutable attempts, 30-minute progress checks.

1. Freeze original artifact hashes, current/native source, runtime/model/GPU,
   exact environments/commands, warmups and schedules. Select six diagnostic
   questions using prompt lengths and original trajectory divergence only;
   both thinking modes, seed42, no accuracy selection.
2. Dense FULL and V31: identical native request order and warmups, two processes
   per pass with crossed audit/clean order. Start with compact, deferred tensor
   snapshots and CPU/CUDA RNG fingerprints; compare to uninstrumented outputs
   to qualify the observer. Find first difference in initial canvas, input
   logits, sampler state and post-sample canvas/self-conditioning. Expand only
   the first differing operation. Different consumers need not agree.
3. Isolate per-step synchronization, geometry, finite checks, trace and CUDA
   events independently. Inspect request reset, graph replay/private generators,
   asynchronous H2D/buffer lifetime and compiled sampler. Preserve every outcome.
   Name the cause only if targeted contrasts and a correction reproduce it.
4. Smallest correction, focused regression tests, new source identity. Require
   same-method output/trajectory equality across processes and audit/clean,
   zero measured captures, fixed budgets, GLOBAL-only changes and native
   settings. Failing qualification blocks both full matrices. If exact equality
   proves unattainable, report mechanism/variation and freeze a justified
   statistical protocol before generation; no silent criterion relaxation.
5. After qualification rerun K256, then K128 (two KV64 prefix blocks per
   query-head/Q128 tile), all164 questions, seed42, thinking on/off. V31, V2,
   V3a, full swaps and direction guard; FULL-fix51994, PIECEWISE, all-kept.
   Work-only V2 calibration per budget/mode; original native/output settings.
   Keep audit and clean separate, verify their trajectory identity before
   associating sparsity with latency. Use resumable 82-question shards and
   independent-process replication on the six-question cohort for every arm.
6. Predeclared additional seed43 on the same six-question cohort, every method,
   both modes/budgets; descriptive only. No paper-level superiority claim from
   seed42 or this small seed check. Official scoring and independent rescoring,
   sanitized per-item metrics, paired question intervals, tails, C/T/N/N:C,
   S/W/S:N, full GLOBAL/LOCAL denominators and overhead; paper table/paragraph.

Any incomplete or failed gate remains explicit and resumable. Do not launch
the expensive matrices merely because all diagnostic jobs terminated.
