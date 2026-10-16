# Replicated native value-refinement evaluation

Frozen source b485866c; parent execution 28714b4a/base79c994ca, plus the
explicit coordinator/test overlay listed in config.json. Method kernels and
native sampling are unchanged. The 71 H100 method tests and 15 coordinator
checks are separately source-bound.

Two common engines per cell; V31 controls before and after four rotated
candidates, with reversed candidate order in replicate2. All structurally
valid outcomes count, including drift. Scores average repeated evaluations;
there is no answer ensemble. Use the mean of both controls. Dense FULL
fix51994, PIECEWISE and all-kept are separate fresh reference paths.

Cohorts: AIME90, LongBench64, MATH100, RULER104. All four workload qualification
stages precede accuracy. Fixed prefix K27 short / K128 long per head and Q128
tile; common mandatory/carry work separate; value changes only at GLOBAL
layer29; LOCAL dense. Clean timing remains paused.

The completed warmup factor found control drift in4/18 control-warmup groups
and2/18 all-method-warmup groups. Every drifting group was the long AIME
item10. This small diagnostic does not establish a warmup benefit or a runtime
repair. The all-method warmup policy was selected before factor results.
The original V31-only history also drifted without any value intervention.

Plan: docs/V31_REPLICATED_NATIVE_PLAN_20261016.md. Older independent-process
and exact-parity panels remain held under their original protocols. This is
a new named comparison design, not a retroactive gate change.

Run/resume using the frozen coordinator and only its execution source on
PYTHONPATH. Completed engine shards verify hashes and are skipped. Interrupted
or structurally failed workers receive new immutable attempt directories.
Output drift does not cause a retry. Check approximately every30 minutes.
One worker at a time; do not resume other paused coordinators.

At03:38 UTC all four two-engine workload qualifications passed with official
scoring. Replicated accuracy has started; no complete accuracy workload yet.
The report-path CPU dry-run passed repetition arithmetic and deterministic
resume using qualification schema; its temporary smoke reports were removed.
