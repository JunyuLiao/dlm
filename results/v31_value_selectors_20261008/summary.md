This study is incomplete. No target accuracy or end-to-end claim is available.

The independent V1/V2/V3 implementation passes 43 H100 formula and integration
tests. V1/V2 have distinct skip-mass state updates; both use a uniform threshold
and no quota. Singleton scores stay fixed, with row-support constraints. Exact
greedy uses cached summaries after each deletion. Batch8 is a separate
approximation with a support-safe single-deletion fallback and exact cleanup.
The later native FA4 consumer and inherited refresh/carry machinery are unchanged.
The V2 pilot failed in warm-up above 128K in the inherited whole-sequence RMS
kernel. It has a retained failure receipt and zero records. A bounded RMS
reduction and reusable selector-only scratch preserve the current-value scale,
projection identity and decisions in independent tests. Model retries are pending.

The completed 15-item preliminary control realizes 76.45235% GLOBAL decode
sparsity over all calls, including exact observations and current canvas tiles.
It is excluded from calibration: 11 items belong to the original holdout. A
supersession note preserves this deviation. Thresholds remain unfrozen; the
calibration reference will come from a completed audited run on the existing
32-item S1 development schedule with seeds 1 and 2.
The one-cell V1 smoke at diagnostic threshold 0.001 passed official binding,
reported zero invalid rows and zero timed CUDA captures, and installed no
LOCAL router. This cell is not quality evidence. New source qualification will
repeat every arm and matched dense/all-kept controls.

At 128K, exact greedy selection remains expensive (roughly 0.38 s per layer).
Synthetic batch8 selection is faster but has higher sketch error on the measured
sample. Timings, matrix identities and approximation error are recorded in the
qualification JSON files; none establish request speedup. Attempt004's
incremental-memory field included the later diagnostic calculation and is
explicitly superseded. Failed attempts remain intact.

LongBench uses the exact supplied official 503-item manifest, seed 1, native
thinking and total cap 16384, with a pre-generation 15/488 development/target
split in the superseded pilot. The restored protocol uses the original 160/343
exploration/holdout definitions and the exact original S1 development schedule;
target evaluation retains the full official 503-item, seed-1 schedule. Prior
pool exposure and the 11 pilot holdout exposures are disclosed. RULER v33 and HumanEval generation remain
blocked by missing exact frozen manifests and cell schedules; no substitute
prompts or schedules are generated. The study has no final completion marker.
