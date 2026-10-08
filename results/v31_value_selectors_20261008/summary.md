This study is incomplete. No target accuracy or end-to-end claim is available.

The independent V1/V2/V3 implementation passes 40 H100 formula and integration
tests. V1/V2 have distinct skip-mass state updates; both use a uniform threshold
and no quota. Singleton scores stay fixed, with row-support constraints. Exact
greedy uses cached summaries after each deletion. Batch8 is a separate
approximation with a support-safe single-deletion fallback and exact cleanup.
The later native FA4 consumer and inherited refresh/carry machinery are unchanged.

The completed 15-item development control realizes 76.45235% GLOBAL decode
sparsity over all calls, including exact observations and current canvas tiles.
This measured value is the calibration target. Thresholds remain unfrozen.
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
split. Prior pool exposure is unknown. RULER v33 and HumanEval generation remain
blocked by missing exact frozen manifests and cell schedules; no substitute
prompts or schedules are generated. The study has no final completion marker.
