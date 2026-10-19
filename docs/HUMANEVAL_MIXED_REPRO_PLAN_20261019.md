# Remaining mixed-method native reproducibility

The two residual diagnostics in humaneval_residual_repro_20261019/attempt001
are complete and their observed/unobserved residual bookends agree. Their184
sampler calls and every production-cache/correction fingerprint agree across
audit/clean. Both reproduce the earlier clean result, not the earlier audit
result. The original residual discrepancy was not captured internally.

Comparing the unobserved prefixes against their original independent engines
finds changes in the audit residual target, clean measured V3a question116,seed42,
and clean V31 question129,seed43. One clean full-swap warmup also changes. Thus
the remaining issue is not established as residual-specific. All recorded
Inductor kernels still report deterministic settings. The earlier six-question
baseline diagnostic remains complete within its scope, not a universal guarantee.

Freeze a new diagnostic attempt. Run two independent clean engines with identical
warmups, source, request order, K256, native settings and numerical policy. Keep
the same original clean prefix through question129,seed43. Repeat that complete
six-arm group twice: observed then unobserved. This gives per-arm unobserved /
observed / unobserved repeats, plus both V31 bookends inside every group. Do not
attribute an observed difference to native behavior until its instrumentation
relationship is assessed. Keep every failed or inconclusive contrast.

Augment the qualified sampler-state observer with a two-seed128-bit integer
fingerprint of every raw logit word. The new GPU diagnostic performs no floating
arithmetic, RNG draw or forward. Fixed-shape integer reductions fold modulo2^64;
an independent NumPy oracle, repeated reads and a final-word bit change qualify
its implementation at short, partial-block and multistage lengths. Retain only
two words per sampler snapshot. This noncryptographic locator cannot prove
equality; a detected change requires targeted actual tensor comparison/replay.
It does not alter native logits or replace official output/scoring hashes.

Trace the first unequal logits, pre/post sampler state or production residual
cache. Distinguish model arithmetic from RNG/control-flow divergence. Keep the
full accuracy matrices and K128 queue held. No compiler/library flag or sampler
change is promoted based only on a favorable final output.

At most two750-second native reservations, plus the separately receipted90-second
maximum fingerprint preflight, charge existing phasek256 within its16200s cap.
The total43200s allocation remains fixed. No automatic gate relaxation or
unbounded diagnostic loop. Additional probes need a new evidence-supported
bounded stage with explicit remaining-budget accounting. Neither diagnostic
timing nor partially reproducible native results support a clean speedup claim.
