# Optimized native boundary localization, same declared contrasts

Use the same158-request schedule as graph attempt009, twice independently:
80 full sampler-observed requests,72 graph/logit-only repetitions in their
original order,6 pure native bookends. Complete seed43 light cases and pure
bookends that were absent when009 expired. No new correctness-driven cohort,
threshold, method, budget or generation setting is selected.

Relative to009, use the checksum's runtime-length implementation and batch its
small uint64 results into one request-finished CPU transfer with signed-bit views
and unsigned restoration. All12 CPU/H100 checks pass in13.898s under
humaneval_dynamic_fingerprint_preflight_20261019/attempt003: independent NumPy,
original-kernel equality, repeats, tail-bit sensitivity, actual CUDA high-bit
batch transfer, native wrapper/capture/binding/scalar/drain checks. Runtime cache
receipts show one words and one fold variant on the test sizes. Source-bind the
exact two tested files. This is an observation variant, not a model repair or
an inference speed result. Never infer equivalence merely from equal checksums.

Keep native graph calls exactly once, no bypass, no extra native model forwards,
original sampler Python helpers and the explicit compiler policy10341ab2. The
rejected global Torch guard/BLAS/adapter candidates are not promoted. Interpret
full and light observation precisely as in HUMANEVAL_OBSERVER_SEMANTICS_20261019.md.
Retain partial-argument/padded-storage limitations. Require actual tensor capture
and same-input operator replay before a causal operator claim.

Freeze a new immutable attempt. At most two1500-second native reservations,
charging phasek256 after the explicit continuation allocation is bound. The
total43,200-second ceiling is unchanged, as are historical attempt limits.
Record all failures/startup/warmups and actual GPU reservation seconds. Do not
launch full accuracy on a failed or merely matching bounded diagnostic. K128
and older campaigns stay held; no technical-repeat criterion is relaxed.
