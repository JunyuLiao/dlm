# Attention-level localization of the remaining native differences

Full-cohort trace attempt002 completed877.086/867.093s. Across72 observed
same-method process contrasts, two differ. V3a question116,seed42 first changes
raw-logit fingerprints before sampler call229 (zero-based), step5. V31's
control-after on question116,seed43 first changes before call128, step11.
Both are denoising calls with valid canvas length256; initial state and all
earlier sampler states/logit fingerprints match. The first changed before-field
is logits; default CPU/CUDA RNG fingerprints still agree. This localizes the
recorded difference upstream of that sampler update, not to a changed RNG state.

Both jobs also have observed/unobserved output/trajectory differences, so their
instrumentation relationship is not globally qualified. Preserve these failures;
do not present the trace as proof of a native causal operator or promote a repair.
First-difference integer hashes need actual tensor confirmation/replay.

Freeze a targeted attempt003 with two identical clean engines. Keep the original
80-request mixed schedule unchanged and append one unobserved repeat for each
of the two affected cases (82 requests). Observe the original V3a116/42 and
V31-control-after116/43 only. Retain full sampler/logit/cache fingerprints and
add per-layer Q/current-K/current-V/output integer fingerprints, GLOBAL logical
cache fingerprints, actual FA4 block lists and observation masses, plus progress
state/trigger identity. Do not hash undefined output buffers before a forward.
Do not import observer classes before the native mask fixes have been installed.
Install the forward observer after adapter patching so it sees sparse GLOBAL
paths as well as native LOCAL paths. Return the original forward result exactly.

No new attention, model or sampler forward, RNG draw, selector definition,
budget, layer scope or native graph change. The observer adds integer kernels;
none of its times are clean performance evidence. Check actual coverage/layer
identity and observation agreement. Find the earliest unequal input, support,
cache or output, then freeze a separate targeted raw-tensor operator replay.
Optional raw-capture support is disabled in this first locator attempt.

Two750-second reservations initially charge existing phasek256. Closed charge
before this stage7595.879s, remaining8604.121s under the16200s cap; total43200s
is unchanged. Both full matrices and the K128 queue remain held. Do not discard
the failed original or intervening inconclusive diagnostic pairs.
