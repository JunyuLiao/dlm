# Shared-memory bank layout correction, 2026-10-21

Synthetic-only follow-up; the native benchmark remains paused. Prior wide-QK
attempt005 was numerically correct but costs1282.18us complete refresh at8K.
Static address analysis identifies same-bank parking stores/loads for the
per-thread64-element output array and strided FP32 raw scores. This variant
transposes output parking across threads and swizzles raw score columns by row,
using paired FP32 stores/loads. No hardware-counter stall inference is made.

qualification001 passes80changed-input graph replays with exact masks and
maximum relativeL2<0.003. refresh001 complete refresh679.03us vs362.10us plain
dense and287.98us native paged. This improves the poor first layout but remains
a negative. All captures happen during warmup, actual timed capture count0.
