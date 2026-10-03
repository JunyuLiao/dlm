# First independent 32K cost diagnostic

Four fresh engines completed successfully, one same question and engine seed per arm,
with an unprofiled warm request before the selected profiled request. Reserved GPU
seconds:1081.6898, including startup, warmup, profiling and teardown. This is a
diagnostic resource receipt, not request timing evidence. No quality evaluation.

| Arm | Actual denoising N | Commits C | Prefill forwards | C/N |
|---|---:|---:|---:|---:|
| Default dense |137|10|3|.07299|
| Matched native |374|19|3|.05080|
| All-kept |150|11|3|.07333|
| Main |115|9|3|.07826|

Each actual N includes one executed but scheduler-unretired final speculative
forward. Equal seed does not require equal sampling trajectories. These single
profiled cells neither estimate panel effects nor support a speed comparison.

Attribution is incomplete. The first report tool recognized `decoder.layers`
instead of the official shared `model.layers` backbone, so eager GLOBAL/LOCAL
labels are missing. Default FULL graph replay also lacks eager layer scopes.
Direct CPU-to-GPU kernel association covers only part of recorded CUDA activity;
GPU subcategory durations and percentages are therefore withheld. Missing means
unknown, never zero. Observation and selector CPU scopes are nested, and the
selector includes both DP build and route; their counters are not additive.

The legacy field name `host_copy_enqueue` must not be read as pure enqueue:
CUDA memcpy API CPU duration can contain implicit waits for preceding work.
Transfer activity duration differs from API/host duration. Side-stream route
counters establish use of that path, not overlap or critical-path savings.

The unchanged outputs and counters are preserved privately. A corrected, newly
frozen independent diagnostic is needed for cost attribution. No original
protocol/source/results were edited, and these records cannot enter formal
W/S/SN or accuracy summaries. All published fields are anonymous aggregates.
