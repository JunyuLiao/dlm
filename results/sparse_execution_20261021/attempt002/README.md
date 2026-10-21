# Further true-online refresh optimization, 2026-10-21

This is a synthetic optimization attempt, not a completed native benchmark.
The prior campaign attempt004 remains paused and immutable. Native evaluation,
calibration and promotion remain gated on refresh and combined qualification.
Short benchmark V31 budgets remain K128; LongBench-v2 remains K8192.

New scheduling designs preserve ascending KV64 traversal, Q128 admission,
current Gaussian32 values, fixed threshold and conditional full-dimensional PV.
The Q32 unique producer computes all output channels using warp MMA; Q64 two
channel groups halve duplicated routing relative to the four-group reference.
Build directories pin sources, compiler output and options before execution.

At Q256/H16/Hkv2/D512 BF16, approximately 50% physical sparsity, 8K keys:

| Variant | Complete refresh, us | Plain dense in same screen, us |
|---|---:|---:|
| Q32 unique warp-MMA | 2968.58 | 357.76 |
| Retained-only dual-router queue | 602.65 | 363.52 |
| Q64 two-channel, register P | 459.98 | 359.88 |
| Q64 two-channel, shared P | 677.46 | 360.56 |
| Q64 hardware cluster barrier, schedule 2 | 521.33 | 359.00 |
| Unique Q64 cooperative global queue | 539.65 | 359.00 |
| PV/next-QK overlap, no spills, schedule 2 | 515.70 | 362.68 |
| Aligned-shape specialization, unroll32 | 433.11 | 359.32 |
| Aligned-shape specialization, unroll1 | 507.47 | 360.43 |
| Aligned-shape plus hoisted descriptors | 434.10 | 366.08 |
| Interior/tail specialization | 429.32 | 365.77 |

Q32 is a negative: register spills and warp-level operand handling overwhelm
removed repeated work. Shared P also spills and loses. Hardware cluster barriers
lose to release/acquire votes. Aligned unroll32 improves the prior449us best,
but remains20.54% slower than plain dense and50.14% slower than native paged
dense288.48us in its matched screen. Its generic masked fallback remains.
These are separately warmed CUDA-graph synthetic kernel pipelines. First rows
use exact approximately 50% masks; barrier screen uses 50.49%. No native request
speed claim follows. Projection, sketch packing and compact lists are charged in
complete refresh, with no dense-output simulation or two-pass replacement.

Sources and numeric evidence are probe001/receipt.json,
dual_kept_queue001/receipt.json, q64_refresh001/receipt.json,
q64_sharedp_refresh001/receipt.json and q64_barrier002/receipt.json.
q64_barrier001 failed because its historical reference-library argument was
incompatible; its frozen source is retained. tests001 has 10 passes, 2 skips and
one wrapper-validation failure, subsequently corrected. q64_tests001 records a
sandbox-only all-skipped run, not qualification. GPU review timeouts occurred
before the original q64_tests001 command could execute. Qualification continues
in new named test directories.

Q64 register-P passes all13operator/current-V/changing-graph tests in
q64_tests002/junit.xml. Cooperative queue and non-spilling PV/QK pipeline each
pass80changing-Q/K/V/Z/held graph replays with exact masks and maximum relative
L2 below0.003 in global_queue_qualification001 and pv_qk_qualification002.
The unroll32 overlap build spills heavily and was not GPU-qualified because
automatic approval review timed out twice before execution. Reduced unroll1
removed those spills, qualified and produced the slower515.70us result above.

Erratum for probe001 only: its zero-capture field was inferred from the replay
harness, not checked with an actual capture counter. Treat its Q32 timing as
a weaker diagnostic, not clean-qualified evidence. The script now checks an
actual capture counter; existing probe001 is immutable and not retro-certified.
All sparse_execution_refresh_screen rows above use actual capture counters.

Interior/tail200354cb passes16kernel tests including Q256/1025-key changing
graphs (interior_tests002); its8K screen is interior_refresh001. It improves
the initial449.02us best by4.39%, still17.37% slower than matched plain dense
and48.68% slower than native paged288.75us. Cached-prefix projection414.12us
is a separate within-canvas optimization, not the full-current-V number.
Full12-cell qualification continues under attempt018. Source checkpoint
5a5611e2 preserves the pending implementation and sanitized evidence.
