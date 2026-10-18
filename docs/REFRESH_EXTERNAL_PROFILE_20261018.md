# Bounded external tracing check — 2026-10-18

The runtime-headroom study completed its clean controls, but all six instrumented
profile requests differ from the corresponding clean trajectories. This blocks
direct transfer of their phase costs to clean request latency. The complete
shared scope inventory and small progress-refresh spans remain descriptive
evidence, not a clean critical-path bound.

Use at most one external Nsight Systems trace of the original shared-clean
schedule to test whether useful runtime measurements can be collected without
the Python audit hooks. This is an instrumentation diagnostic on the already
selected inputs, not a new method or benchmark expansion.

## Frozen execution and budget

Commit the standalone runner, tests and this specification before preparing a
new immutable leaf under `results/v31_refresh_diagnostic_20261018/external_headroom/`.
Bind the original attempt001 engine002 protocol, complete artifacts, runtime,
model and inputs. Execute its byte-identical frozen worker and source snapshot.
Keep the same implicit warmup, two explicit warmups and eight measured requests
(dynamic/fixed16, two inputs, two repetitions), seeds, native sampler and output
limits. Substitute only new output and copied-cache locations. Preserve all
original files and cache contents.

Do not install headroom, factorial or native audit hooks: REFRESH_AUDIT=0,
HEADROOM_PROFILE=0, VALUE_AUDIT=0, VALUE_CLEAN_TIMING=1 and
VALUE_BOUNDARY_TIMING=1. Use external CUDA tracing with graph-node activities,
CPU sampling/context-switch tracing disabled, and no Python/NVTX instrumentation
added to the worker. Record the nsys version, binary hash and exact options.
Graph-node tracing can add overhead; this run is not a clean latency measurement.

Launch only after the bounded factorial coordinator reaches a terminal state.
Acquire the same GPU lock, verify the same idle device, and use the same campaign
reservation ledger. Allocate at most the remaining 600-second reserve, including
startup, warmups, tracing, failure and export/analysis. The overall cap remains
7,200 GPU-reservation seconds across every attempt. Use an owned process group
and enforced deadline. No second traced GPU attempt follows a failed check.

## Acceptance and interpretation

Require source/runtime/input identity, native structural receipts, zero request
graph captures and exact measured reproduction of the original clean run's
output hash, N, C, T, finish reason, seed and relevant adapter counters. Preserve
all mismatches. A mismatch blocks a matched-trajectory cost claim; it must not be
reinterpreted as an accuracy or method result.

Export and analyze the private trace on CPU after the GPU worker exits, within
the same deadline. Attempt request/prefill/decoding windows only when the frozen
worker's explicit synchronization order and recorded boundaries map uniquely to
CUDA API events. Duration resemblance alone is insufficient. Otherwise retain
whole-process scope, including startup/warmup, and declare request attribution
unresolved. Keep unknown kernel categories and overlaps explicit. Generic GEMMs
are not automatically MoE and GPU gaps do not prove host causality.

Even exact trajectory reproduction and valid request windows do not identify
every refresh/commit operation or its removable critical-path contribution.
Report that limitation if attribution remains incomplete. This check cannot
by itself promote a refresh method, establish noninferiority, or authorize an
older full503 continuation. The original factorial and its decision rules remain
unchanged.
