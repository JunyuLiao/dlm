# Bounded native repeatability and refresh attribution

Authorized 2026-10-18 after the completed first-refresh factorial. This is a new
diagnostic campaign, not a selector trial or a resumption of either older panel.

## Budget and source boundary

One idle H100, one owned process group at a time, shared GPU lock. Hard cap 2700
reservation seconds (45 minutes): capture 900, isolated replay 600, attribution
900, recovery 300. Startup, warmups, failures and in-reservation processing count.
No automatic phase borrowing. Recovery requires a preserved failed attempt, a
concrete repair and a new source-bound leaf. Timeout stops the owned worker,
never shortens native generation limits. CPU preparation and final analysis do
not reserve the GPU. Stop after this protocol, without a new selector.

Freeze these diagnostics and tests before preparing immutable attempt001 under
results/v31_refresh_repeatability_20261018. Retain original native source 6aab8632
and input/runtime/model bindings from the previous campaign. Copy original caches
and execution sources into each new worker context; preserve original contents.
All process-local wrappers are separately fingerprinted. No installed package or
production selector is edited. Private snapshots, traces and generations remain
ignored; public receipts contain metadata, hashes and numeric summaries only.

## Numerical diagnostic

Capture the first genuine progress refresh at GLOBAL layer 5 on LongBench inputs
125/seed1 (the earlier control-drift input), 329/seed42 and 254/seed42 (the runtime
inputs), twice each. Use one implicit and one explicit original V31 warmup. Native
sampler, generation limits, GLOBAL fixed K8192, Q128/KV64, carry, sticky=1.386,
LOCAL/prefill/commit dense and all other production settings stay unchanged.
Capture is diagnostic, not clean timing or an exact replay of the old engine's
request-order history. Preserve full generated requests and report exposure.

At that single operator event per measured request, return the original output
unchanged and save private Q/K/V values, original shape/dtype/strides/storage
offset/alignment, scale/auxiliary values, resolved native configuration, compile
keys/object identity, actual intermediate FP32 outputs/LSE, output and z. The
Python return-frame observer only retains references; CPU copies follow the
original call. Add no model forward, sampler call or RNG draw. Captured requests
may be perturbed by instrumentation; do not pool their accuracy into the old panel.

After the model exits, replay each exposed snapshot in two independent operator
processes. Preserve input layouts and alignment; verify input hashes before/after.
Use unchanged observe_dense with auto splitting, two warmups, eight repetitions
on fixed input buffers and eight on fresh layout-preserving allocations. Compare
exact hashes, maximum absolute error, RMS/relative error, nonfinite patterns and
fraction of unequal elements for output, z, partial outputs and partial LSE.
Record native resolved settings and loaded compile-object/key identities; object
identity is process-local, not proof of equal machine code across processes.
If auto output or intermediates vary, run eight repetitions with explicit split=1
as a separately labelled localization control within the same deadline. It is not
a production fix or evidence for unchanged auto behavior. Do not require unused
partial-output slots with -inf LSE to be initialized; compare only active slots
and report inactive storage separately.

Repeatability go gate: all six states exposed, layout/input/RNG checks pass,
same configuration and active intermediates/output repeat exactly within each
process, and replay output matches captured output plus the second process.
Any unexplained difference, input mutation, missing configuration or incomplete
coverage blocks causal/selector promotion. A passing sample does not prove global
determinism or explain the old drift. Quantify every mismatch rather than dropping it.

## Runtime attribution

Run the exact previous shared-clean schedule: two explicit warmups and eight
measured V31/fixed16 requests on 41K/112K inputs, plus implicit warmup. First run
unannotated as a new clean control, then the same schedule with CPU-only NVTX
ranges and external Nsight CUDA/NVTX tracing. Keep audit flags off and boundary
timing unchanged. No CUDA events, tensor reads, per-step synchronizations, sampler
hooks, or numerical observation are installed in this trace.

Annotate eager attention calls and nested KV copying, full selection and dense
observation. Associate the parent after execution with native counter changes:
initial selection, progress refresh, bootstrap, held or carried use; separate
prefill/commit and LOCAL. Save scope metadata at process exit. Classify GPU
activities through CUDA launch correlation and enclosing CPU ranges, not kernel
names. Keep unlinked graph work/activities explicit. Nested children are not
added to their parent. Require exact per-phase scope counts versus native
counters and uniquely qualified original request boundaries.

Attribution gate: exact output/N/C/T/finish/native-counter parity against both
the original clean run and new clean control, zero measured graph captures,
no monitored JIT during measurement, complete scopes, valid launch correlation
and no ambiguity. A failure leaves attribution diagnostic only.

Report clean W/S/N/C/T, inclusive CPU spans, GPU activity sums/unions and child
breakdowns. A deliberately generous fixed-trajectory deletion screen sums parent
CPU elapsed time and associated GPU activity duration (double counting overlap
is conservative) separately for progress refresh and for all eligible dense
attention phases. It assumes deleting those computations does not alter later
work and is not a universal algorithmic speedup bound. A measured saving would
still require replacing the computation and timing the complete method.

Research go gate: qualifying attribution must show plausible >=10% incremental
clean request benefit after proposed replacement cost, with matched quality.
If even the generous deletion screen is below 10%, stop this named regime for
that optimization scope. Unlinked work or failed parity makes the gate unresolved,
not positive. No batch>1 or alternative-model claim follows from this campaign.
