# HumanEval numerical mechanism — baseline qualification complete

Native source20e6c0be; investigation plan c467882a and unchanged total ceiling
43200 H100 reservation seconds. Canonical root:
results/humaneval_reproducibility_20261019/attempt001. The original K256 campaign
and older paused campaigns remain intact. There is no corrected full panel yet.

Independent native clean FULL, PIECEWISE and inherited V31 engines each change
all12 measured output pairs on six diagnostic questions. Within-engine audit
interventions agree. Actual loaded parameters/buffers, valid initial model
inputs and initial sampler noise match. Audit synchronization is not established
as the cause. Previous native attention snapshots locate one pair's first change
after layer0 attention and by layer1 Q/K/V.

The supported BF16 accumulation-only contrast still changes12/12 independent
outputs and N. The tuple split-K flag with cuBLAS fails before generation; no
BF16 policy has been promoted.

## Same-input operator evidence

The address-preserving first-segment observer (24cb5a57, frozen source019) has
identical native clean/observed/clean question0 bookends in two processes; five
additional unobserved question groups also agree. Only that first prefill
segment bypasses its CUDA graph to expose the existing compiled Python calls.
No extra model forward or sampler change occurs. This is a diagnostic dispatch,
not production timing. The first attempted bypass returned fresh output pointers
and failed its bookend check; it is excluded from native trajectory claims.

An earlier first-segment record uses R0_BLOCK1024 for
triton_red_fused_add_mul_rms_norm_2; both qualified records use512. All valid
operands before this kernel agree. Replaying its byte-identical generated Triton
source on the exact same saved inputs, with XBLOCK1 and8 warps fixed, gives:

| Output | Changed elements /481536 valid elements | Maximum absolute change |
|---|---:|---:|
| Layer output, BF16 | 11 | 0.0078125 |
| Internal normalized state, FP32 | 86596 | 0.0001220703125 |
| Input to the next Q/K/V projection, BF16 | 10 | 0.03125 |

Each configuration repeats exactly. The512 replay exactly matches all three
saved outputs in both qualified processes. The1024 replay exactly matches the
earlier record's outputs, recorded before its downstream pointer-dispatch defect.
That earlier record supports this bounded operator comparison, not native final
accuracy or observer qualification. See analysis/attempt001/reduction_replay_native_link.json.
The successful operator replay consumes3.776s and runs zero native forwards;
its failed API-preflight attempt consumes3.174s and is preserved.

This establishes a numerical effect of runtime reduction configuration. It does
not yet establish that this is the only source of the full campaign discrepancy.
The two qualified processes have equal valid outputs throughout their first
segment (both512), yet their final generations differ. Other compiled segments
and sampler reductions therefore still need the end-to-end qualification gate.
Unused padded positions differ and are excluded from the valid-row comparison.

## Qualified correction and remaining panel gates

Pinned Torch's triton_heuristics.py explicitly filters reduction configurations
for deterministic numerics and forbids reduction block/warp tuning in that mode.
The environment-only TORCHINDUCTOR_DETERMINISTIC=1 contrast was ineffective:
actual generated-kernel metadata still reported40 nondeterministic kernels.
Explicit model options alone left24 native helper/sampler kernels uncontrolled.
Both failed stages are preserved.

Qualified source10341ab2, frozen private/source_addenda/attempt027/worker.py,
binds deterministic compiler options to the model and recompiles the three
unchanged original native helper Python callables with the same options.
Actual53/53 recorded kernels are deterministic. Horizontal combo fusion and
its unsupported benchmarking are disabled in this named numerical/compiler
policy. Native FULL/PIECEWISE graphs, model and sampler formulas, stopping
settings, selector budgets, and layer scopes are preserved. Its latency must
be measured with matched controls; this is not a claim about original defaults.

Two independent unobserved native clean FULL engines agree on all12 output/N/C/T
pairs. Eleven additional qualification engines cover dense/V31, both thinking
modes, independent same-mode and audit/clean comparisons. All declared full
pre/post sampler-state, valid history, emission and CPU/CUDA RNG comparisons
pass, as do clean/observed/clean bookends; measured CUDA captures are zero.
Authoritative marker: qualification/attempt001/baseline_runtime_qualified.json.
This completes the six-question baseline diagnostic, not full164 method accuracy.

Every requested method must still pass independent process-prefix and audit/clean
comparisons plus accounting checks in each corrected panel. No eager fallback
or reproducibility-gate relaxation is permitted. K256 uses a new frozen protocol;
K128 follows verified K256 completion and requires fresh work-only V2 calibration.

Authoritative sanitized operator evidence: operator_diagnostics/attempt002/summary.json
and operator_complete.json; valid-row comparison:
analysis/attempt001/first_native_segment_comparison_valid_rows.json.
