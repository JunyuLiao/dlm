# HumanEval numerical mechanism — diagnostic, qualification pending

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

## Candidate correction and gates

Pinned Torch's triton_heuristics.py explicitly filters reduction configurations
for deterministic numerics and forbids reduction block/warp tuning in that mode.
The next contrast sets the supported TORCHINDUCTOR_DETERMINISTIC=1 policy only.
It retains native CUDA graph modes, model formulas, sampler/stopping settings,
selector budgets and layer scopes. This is a named numerical-runtime variant;
the actual compiler policy and selected configurations must be receipted.

Do not call the discrepancy resolved until independent same-method process
repeats and audit/clean pairs agree, for dense/V31 and both thinking modes,
followed by the requested methods and accounting checks. Do not silently switch
to eager execution. Measure any deterministic-policy latency cost against matched
controls before reporting a method speedup. No K256/K128 full rerun is released
while qualification fails; K128 still requires fresh work-only V2 calibration.

Authoritative sanitized operator evidence: operator_diagnostics/attempt002/summary.json
and operator_complete.json; valid-row comparison:
analysis/attempt001/first_native_segment_comparison_valid_rows.json.
