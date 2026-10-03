# Independent 32K CUDA event diagnostic003

Frozen source `54163f4e7c8766aaa4187807420b3cf3167dba81`; protocol
`v28_v29_cost32k_events_20261002_003`. LongBench-v2 existing E14/V18b-confirmed
32K singleton, reused diagnostic question (not held-out quality evidence).
vLLM0.30.0, Torch2.13.0+cu130. Each engine: seed28001, one unprofiled warm then
one profiled request (zero-based ordinal1), with8192-token budget and unchanged
native stopping. All four arms use PIECEWISE compilation and graph mode.

Four fresh PIECEWISE engines completed rc0: dense without adapter, matched
native, all-kept and main. Same frozen singleton/engine28001, unprofiled warm,
complete8192 budget/native stopping, Q128, alias2, original Torch copy/merge,
legacy canvas buffers and request-clear. Captures0; adapter order errors0.
Reserved GPU seconds1092.569460863946 (startup, warmup, profiling and teardown).
This is a resource receipt, not W/S timing evidence. No quality evaluation.

| Arm | Actual N | C | Prefill P | Denoise GLOBAL span/N ms | Denoise LOCAL span/N ms |
|---|---:|---:|---:|---:|---:|
| PIECEWISE dense, no hook |374|19|3|unknown phase|unknown phase|
| Matched native |99|9|3|4.494699|6.461866|
| All-kept |150|11|3|4.894036|6.618996|
| Main |115|8|3|4.329062|6.611120|

Every N includes1 executed, scheduler-unretired denoising forward. Expected
attention counts match:5 GLOBAL and25 LOCAL per actual forward. No-hook has
1980 GLOBAL/9900 LOCAL calls, covering N+C+P, but no exact per-call phase tags.
Matched arms separate denoise and encoder; encoder includes prefill plus commits.
Their encoder span/C fields are amortized totals including prefill, not isolated
commit prices. Missing phase/scope means unknown, not zero. Inactive method
scopes are naturally absent in controls.

Different N/trajectories preclude a paired speed claim. Main/native GLOBAL span/N
is.963149 (about3.7% lower); LOCAL is1.023258 (about6.61 versus6.46ms/N).
These are independent inclusive diagnostic scopes, not formal request speed,
identical-input kernel efficiency or causal explanations of panel S/N. No-hook
374/19 matched the old002 native counts, but003 native99/9 differs. Equal seed
does not establish equal trajectories; the cause is not established here.

Main leaf scope counts match receipts: fused observation40, DP build40, DP
route105; asynchronous observation routes40. Five active GLOBAL layers imply8
observed-canvas equivalents; prefix builds45 cover9 canvas equivalents,
including the final unused canvas entry. No extra observation is assumed there.

| Main scope | Calls | Inclusive span ms | Mean ms per call | Sum ms / observed canvas |
|---|---:|---:|---:|---:|
| Fused observe |40|99.534944|2.488374|12.441868|
| DP build, side stream |40|68.427264|1.710682|8.553408|
| DP route |105|42.426816|.404065|not once per canvas|
| Prefix build |45|26.009920|.577998|different denominator|
| Tail refresh |530|13.744416|.025933|not once per canvas|

Fused observation99.53494 / GLOBAL497.84218 =19.99% is a share of the same
main-stream inclusive GLOBAL span. It is **not20% extra overhead**: this fused
range includes the normal dense attention output plus observation work. No
matched identical-state FA4 counterfactual was timed to isolate its incremental
cost. DP build/route run on the existing side stream where applicable and may
overlap main-stream work; do not add their spans to observation as wall cost.
The outer observation span116.54109ms also includes setup around the fused leaf.

Selector210 and consumer1065 calls contain nested duplicate ranges (535 alias
consumer calls plus530 core consumer scopes); their sums are explicitly not
exclusive costs. GLOBAL includes its observation/copy/consumer subranges. Event
spans include host dispatch gaps, streams/nesting overlap and instrumentation
can perturb execution. They cannot be summed into request time or treated as a
causal percentage of unprofiled latency. Cross-stream overlap/critical path is
not measured; default FULL graph costs remain from002 with incomplete attribution.

CUDA copy API CPU duration includes implicit waits and is not pure enqueue or
physical DMA. For main, metadata CPU989.595ms and its memcpy API899.987ms coexist
with only.627762ms total D2H activity. This is observed waiting/accounting, not a
separate additive copy cost. Host diagnostic fields remain separate in JSON.

Installed official Gemma4 source scaling is1.0 and the adapter preserves
`impl.scale`; this is source evidence, not a runtime scale distribution. Alias
LSE uses natural logarithms and merges in FP32. Validation of actual model-state numerics
and task quality requires separate qualifications; this profile does not supply it. All public
fields are anonymous aggregates; original sources/protocol/results remain frozen.
