# Dense controls, integration scope and next execution diagnostic

## What each control means

Default dense, native-hook and main all run inside native vLLM0.30.0. Main and
Q64 are already request-level vLLM implementations; they are not HF-only plans.
Default dense uses default vLLM graph configuration without our adapter hooks.
Native-hook retains full native attention, uses PIECEWISE execution, and executes
our clock/sample hooks. It does not do adapter KV copies or sparse selection.
All-kept additionally retains the adapter copies and alias2 consumer while keeping
every block. Main enables the frozen risk-based sparse method on five GLOBALs.

PIECEWISE concerns scheduling pieces of a model forward, not splitting the text
into independent examples or shortening context. Hooks are callback insertion
points. Equal mathematical dense attention does not prove identical floating-point
or sampling trajectories across execution configurations. V18b changed N as well
as time, so do not interpret its W differences as isolated engineering costs.

Main not consistently beating native-hook means its incremental end-to-end
benefit is not established. A lower W versus default dense is a result for that
measured comparison, not license to ignore another valid dense reference. A
stable claim must survive adequately sampled strong dense controls. Choose a
reference configuration using disclosed qualification/development evidence,
not by selecting the worst control or a favorable seed after scoring.

## Integration is required for the claim being made

Not every sparse-attention publication uses vLLM. The
[SparseD official implementation](https://github.com/INV-WZQ/SparseD) supplies
Dream/LLaDA generation scripts using FlexAttention, while
[PBS-Attn](https://github.com/xinghaow99/pbs-attn) exposes a Hugging Face prefill
patch and separate efficiency/evaluation scripts. These examples show different
implementation substrates; they do not establish performance on our model.

For this study, a vLLM end-to-end claim requires actual vLLM integration with
unchanged native sampling/stopping, quality scoring and all attributable costs.
Rejected components can remain component negatives. Standard copy/merge gains
must also be applied to eligible matched controls. New models should use their
qualified official stacks, rather than being forced into vLLM for uniform branding.

## Predeclared fifth diagnostic: PIECEWISE dense without hooks

This is a separate diagnostic after the frozen four-arm profile. Do not modify
that running protocol or append its records to a formal panel. Use a new own
directory and binding, same source/code and original32K index0 as the four-arm
profile, engine seed28001, full native8192 budget, warm1/profile1, OMP1, q128
contract and matching model/FA4/memory settings. Set `arm=dense` and
`arm_settings.dense.cudagraph_mode=PIECEWISE` (and matching compilation field)
using the existing panel interface. No adapter is constructed on the dense arm.
Label this result `dense_piecewise_nohook`, never default dense.

Compare it with default dense to explore graph configuration, and with native-hook
to explore hook effects; different trajectories still preclude a causal cost
estimate by subtracting whole-request times. Require actual phase/graph/source
receipts and profiler scope coverage. One profiled request is diagnostic only,
not a new baseline winner, quality result or request speed claim. The existing
sampler config and source pin must remain exact. Any extra trace or numerical
replay requires a separately described protocol.

The expanded four-arm, eight-engine-seed main confirmation is a separate study.
Do not silently replace its control or add a posthoc fifth arm after generation.
