# V29 default dense / native-hook audit — 2026-10-03

This is a source and closed-evidence audit, not a new GPU result. No formal
worker, frozen source, sampler or environment is changed. `native` is the
matched official-dense path with adapter hooks; it is not an alternative dense
attention implementation. Its hooks and execution policy remain material costs
and possible sources of trajectory differences that need separate investigation.

## Actual paths and timing boundaries

Locations below refer to the current repository; official module locations are
from the campaign installation, vLLM0.30.0 / torch2.13.0+cu130. No private paths
or private input hashes are published.

| Evidence | Verified behavior |
|---|---|
| `scripts/v27_vllm_panel_run.py:118–134` | Dense does not install the method adapter; native/all-kept/main do. All construct the same official LLM with the same model binding, BF16, length, chunk, block-size, seed and memory settings. Only non-default graph settings are explicitly overridden. |
| `scripts/v29_expanded_panel.py:78–81` | The running formal protocol requests default dense and PIECEWISE native/all-kept/main. Sparse already uses PIECEWISE; it does not need a new PIECEWISE switch. |
| `experiments/numerical_qk_reuse/vllm_adapter.py:184–198` | Native returns before installing the mathematical runtime; its runtime stays `None`. It does not replace model weights or model architecture. |
| `vllm_adapter.py:508–528` | The metadata hook reads exact GPU phase/step/length with `stack(...).tolist()`, then calls original preparation. The sample wrapper calls the original compiled sampler first and returns its exact result. Native's `on_sample` only counts calls (`343–350`); method logits observation requires a non-null runtime. |
| `vllm_adapter.py:530–553` | Active native GLOBAL calls forward every original argument to the captured official `FlashAttentionImpl.forward`. LOCAL/encoder/unbound calls also pass through. Native does not call adapter KV gathering, alias sparse consumers, observation or DP selection. |
| `vllm_adapter.py:336–337` | Native's HF-shaped bookkeeping canvas is zeros; it does not overwrite the official sampler canvas or consume random draws. |
| `scripts/v27_vllm_panel_run.py:163–177` | The same request-boundary synchronization and actual-execution tracker apply to all arms. Adapter begin/end are inside W. No hidden exclusion of hook cleanup is established. Completion decoding is outside W (`201`). |
| `scripts/v27_vllm_panel_run.py:186–209` and V28 wrapper | Timed graph/compile deltas are checked. Existing async snapshots count actual N, including unused terminal execution; scheduler-consumed N is separately stored. V28 also checks Triton/CuTe compilation events. |

The source establishes native argument passthrough. It does not prove equality
of every GPU intermediate or random trajectory under different graph/wrapper
paths. Hook D2H waiting is a real difference. Native formal profile flags are
false, so the adapter does not create its optional timing events in that path.
No confirmed native attention/sampler replacement bug was found in this audit.

W is request time; S is the decode span including commits and host/scheduling
work; N excludes commits. S/N is an amortized cost, not the price of a single
forward or an isolated FA4 kernel. C/N, output length and N must accompany it.
An arm that stops sooner may have lower W despite greater S/N. Different phase
counts also change the amortized GLOBAL/LOCAL mixture.

## Graph configuration is not a per-forward receipt

Read-only closed-log inspection found initialization `FULL_AND_PIECEWISE` for
002 default dense, and `PIECEWISE` for 002 native/all-kept/main. All four 003
engines initialized with `PIECEWISE`. These are resolved initialization fields,
not a histogram of actual runtime modes.

The actual new GPU runner uses `vllm/v1/worker/gpu/model_runner.py`, not the
older similarly named `gpu_model_runner.py`. At `1685–1699` it dispatches from
CPU batch descriptors. At `1811–1821` it passes `batch_desc.cg_mode` into the
model-state attention preparation, and `1901–1909` explicitly replays FULL only
when that runtime descriptor is FULL. Its alternative path uses the same
runtime mode in forward context (`1918–1922`).

`vllm/v1/worker/gpu/cudagraph_utils.py:248–258,340–385` constructs decode/mixed
candidates; `496–527` returns the first compatible captured descriptor or NONE.
The DG state does not force denoising to PIECEWISE in its inspected preparation:
`vllm/model_executor/models/diffusion_gemma.py:994–999` uses padded dimensions
only for actual FULL, and `1004–1031` creates the phase-dependent causal GPU
metadata. The initialization mode alone cannot establish which denoise,
commit or prefill calls selected FULL. The historical public request records
store the requested graph setting, not a phase-resolved runtime dispatch log.
The old traces were not exported and cannot recover this missing evidence.

## Completed paired evidence and historical warm confound

Ratios are candidate native / reference default dense. Below one means less
of that quantity. These are completed panels, not qualification singletons.

| Panel / stratum | Paired population | W | S/N | N |
|---|---|---:|---:|---:|
| V18b32K | 4 questions,16 requests/arm | .5813 | 1.0504 | .5228 |
| V18b64K | 4 questions,16 requests/arm | .8660 | 1.0209 | .7967 |
| V18b96K | 4 questions,16 requests/arm | .8764 | 1.0180 | .7966 |
| V28 preview, all lengths | 6 questions,48 requests/arm | .78977 | 1.02514 | .73670 |
| V28 preview32K | 2 questions,16 requests/arm | .50863 | 1.05712 | .44349 |
| V28 preview64K | 2 questions,16 requests/arm | .94417 | 1.01440 | .91068 |
| V28 preview96K | 2 questions,16 requests/arm | 1.02575 | 1.00466 | .98995 |

Sources: [V18b interpretation](V18B_INTERPRETATION_V28_20261002.md) references
the published V18b summary and its original generation source; [V28 preview
interpretation](V29_V28_PREVIEW_INTERPRETATION_20261002.md) references the
published seed4 preview. Point estimates do not by themselves establish a
universal advantage; V28 96K native W is above one, and V18b native S/N is above
one in all three strata. Historical statements that dense is always slower or
that native achieves a faster denoising kernel are not supported.

V18b has a concrete protocol confound: `v27_vllm_panel_run.py:113–114` filters
controls before warm scheduling (`148–153`). The frozen V18b spec contains59
main/dense questions but only12 native/all-kept control questions. Thus dense
warms59 requests while native warms12. Subsequent timed histories also differ
because dense processes the full inventory. Equal engine seed does not give
these arms the same RNG position at a paired control question. This is stronger
than merely saying there were few seeds; it directly limits the historical
native/default-dense attribution. It does not invalidate the unchanged
main/default-dense comparison's shared warm inventory.

V28 preview has the same6 warm questions across all arms; its summary validator
requires the full selected inventory in control indices
(`scripts/v28_preview_summary.py:131–135`). The current V29 expanded formal
protocol also retains the same inventory for every arm
(`scripts/v29_expanded_panel.py:69–88`). This removes the V18b warm-coverage
confound. It still does not establish identical random consumption during warm
or timed requests if the trajectories differ.

## 002 / 003 discrepancy is unresolved

The following are one warm + one profiled request in fresh engines, same
32K singleton and engine seed28001. They are diagnostic observations with
unknown profiler overhead, not paired speed/quality evidence.

| Diagnostic | Arm | Requested / initialization graph mode | Actual N | C | P | Output-token count |
|---|---|---|---:|---:|---:|---:|
| 002 | dense | default / FULL_AND_PIECEWISE | 137 | 10 | 3 | 2497 |
| 002 | native | PIECEWISE / PIECEWISE | 374 | 19 | 3 | 4761 |
| 003 | dense no hook | PIECEWISE / PIECEWISE | 374 | 19 | 3 | 4761 |
| 003 | native | PIECEWISE / PIECEWISE | 99 | 9 | 3 | 2107 |
| 002 / 003 | all-kept | PIECEWISE / PIECEWISE | 150 / 150 | 11 / 11 | 3 / 3 | 2734 / 2734 |
| 002 / 003 | main | PIECEWISE / PIECEWISE | 115 / 115 | 9 / 8 | 3 / 3 | 2205 / 2006 |

The authoritative numeric accounting is now available in
[request rows](../results/v29_20261002/dense_native_accounting001/requests.csv)
and its [README](../results/v29_20261002/dense_native_accounting001/README.md).
[002 raw evidence](../results/v29_20261002/cost32k_002/raw/README.md) and
[003 raw evidence](../results/v29_20261002/cost32k_events003/raw/README.md)
retain actual measurements and limitations. Equal N/C/output counts do not
establish identical output text or intermediate logits.

Read-only equality checks of the two closed private bindings verified the same
cells and generation manifests. The deployed panel loop, V28 wrapper, metrics,
adapter and JIT-receipt source bytes were identical. Git comparison
`39e08c521..54163f4e7` confirms these execution/core sources did not change.
The profiler did change: corrected layer labels, direct CUDA event recording,
three leaf scopes and event-setting validation were added. Native's model,
config and hook source therefore did not change, but its instrumentation did.
Warm N and RNG boundary state were not saved. Profiler effects, kernel/runtime
nondeterminism and RNG-state differences have not been excluded or assigned
causal shares. Native374→99 and 003 no-hook374 versus native99 cannot be explained
solely by graph configuration or V18b's different warm subsets. Repeating the
seed label is not a reproducibility proof. The coincidence between 002 native
and 003 no-hook counts remains an observation, not an explanation.

The actual official DG sampler is compiled (`diffusion_gemma.py:469–470`) and
accepts no per-request generator in its argument list (`470–510`). It calls
`torch.rand_like` for Gumbel noise (`529–533`), `torch.randint` to reinitialize
canvas state (`572–574`), and `init_canvas` also calls `torch.randint`
(`735–751`). The runner supplies no request sampling seed (`panel_run.py:160`),
and marks `seed_applied=False` (`141`). Unequal request histories can consume
unequal GPU random draws. This source fact does not prove which cause produced
the measured differences, nor whether compiled RNG and graph paths matched.

## Next independent diagnostic, not a replacement formal baseline

003 already contains the PIECEWISE dense-nohook reference. It is missing from
the running formal four-arm panel, not missing from all existing diagnostics.
Retain the strongest default dense baseline; add a separate reproducibility
check with three references: default dense, PIECEWISE dense without method
hooks, PIECEWISE official dense with native hooks. Use the same complete warm
inventory, order, budget and several fresh-engine seeds, with a repeated
fresh-engine run at each same seed before attributing any arm difference.

The new diagnostic must record resolved engine graph configuration and actual
runtime graph mode per execution phase. Match a real execution descriptor to
the already-existing async sample-count snapshots after the boundary sync:
prefill has no scheduled drafts, decode with zero emitted tokens is denoising,
and a positive emitted count is commit. Do not use scheduler-retired identity
for a submitted/unused forward. Record N/C/P and output counts, and retain zero
new timed captures/JIT receipts. This does not require changing the sampler or
consuming random draws.

RNG boundary diagnostics must only read state, never reseed/restore it in this
protocol. Read CUDA RNG state at the already-synchronized request boundaries;
keep state bytes/fingerprints private and publish only equality results. Such
instrumentation is explicitly diagnostic and cannot be mixed into formal W.
If states or same-seed trajectories disagree, investigate that boundary before
claiming kernel or graph-mode speed. A later fixed-state Q/K/V or logits replay
can isolate math; it is a separate experiment, not a silently modified sampler.

No new GPU run or phase-resolved graph/RNG receipt is claimed here. CPU native
argument-passthrough tests can verify wrapper contracts, but cannot qualify GPU
numerics, compilation, stopping or random trajectory equivalence.

CPU validation: six tests in `tests/test_v29_native_passthrough.py` pass with
the bundled dependency-light Python. They execute the actual nested adapter
patches against stand-ins for the official modules, checking all attention
argument identities, exact sampler return identity, native logits non-access,
encoder/unbound paths and preparation passthrough. GPU behavior remains untested.
