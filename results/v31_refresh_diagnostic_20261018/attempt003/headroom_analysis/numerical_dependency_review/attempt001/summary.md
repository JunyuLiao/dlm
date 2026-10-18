# Dense refresh numerical-dependency review

The engine11 item125/seed1 control pair has identical recorded prefix metadata and first-GLOBAL Q/K/V byte fingerprints, but different returned dense-attention output hashes. This identifies a numerical-repeatability issue at or within the observed refresh call boundary. It does not establish a FA4 bug, a changed kernel configuration, or a particular cause. The magnitude of the control-output difference is not available from hashes alone.

This is a completed read-only source review. It adds no GPU execution, method variant or source changes. The evidence comes from completed engine11, whose completion marker verifies96 measured records and the public records/scored-file hashes. This review makes no claim about whole-campaign completion.

## Recorded observation

At canvas5, zero-based call9, first GLOBAL layer5, prefix70815/query_rows256:

| Request | N | C | T | Official score | First-GLOBAL returned dense-output SHA256 |
|---|---:|---:|---:|---:|---|
| control_before (request65) | 51 | 4 | 890 | 1 | `41f5d5da49f5f4744627fe1ad5c01cb7fc9b4d3077e743f5f174445e07b2c7c8` |
| dense_fresh (request68) | 51 | 4 | 890 | 1 | `41f5d5da49f5f4744627fe1ad5c01cb7fc9b4d3077e743f5f174445e07b2c7c8` |
| control_after (request70) | 60 | 4 | 899 | 1 | `4feda779da9711b64acbb7747b9058501609ae8a5bd389cdd4ec6e71c1d2c3d5` |

The full recorded prefix metadata agrees across these requests, including input fingerprints, selected sampler-state/RNG fingerprints, all held support hashes and progress state. The two control anchors agree on every recorded field except returned output hash, including old/next support hashes, one inherited call, unchanged RNG and zero sparse-oracle calls. Every row reports zero measured CUDA captures. The receipt retains the exact public-record projections and source/evidence hashes.

## Exact return path

Frozen source `db755c43be763a8af9edfe8a147ec333d8b8542b`:

1. `experiments/numerical_qk_reuse/vllm_adapter.py:1129–1152`: a due refresh calls `_select_initial_refresh`, updates the selected support and returns its `out`.
2. Adapter `:1190–1192` and `:1438–1450`: the plain selector calls `_mage_select_fa4`, which obtains `out` from `observe_dense` on transposed contiguous-K/V-buffer views.
3. `experiments/numerical_qk_reuse/v31_fa4_observe.py:100–127`: observation calls native `_flash_attn_fwd` with explicit softmax scale, causal=True, a cached dynamic-causal=False tensor, pack_gqa=False, num_splits=0(auto), auxiliary z/scale tensors, and no paged table/sequence-used/sparse mask.
4. Native `vllm/vllm_flash_attn/cute/interface.py:932–955`, `:979–993`, `:1849–1861`: resolve the forward configuration; when splitting is selected, allocate FP32 partial outputs/LSE and run native forward-combine.

This is the dense observation output, not the adapter's two-way paged sparse-consumer merge. The FP32 tail matmul and attention-mass selection happen after `out` is computed (adapter `:1451–1479`); they do not mathematically modify it. Old/fresh masks control subsequent support and therefore do not constrain the dense output arithmetic at this call. The source specifies automatic native splitting; the resolved split count was not measured at the captured event and is not asserted here.

## What the fingerprints do and do not establish

Worker `scripts/v31_refresh_diagnostic_worker.py:17–20` and runtime `experiments/numerical_qk_reuse/v31_refresh_diagnostic.py:25–29` hash detached, contiguous CPU bytes. They do not encode the original tensor shape, dtype, stride, storage offset, alignment, pointer, device/stream, scalar scale, auxiliary values or resolved kernel configuration. Geometry/model/source constraints narrow these possibilities but are not a complete kernel-input receipt. Dense_fresh separately records output shape[1,256,16,512]/BF16; control anchors record only the output hash.

Concrete dependencies not captured by the event proof:

| Dependency | Source | Limit of present evidence |
|---|---|---|
| Input shape/dtype/strides/storage offset/alignment and possible canonicalizing copy | Native `cute/cute_dsl_utils.py:75–124`; `cute/interface.py:680–705` | `maybe_contiguous` branches on pointer/stride alignment. The original layout and copy decision are not recorded. |
| Softmax scale, cached GPU dynamic-causal/scale tensors and auxiliary ABI | Frozen `v31_fa4_observe.py:84–127`; native `cute/cute_dsl_utils.py:209–267` | Source initializes constant cached values, but values/layouts at this event are not fingerprinted. |
| Resolved tile sizes, splits, architecture/SM-count override, MMA/PV overlap flags | Native `cute/interface.py:345–459`, `:932–955`; `cute/cute_dsl_utils.py:53–67` | These select an algorithm/reduction path; actual resolved values are absent from the event receipt. |
| Forward/combine compile keys and loaded compiled artifact identity | Native `cute/interface.py:1326–1403`, `:3978–4083`; `cute/cache_utils.py` | Source identity does not itself record which compiled artifact executed. |
| Output/partial/LSE allocations, complete writes and stream ordering | Native `cute/interface.py:839–855`, `:979–993`, `:1738–1861` | The intermediate tensors and launch ordering were not captured. Fresh uninitialized allocation is normal only if the kernel fully writes all required elements; this review has not demonstrated a write defect. |

Most listed dependencies are expected to remain constant under this model and single engine. None has been observed to change. They are missing checks, not diagnosed causes. Runtime helper files not individually present in the original protocol's runtime_sources are hashed at review time and explicitly distinguished in the receipt from execution-frozen per-file fingerprints.

## Next diagnostic gate

Before another method sweep, run a separately authorized bounded repeat check on the failing operator with identical tensor values **and original layouts**, explicit scalar/auxiliary inputs, the same native observation path and fully warmed shapes. Record the resolved configuration and compiled forward/combine identity; compare output, FP32 partials and LSE using both exact hashes and numerical error magnitudes. Include repeated no-op controls and allocation/request-order context. Preserve a diagnostic with explicitly fixed split configuration as a separately named follow-up only if needed; it would not be evidence about the unchanged path by itself.

This check should first establish repeatability of the observation boundary before interpreting small causal differences or beginning further selector sweeps. No such additional execution was performed by this review.
