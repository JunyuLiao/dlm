# V28 CPU metadata boundary audit — 2026-10-02

This is a source audit, not an implemented optimization or a performance result. It concerns the per-forward GPU-to-CPU metadata read in `experiments/numerical_qk_reuse/vllm_adapter.py:468–479`. The frozen preview and its native sampling/stopping behavior are unchanged.

The official sources inspected are the installed **vLLM 0.30.0** modules in the campaign environment (torch **2.13.0+cu130**), rather than the repository's older third-party trees. All official locations below are module-relative paths and one-based line numbers from that installation; no host paths are needed. The adapter reference is the V28 source containing the existing hook.

## Finding

There is **no demonstrated equivalent replacement of the complete `(phase, step, seq_len)` tuple using the existing CPU scheduler or AsyncOutput fields**. Existing CPU metadata can identify prefill versus scheduled draft work, and an async sample-count snapshot can classify an already executed forward. It cannot distinguish a denoising forward that has just converged from one that will continue denoising before the next attention call. CPU sequence length is explicitly an upper bound. Waiting for an async copy inside the next prepare hook could move the synchronization instead of removing it.

The current adapter obtains the tuple with `torch.stack(...).tolist()` at lines 476–477, combines phase with CPU `num_draft_tokens` at lines 478–479, and skips this read for CUDA graph capture. Its consumers are consequential: encoder/commit routing invalidates canvas state, while denoising calls use the exact step in `state.begin` (`vllm_adapter.py:310–329`). A guessed phase or step would change the method's clock or cache lifecycle.

## Official source evidence

| Source | Observed behavior | Consequence |
|---|---|---|
| `vllm/model_executor/models/diffusion_gemma.py:559–569` | Sampling reads the current `is_encoder_phase` as commit status. Denoising increments step; commit resets step to zero. | A CPU shadow needs the actual commit transition, not just a forward count. |
| `vllm/model_executor/models/diffusion_gemma.py:602–610` | A current commit emits its actual canvas token count; a denoising forward emits zero sampled tokens. | Ordinary denoising and just-converged denoising have the same existing CPU sample-count output. |
| `vllm/model_executor/models/diffusion_gemma.py:619–626` | GPU stability/confidence/history/max-step conditions set the next encoder/commit phase. | The next phase is not recoverable from the existing sample count alone. Reimplementing those conditions on CPU is not an equivalent metadata substitution. |
| `vllm/model_executor/models/diffusion_gemma.py:1309–1313` | The sampler clones current commit status before the compiled sample step mutates the persistent phase state. | Current-execution identity and next-state identity must be kept separate. A live view of persistent state can become stale. |
| `vllm/v1/worker/gpu/sample/output.py:15–22` | `SamplerOutput` includes sampled IDs, sample/rejection counts and sampling-mask fields, but no phase or step snapshot. | Existing AsyncOutput has no complete native clock receipt to reuse. |
| `vllm/v1/worker/gpu/input_batch.py:74–76` | GPU `seq_lens` is separate from CPU `seq_lens_cpu_upper_bound`; the latter is explicitly an upper bound. | Substituting the CPU field for the adapter's exact GPU sequence length is not justified. |
| `vllm/v1/worker/gpu/model_runner.py:1184–1189, 1412–1420, 1442–1448` | CPU computed-token state comes from scheduler output; CPU sequence upper bounds add scheduled-token upper bounds. Both exact GPU lengths and CPU bounds enter the input batch separately. | CPU scheduled progress is not a guarantee of actual execution progress. |
| `vllm/v1/worker/gpu/input_batch.py:358–366, 606–609` | GPU sequence length uses actual computed length and query length. Post-update changes computed length by query length minus rejected tokens. | The exact length depends on GPU execution outcomes, not just optimistic scheduled lengths. |
| `vllm/v1/worker/gpu/async_utils.py:115–165` | AsyncOutput's copy stream waits on execution and asynchronously copies sampled tokens and counts, including `num_sampled_tokens_np` at line 149. | It is an existing route for execution receipts, but does not currently carry the needed clock tuple. |
| `vllm/v1/worker/gpu/async_utils.py:167–178` | `get_output` synchronizes the copy event before reading CPU lists and trimming them by sampled count. | Reading the pinned buffer earlier is unsafe. Adding this wait to `prepare_attn` does not establish a synchronization-free optimization. |

Pure prefill is more amenable to CPU classification: `diffusion_gemma.py:1152–1169` determines completed prefill from CPU prefill lengths and scheduled work, then initializes the canvas. This does not prove that the exact sequence-length substitution or the complete decode tuple is available on CPU.

## Async execution and identity boundaries

The earlier controlled diagnostic observed **350 actual denoising forwards versus 349 scheduler-consumed denoising forwards**, with six prefill and 23 consumed commit updates and no adapter ordering errors. This is evidence of a terminal executed-but-unretired forward, not a universal fixed offset. The existing `scripts/v27_vllm_metrics.py:126–189` execution hook retains the already-created async sample-count snapshots and reads them after the request boundary synchronization. It is appropriate for final accounting, not for predicting the next forward's routing.

A future snapshot must distinguish request ID, request epoch, execution ordinal and slot. Slot alone is unsafe after request retirement/reuse; scheduler-consumed ordinal is unsafe when execution runs ahead. Phase/step tensors are persistent GPU state, so retaining a view does not freeze their value. CPU buffers must not be reused before the intended reader is done. Capture, warm-up, partial final commits and unused terminal executions also need explicit identities.

## Candidate boundary and validation proposal — not implemented

One research candidate is to extend native execution output with immutable current/next phase and step metadata near the sampler return / AsyncOutput construction, and to obtain exact computed-length metadata after its GPU post-update. It could share the existing async copy path. Those are candidate insertion boundaries, not proof that the next `prepare_attn` can use the receipt without waiting. Async scheduling may submit that next forward before the scheduler retires the preceding result. Merely adding a copy-event wait at the prepare hook would relocate the present stall and could reduce overlap.

Before any routing change, a read-only shadow trace should compare the candidate tuple with the current exact read at the same execution boundary. It should cover first/last prefill, denoising convergence to commit, commit to the next canvas, partial final commit, terminal unused forward, request/slot reuse and capture exclusion. Verify actual versus consumed counts separately, method begin/observe/invalidation order, and private completion equality without changing RNG or native stopping. A candidate must also demonstrate that the needed snapshot is naturally ready before use; otherwise a GPU-resident routing design or scheduling change would be a separate implementation project.

No shadow trace, metadata optimization, new GPU experiment or additional test was implemented for this audit. Existing preview results remain results of the frozen adapter, including its metadata-read cost.
