# Clean/profile execution-path source review

This is a completed read-only source review, not a completed experimental panel or a timing result. The profile execution is internally repeatable on the repeated short request but does not reproduce the clean trajectory. Its phase costs cannot be transferred into a clean request-latency saving or critical-path bound. The initial source of divergence remains unresolved. The external trace of the original clean worker is pending at this review snapshot.

## Frozen identity and review scope

The review compares original attempt001 (source `6aab86325990edf6e5ab6bb72bcd909781234e97`) with successor attempt003 (source `db755c43be763a8af9edfe8a147ec333d8b8542b`). `receipt.json` records both protocol hashes, actual hashes of every reviewed frozen file, the verified native runtime source hash, and immutable projections/prefix hashes of the public evidence read. No GPU work, model execution, or edits to execution source or prior analysis were performed.

The benchmark, diagnostic worker, V31 adapter and causal runtime have identical hashes in both frozen source bundles. Changed frozen files are the profiling collector, coordinator, report, their tests, and the recovery specification. The inherited V31 control remains Yuhan's attention-mass selection and cross-step reuse; this review introduces no method variant.

## Concrete execution differences

Source pointers below refer to files inside the frozen execution bundle for the stated attempt, not to an assumed current checkout. Runtime pointers refer to the pinned package file whose hash is in the receipt.

| Difference | Exact source pointer | Implication |
|---|---|---|
| Clean/profile flags change together: clean uses VALUE_AUDIT=0, VALUE_CLEAN_TIMING=1, VALUE_BOUNDARY_TIMING=1; profile uses the inverse values and HEADROOM_PROFILE=1. | attempt003 `scripts/v31_refresh_diagnostic_study.py:555–572` | Headroom hooks are not the only changed execution condition. |
| Clean synchronizes at request/prefill boundaries; profile synchronizes after each engine step. CPU/CUDA generators are reseeded before every request in both. | both `scripts/v31_vllm_paired_bench.py:400–426` | Launch/host scheduling differs; no different native seed policy was identified. |
| Profile adds GLOBAL/LOCAL CUDA events, physical-tile accounting tensors and nonfinite attention reductions. Clean disables split-map accounting. | both `experiments/numerical_qk_reuse/vllm_adapter.py:1024–1062`, `2115–2135`, `2341–2379` | More launches, allocations and retained audit tensors; these branches do not directly change the selected support or returned attention tensor. |
| Dense profile installs geometry metadata GPU-to-CPU reads and nonfinite-logit reductions. | both `experiments/numerical_qk_reuse/v31_value_native_audit.py:37–40`, `76–85`; benchmark `:293–305` | Dense profile also differs from clean beyond CUDA-event timing. |
| Headroom adds current-stream CUDA events; torch.profiler/record_function is entered for one frozen long request. Inactive hooks call the original implementation directly. | attempt003 `scripts/v31_refresh_headroom_profile.py:221–269`, `353–365`, `398–407` | Hook overhead and allocation/scheduling changes are present, but source review found no direct model-tensor mutation or added random draw. |
| Headroom disables factorial state tracing; control/fixed16 call inherited `_mage` directly. | both `scripts/v31_refresh_diagnostic_worker.py:133–150`; `experiments/numerical_qk_reuse/v31_refresh_diagnostic.py:173–208` | No extra factorial sparse oracle or extra native model/sampler call runs in these headroom requests. |
| The same requested graph modes are selected by family, and audit only records native preparation modes. | attempt003 coordinator `:557–558`; both adapter `:2260–2273`; native audit `:76–85` | No source-level audit branch selecting a different graph mode was found. |

Audit receipt reductions read tensors but do not replace native attention output. The scan found no new forward, RNG draw, native sampler substitution, or direct mutation of native model tensors in the headroom hooks. This is a bounded source-audit finding, not proof that the added asynchronous operations cannot expose a runtime issue.

## Evidence that narrows the explanation

For LongBench-v2 item339, the clean dynamic control has N=325/T=3965 and the profile dynamic control has N=276/T=3700. Clean fixed16 has N=146/T=2024; profile fixed16 has N=263/T=3139. Within each engine, each explicit warmup has the same output hash/N/C/T as its corresponding measured short request. The profile engine's short requests precede its single torch.profiler session (long dynamic request index5). Thus that single torch.profiler session alone cannot explain the already observed short-request divergence. CUDA-event instrumentation and other audit changes are active on those short requests.

The dense FULL short request similarly repeats internally but differs across passes: clean N=234/T=3452 versus profile N=354/T=4309. The native long dynamic profile records PIECEWISE for all190 denoising forwards; the native dense long profile records FULL for all993 forwards. Every reviewed clean/profile public row records zero new CUDA captures. Clean does not record native mode-dispatch counts, so the mode evidence is bounded by its requested/resolved mode and profile's observed preparation mode.

Compiler context also differs. Original shared clean engine2 loaded an existing AOT compiled function (`attempt001/headroom/engine002_longbench_v2_shared_clean/attempt001/private/worker.log:59–61`); successor shared profile engine6 compiled a graph afresh (`attempt003/headroom/engine006_longbench_v2_shared_profile/attempt001/private/worker.log:59–64`). Both original and successor FULL engines compiled separately (engine3 `:59–64`, engine7 `:59–64`). Cache roots are attempt-specific (`scripts/v31_v2_local_threshold_sweep.py:217–223`). No generated-artifact numerical equivalence was established. Cache/compiler state and the whole audit/timer bundle remain confounded; these observations do not identify a cause.

## Separate cross-engine context example

Item422, panel seed42, provides additional context from audited qualification and the still-running main factorial engine9. It is not a clean/profile isolation experiment.

| Context | Control before / dense_fresh / control after | N | C | T | Output hash |
|---|---|---:|---:|---:|---|
| attempt001 qualification engine1 | All three agree | 515 | 24 | 6035 | `f6e0aa2a94f74b84` |
| attempt003 factorial engine9, captured complete rows | All three agree | 403 | 14 | 3357 | `b2faa42a5f82b590` |

These rows match on item, seed, computed RNG seed, prompt/manifest hashes, output budget, context limit, chunk/page size and step-synchronized timer policy; all record zero captures. They run at different positions (qualification request indices5/8/10 versus factorial35/36/40), with different warmup/cohort history and per-attempt compiler/cache context. The evidence shows that within-engine no-op/A-A agreement does not establish cross-engine reproducibility. It does not identify warmup, cache, order, instrumentation, or any specific run-configuration component as the cause. Planned independent main-engine repetitions will assess stability if completed; this review does not assume their result. Engine9 was still active when the complete-record prefix was captured, and its file may subsequently grow.

## Native amplification and unresolved boundary

The pinned `vllm/model_executor/models/diffusion_gemma.py:528–548` uses Gumbel argmax and entropy confidence, `:550–556` forms discrete entropy-sorted acceptance, and `:614–646` combines exact token-history equality with confidence for stopping and writes probability-derived self-conditioning for the next step. These operations provide a concrete mechanism by which a small upstream numerical difference can change subsequent tokens, N/T, refresh events and later RNG consumption. They do not locate the first difference. The free-memory-dependent sampler grouping branch at `:1323–1330` cannot change the group size here: batch1 forces group1.

The supported conclusion is: the instrumented pass is internally repeatable on its duplicated short inputs but does not reproduce the clean trajectory; its costs are diagnostic for that execution. The cause is not isolated, and a clean critical-path refresh saving remains unresolved. The prepared external trace preserves the original clean worker, audit flags, native limits and copied cache; its parity/segmentation results are pending and must not be anticipated.
