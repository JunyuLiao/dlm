# Timing boundaries (v26)

| Metric | Where measured | Instrumentation |
|---|---|---|
| Quality | first output of each cell (scored offline with the frozen scorer) | none |
| Whole-request wall (W) | accepted warm repeat of the same cell, same GPU, same config; synchronized boundary | none (no profiler, no per-step sync) |
| Calls, canvases, tokens, B0/BO/A/D/H | first-output receipt counters (host-side) | counters only |
| W/N (per call) | W divided by decoder calls of the same accepted cells | derived; amortized request cost, not a direct forward price |
| Direct model_forward / denoising_step | separate v24/v25 replay profiles on captured native states | CUDA events around whole calls; not in request runs |
| Attribution of the remaining time | not measured yet (W-trace pending) | would use a separate profiler run |

A GPU timeline span (generation_gpu_timeline_seconds) includes host gaps and later encoder work; it is not TBT. TBT is N/A (no committed-chunk emission timestamps).
