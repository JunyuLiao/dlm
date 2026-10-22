# Optimized refresh native panel, 2026-10-22, attempt005

Status: gated campaign running; no completed benchmark claim. See state.json, coordinator.log, and final_complete.json when it exists.

Source checkpoint: 6ecdfbf6. Best V2 SM90 interior/tail library 200354cb919fb6b8 with fused current-V Gaussian32 projection/packing, true-online conditional PV, and resident direct-paged held/carry consumer. V31 uses fused FA4 observation/top-k, direct paged above 4K. All-kept uses the same resident consumer and full physical support. Synthetic V2 refresh remains slower than dense; no speedup is inferred.

Coverage: AIME26 30xseeds42/43/44 K128; MATH500 500x42 K128; HumanEval164x42 K128; LongBench-v2 503x42 K8192. V2 calibrated separately per workload against full-request actual held GLOBAL eligible/kept tiles, target40-50%.

Every measured input gets a separate same-input/seed warm request. Audit and clean passes are separate, actual CUDA capture and universal JIT event counters apply to all arms, and full panels require zero-event qualified smoke pairs first. Native sampler, LOCAL, prefill and commits are preserved. Full model content hashes verified. Shards contain at most128 requests to amortize startup and bound recovery.

The corrected resident all-kept HumanEval audit/clean pair already passes:133120/133120 GLOBAL tiles kept,280 calls,0LOCAL skips,0K/Vcopies,0measuredJIT/captures,identical output hash and denoising-call trajectory. These are qualification diagnostics, not benchmark outcomes.

The prior 2026-10-21 attempt004 remains paused/partial. All newer failed preparations and partial qualification attempts001-004 remain preserved. Attempt004 stopped on an omitted all-kept tile-total receipt and launched no calibration or scored panel. Raw inputs, completions and logs remain private.
