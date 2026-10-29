# Optimized V2/V31 native panel, 2026-10-22

Use the best qualified SM90 interior/tail online V2 library from sparse_execution attempt018. Native V2 uses fused current-V Gaussian32 projection and sketch packing, online full-QK/softmax/risk, conditional full PV, and compact-mask construction. The refresh kernel is unchanged between audit and clean passes.

Use the resident direct-paged FA4 consumer on V2 and V31 held/carry steps. Held calls do no selector work, list construction, or K/V copy. Keep graph-bucket capture disabled for the timed panel; the worker's ordinary model warmup precedes measured cells. V31 uses FA4 score observation and fused top-k, with direct-paged refresh above 4K keys.

Compare V2, V31, corrected FULL dense FA4, PIECEWISE dense FA4, and all-kept FA4 under the native sampler. Thinking is on; canvas=256, maximum denoising calls=48, entropy bound=0.1, confidence=0.005, stability=1, temperature=0.8 to 0.4. GLOBAL is sparse; LOCAL and prefill remain dense.

V31 budgets: AIME26 K128 (30 questions x seeds 42/43/44), MATH500 K128 (500 x seed 42), HumanEval K128 (164 x seed 42), LongBench-v2 K8192 (503 x seed 42). Calibrate V2 separately per workload against actual held-phase GLOBAL eligible/kept Q128 x KV64 tile counts. Target held physical sparsity is 40-50 percent. Preserve every calibration candidate and report refresh, carried, first-canvas, held, and whole-request work separately.

Gate native panel launch on synthetic parity/changing-input graph checks, native audit/clean trajectory agreement, exact kernel/selector receipts, zero extra model forwards, zero LOCAL skips, and zero captures or JIT events in clean timing. Keep audit and clean timing as separate passes. Score with the official benchmark scorers and report paired/item-clustered uncertainty, accuracy, N, N/C, T, S/N, S, W, refresh/held latency, physical tile work, and source/runtime identities.

Calibration uses the full native generation budget on four deterministic prompt-length strata per workload. The same calibration questions remain in mandatory full coverage and are disclosed as transductive. Every measured request is preceded by a separate untimed warm request with the same input and seed; all warm request cost remains in resource accounting. Universal request JIT events and actual CUDA capture counters apply to every arm, including native dense and resident all-kept. All-kept FA4 uses the same resident direct-paged consumer with full physical support. No final-panel launch unless calibration held GLOBAL physical sparsity is within 40-50 percent.
