# Optimized refresh native panel, 2026-10-22, attempt004

Status: launched, gated, incomplete. See state.json and coordinator.log. The completion claim requires final_complete.json and all four scored suite summaries.

The frozen source checkpoint is 06ee4806. The best V2 library is unchanged from sparse_execution attempt018; its full-current-V 8K synthetic refresh remains slower than dense. The native integration now uses the matching fused projection/pack wrapper and resident direct-paged held execution. V31 direct paged refresh is used above 4096 keys. Resident all-kept uses full support in the same held consumer.

Coverage: AIME26 30x42/43/44, MATH500 500x42, HumanEval 164x42, LongBench-v2 503x42; V31 K128 for short benchmarks and K8192 for LongBench. V2 calibrates independently per workload to 40-50% actual held GLOBAL physical sparsity over full native requests.

The durable service runs qualification, calibration, panel, scoring in order, stopping on any failed gate. Every measured input has an explicit preceding warm request. Audit and clean are separate. Universal JIT and actual CUDA capture counters are mandatory for all arms. Logs, raw inputs and completions remain private. Source, protocol, runtime, model and input hashes are frozen.

The old 2026-10-21 attempt004 remains paused/partial. New attempt001 failed preparation, attempt002 exposed a list-vs-zero validator defect, and attempt003 is preserved as a 27-row partial qualification diagnostic. No records from those attempts form a completed final panel.
