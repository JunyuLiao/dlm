# Stable cheap refresh for V31 — 2026-10-13

Frozen development study: 64 LongBench-v2 seed-1 cells and 30 AIME26 problems at seeds 42/43/44, 154 cells per arm and 2,002 native cells across 13 arms. Previously examined development data; the 503-cell audit is a later decision. The inherited cross-step lifecycle is attributed to Yuhan. This study owns the separately named cheap-refresh variants.

The pinned DiffusionGemma/vLLM/SM90 FA4 runtime, sampler, protected support, GLOBAL-only layer scope and per-head/Q128 budget remain fixed. K is 128 prefix KV64 tiles, capped for all-kept short prefixes. LOCAL, prefill and commits remain native dense. Output computation uses the native current attention path; selector cache statistics never replace model K/V or outputs.

The operator-only choice is hysteresis 0, persistence 2, event cap 2, cumulative cap 8, removal-mass tolerance 0.025 and no outside probes. Zero hysteresis is explicitly reported; stability comes from persistence, caps and the row guard. Every required no-progress frequency uses these same frozen parameters. Bounded B8 and probing remain offline diagnostics.

Correctness gates: 38/38 pinned H100 tests; 620 real snapshots across both workloads and all five GLOBAL layers; all 670 Q/K/V files, including 50 immutable prefixes, are fingerprinted. Retained statistics match dense observation within 1.91e-6. Two three-cell native repetitions have zero output-hash, denoising-N or output-length differences, and updates-disabled collection changes no masks.

Controls: dense_full_fix51994, dense_piecewise, allkept_fa4, v31_progress_dense_control. The nine required method arms and aggregate trajectory criteria are fixed in specs/protocol.json. Source identity, selected parameters and official scoring inputs are frozen under specs/. Operator attempt004 is the production-matched replay. Previous replay attempts remain immutable and exploratory frequencies are labelled separately.

Generation uses four mixed-workload shards with rotated arm order and one H100 worker. Paths are native/shardNN/arm_name/attemptNNN/. Failed attempts remain intact; only completed, source-validated leaves may be reused. Official scorers and problem-clustered intervals run after the complete matrix. No development accuracy has been scored at this freeze.

Reports will separate operator diagnostics, official native accuracy, denoising trajectory, selector work/overhead, and the intentionally absent clean timing. No speedup is claimed. Clean timing stays paused. Private tensors, manifests and completions remain excluded from Git. Predecessor corrections are in specs/first_iteration_correction.md and specs/source_audit.json.
