# v23 Track E: GQA K replication at score observations

**Source fact (HEAD `6b795ca6`).** `Attention.observe_scores()` expanded K with `k.repeat_interleave(Hq // Hkv, dim=1)` before the BF16 score matmul. This runs only at true score observations (A calls, and the new bootstrap observation). The sparse consumer already indexes the KV head and never repeats K.

**Change.** An opt-in `observe_scores_grouped` reshapes Q to `[B, Hkv, G·Q, D]` (only the small Q is copied), multiplies it by native `K^T`, then views the result as `[B, Hq, Q, K]`. The BF16 scaling, mask, finite handling, `-inf` fill and FP32 publication are shared code (`_finish_scores`). It is enabled per arm by `observation_producer='grouped_q'` in the v21 wrapper config. Absent means the legacy producer, so existing arms and their fingerprints are unchanged.

**Qualification (stage `v23_obs_b80d116`, both H100s, 80.1 GPU-s).** Both producers ran on the same captured M3 method-path Q/K at 23 reached GLOBAL layer-5 states. Scores were compared bitwise, including `-inf` and NaN payloads via int32 views, and the unchanged route kernel was run on each output. Timing: 5 rotated repetitions after a warmup, no JIT miss, with peak allocation tracked above the pre-call baseline.

| Task | States | Keys | Scores and route bitmap bitwise equal | Producer ms, repeat → grouped | Median ratio | Peak transient MiB, repeat → grouped |
|---|---:|---|---|---|---:|---|
| LongBench-v2 | 8 | 13,703 / 17,540 | 8/8 yes | 1.45–1.49 → 1.16–1.28 | 0.83 | 856–1096 → 646–826 |
| RULER-4K | 7 | 4,084 / 4,216 | 7/7 yes | 0.34–0.37 → 0.30–0.32 | 0.86 | 255–264 → 196–202 |
| AIME26 | 8 | 365 / 608 | 8/8 yes | 0.12–0.16 → 0.12–0.15 | 0.98 | 23–40 → 22–33 |

**Accepted as exact.** Bitwise identity held on every tested state, and cuBLAS did not change rounding at these shapes. This is a measured property of these shapes on this stack, not a theorem. The frozen bootstrap6 panel applies the grouped producer to every method arm that uses this observation (incumbent M3 and the bootstrapped M1/M3/B). T_scope uses Junyu's own producer and is unchanged.

**Size of the effect.** About 0.2–0.3 ms per GLOBAL layer per observation on LB, i.e. roughly 1–1.5 ms per A call over the five GLOBAL layers. A calls are only every 8th call, plus one bootstrap observation per canvas, so the amortized effect on a request is on the order of 0.1% of decoder time. The ~210–270 MiB drop in transient peak per observed layer is the more tangible benefit for long contexts. Complete A-call, model-forward and denoising-step prices with this producer were not separately profiled in this window. The repeated-K tensor is 8× the K payload (Hq16/Hkv2); for example 17,540 × 512 × 2 B × 2 heads × 8 ≈ 274 MiB. That is size arithmetic, not a bandwidth measurement.

This is engineering hygiene, not a method contribution.
