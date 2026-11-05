# Aggressive BLASST: completed two-model evaluation

All six new panels (768 requests) are complete. All earlier results remain unchanged.

| Model | Benchmark | Native dense | All-kept FA4 | V31 | V3a | V2-CTI | Aggressive BLASST | BLASST sparsity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llada22 | aime26 | 40.00% | 32.22% | 23.33% | 27.78% | 30.00% | 0.00% | 41.12% |
| llada22 | humaneval | 85.98% | 84.15% | 50.61% | 62.80% | 72.56% | 1.22% | 46.29% |
| llada22 | ruler128k | 85.44% | 84.82% | 86.01% | 85.03% | 85.21% | 84.01% | 48.20% |
| trado | aime26 | 8.89% | 4.44% | 3.33% | 2.22% | 8.89% | 10.00% | 50.69% |
| trado | humaneval | 79.27% | 78.05% | 42.68% | 47.56% | 74.39% | 74.39% | 51.96% |
| trado | ruler32k | 64.64% | 64.85% | 63.49% | 64.96% | 64.72% | 63.79% | 48.03% |

AIME is exact-match accuracy over 30 questions × three seeds (42/43/44). HumanEval is official functional pass@1 over all 164 questions, seed 42. RULER is official fractional retrieval credit over 130 questions, 10/task across 13 tasks, seed 42; 128K on LLaDA and 32K on TraDo.

New BLASST work and termination statistics:

| Model | Benchmark | Full credit | Mean output tokens | N total | C total | Steps/request | Canvases/request | Steps/canvas | Sparsity | EOS / caps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llada22 | aime26 | 0/90 | 32.3 | 1232 | 216 | 13.69 | 2.40 | 5.704 | 41.12% | 90 / 0 |
| llada22 | humaneval | 2/164 | 32.5 | 1516 | 406 | 9.24 | 2.48 | 3.734 | 46.29% | 164 / 0 |
| llada22 | ruler128k | 98/130 | 28.3 | 1049 | 236 | 8.07 | 1.82 | 4.445 | 48.20% | 130 / 0 |
| trado | aime26 | 9/90 | 1030.5 | 46787 | 23256 | 519.86 | 258.40 | 2.012 | 50.69% | 87 / 3 |
| trado | humaneval | 122/164 | 114.8 | 9568 | 4832 | 58.34 | 29.46 | 1.980 | 51.96% | 164 / 0 |
| trado | ruler32k | 70/130 | 29.9 | 2652 | 1078 | 20.40 | 8.29 | 2.460 | 48.03% | 127 / 3 |

Calibration uses only small, count-only native trajectories. Short benchmarks use the same six prompt-length strata as V2-CTI; RULER uses the existing 13 independent retrieval development questions. No V2 thresholds, gold or scores select BLASST parameters.

| Model / benchmark | log κ | Calibration sparsity | Full sparsity | Target status |
|---|---:|---:|---:|---|
| llada22 / aime26 | 22.5 | 42.44% | 41.12% | Calibration target missed; full result in allowed 40–60% range |
| llada22 / humaneval | 15 | 40.98% | 46.29% | Calibration target missed; full result in allowed 40–60% range |
| llada22 / ruler128k | 2.3977134 | 48.14% | 48.20% | 48–52% calibration target reached |
| trado / aime26 | 3.2698565 | 48.68% | 50.69% | 48–52% calibration target reached |
| trado / humaneval | 6.2121233 | 49.18% | 51.96% | 48–52% calibration target reached |
| trado / ruler32k | 2.8654441 | 48.70% | 48.03% | 48–52% calibration target reached |

**The two LLaDA short-task cells are not exact 50% work matches.** Their ten-candidate searches reached the first/current-tile protection floor after outputs became short: HumanEval calibration plateaued at 40.98%, AIME at 42.44%. The frozen policy selected the closest measured sparsity; full panels reached 46.29% and 41.12%, inside the predeclared 40–60% acceptance range, so no target-driven rerun occurred. This does not prove no alternative threshold can reach 50% on another trajectory.

LLaDA BLASST outputs averaged 32.5 HumanEval and 32.3 AIME tokens, versus 219.3 and 5964.2 for current-tile V2. All 164/90 ended by native EOS; none reached the cap. These measurements support a short-output failure on these two cells; this study does not identify an exact causal EOS perturbation. TraDo HumanEval equals V2-CTI at 122/164, with different per-item outcomes; paired difference 0.00 pp, 95% CI [−5.49, +5.49].

BLASST uses the existing preceding maximum over all seen tiles, including skipped tiles, strict all-row skipping, first/current protections and λ(L)=exp(log κ)/L. Finite λ>1 is allowed; this is an aggressive extension without an accuracy guarantee. It uses no value projection or CTI initialization. The new model ports use FP32 accumulated scaled QK scores consistent with their FA4 consumer; the historical Gemma implementation had its own BF16 logit rounding.

Scope matches current experiments: all denoising layers, Q128/KV64 tiles, four-call map reuse, no cross-canvas carry, full-D128 FA4 output and dense prefill/commit. Native samplers, EOS and editing are unchanged. LLaDA uses 32-token blocks, steps 32, threshold 0.5, editing threshold 0, max post steps 16, temperature 0; TraDo uses 4-token blocks, 4 steps, temperature 1, top-p 1, top-k 0 and confidence 0.9. Caps: LLaDA AIME 32768, HumanEval 1024; TraDo AIME/HumanEval 2000; both RULER 128. BF16, batch 1, one H100.

Sparsity is one minus pooled actual kept/eligible denoising tiles, including mandatory tiles and selection/reuse calls. Dense prefill/commits and selector discovery are outside that denominator. Discovery QK pairs and instrumented GPU-worker seconds remain available in canonical evidence; no clean latency or speedup claim is made.

Validation: 11 focused CPU tests, 120 independent CUDA mask/FP32 consumer cases, six exact historical all-kept replays, 24 complete shards, 768 private-token/source/config joins and unchanged historical-file hashes. An additional 18-case CPU comparison against the original BLASST core matches exactly. Two pre-calibration integration failures are preserved; they contributed no benchmark records.

Canonical [full method/work table](../analysis001/README.md), [summary](../analysis001/summary.json), [paired question-cluster intervals](../analysis001/paired.json), [official audit](../analysis001/receipt.json), [frozen protocol](../protocol.json), and [completion](../final_complete.json). Calibration exposure, lambda ranges and failed attempts remain documented.
