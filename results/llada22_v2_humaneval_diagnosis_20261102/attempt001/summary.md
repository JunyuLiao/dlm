# LLaDA2.2 V2 HumanEval diagnosis

## Conclusion

The low score is a real V2 trajectory failure on long-form code completion, not an output-cap truncation or scorer/extractor artifact. The proximate failure is early EOS and incomplete code. This is a task-format sensitivity of the sparse attention trajectory: V2 at 49.22% achieved consumer sparsity still emitted EOS in every request, usually before completing a parsable function. The current evidence supports that diagnosis strongly, but does not yet isolate the exact attention perturbation that flips a particular EOS decision.

## Full-panel evidence

| LLaDA2.2 HumanEval arm | Correct /164 | Mean output tokens | Median | Max | EOS finished | At 1,024-token cap | Syntax errors | Parsed, entry point present | Docstring-only / pass-only parsed functions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native dense | 141 | 243.1 | 224 | 783 | 164 | 0 | 2 | 162 | 0 / 0 |
| All-kept FA4 | 138 | 249.2 | 228.5 | 1,024 | 163 | 1 | 2 | 162 | 0 / 0 |
| V31 mass reuse4 | 83 | 218.9 | 193 | 788 | 164 | 0 | 31 | 128 | 5 / 2 |
| V2 Gaussian32 reuse4 | 10 | 62.9 | 58 | 155 | 164 | 0 | 98 | 61 | 21 / 12 |
| V3a singleton reuse4 | 103 | 245.3 | 219.5 | 1,024 | 163 | 1 | 21 | 140 | 3 / 2 |

V2 produced 61 parseable functions; 21 contained only a docstring, 12 were pass/ellipsis stubs, and 28 had substantive bodies. Only10tasks passed the official tests, so shortness is the leading failure mode rather than the sole failure mode. Another 98 outputs did not parse; two parsed outputs lacked the required entry point, and three had no final response. 155/164 outputs were at most128tokens, while the maximum output was155 tokens, far below the1,024-token cap. All164 ended through the model's EOS stop path.

The contrast is task-specific: LLaDA2.2 V2 averaged52.18 output tokens/request on AIME26 (28/90 correct) and28.12 tokens/request on RULER128K (86.23% official score), where short numeric/retrieval answers suffice. HumanEval requires a complete executable function; dense and V31 typically produced over200 tokens/request. This pattern explains why an EOS-truncated trajectory is much more damaging on HumanEval. It does not prove that the benchmark itself or its scorer is faulty.

## Calibration and implementation checks

HumanEval V2 threshold0.8944271909999159 was selected by physical sparsity on six hand-written development coding prompts; the final confirmation was48.49%. Those six prompts had median169 input tokens, close to HumanEval's161-token median, but their generated outputs averaged69.8 tokens and they were not official HumanEval tasks. Calibration did not score code correctness or require a complete-function/EOS check. The full HumanEval panel achieved49.22% sparsity, within the 48-52% target, so aggregate sparsity drift does not explain the score collapse.

The numerical run used the expected V2 integration: rank-32 Gaussian value projection, online preserve-mass recurrence, mandatory-tile protection, Q128/KV64 maps, reuse every fourth call, and the unchanged native LLaDA2.2 editing/EOS sampler. All20 layers were eligible, the generation cap was1,024, and dense prefill/commit remained outside sparse routing. The V2 output consumer used current full-dimensional V. Runtime and attention module hashes match the pinned LLaDA2.2 HumanEval benchmark configuration.

A source-pinned replay of HumanEval item0 with seed42, the frozen V2 threshold, and the original numerical worker reproduced the saved110-token failing output exactly; repeating it in the same process reproduced the same hash. At a captured layer10 refresh (first call in the canvas), the actual V2 map had34 kept entries among80 eligible head/tile units, including16 mandatory entries. It matched both exact CUDA recomputation and the independent CPU V2 recurrence: zero map mismatches. This rules out a map/count integration discrepancy in that observed refresh and makes a benchmark-output/scoring mismatch unlikely. It does not by itself validate every layer/state or prove which attention difference caused EOS.

## Limit

A direct same-state comparison of EOS logits under V2, V31, and all-kept attention was planned as attempt002, but did not launch: host execution approval timed out before the command ran, including its single retry. Therefore this report does not claim that a denser route would reverse the EOS prediction on an identical state. The existing paired full benchmark, exact V2 replay, and selector recurrence audit support a method/task trajectory diagnosis without that counterfactual.

## Recommendation

Do not use the current LLaDA2.2 V2 HumanEval result as an accuracy-preserving operating point, and do not attribute it to the scorer or the1,024-token cap. Treat V2 as failing long-form code generation at this calibration. Any follow-up should test EOS/code completion behavior alongside physical sparsity; the six-prompt count-only calibration did not cover that quality dimension.

Sources: original campaign `results/multi_model_value_20261030/attempt001/`; this diagnosis's `protocol.json` and `receipt.json`; final aggregate `results/multi_model_value_report_20261102/attempt002/summary.json`. The replay item is previously examined benchmark data, not held-out evidence.
