# Multi-model value-direction evaluation: completed retained scope

Value information can help over attention-mass ranking, but the effect is model- and task-dependent. TraDo V2 on HumanEval is the strongest lead. LLaDA2.2 V3a improves HumanEval over V31, whereas V2 fails badly on that same model/task. No reliable AIME or RULER accuracy improvement is established. These ports do not isolate K/V tying causally.

Completed: 30 panels / 4010 retained benchmark requests. Two original AIME V2 panels are replaced by 180 freshly generated requests. Ultra AIME/RULER were cancelled by the user; TraDo128K is unsupported. The original 40-panel protocol is not reported as complete.

## Accuracy

| Model | Task | Native dense | All-kept FA4 | V31 | V2 | V3a |
|---|---|---:|---:|---:|---:|---:|
| LLaDA2.2-mini | aime26 | 40.00% (36/90) | 32.22% (29/90) | 23.33% (21/90) | 31.11% (28/90) | 27.78% (25/90) |
| LLaDA2.2-mini | humaneval | 85.98% (141/164) | 84.15% (138/164) | 50.61% (83/164) | 6.10% (10/164) | 62.80% (103/164) |
| LLaDA2.2-mini | ruler128k | 85.44% | 84.82% | 86.01% | 86.23% | 85.03% |
| TraDo-8B-Instruct | aime26 | 8.89% (8/90) | 4.44% (4/90) | 3.33% (3/90) | 6.67% (6/90) | 2.22% (2/90) |
| TraDo-8B-Instruct | humaneval | 79.27% (130/164) | 78.05% (128/164) | 42.68% (70/164) | 76.22% (125/164) | 47.56% (78/164) |
| UltraLLaDA | humaneval | 1.22% (2/164) | 1.22% (2/164) | 0.00% (0/164) | 0.00% (0/164) | 0.00% (0/164) |

AIME:30 problems x seeds42/43/44. HumanEval:164 problems, seed42. RULER:130 problems across13 tasks, seed42; official fractional retrieval score rather than all-or-nothing accuracy. Question-clustered bootstrap intervals use10,000resamples and are exploratory, without a multiple-comparison correction. Repeated deterministic-seed outputs are not independent observations.

## What the comparisons support

- TraDo HumanEval V2 versus V31: +33.54pp, 95% problem-clustered CI [25.00, 42.07]pp.
- LLaDA HumanEval V3a versus V31: +12.20pp, 95% problem-clustered CI [3.05, 21.34]pp.
- LLaDA AIME V2 versus V31: +7.78pp, 95% problem-clustered CI [-6.67, 22.22]pp.
- TraDo AIME V2 versus V31: +3.33pp, 95% problem-clustered CI [-1.11, 8.92]pp.
- LLaDA RULER V2 versus V31: +0.22pp, 95% problem-clustered CI [-2.13, 2.62]pp.

TraDo V2 HumanEval reaches125/164, versus V3170/164 and native dense130/164. Its -3.05pp difference from dense has CI[-7.93,+1.22]pp; this does not establish noninferiority. LLaDA V3a HumanEval reaches103/164 versus V3183/164, still below native141/164. LLaDA V2 HumanEval is10/164 despite49.22% sparsity; K/V separation alone does not resolve value-router failures.

## Denoising and output work

N is total denoising forwards; C is total canvases; T is emitted output tokens. N/C below is pooled across the full cohort. Prefill/commit forwards remain separate. The full30-row table is [summary.md](summary.md); per-request mean N/C, refinement/edit counters, discovery work and paired intervals are in summary.json and paired.json.

| Model/task | Arm | N | C | N/C | T |
|---|---|---:|---:|---:|---:|
| LLaDA/aime26 | V31 | 119064 | 13271 | 8.972 | 422252 |
| LLaDA/aime26 | V2 | 122995 | 13290 | 9.255 | 422632 |
| LLaDA/aime26 | V3a | 125912 | 14869 | 8.468 | 473315 |
| TraDo/aime26 | V31 | 39895 | 17583 | 2.269 | 70081 |
| TraDo/aime26 | V2 | 48514 | 23887 | 2.031 | 95292 |
| TraDo/aime26 | V3a | 38130 | 16180 | 2.357 | 64451 |
| LLaDA/humaneval | V31 | 6029 | 1285 | 4.692 | 35900 |
| LLaDA/humaneval | V2 | 2331 | 503 | 4.634 | 10317 |
| LLaDA/humaneval | V3a | 6941 | 1422 | 4.881 | 40227 |
| TraDo/humaneval | V31 | 11616 | 5105 | 2.275 | 19934 |
| TraDo/humaneval | V2 | 10059 | 5362 | 1.876 | 20933 |
| TraDo/humaneval | V3a | 11233 | 5600 | 2.006 | 21901 |
| LLaDA/ruler128k | V31 | 1017 | 237 | 4.291 | 3689 |
| LLaDA/ruler128k | V2 | 1024 | 236 | 4.339 | 3656 |
| LLaDA/ruler128k | V3a | 1040 | 237 | 4.388 | 3662 |

TraDo V2 HumanEval has lower pooled calls and a paired geometric N/C ratio0.851 [0.817,0.885] versus V31. Its paired geometric total-N ratio is1.432 because per-problem work is highly skewed; pooled sums and paired geometric means answer different questions. LLaDA V3a HumanEval uses more total calls and its paired N/C ratio0.998 [0.957,1.041] gives no evidence of faster per-canvas convergence. AIME V2 paired N/C is1.000 [0.900,1.106] for LLaDA and0.862 [0.802,0.918] for TraDo; TraDo produces more canvases/output and uses more total calls than V31. Shorter output accompanying failed answers is not useful acceleration.

## Small calibration and realized sparsity

Each new calibration candidate used six fixed AIME questions, seed42 only. Each model accepted its second new candidate, then ran the final90 requests once. Selection used physical tile counts without scoring; the prompts/sample selection are transductive and do not constitute held-out calibration. The user explicitly removed post-benchmark recalibration.

| Model | V2 threshold | Six-request calibration sparsity | Final90 sparsity | Final correct |
|---|---:|---:|---:|---:|
| llada22 | 0.0124896637018 | 49.39% | 51.72% | 28/90 |
| trado | 0.0251091332518 | 50.65% | 49.78% | 6/90 |

All retained sparse panels are within48–52% realized consumer sparsity. This denominator includes mandatory generation/boundary tiles, over eligible denoising consumers only; it excludes discovery, genuine dense prefill and commit work. It is not whole-model FLOP sparsity. Original AIME V2 target misses remain historical: LLaDA75.21%,9/90; TraDo60.45%,6/90. They are not substituted into the revised comparison.

## UltraLLaDA and scope limitations

Ultra HumanEval completes164requests per arm. Every request uses1024denoising forwards and16canvases and reaches1024tokens without EOS; dense/all-kept each solve2/164 and sparse arms0/164. This low dense operating point is not a useful test of value-aware superiority. Its native sampler uses a fixed schedule, so reaching the cap alone is not an adaptive convergence failure.

Ultra AIME was stopped during calibration (25 completed development records each used4096forwards/64canvases); no completed AIME benchmark exists. Ultra RULER was also stopped during calibration: three completed examples each used128forwards,2canvases,128tokens, no EOS, and approximately4.7hours. All three receive full official retrieval credit. This is development-only3/3, not a130-question benchmark estimate. Preserve the incomplete attempt and cancellation receipts. TraDo128K exceeds the supported context and was not truncated or extrapolated. LLaDA RULER actual prompt lengths range115092–130776tokens (median130590); all fit the131072limit with128output tokens, without truncation.

## Execution, checks, and provenance

Native masks/samplers/caps are fixed within model/task; all denoising layers are eligible. Dense prefix/commit attention remains dense. Sparse selection uses Q128/KV64 tiles and reuse every fourth call, resetting at canvas/structural changes. Gaussian32 uses seed1729pluslayer; V3a singleton risk uses full-support projected summaries, while retained attention always consumes current full-dimensional V. Gaussian32 is Junyu's method; initial cross-step reuse machinery is inherited from Yuhan; these model ports are distinct variants.

AIME execution deviation: the pinned worker imported the active checkout's newer shared runtime/attention modules because snapshot experiments lacked the top-level package initializer. Actual runtime SHA561b9d8c and attention SHA1e9741b4 differ from original controls. No benchmark was repeated. After generation, source inspection verified identical LLaDA/TraDo generation AST and unchanged selector/consumer dependencies;216synthetic H100 cases showed bitwise-equal outputs/maps and exact work counts, including held steps, refreshes and invalidation. Changes relevant to these models concern cached integer tile accounting, default-equivalent statistics grouping and health callbacks; Ultra-only sampler changes are unreachable here. This after-execution qualification supports the comparison, but the reruns are not exact-source reproductions. The failed first final audit is preserved in attempt001.

Independent original audits pass15/1920LLaDA,10/1270TraDo and5/820Ultra panels/requests. The final audit checks180fresh identities, prompt/cap bindings, row/config/source hashes, physical accounting and scorer/raw agreement; no target misses remain. Eleven focused CPU tests pass. Numerical qualification uses existing independent selector/consumer gates plus the additional module-parity receipt. Optional observer-on/off reuse diagnostics from the original finalizer were not run after the user narrowed the scope; primary refresh/held counts are available, but no causal reuse-benefit claim is made.

Canonical original benchmark receipts total167827.126worker-seconds (46.619h); fresh AIME replacements add20518.392s (5.700h). These are allocated benchmark-worker wall times, excluding loading, calibration, cancelled/failed attempts and qualification. They are not clean request latency or GPU-utilization integrals. No end-to-end speedup claim is made.

Primary recommendation: prioritize TraDo V2 for follow-up accuracy-preserving work and retain LLaDA V3a as a task-specific comparison. Do not claim a universal value-direction gain, a proven K/V-tying explanation, or faster denoising from fewer calls alone.

Artifacts: final_complete.json, receipt.json, evidence_index.json, summary.json, paired.json and sanitized scored_records.jsonl. Raw prompts, gold, tokens and generations remain private.
