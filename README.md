# Value-aware sparse attention across diffusion LLMs

This branch evaluates value-direction-aware attention on **LLaDA2.2-mini** and **TraDo-8B-Instruct**, comparing online Gaussian32 selection with attention-mass ranking, full-information singleton selection and two dense controls. The framework retains each model’s native denoising sampler and uses physical block-sparse attention on one H100.

**Results through November 4, 2026:** all six current-tile V2 model/benchmark panels are complete, as are the four additional TraDo RULER32K baseline panels. Existing baseline results are retained. DiffusionGemma follow-up experiments were stopped and are outside this completed scope.

## Methods

| Method | Selection and output attention |
|---|---|
| Native dense | Native SDPA attention; reference decoding trajectory. |
| All-kept FA4 | The sparse methods’ full-dimensional FA4 consumer with every eligible tile retained; separates consumer differences from sparsification. |
| V31 | Rank physical tiles by attention mass and retain a context-dependent fixed budget. |
| Original V2 | Online Gaussian32 value-direction selector; chronological scan begins with the first prefix tile. Threshold determines the retained set. |
| V3a | Rank full-information singleton deletion risk relative to the full-support projected output, with a fixed budget. |
| Current-tile V2 | Initialize the same online V2 recurrence from the mandatory current tile, then scan prefix tiles `0..J-2`. Recalibrate the threshold for this order. |

The current-tile variant removes the forced first-prefix-tile initialization. It preserves V2’s rank-32 Gaussian projection, preserve-mass skip semantics and full-dimensional current-value output. It does not alter EOS, acceptance, editing or stopping, and adds no model forward. Gaussian32 is Junyu’s prior method; cross-step map reuse is inherited from Yuhan. The initialization change is a separate named variant.

## Accuracy and retrieval results

| Model | Benchmark | Native dense | All-kept FA4 | V31 | Original V2 | V3a | Current-tile V2 |
|---|---|---:|---:|---:|---:|---:|---:|
| LLaDA2.2-mini | AIME26 | 40.00% | 32.22% | 23.33% | 31.11% | 27.78% | 30.00% |
| LLaDA2.2-mini | HumanEval | 85.98% | 84.15% | 50.61% | 6.10% | 62.80% | 72.56% |
| LLaDA2.2-mini | RULER128K | 85.44% | 84.82% | 86.01% | 86.23% | 85.03% | 85.21% |
| TraDo-8B-Instruct | AIME26 | 8.89% | 4.44% | 3.33% | 6.67% | 2.22% | 8.89% |
| TraDo-8B-Instruct | HumanEval | 79.27% | 78.05% | 42.68% | 76.22% | 47.56% | 74.39% |
| TraDo-8B-Instruct | RULER32K | 64.64% | 64.85% | 63.49% | — | 64.96% | 64.72% |

AIME26 is exact-match accuracy over **30 problems × 3 seeds = 90 requests**. HumanEval is official functional **pass@1 over all 164 problems**, with one seed. RULER is official **fractional target-retrieval credit over 130 problems**, balanced at 10 per task across 13 tasks; it is not binary accuracy. No original-V2 TraDo RULER32K evaluation exists.

Key observations:

- **LLaDA2.2-mini HumanEval:** current-tile V2 recovers **119/164** passes from original V2’s **10/164**, compared with V31 **83/164**, V3a **103/164** and native dense **141/164**. Mean output length rises from **62.9 to 219.3 tokens**. This is initialization plus recalibration: achieved sparsity changes from **49.22% to 55.67%**, so it is not an exactly matched-50% or fixed-threshold ablation.
- **TraDo HumanEval:** current-tile V2 scores **122/164**, versus original V2 **125/164**, V31 **70/164**, V3a **78/164** and native dense **130/164**. The initialization change does not establish a universal improvement.
- **TraDo RULER32K:** V31 and V3a both achieve **49.97%** sparsity. V3a exceeds V31 by **1.47 percentage points**, but the paired problem-bootstrap 95% CI **[−0.32, +3.78]** and task-cluster CI **[−0.71, +4.42]** cross zero. This does not establish superiority or noninferiority.

## Denoising and achieved sparsity

The table below describes the accepted **current-tile V2** panels. Full work statistics for all **35 method/benchmark combinations** are in the [combined metrics](results/trado_ruler32k_baselines_20261104/attempt002/analysis002/combined_35_rows.json).

| Model | Benchmark | Denoising calls N | Canvases C | Mean output tokens | N/C | Achieved sparsity | EOS / cap hits |
|---|---|---:|---:|---:|---:|---:|---:|
| LLaDA2.2-mini | AIME26 | 130126 | 16858 | 5964.2 | 7.719 | 51.95% | 87 / 3 |
| LLaDA2.2-mini | HumanEval | 4278 | 1275 | 219.3 | 3.355 | 55.67% | 164 / 0 |
| LLaDA2.2-mini | RULER128K | 1041 | 236 | 28.0 | 4.411 | 49.63% | 130 / 0 |
| TraDo-8B-Instruct | AIME26 | 47516 | 22972 | 1018.1 | 2.068 | 50.51% | 88 / 2 |
| TraDo-8B-Instruct | HumanEval | 9838 | 5137 | 122.2 | 1.915 | 53.31% | 164 / 0 |
| TraDo-8B-Instruct | RULER32K | 2767 | 1113 | 31.0 | 2.486 | 49.45% | 126 / 4 |

`N` counts denoising forwards; `C` counts native generation canvases; `N/C` is their pooled ratio. Output length counts emitted tokens, including emitted EOS. Cap hits indicate non-EOS outputs reaching the configured limit. Fewer calls caused by premature termination are not useful acceleration.

Consumer tile sparsity is `1 - sum(kept tiles) / sum(eligible tiles)` over actual denoising calls, including mandatory tiles and calls that reuse a map. It excludes selector discovery, dense prefill and block commits. It measures geometric attention sparsity, not whole-model FLOPs or preserved attention mass. The runs are instrumented; **no clean end-to-end latency or speedup claim is made**.

## Evaluation setup

| Setting | LLaDA2.2-mini | TraDo-8B-Instruct |
|---|---|---|
| Model / frozen revision | `inclusionAI/LLaDA2.2-mini` / `e4f77d25` | `Gen-Verse/TraDo-8B-Instruct` / `2d37bd3e` |
| Native block length | 32 tokens | 4 tokens |
| Sampler | Temperature 0; confidence threshold 0.5; editing threshold 0; 32 steps; maximum 16 post-steps; native editing and EOS | Temperature 1; top-p 1; top-k 0; confidence threshold 0.9; 4 denoising steps; native EOS |
| Output caps: AIME / HumanEval / RULER | 32,768 / 1,024 / 128 | 2,000 / 2,000 / 128 |
| RULER context limit | 131,072 tokens (128K) | 32,768 tokens (32K) |
| Actual RULER prompt lengths | 115,092–130,776 tokens | 29,798–32,472 tokens |

Both models use **BF16, batch size 1, one H100**, native prompt adapters and source-pinned runtimes. AIME seeds are **42/43/44**; HumanEval and RULER use **42**. RULER inputs fit without truncation. Different context lengths and tokenizer-specific RULER prompts make cross-model scores different evaluation conditions. Output budgets also differ by model.

All denoising attention layers are eligible; prompt prefill and committed-block attention stay dense. Selection uses **Q128 × KV64** physical tiles per query head/block, a fixed Gaussian32 projection seeded by `1729 + layer`, a refresh every fourth denoising call, and no cross-canvas carry. Maps reset at canvas boundaries or structural index changes; padded query rows do not vote. Retained attention always uses full-dimensional current values. V31/V3a budgets follow effective context length: `max(round(J/2), mandatory_count)`, where `J` is the number of available KV tiles and mandatory current tiles count toward the budget. Native dense and all-kept FA4 remain separate controls because their numerical trajectories can differ.

Current-tile V2 thresholds are calibrated by physical tile counts, without gold answers or accuracy-guided tuning. AIME/HumanEval use six fixed benchmark prompts selected by length strata: **transductive, exploratory calibration**. RULER uses 13 separately generated development prompts, one per task, with seed 142. Calibration never runs the full 90-request AIME panel. The target is approximately 50%; accepted full-panel sparsity is 40–60%. All six accepted panels fall in that range. Thresholds and calibration histories are in the [six-panel report](results/current_tile_multimodel_20261103/attempt001/publication001/README.md).

## Evidence and limitations

- [Six-panel V2 report](results/current_tile_multimodel_20261103/attempt001/publication001/README.md): calibration, historical comparisons, paired intervals, source identities and maintenance recovery. Its original TraDo RULER32K baseline gaps are filled by the later supplement below.
- [Completed TraDo RULER32K supplement](results/trado_ruler32k_baselines_20261104/attempt002/publication001/README.md): native dense, all-kept FA4, V31 and V3a results; per-task scores; exact half-budget audits; existing V2 re-scored without new V2 generation.
- Canonical completion markers: [six V2 panels](results/current_tile_multimodel_20261103/attempt001/final_complete.json) and [TraDo baseline supplement](results/trado_ruler32k_baselines_20261104/attempt002/final_complete.json). Frozen results and failed attempts remain unchanged.
- These are previously examined, exploratory cohorts. Paired intervals do not measure all runtime or seed uncertainty; AIME resampling clusters its three seeds by problem. Similar sparsity does not establish identical attention work or equivalent model quality.
- **TraDo historical V2 replay limitation:** fresh original-V2 replays differ from saved historical outputs despite matching pins; fresh repeats agree with each other, and all-kept controls replay exactly. Historical original-V2 comparisons are descriptive, not a bitwise reproduced reference. The reports disclose this qualification limitation.
- LLaDA RULER128K combines 66 verified records from two complete pre-maintenance shards with 64 records from fresh shards at the frozen threshold. The 21 partial records are preserved and excluded. An earlier 61.35%-sparse LLaDA HumanEval development result is also preserved separately and is not the accepted result above.
- Raw prompts, gold answers, tokens, completions and model weights remain private. Public evidence contains sanitized records, aggregates, hashes and execution receipts.

## Implementation and reproduction

| File | Role |
|---|---|
| [multi_model_runtime.py](experiments/numerical_qk_reuse/multi_model_runtime.py) | Model-specific native sampler integration and numerical execution. |
| [multi_model_attention.py](experiments/numerical_qk_reuse/multi_model_attention.py) | Attention routing, full-dimensional sparse consumers and work counters. |
| [v2_current_tile_init.py](experiments/numerical_qk_reuse/v2_current_tile_init.py) | Independent CPU recurrence and optimized current-tile initialization. |
| [current_tile_multimodel_worker.py](scripts/current_tile_multimodel_worker.py) | Small-sample calibration, threshold freeze and sharded generation. |
| [current_tile_retained_report.py](scripts/current_tile_retained_report.py) | Publish the retained two-model comparison with preserved historical evidence. |
| [trado_ruler32k_baselines.py](scripts/trado_ruler32k_baselines.py) | Source-bound TraDo RULER32K baseline runner and qualification. |
| [trado_ruler32k_baseline_report.py](scripts/trado_ruler32k_baseline_report.py) | Official RULER scoring, paired comparisons and combined 35-row metrics. |
| [test_v2_current_tile_init.py](tests/test_v2_current_tile_init.py) | CPU recurrence, initialization, padding and budget checks. |

The [original multi-model protocol](docs/MULTI_MODEL_VALUE_EVALUATION_20261030.md), [current-tile protocol](docs/CURRENT_TILE_MULTIMODEL_20261103.md) and [TraDo baseline protocol](docs/TRADO_RULER32K_BASELINES_20261104.md) define frozen source, sampler, scorer and qualification requirements. Later scope changes and completion markers take precedence over the original nine-panel schedule. Reproduction requires the specified model/runtime revisions and authorized private inputs; one generic Transformers environment does not cover every model. In particular, the TraDo supplement uses its pinned Transformers 4.52.4 compatibility environment.

## Historical DiffusionGemma work

Earlier studies remain available in the [Hopper implementation guide](experiments/value_direction_hopper/README.md), [value-aware method documentation](docs/research_week_20261015/01_VALUE_AWARE_METHODS.md), [HumanEval reproducibility record](docs/HUMANEVAL_REPRODUCIBILITY_FIXES_20261020.md) and [results ledger](docs/RESULTS_LEDGER.md). Their models, dense baselines, layer scopes and native samplers are separate from the HF multi-model results summarized above.

## License

[Apache-2.0](LICENSE).
