# Current-tile V2: final retained LLaDA and TraDo results

Completed six panels /768 benchmark requests. DiffusionGemma was stopped by the user; its queued HumanEval attempt failed during execution-link setup before GPU generation. Its three panels are excluded. The original nine-panel protocol is scope-superseded, not declared complete. H100 is idle; other methods and historical evidence remain unchanged.

| Model | Benchmark | Native dense | All-kept FA4 | V31 | Original V2 | V3a | V2 current-tile init |
|---|---|---:|---:|---:|---:|---:|---:|
| LLaDA2.2-mini | AIME26 | 40.00% (36/90) | 32.22% (29/90) | 23.33% (21/90) | 31.11% (28/90) | 27.78% (25/90) | 30.00% (27/90) |
| LLaDA2.2-mini | HumanEval | 85.98% (141/164) | 84.15% (138/164) | 50.61% (83/164) | 6.10% (10/164) | 62.80% (103/164) | 72.56% (119/164) |
| LLaDA2.2-mini | RULER128K | 85.44% | 84.82% | 86.01% | 86.23% | 85.03% | 85.21% |
| TraDo-8B-Instruct | AIME26 | 8.89% (8/90) | 4.44% (4/90) | 3.33% (3/90) | 6.67% (6/90) | 2.22% (2/90) | 8.89% (8/90) |
| TraDo-8B-Instruct | HumanEval | 79.27% (130/164) | 78.05% (128/164) | 42.68% (70/164) | 76.22% (125/164) | 47.56% (78/164) | 74.39% (122/164) |
| TraDo-8B-Instruct | RULER32K | — | — | — | — | — | 64.72% |

AIME is exact-match accuracy over30 problems x3seeds; HumanEval is official functional pass@1 over164 problems x1seed; RULER is fractional retrieval credit over130problems (10/task across13tasks), not binary accuracy. TraDo32K is new and has no historical baseline. All other methods are reused without regeneration.

| Model | Benchmark | Mean output tokens | N total | C total | N/request | C/request | N/C | Sparsity | EOS / cap hits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaDA2.2-mini | AIME26 | 5964.2 | 130126 | 16858 | 1445.84 | 187.31 | 7.719 | 51.95% | 87 / 3 |
| LLaDA2.2-mini | HumanEval | 219.3 | 4278 | 1275 | 26.09 | 7.77 | 3.355 | 55.67% | 164 / 0 |
| LLaDA2.2-mini | RULER128K | 28.0 | 1041 | 236 | 8.01 | 1.82 | 4.411 | 49.63% | 130 / 0 |
| TraDo-8B-Instruct | AIME26 | 1018.1 | 47516 | 22972 | 527.96 | 255.24 | 2.068 | 50.51% | 88 / 2 |
| TraDo-8B-Instruct | HumanEval | 122.2 | 9838 | 5137 | 59.99 | 31.32 | 1.915 | 53.31% | 164 / 0 |
| TraDo-8B-Instruct | RULER32K | 31.0 | 2767 | 1113 | 21.28 | 8.56 | 2.486 | 49.45% | 126 / 4 |

N is denoising forwards, C native canvases, T emitted output tokens including emitted EOS. N/C is the pooled ratio. Cap hits are requests ending without EOS at the configured output limit. Lower N from failed short output is not useful acceleration. All new panel sparsities are inside the user-approved40–60% range; no completed panel required another sparsity-driven rerun.

| Model | Benchmark | Threshold | Calibration prompts/candidate | Candidates | Calibration sparsity | Final sparsity |
|---|---|---:|---:|---:|---:|---:|
| LLaDA2.2-mini | AIME26 | 0.0030248896724 | 6 | 8 | 48.22% | 51.95% |
| LLaDA2.2-mini | HumanEval | 0.213270966505 | 6 | 4 | 50.65% | 55.67% |
| LLaDA2.2-mini | RULER128K | 2.21165362353e-05 | 13 | 4 | 49.62% | 49.63% |
| TraDo-8B-Instruct | AIME26 | 0.0133616799226 | 6 | 9 | 49.07% | 50.51% |
| TraDo-8B-Instruct | HumanEval | 0.127350308102 | 6 | 4 | 51.15% | 53.31% |
| TraDo-8B-Instruct | RULER32K | 0.000139002585453 | 13 | 4 | 49.88% | 49.45% |

Calibration uses physical tile counts only. AIME/HumanEval use six fixed prompts chosen by character-length strata from the examined benchmark, seed42: transductive and exploratory. Retrieval uses13 separately generated development prompts, seed142. No gold or functional score selects thresholds. Calibration runs only those small samples, never the90-request AIME benchmark.

The new selector processes the mandatory current KV64 tile first, then chronological tiles0..J-2, once each. Gaussian32 projection, conditional normalized-state updates, preserve-mass skip rule, full-dimensional current-V output, Q128/KV64 geometry and four-call support reuse remain. No forced first prefix tile, mass-ranking repair, EOS override or extra model forward is added. All denoising layers are eligible; prompt prefill and committed-block attention remain dense.

| Model | Native block | Native generation settings | Output cap: AIME / HumanEval / RULER | RULER prompt-token range |
|---|---:|---|---:|---:|
| LLaDA2.2-mini |32|Greedy temperature0; M2T confidence0.5; editing threshold0; max poststeps16; steps32; native editing and EOS|32768 /1024 /128|115092–130776|
| TraDo-8B-Instruct |4|Temperature1; top-p1, top-k0; dynamic confidence0.9; four denoising steps; native EOS|2000 /2000 /128|29798–32472|

Both use batch1,BF16 on the same H100, the original model revisions, native chat/prompt adapters and source-pinned numerical runtimes. Seeds42/43/44 for AIME,42 for HumanEval/RULER. RULER fits within each model limit without truncation; LLaDA128K and TraDo32K are different tokenizer-specific prompts.

Sparsity is1−sum(kept)/sum(eligible) across actual native denoising consumers, all eligible layers/heads, including mandatory current tiles and refresh/reuse calls. It excludes selector discovery, dense prefill and commits; it is geometric tile sparsity, not whole-model FLOPs or attention mass. Original dense SDPA and all-kept FA4 remain separate controls. Timings were instrumented; no clean latency/speedup claim is made.

The largest gain is LLaDA HumanEval:10->119passes,62.9->219.3 mean output tokens and4.634->3.355steps/canvas. Its new55.67% sparsity differs from original49.22%, so this is an initialization-plus-recalibration result within the accepted band, not an exactly matched50% or fixed-threshold ablation. It remains below native141/164. TraDo HumanEval is122/164 versus original125/164; AIME and LLaDA retrieval do not establish a universal improvement.

Paired problem-bootstrap intervals (10000resamples; AIME clusters its three seeds by problem) are in ../analysis001/paired.json. LLaDA HumanEval gain over V31 is+21.95pp[12.80,31.10], over original V2+66.46pp[59.15,73.17]. TraDo HumanEval difference from original V2 is−1.83pp[−7.93,4.27]. These intervals are exploratory and do not remove sparsity or runtime reproduction limitations.

TraDo control limitation: original V2 historical replay differs despite matching pinned source, prompt, seed and threshold; fresh repeats match one another. All-kept replay exactly reproduces historical hashes/counts. Every TraDo panel records this comparison and the disclosed qualification policy. Historical V2 comparison is descriptive, not a bitwise reproduced reference. No intrinsic model explanation is inferred.

Maintenance recovery: LLaDA RULER128K attempt001 contains two complete33-request shards and21partial records. Attempt002 inherits only the66complete hash/token-verified records and regenerates both remaining shards at the frozen threshold. The21partial records are preserved and excluded. Its completed130-request panel has explicit mixed-attempt lineage. The earlier outside-band LLaDA HumanEval61.35% development result is preserved separately; the accepted campaign panel uses its own calibration.

Verification:11 focused CPU tests;96 CUDA recurrence cases before and again after maintenance; independent private-token hashes and record/config/protocol/threshold joins;24 complete shard references across6panels; official scorer;768 unique planned cells;138 protected historical files unchanged. No campaign GPU processes remain.

Full method-by-method work table follows. Every historical denominator belongs to its own native trajectory.

| Model | Benchmark | Method | Score | Mean T | N total | C total | N/C | Sparsity | EOS / caps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| LLaDA2.2-mini | AIME26 | Native dense | 40.00% | 7018.6 | 138512 | 19813 | 6.991 | 0.00% | 85 / 5 |
| LLaDA2.2-mini | AIME26 | All-kept FA4 | 32.22% | 7356.3 | 154593 | 20767 | 7.444 | 0.00% | 86 / 4 |
| LLaDA2.2-mini | AIME26 | V31 | 23.33% | 4691.7 | 119064 | 13271 | 8.972 | 50.00% | 90 / 0 |
| LLaDA2.2-mini | AIME26 | Original V2 | 31.11% | 4695.9 | 122995 | 13290 | 9.255 | 51.72% | 90 / 0 |
| LLaDA2.2-mini | AIME26 | V3a | 27.78% | 5259.1 | 125912 | 14869 | 8.468 | 50.00% | 90 / 0 |
| LLaDA2.2-mini | AIME26 | V2 current-tile init | 30.00% | 5964.2 | 130126 | 16858 | 7.719 | 51.95% | 87 / 3 |
| LLaDA2.2-mini | HumanEval | Native dense | 85.98% | 243.1 | 4717 | 1405 | 3.357 | 0.00% | 164 / 0 |
| LLaDA2.2-mini | HumanEval | All-kept FA4 | 84.15% | 249.2 | 5153 | 1439 | 3.581 | 0.00% | 163 / 1 |
| LLaDA2.2-mini | HumanEval | V31 | 50.61% | 218.9 | 6029 | 1285 | 4.692 | 49.98% | 164 / 0 |
| LLaDA2.2-mini | HumanEval | Original V2 | 6.10% | 62.9 | 2331 | 503 | 4.634 | 49.22% | 164 / 0 |
| LLaDA2.2-mini | HumanEval | V3a | 62.80% | 245.3 | 6941 | 1422 | 4.881 | 49.99% | 163 / 1 |
| LLaDA2.2-mini | HumanEval | V2 current-tile init | 72.56% | 219.3 | 4278 | 1275 | 3.355 | 55.67% | 164 / 0 |
| LLaDA2.2-mini | RULER128K | Native dense | 85.44% | 28.2 | 1015 | 237 | 4.283 | 0.00% | 130 / 0 |
| LLaDA2.2-mini | RULER128K | All-kept FA4 | 84.82% | 28.2 | 1044 | 237 | 4.405 | 0.00% | 130 / 0 |
| LLaDA2.2-mini | RULER128K | V31 | 86.01% | 28.4 | 1017 | 237 | 4.291 | 50.00% | 130 / 0 |
| LLaDA2.2-mini | RULER128K | Original V2 | 86.23% | 28.1 | 1024 | 236 | 4.339 | 49.19% | 130 / 0 |
| LLaDA2.2-mini | RULER128K | V3a | 85.03% | 28.2 | 1040 | 237 | 4.388 | 50.00% | 130 / 0 |
| LLaDA2.2-mini | RULER128K | V2 current-tile init | 85.21% | 28.0 | 1041 | 236 | 4.411 | 49.63% | 130 / 0 |
| TraDo-8B-Instruct | AIME26 | Native dense | 8.89% | 1072.8 | 48516 | 24203 | 2.005 | 0.00% | 86 / 4 |
| TraDo-8B-Instruct | AIME26 | All-kept FA4 | 4.44% | 1015.7 | 47348 | 22921 | 2.066 | 0.00% | 89 / 1 |
| TraDo-8B-Instruct | AIME26 | V31 | 3.33% | 778.7 | 39895 | 17583 | 2.269 | 50.00% | 88 / 2 |
| TraDo-8B-Instruct | AIME26 | Original V2 | 6.67% | 1058.8 | 48514 | 23887 | 2.031 | 49.78% | 88 / 2 |
| TraDo-8B-Instruct | AIME26 | V3a | 2.22% | 716.1 | 38130 | 16180 | 2.357 | 49.97% | 89 / 1 |
| TraDo-8B-Instruct | AIME26 | V2 current-tile init | 8.89% | 1018.1 | 47516 | 22972 | 2.068 | 50.51% | 88 / 2 |
| TraDo-8B-Instruct | HumanEval | Native dense | 79.27% | 124.3 | 9612 | 5225 | 1.840 | 0.00% | 164 / 0 |
| TraDo-8B-Instruct | HumanEval | All-kept FA4 | 78.05% | 120.7 | 9425 | 5076 | 1.857 | 0.00% | 164 / 0 |
| TraDo-8B-Instruct | HumanEval | V31 | 42.68% | 121.5 | 11616 | 5105 | 2.275 | 49.55% | 163 / 1 |
| TraDo-8B-Instruct | HumanEval | Original V2 | 76.22% | 127.6 | 10059 | 5362 | 1.876 | 48.84% | 164 / 0 |
| TraDo-8B-Instruct | HumanEval | V3a | 47.56% | 133.5 | 11233 | 5600 | 2.006 | 49.67% | 162 / 2 |
| TraDo-8B-Instruct | HumanEval | V2 current-tile init | 74.39% | 122.2 | 9838 | 5137 | 1.915 | 53.31% | 164 / 0 |
| TraDo-8B-Instruct | RULER32K | V2 current-tile init | 64.72% | 31.0 | 2767 | 1113 | 2.486 | 49.45% | 126 / 4 |

Raw prompts,gold,token IDs,completions and unsanitized process paths remain private. Sanitized per-cell scores,complete counts,calibration histories,source identities and completion receipts are retained.
