# Shared-Page Attention (SPA): oracle feasibility report

## Decision

**No-go at the oracle gate.** A quality-safe group-shared support does not
exist at the required work budget for the tested LLaDA-8B-Instruct setting.
The strongest target-length, work-eligible `G=64` policy retained only 71.9%
P1 and 79.9% mean attention mass in its worst layer, versus the required 95%
and 99%. The irregular shared-column upper bound also failed (73.2% P1,
81.0% mean), so 64-key page contiguity is not the primary obstruction.

The result is therefore preserved as a negative finding. Per the experiment
brief, complete-trajectory sparse replacement, temporal refresh tuning, the
cheap pre-QK estimator, H100 sparse-kernel microbenchmarks, and WGMMA/TMA
implementation were not authorized. No production-kernel speedup is claimed.

## Current hypothesis and how it changed

Initial hypothesis: exact normalized attention mass would reveal a small,
head-specific key support shared by 64 adjacent queries; selecting contiguous
64-key pages would then create full 64x64 Hopper tiles, while one packed sink
page would recover scattered high-value columns.

Measured conclusion: adjacent queries do share support moderately, but the
mass outside that common support is much too large. The failure is more basic
than page layout: even the shared-column upper bound misses the oracle mass
gate. Sink columns and minority-aware selection improve isolated statistics
only slightly and do not approach acceptance.

## Literature and design comparison

SPA is not presented as novel merely because it shares key support within a
query group. The closest reference mechanisms already do this.

| Work | Relevant implementation logic | Consequence for SPA |
|---|---|---|
| [PulseCol](https://arxiv.org/abs/2605.20813) | At exact refresh steps, average normalized attention weights over each query group, select group-shared top-K columns, reuse them between refreshes, and stop refreshing after an early window. Its kernel gathers indexed K/V columns and maintains online softmax. | Format A is the direct quality upper bound; periodic refresh would only be justified after current-step support succeeds. PulseCol also reports its best quality at group size 32, which is not the primary Hopper shape here. |
| [MAGE](https://arxiv.org/abs/2602.14209) | For block diffusion, run exact attention at the initial all-mask step, average scores over the queries sharing a KV head/block, select a per-block KV subset, and reuse it for the whole block trajectory. Selection is overlapped with the FFN on another CUDA stream. | Motivates the first-step anchor, but its block-diffusion stability cannot be assumed for fully bidirectional LLaDA. |
| [SparseD](https://arxiv.org/abs/2509.24014) | Build head-specific block patterns, select prompt and generation regions separately, keep early denoising steps dense, and reuse the selected blocks later. | Confirms that heads must remain isolated and that early-step sensitivity must be measured, not averaged away. |
| [LoSA](https://arxiv.org/abs/2604.12056) | Rank tokens by cross-step representation change, recompute sparse prefix attention for active queries, reuse cached prefix-attention statistics for stable queries, compute within-block attention densely, then merge contributions with online softmax. | Stable-query output reuse is relevant only to block/prefix structures that expose a reusable component. It was not evaluated after SPA's oracle gate failed. |
| [Focus-dLLM](https://arxiv.org/abs/2602.02159) | Predict active queries from previous-step confidence, preserve local windows, identify sink tokens at a dense cutoff layer, reuse sink locations in deeper layers, and select prompt blocks using mean-pooled representative keys. | Motivates mandatory sink tests, but broad cross-layer sink sharing needs direct validation. |
| [BA-Att](https://arxiv.org/abs/2605.19726) | Mean-pool Q/K blocks, score the compact block map, reduce within-block dispersion with norm sorting, and correct pooled logits using diagonal Q/K variance as a covariance approximation before top-block selection. | Provides a training-free estimator design if exact SPA passes. Norm-based regrouping was not repeated because this repository already measured regrouping overhead and the SPA oracle failed first. |
| [HiLS-Attention](https://arxiv.org/abs/2607.02980) | Defines the correct retrieval target as chunk LogSumExp/attention mass, learns landmark-derived chunk summaries, and factorizes attention into inter-chunk mass and intra-chunk softmax. | Supports normalized chunk mass as the oracle target and explains why mean/max logits can mis-rank chunks. Its learned landmarks and continued training are outside the training-free scope. |

## Implementation

The implementation is intentionally separate from the existing dense and
BLASST paths.

- `spa/reference.py` implements:
  - formats A/B/C: shared columns, physical 64-key pages, and pages plus 0/1/2
    head-shared sink pages;
  - mean-mass, tail-aware, and coverage-aware selectors;
  - local, first, and last mandatory-page ablations;
  - strict request/head isolation, with broader head sharing exposed only as
    an explicitly named negative ablation;
  - exact selected-key attention that recomputes selected scores, applies
    softmax only over retained keys, and multiplies only those probabilities
    by retained V rows;
  - normalized retained-mass and attention-output metrics.
- `spa/analyzer.py` wraps real LLaDA FlashAttention calls for read-only oracle
  analysis, records group/layer/step overlap, and contains the gated exact
  reference controller. Phase-A multi-policy analysis uses the algebraically
  equivalent conditional dense probabilities only to avoid recomputing full
  QK for every selector; the Phase-B reference itself uses explicit selected
  scores and selected-key softmax.
- `scripts/eval_shared_page_attention.py` provides the requested opt-in flags:
  `--enable-shared-page-attention`, `--spa-group-size`, `--spa-support`,
  `--spa-density`, `--spa-selector`, and `--spa-sink-pages`, plus ablation and
  refresh controls. Density is a hard upper bound: whole-page and column
  counts are rounded down.
- `tools/analyze_spa_results.py` aggregates the layer/step records and audits
  sensitive-layer exclusions.
- `tests/test_spa_reference.py` covers dense equivalence, a single retained
  page, partial final pages, invalid rows/keys, causal and bidirectional masks,
  head isolation, request isolation, explicit dense-mask equivalence for all
  formats, and minority-row coverage selection.

No Python loop or dense score construction is claimed as a production path.
These are correctness and feasibility references.

## Experimental setup

| Item | Setting |
|---|---|
| Model | `GSAI-ML/LLaDA-8B-Instruct` |
| Device | NVIDIA H100 80GB HBM3, compute capability 9.0 |
| Model precision | BF16; oracle scores/softmax and metrics in FP32 |
| Target shape | sequence 4096, 32 MHA heads, head dimension 128, 32 layers |
| Page/group isolation | Per request, layer, and head; no request/head mixing |
| Primary support | 64 query rows x shared physical 64-key pages |
| Target-length screen | 13 policies x all 32 layers at the initial all-mask state |
| Complete oracle trajectories | 2 prompts x 2 seeds x 8 complete reveal steps x 32 layers = 1,024 layer-step-context records |
| Full ablation diagnostic | 42 policies x all 32 layers at sequence 512 |
| Density accounting | Actual unique retained keys, including sink columns |

The eight-step runs are complete coarse denoising trajectories: each step
reveals a scheduled fraction of the initial mask set using dense confidence,
ending with no masks. They are teacher-forced for support analysis, not a
claim that eight steps match the model's standard benchmark generation recipe.

## Oracle quality/work Pareto frontier

Target-length initial-state results are worst values across all 32 layers.

| Policy | Actual key density | Work reduction | Worst P1 mass | Worst mean mass | Max output rel. L2 | Gate |
|---|---:|---:|---:|---:|---:|---|
| G64 pages, 50% budget | 50.00% | 50.00% | 54.34% | 66.41% | 0.1804 | Fail |
| G64 pages, 60% budget | 59.38% | 40.62% | 63.20% | 73.29% | 0.1322 | Fail |
| G64 pages, 70% budget | 68.75% | 31.25% | 71.93% | 79.91% | 0.0925 | Fail |
| G64 columns, 70% budget | 69.995% | 30.005% | 73.17% | 80.95% | 0.0833 | Fail |
| G64 pages, 70%, tail (`beta=.5,q=.95`) | 68.75% | 31.25% | 71.93% | 79.90% | 0.0926 | Fail |
| G64 pages, 70%, P1 coverage greedy | 68.75% | 31.25% | 71.88% | 79.80% | 0.0936 | Fail |
| G32 pages, 70% budget | 68.75% | 31.25% | 71.98% | 80.02% | 0.0919 | Fail |
| G128 pages, 70% budget | 68.75% | 31.25% | 71.82% | 79.67% | 0.0937 | Fail |

Required gate: at least 30% work reduction, P1 retained mass at least 95%,
mean retained mass at least 99%, and small output error across all layers. No
configuration is close on mass. Smaller groups do not reveal hidden oracle
headroom at the target shape.

### Selector and mandatory-support ablations

The sequence-512 full grid exists to compare all prescribed choices cheaply
before the target-length screen. At 50% density and `G=64`:

| Policy | Worst P1 mass | Worst mean mass | Max output rel. L2 |
|---|---:|---:|---:|
| Mean pages | 44.87% | 79.79% | 0.2710 |
| Best tail (`beta=.5,q=.95`) | 45.70% | 79.76% | 0.2711 |
| P1 coverage greedy | 45.83% | 79.07% | 0.2742 |
| P5 coverage greedy | 46.64% | 79.63% | 0.2741 |
| Mandatory local page | 42.83% | 79.42% | 0.2746 |
| Mandatory first page | 44.54% | 79.78% | 0.2699 |
| Mandatory last page | 41.94% | 77.72% | 0.2956 |
| Experimental support shared across heads | 22.86% | 75.30% | 0.3419 |

Tail and coverage policies shift less than two P1 percentage points and trade
away mean mass. Mandatory pages do not help. Sharing support across MHA heads
is strongly harmful, agreeing with SparseD's head-specific observation.

## Complete-trajectory oracle analysis

The primary trajectory policy is `G=64`, page-only mean selection, 60% hard
budget (38/64 pages = 59.375% actual density, 40.625% logical work reduction).

| Step | Worst P1 mass | Worst mean mass | Max output rel. L2 | Mean previous-step Jaccard |
|---:|---:|---:|---:|---:|
| 0 | 63.20% | 73.29% | 0.1383 | anchor |
| 1 | 64.89% | 78.80% | 0.2089 | 0.584 |
| 2 | 64.64% | 76.08% | 0.1911 | 0.787 |
| 3 | 65.42% | 74.07% | 0.1747 | 0.750 |
| 4 | 63.66% | 73.21% | 0.1520 | 0.710 |
| 5 | 62.66% | 73.10% | 0.1368 | 0.756 |
| 6 | 63.14% | 73.94% | 0.1168 | 0.733 |
| 7 | 63.42% | 74.04% | 0.1043 | 0.720 |

Additional findings:

- Masked rows: worst P1 62.09%, worst mean 72.62%, minimum row 9.17%.
- Revealed rows: worst P1 52.21%, worst mean 71.34%, minimum row 0.46%.
- Masked/revealed maximum attention-output relative L2 is 0.2084/0.2117;
  minimum cosine similarity is 0.9780/0.9777.
- Mean adjacent-query-group support Jaccard ranges from 0.753 to 0.821.
- Mean adjacent-layer Jaccard ranges from 0.553 to 0.638.
- No one of the 32 layers independently reaches both the 95% P1 and 99% mean
  mass thresholds. Sensitive-layer exclusion therefore leaves no sparse layer.

The first-step support is not an adequate MAGE-style anchor: its first reuse
transition has only 0.584 mean Jaccard, and later adjacent-step overlap remains
well below identity. More importantly, even exact current-step selection is
already invalid, so no refresh frequency can repair the base approximation.

## Sink-page analysis

At the 4096-token initial state, 60% ordinary-page budget:

| Sink pages | Sink columns | Actual total density | Work reduction | Worst P1 mass | Worst mean mass | Max output rel. L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 59.375% | 40.625% | 63.20% | 73.29% | 0.1322 |
| 1 | 64 | 60.56% unique | 39.44% | 63.41% | 73.89% | 0.1255 |
| 2 | 128 | 61.80% unique | 38.20% | 63.67% | 74.49% | 0.1196 |

The packed columns are head-shared across every query group, so their reuse is
maximal by construction. One and two pages recover only 0.60 and 1.20
percentage points of worst-layer mean mass. The tested 64/128-column sink
budgets are insufficient; the result does not justify a group-specific gather
or broad cross-layer sink scheme.

At sequence 512 and 50% ordinary pages, individual columns improve P1 more
than physical pages (60.56% versus 44.87%), showing that scattered minority-row
keys do exist. At target length and 70% total budget, however, columns and
pages are almost tied and both fail badly. Thus page contiguity is imperfect
but not the dominant no-go reason; the shared support itself is too diffuse.

## Relationship to existing negative baselines

The repository's existing measurements are retained unchanged:

- BLASST's heterogeneous 4096-token schedule skips 25.99% of physical 128x64
  tiles and yields 1.032x full-model speedup, but masked-token agreement is
  roughly 95.5-96.7%, below the new trajectory-quality bar.
- Query regrouping gains only 3.56 percentage points of physical sparsity and
  5.63% fewer P@V/V tiles, while its measured 0.305 ms/layer overhead is about
  five times the optimistic tile-work saving.
- The 32-row microgroup H100 path is slower than both dense and baseline
  sparse kernels because it loses the favorable 128-row WGMMA mapping.
- Active-Voter pruning reaches only 93.32% worst teacher-forced top-1 at its
  chosen margin and 61.63% final agreement with BLASST; hidden error reaches
  9.6%. Its quality-safe upper-bound complete-attention speedup is below 0.03%.

SPA avoids repeating those mechanisms, but its own normalized-mass oracle
fails earlier than their hardware/trajectory stages.

## Gate status by phase

| Phase | Status | Evidence / reason |
|---|---|---|
| A: current-step oracle | **Fail** | Best work-eligible G64 pages: 71.9% P1, 79.9% mean at target length. |
| B: exact sparse reference | Implemented and unit-tested as a correctness aid; not promoted | Selected-key softmax semantics verified. |
| C: sparse complete-trajectory quality | **Not authorized** | Oracle screening is necessary and failed. No SPA top-1/task-quality claim. |
| D: anchor/refresh | **Not authorized** | Temporal overlap measured diagnostically; current-step support already fails. |
| E: sink analysis | Diagnostic 0/1/2-page oracle completed; **fail** | At most +1.20 pp worst mean mass with 128 sink columns. |
| F: cheap pre-QK selector | **Not implemented** | Exact support failed; estimator error can only worsen it. |
| G: H100 sparse microbenchmark | **Not run** | No quality-passing policy to benchmark. |
| H: WGMMA/TMA implementation | **Not implemented** | Algorithmic and hardware gates were not passed. |

## Direct answers

**Does a quality-safe group-shared support exist for 64 query rows?**  
No, not within the tested 20-70% budgets. The best work-eligible target-length
G64 result misses the mass thresholds by more than 20 percentage points.

**Are important keys naturally contiguous in 64-key pages?**  
Only partially. Columns help minority rows at short context, but at target
length the column upper bound and pages are similarly poor. Diffuse shared
mass, not only contiguity, is the blocking issue.

**How many scattered columns require a sink page?**  
Neither 64 nor 128 head-shared sink columns is enough. They recover only 0.60
and 1.20 points of worst-layer mean mass. The experiment does not support
extrapolating a larger packed sink because the base shared-column bound fails.

**Does the initial all-mask state predict later shared support?**  
Not reliably enough: first-transition mean Jaccard is 0.584 and subsequent
adjacent-step values are about 0.71-0.79. Exact current-step support also fails.

**How often must supports be refreshed?**  
No valid frequency exists for this candidate: refreshing every step still
fails the mass gate. Periodic/fixed schedules cannot improve on that oracle.

**Can a cheap pooled estimator approximate normalized page mass?**  
Not evaluated, by design. The exact target support failed, so estimator
quality and latency experiments were stopped.

**Does the sparse structure create full WGMMA tiles naturally?**  
Yes geometrically: G64 x 64-key pages are full 64x64 tiles, and G128 can use two
64-row subtiles. No quality-compatible sparse tile set was found.

**Does the complete H100 implementation improve end-to-end latency?**  
Unknown and intentionally unclaimed. A complete kernel was not authorized.

## Tests and reproduction

Correctness and repository regression tests:

```bash
conda run -n ljy_dlm python -m unittest tests.test_spa_reference -v
conda run -n ljy_dlm python -m unittest discover -s tests -v
```

Target-length gate screen:

```bash
conda run -n ljy_dlm python scripts/eval_shared_page_attention.py \
  --enable-shared-page-attention --mode oracle --screening-grid \
  --context-length 4096 --steps 1 --num-contexts 1 --seeds 20260721 \
  --output outputs/spa/oracle_screen_4096_step1_budgeted.json
```

Complete oracle trajectories:

```bash
conda run -n ljy_dlm python scripts/eval_shared_page_attention.py \
  --enable-shared-page-attention --mode oracle \
  --context-length 4096 --steps 8 --num-contexts 2 \
  --seeds 20260721,20260722 --spa-group-size 64 \
  --spa-support pages --spa-density 0.6 --spa-selector oracle-mean \
  --output outputs/spa/oracle_trajectory_4096_g64_d60_final.json

conda run -n ljy_dlm python tools/analyze_spa_results.py \
  outputs/spa/oracle_trajectory_4096_g64_d60_final.json \
  --output outputs/spa/oracle_trajectory_4096_g64_d60_final_summary.json
```

Sink and full selector ablations:

```bash
conda run -n ljy_dlm python scripts/eval_shared_page_attention.py \
  --enable-shared-page-attention --mode oracle --sink-grid \
  --context-length 4096 --steps 1 --num-contexts 1 --seeds 20260721 \
  --output outputs/spa/sink_grid_4096_step1_budgeted.json

conda run -n ljy_dlm python scripts/eval_shared_page_attention.py \
  --enable-shared-page-attention --mode oracle --full-ablation \
  --context-length 512 --steps 1 --num-contexts 1 --seeds 20260721 \
  --output outputs/spa/oracle_ablation_512_step1_budgeted.json
```

## Artifacts

- `outputs/spa/oracle_screen_4096_step1_budgeted.json`: target-length Pareto
  screen and per-layer records.
- `outputs/spa/oracle_trajectory_4096_g64_d60_final.json`: four complete
  dense-driven support trajectories.
- `outputs/spa/oracle_trajectory_4096_g64_d60_final_summary.json`: compact
  temporal, layer, and token-state analysis.
- `outputs/spa/sink_grid_4096_step1_budgeted.json`: 0/1/2 sink-page comparison.
- `outputs/spa/oracle_ablation_512_step1_budgeted.json`: 42-policy selector,
  mandatory-support, format, group, density, and head-sharing ablation.

## Failed configurations and next justified experiment

Failed configurations include all requested G32/G64/G128 page budgets from
20% through 70%, columns/pages/hybrid, all tail beta/quantile combinations,
all four coverage objectives, mandatory local/first/last pages, 0/1/2 sink
pages, and experimental broader head sharing. The existing BLASST negatives
cover query regrouping, smaller physical tiles, and Active-Voter row masking.

There is no justified next experiment inside this SPA/kernel ladder. A future
project would need a materially different model-side structure or acceptance
criterion, not a weaker selector or more refreshes. Under the present rules,
the correct next action is to stop and retain the negative result.
