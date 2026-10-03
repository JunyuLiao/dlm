# Junyu value-aware branch review — 2026-10-03

Publication update: the later `b890ff494` commit supplies the previously missing
implementation and tests. See [the pinned code review](JUNYU_VALUE_AWARE_CODE_REVIEW_20261003.md).
The analysis below remains a historical review of `859c0c8fc`, not a statement
that the newer implementation is absent.

Read-only review of `origin/ljy/value_aware` at `859c0c8fc2a4509a148a3270915f79961e565dee`. No checkout/merge, peer code copy, environment execution, GPU launch, or shared-file modification. References below name files **inside that pinned ref**, not similarly named files in our current worktree.

## Publication boundary

The three newest commits (`86269d901`, `f51244b41`, `859c0c8fc`) change only root `README.md` and `experiments/value_direction_hopper/README.md`; their code predecessor is `47c47d9d7`. `git ls-tree -r` lists only the README inside the Hopper directory. The new README's implementation/kernel/ABI/cache/C_gate/graph statements therefore describe a design whose implementation is not published at this ref. Our worktree's similarly named files do not repair this provenance gap.

All Markdown local links in root README, `experiments/JL_VALUE_DIRECTION_AWARE.md` and the Hopper README were checked with `git cat-file -e REF:path`: 44 occurrences, 25 missing occurrences, 15 unique missing targets. Historical JL reference links and completed report links exist. Commands additionally name a missing `tests/test_value_direction_hopper_report.py` (Hopper README:188–191); bare command paths are not counted as Markdown links.

## What the published code establishes

| Evidence at pinned Junyu ref | Observation and consequence |
|---|---|
| `experiments/diffusion_gemma_jl_output_aware/projections.py:20–38` | Fixed per-native-KV-head Gaussian/sign/identity matrices, hashed seed construction, FP32 generation and recorded matrix identity. The value-direction basis is established peer work, not an invention of our temporal reuse system. |
| `projections.py:49–53, 89–113` | Only complete unchanged prefix blocks reuse sketches. Exact value/validity equality authorizes reuse, with comparison elements and cloned validation copies accounted. Boundary and canvas refresh. This is V-sketch caching, not stale QK/block-softmax/weighted-mean reuse. `torch.equal` returns a host decision and therefore introduces a synchronization boundary on CUDA. |
| `routing.py:40, 58–70` | The router prepares the **current** QK scores and valid mask, then forms current candidate block softmax/projected PV. Information is available after current QK computation, not free before QK. |
| `reference.py:32–74`; `kernels.py:57–100` | Centered candidate impact is measured against the previously **retained** running state, first-support and threshold ties retain, skipping leaves that state unchanged. Physical vote is conservative over valid rows, with sequential retained-state dependence. |
| `routing.py:71–85` | Near-boundary guard uses a host `.item()` and can recompute trusted reference statistics. This accuracy guard and its synchronization must be counted in any speed evaluation. |
| `routing.py:100–117, 120–136` | Native output is retained original-V attention, but the reference execution still uses a dense-shaped masked PV. Additional full-dimensional dense/sparse output diagnostics occur **after** selection. It does not route by projecting a completed full-dimensional block PV, and it does not demonstrate that physical skips eliminate that dense-shaped arithmetic. |
| `experiments/JL_VALUE_DIRECTION_AWARE.md:56–71` | AIME30/LongBench100 historical cohort, inherited scorer/prompt, AIME2048/LongBench4096 output budgets, 32K input cap, calibration examples in headline scores, native temperature0 sentinel, no FlashAttention speedup claim. |

## Latest Hopper/C_gate statements that remain unverified

Root `README.md:19–52` and Hopper `README.md:18–44` describe Gaussian32 current-block output-risk routing and retained original-BF16 PV omission. The historical implementation supports the routing mathematics, but does not verify the claimed newer SM90 retained-PV kernel or its latency.

Root `README.md:54–89` and Hopper `README.md:62–103` specify C_gate: first-call maximal protection, completed-call confidence, renoise EMA, stable accepted run reset on a top-1 flip, and bounded sensitivity added before row maximum. The stated hook runs after acceptance; current-call logits cannot affect that call's route. These are causal **previous completed sampler outcomes**, distinct from the router's fresh current QK statistics. We cannot source-check the hook, accepted-mask interpretation, ABIv4 pointer, reset/canvas boundaries, or formula tests at this ref because `query_adaptive.py` and its test are absent.

Hopper `README.md:154–173, 182–202` requires explicit prebuild and claims binary/ABI/Torch identity checks plus GPU graph tests. Missing loaders/bridge/tests mean build or CUDA-graph qualification is not independently established here. Source `README.md:250–263` requests numerical and synchronized whole-run accounting; it is a protocol description, not a public latency result.

Hopper `README.md:261–270` explicitly identifies native SDPA as the dense dispatch and distinguishes beating that fallback from FlashAttention/TensorRT acceleration. A common CUDA ABI alone does not establish a TensorRT backend (`README.md:169–173`).

## Relation to our current main and regroup

Our value sketch, centered output-risk idea and inherited query sensitivity have peer provenance. Cite Junyu for the value-direction basis and the group member for C_gate; importing a causal formula is a collaboration candidate, not our novelty.

Our `experiments/numerical_qk_reuse/v27_dense_prefix.py:1–20, 35–40, 219–230` uses a prebuilt risk table against **all earlier dense-prefix** statistics rather than the retained state. Its decision stage combines stored log-risk with the current causal sensitivity. This avoids sequential retained-state recomputation during selection but changes the risk surrogate; an equal threshold does not make it numerically identical to Junyu's retained-state router. DP, threshold shifts, observer precision, summary reuse, carry-first and the FA4 consumer must be evaluated as our distinct configured system, with exact effective fields reported.

Our `integration.py:1–5, 26–32, 759–807` separates score and decision clocks and can consume previously stored routing/output scores; cached scores are not fresh current QK. The observer refresh and canvas/boundary protection are the relevant integration boundary. Our main temporal T reads previous completed flips, while the documented C_gate also requires confidence and accepted/renoised-run history. `v21.py:235–239, 466–469, 626` already labels optional C_gate as a cited port, and labels the default as temporal_T. Neither this review nor README-only publication verifies equivalence of our port to an unpublished newer peer implementation.

No query regroup/permutation implementation was found in the audited historical JL router: `routing.py:60–61` iterates natural contiguous 128-query groups, and the published docs describe physical128×64 tiles. No latest source establishes a new low-reordering consumer. Our q64/regroup attribution to chw remains separate (`v27_dense_prefix.py:267–277`). The new sparse-cycle proposal changes movement cost for the same frozen held order; it must not claim a new value gate or appropriate a peer grouping method. Natural uses the same fused LSE merge, so generic merge gains cannot be attributed to regroup.

## Accuracy and speed evidence: fair interpretation

`results/diffusion_gemma_jl_aime_seed43_v12/report.md:3–9` reports150 fresh outputs on the same30 examples with seed43, unchanged seed42 thresholds and projection seed1729. The report's table (`:16–22`) gives dense17/30, full-centered17/30, Gaussian8 15/30, mass16/30 and aggressive BLASST8/30; these are rank8 historical results, **not** a Gaussian32+C_gate result. Its noncalibration24 table (`:28–34`) and limitations (`:51–59`) explicitly disclose development exposure, exploratory paired bootstrap and only two generation seeds. A CI including zero does not prove noninferiority.

The historical reports preserve sampler settings: canvas256, max48 denoising steps, thinkingFalse and temperature0 as the native0.4–0.8 schedule. Our V29 AIME uses a distinct full8192 budget, eight predeclared engine seeds, a vLLM/FA4 base and explicit native graph settings. Comparing a historical2048-token HF/SDPA quality or tile count directly to our8192-token vLLM component/request timings would confound budget, backend, schedule, graphs, attention scope and observer work. The scorer should be pinned unchanged, unanswered/capped cases kept as errors, and output length/native forward count reported alongside wall time.

`experiments/diffusion_gemma_jl_lowrank_multibench_v2.py:271` discloses330 new final slots plus930 compatible cached comparisons; imported evidence is not1260 independently fresh runs. Seed43 adds150 fresh paired runs (`aime_seed43.py:162`). Seed43 shared-QKV diagnostics reuse32 selected early seed42 states while execution-local measurements are fresh (`v12/report.md:51–57`). Same-QKV operator error, generated-trajectory disagreement and end-task accuracy answer different questions.

No hardware speedup is claimed in `JL_VALUE_DIRECTION_AWARE.md:68–71`, `v12/report.md:57` or `results/diffusion_gemma_ruler8k_jl130_v14/report.md:105`. The latter states native RULER scorer and distinct official output budgets (`:9`). Missing C_gate result/kernel/test artifacts prevent a fair claim that the new gate reduced real step count or latency relative to our T/DP main.

## Worthwhile cooperation/verification next

1. Obtain a published/pinned C_gate implementation and its immutable per-condition native forward/output-length/quality/timing evidence before asserting port fidelity. On the same backend compare T vs C_gate with frozen local/global calibration policy, causal reset semantics, matched information availability, counted confidence/hook overhead, and fresh question-cluster inference. Do not retune using final scores.
2. Test the sparse-cycle movement backend only as an independent same-state consumer intervention. Keep the existing held order/support and same fused natural reference; pay forward+inverse restoration and online construction. This isolates remaining movement potential, without using undocumented peer kernel behavior or treating the approximately one-second historical CPU search as free.

## Missing Markdown targets at this pin

| Target | Referencing document/line |
|---|---|
| `experiments/value_direction_hopper/aime_query_sensitivity_gated.py` | README.md:127 |
| `experiments/value_direction_hopper/aime_query_sensitivity_uniform.py` | README.md:126 |
| `experiments/value_direction_hopper/aime_uniform_baselines.py` | README.md:125 |
| `experiments/value_direction_hopper/csrc/torch_bridge.cpp` | README.md:119, experiments/value_direction_hopper/README.md:122 |
| `experiments/value_direction_hopper/csrc/value_direction.cu` | README.md:116, experiments/value_direction_hopper/README.md:118 |
| `experiments/value_direction_hopper/csrc/value_direction.h` | README.md:117, experiments/value_direction_hopper/README.md:119 |
| `experiments/value_direction_hopper/cuda.py` | README.md:118, experiments/value_direction_hopper/README.md:121 |
| `experiments/value_direction_hopper/experiment.py` | README.md:128 |
| `experiments/value_direction_hopper/integration.py` | README.md:122, experiments/value_direction_hopper/README.md:111 |
| `experiments/value_direction_hopper/projection.py` | README.md:121, experiments/value_direction_hopper/README.md:115 |
| `experiments/value_direction_hopper/query_adaptive.py` | README.md:86, README.md:123, experiments/value_direction_hopper/README.md:124 |
| `experiments/value_direction_hopper/query_sensitivity_uniform.py` | README.md:124, experiments/value_direction_hopper/README.md:126 |
| `experiments/value_direction_hopper/torch_build.py` | README.md:120, experiments/value_direction_hopper/README.md:121 |
| `tests/test_query_adaptive.py` | README.md:130 |
| `tests/test_value_direction_hopper.py` | README.md:131 |
