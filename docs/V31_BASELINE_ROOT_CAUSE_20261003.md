# V31: why "dense" trajectories differed in vLLM, and what the dense baseline is (2026-10-03)

Branch `research/vllm-paired-20261003`, worktree `E:/dlm/vllm_paired_20261003`, parent
`research/cooperative-sparsity-20261003` (V30). Evidence: `results/v31_20261003/graphmode001/README.md`.

## The confusion

V18b–V30 compared the method with two dense arms inside vLLM 0.30.0:
- `dense`: vLLM's default execution, FULL CUDA graphs for decode;
- `native`: the same official dense attention, run in PIECEWISE mode with our hooks.

`native` and every other PIECEWISE arm needed 20–30% fewer denoising forwards and produced shorter outputs than
`dense`, so `native` looked like a "faster dense". Its per-step cost was in fact equal or slightly higher.

## Root cause: an upstream vLLM bug in FULL-graph decoding of DiffusionGemma

- [vLLM PR #51994](https://github.com/vllm-project/vllm/pull/51994), merged 2026-09-30 and not in the latest
  release 0.30.0 (2026-09-22): "Fix DiffusionGemma silently freezing attention mask under CUDA graph replay".
- Mechanism:
  - `DiffusionGemmaModelState._causal_buf`, the per-request causal (commit / encoder) vs bidirectional (denoise)
    flag, was a bool tensor.
  - `FlashAttentionMetadataBuilder.build()` cast it out of place to int32 on every call.
  - FULL CUDA graphs bind the capture-time copy, so every replayed step reads a frozen mask.
- Our measurements match this:
  - With per-request reseeding, FULL and PIECEWISE are bitwise identical at the first denoising call and diverge
    from the second.
  - Over 6 cells, eager / PIECEWISE / FULL need 10.5 / 10.3 / 15.7 denoising calls per canvas.
  - On the V30 short-task panel, FULL dense has more forwards and longer outputs than every PIECEWISE arm.

## Consequences for the earlier vLLM results (V18b, V28, V29, V30)

- Every "method vs default dense" ratio there compares against a dense that runs a frozen attention mask. Those
  ratios mix a real per-step effect with the bug's extra forwards. **Do not quote them.**
- "method vs native" compared against a correct dense, but without seed pairing and with few items. The V18b
  `native` arm also had a smaller warm-up inventory. These are previews only.
- PIECEWISE arms (native, all-kept, main) were not affected. They were run correctly, though unpaired.

## Randomness is controllable

vLLM rejects per-request `seed` for diffusion models, but its sampler draws the initial canvas and all Gumbel noise
from the default torch generators. Reseeding them before each request makes batch-1 runs token-identical across
repeats; verified for 6 cells × 2 repeats in every mode. The method uses private generators, so equal seeds give
equal noise to every arm. All v31 panels use `seed = sha256(base, dataset, index, panel seed, repeat)`.

## The dense baseline from now on

**Primary: vLLM 0.30.0 default execution (FULL decode graphs) plus the exact upstream fix of PR #51994**, seed-paired.
- This is the official default configuration with a merged upstream bug fix, not a hand-made baseline.
- The fix is applied at runtime (`FIX_51994=1` in `scripts/v31_vllm_paired_bench.py`: the causal buffer is allocated
  as int32, so vLLM's own slice assignment updates it in place). No installed file is modified.
- Also reported:
  - PIECEWISE dense, no hooks: the method's own execution mode;
  - the unfixed default, to document the bug's effect.
- Optional cross-check once a GPU is free: a vLLM nightly that contains the fix on one host; then vLLM 0.31 when
  released.

## Metrics on paired requests

- Per-forward cost: S/N (decode span / denoising forwards), plus direct timing on common states.
- Forward count: N = C (canvases) × N/C (denoising forwards per canvas).
- Request time: W (with prefill) and S (generation only).
- Accuracy with the panels' unchanged LongBench-v2 scorer (`scripts/v31_score_paired.py`, on mpk).
- Summary: `scripts/v31_paired_summary.py`, with paired geometric means and item-clustered 95% CIs.

## Running now (started 2026-10-03 05:22 UTC−5)

- E14 LongBench-v2 32K + 64K items × 2 panel seeds = 96 cells, sharded across dllm / mpk / dlm2. Every arm of a
  cell runs on the same host.
  - Panel a: method (main), dense PIECEWISE, dense default (unfixed).
  - Panel b: dense default with the fix.
  - Panel c, all with the fix: dense PIECEWISE, main, main + C gate, base threshold ± C gate, +ln2 threshold
    ± C gate.
- The C gate in vLLM uses the sampler's acceptance mask, recomputed from its own logits with the official
  entropy-bound rule (`vllm_adapter.accepted_mask`, `tests/test_v31_accepted_mask.py`). Variant configs come from
  `scripts/v31_make_variant_config.py`: only `threshold_shift` / `sensitivity` change, validated by v21.
- Then on dllm: `scripts/v31_packgqa_sparse_bench.py`. This is the regroup lever: sharing K/V tiles across the 8
  query heads of a KV head (pack-GQA), which our per-head lists currently disable.

## Prior art and the MAGE baseline (added 2026-10-03 06:40 UTC−5)

- Novelty check against current preprints. The closest prior art:
  - **MAGE** ([arXiv 2602.14209](https://arxiv.org/html/2602.14209), Feb 2026): exact attention at the first step of a
    block, per-KV-head top-k (fixed budget, 512/1024 tokens) reused for the whole block. FlashInfer dense baseline;
    6.82× at 128K. This overlaps our "observe once per canvas, hold the maps" core.
  - **LoSA** ([arXiv 2604.12056](https://arxiv.org/html/2604.12056v1), Apr 2026): stable vs active query tokens by
    query change between steps; stable tokens reuse cached prefix attention. This overlaps the query-protection idea.
  - Also FlashBlock (2602.05305) and PulseCol (2605.20813).
  - None of them analyses how sparsity changes the denoising trajectory or the number of forwards.
- **MAGE port as a baseline arm** (`vllm_adapter.py` arm `mage`; tests `tests/test_v31_mage_port.py`):
  - the same paged split FA4 execution as our method; only the selection rule differs;
  - first call of each canvas: exact dense output plus MAGE eq. 5 at 64-key tile granularity;
  - later calls reuse the selection; the canvas tiles are always kept;
  - all 5 GLOBAL layers are sparsified (MAGE keeps layers 1–2 dense; no GLOBAL layer is among them).
- Panel d (all three hosts, after panel c): MAGE at k = 1024 and 4096 tokens, the same 96 cells and seeds.
- Proposed contribution framing, conditional on panels c / d:
  1. a forward-count-aware evaluation (latency = forwards × per-forward cost, with seed-paired trajectories);
  2. step-stable sparsity: query protection (C gate, collaboration with Junyu) on top of cached reuse, at higher
     sparsity;
  3. sparse execution that matches the strongest production dense path (alias split, pack-GQA).

## Panel a (seed-paired, 48 cells per length, all three hosts) — 2026-10-03 06:10 UTC−5

Reference: PIECEWISE dense, no hooks. Paired geometric means arm/ref with item-clustered 95% CIs
(`scripts/v31_paired_summary.py`; private inputs `E:/dlm/v31_private/paired/`).

| arm | bin | W | N/C (forwards per canvas) | S/N (mean per-forward) | median step | identical outputs |
|---|---|---|---|---|---|---|
| vLLM default dense (FULL, **unfixed**) | 32K | 1.185 [1.04, 1.37] | **1.192 [1.12, 1.26]** | 0.971 | – | 0/48 |
| vLLM default dense (FULL, **unfixed**) | 64K | 1.222 [1.11, 1.37] | **1.275 [1.18, 1.37]** | 0.979 | – | 0/48 |
| main (PIECEWISE) | 32K | 1.006 [0.86, 1.17] | 0.982 [0.93, 1.03] | 1.086 [1.02, 1.18] | 0.955 | 0/48 |
| main (PIECEWISE) | 64K | 1.001 [0.91, 1.10] | 1.065 [1.01, 1.13] | 0.966 [0.92, 1.04] | 0.860 | 0/48 |

**The bug at scale.** Seed-paired, the unfixed FULL default needs 19–28% more forwards per canvas than correct dense.
Its per-forward cost is 2–3% lower, from the FULL graph.

**The method against correct dense: no end-to-end gain yet.**
- A typical sparse forward (per-request median step) is 4.5% (32K) and 14% (64K) cheaper, consistent across hosts.
- The mean per-forward cost keeps little of that, because the per-canvas observation and the re-decisions are
  expensive. The fused observation kernel takes 4.2 ms per GLOBAL layer at 65K, versus about 1.7 ms for vLLM's dense
  call (`observe_split_1002`).
- At 64K the method also takes 6.5% more forwards per canvas.
- The earlier HF-substrate gains came largely from a slower dense reference. The vLLM smoke "0.85× at 64K" was a
  median step against the buggy FULL default.

**Next levers, queued or planned:**
- overhead: observe every 2nd canvas (`cc2`), re-decide every 12 calls (`r12`), 64-row maps (`q64`) — panel e;
  a cheaper observation without the projected-V mu (V term is a measured negative), timed in
  `scripts/v31_observe_cost_bench.py`;
- step inflation: the C gate at the main / base / +ln2 thresholds — panel c;
- prior art at its own budgets: MAGE k = 1024 / 4096 — panel d;
- kernel: pack-GQA — `scripts/v31_packgqa_sparse_bench.py`;
- 96K, where attention is a larger share of the step.
