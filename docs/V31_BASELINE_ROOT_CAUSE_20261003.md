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
