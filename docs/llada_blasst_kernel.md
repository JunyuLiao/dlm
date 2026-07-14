# LLaDA bidirectional BLASST kernel

`blasst/triton_bidirectional.py` is a fused BF16 forward kernel for LLaDA's
noncausal self-attention. It follows the reproduced BLASST prefill semantics
while adapting the pruning unit to diffusion attention.

## Intended takeaway

This experiment tests the sparsity mechanism, not a claim of complete BLASST
kernel reproduction. Its intended takeaway is that the same online-max tile
criterion can skip a measurable fraction of LLaDA's full bidirectional 2D
attention tiles while preserving high agreement with an unpruned execution of
the same fused kernel. Dense FlashAttention is the latency baseline; the fused
`lambda=0` path is the numerical reference, which separates pruning effects
from different kernel reduction orders and BF16 rounding.

## 2D tile decision

Each CUDA program owns one `(batch, head, 128-query-row)` tile and traverses
64-column KV tiles from right to left, matching FlashAttention 2.8. It always
computes `QK^T`. Every valid query row compares its tile-local maximum with its
online-softmax running maximum. Only a unanimous vote skips the entire 128x64
tile; a skipped tile avoids exponentiation, row sums, and `P@V`. As in the
paper's compute-bound prefill kernel, V stays on the normal prefetch pipeline.
If any row votes to retain the tile, the whole tile participates in online
softmax.

There is no causal mask: every query tile considers every KV tile. Partial
tail tiles exclude invalid rows and columns from both maxima and voting.

## Diffusion-aware threshold schedule

A global threshold is unnecessarily constrained by early, high-noise steps.
The default schedule uses the native 4096-token calibration:

| Remaining masks | Lambda | Agreement with lambda-zero kernel |
|---|---:|---:|
| 75-100% | 0.03 | 96.71% at 90% masks |
| 25-75% | 0.30 | 95.50% at 50% masks |
| 0-25% | 1.00 | 96.59% at 15% masks |

The schedule selects the largest tested threshold that retains at least 95%
agreement with the fused lambda-zero path in each diffusion regime. Exact
heterogeneous-batch sparsity and latency are recorded by the benchmark below.

## Experimental setup

| Component | Configuration |
|---|---|
| Model | `GSAI-ML/LLaDA-8B-Instruct`, evaluation/inference mode |
| Model attention configuration | `flash_attention=True`, full noncausal self-attention |
| Device and precision | One NVIDIA H100 80GB, BF16 Q/K/V and FP32 online-softmax state |
| Input shape | Batch 3, sequence length 4096, 32 heads, head dimension 128 |
| Diffusion batch | One sequence at each remaining-mask ratio: 0.15, 0.50, 0.90 |
| Dense latency baseline | The installed compiled `flash_attn_func` bound by LLaDA |
| Sparse numerical baseline | This fused kernel at `lambda=0`, so no tile can be pruned |
| Fidelity metric | Argmax-token agreement at masked positions only, reported by mask ratio |
| Sparsity metric | Skipped physical 128x64 `(query, KV)` tiles divided by all such tiles across 32 layers |
| Timing protocol | 5 warmups, then 30 CUDA-synchronized full-model forwards per configuration |
| Statistics protocol | A separate instrumented forward; counter atomics are disabled during timing |
| Launch | 4 warps, 2 pipeline stages |

The tested diffusion schedule assigns per-sequence thresholds in the same
heterogeneous batch: `lambda=1.0` at 15% masks, `0.3` at 50%, and `0.03` at
90%. The fixed-threshold comparison uses `lambda=0.03` for every sequence.

On an H100 with LLaDA-8B-Instruct, native 4096-token contexts, BF16, and a
heterogeneous `[15%, 50%, 90%]` masking batch, the optimized schedule skips
25.99% of physical tiles. Over 30 measured full-model forwards it takes
456.33 ms versus 470.71 ms for installed dense FlashAttention, a 1.032x
speedup. Fixed `lambda=0.03` takes 459.71 ms (1.024x). Scheduled agreement with
the lambda-zero kernel is 96.59%, 95.50%, and 96.71% respectively.

## Kernel-only optimizations (algorithm unchanged)

These changes preserve BLASST's tile order, 128x64 pruning unit, strict
threshold predicate, unanimous 2D vote, and always-QK/conditional-softmax-PV
algorithm:

- Online-softmax scores, maxima, and thresholds use equivalent base-2 units.
  `exp2` maps directly to the H100 fast exponential path and avoids repeated
  natural-to-base-2 conversion around every exponential.
- The unanimous Boolean vote is expressed as one maximum-gap reduction. A
  tile is skipped iff that maximum is below `log2(lambda)`, exactly equivalent
  to AND-reducing every valid row's predicate.
- The running maximum is updated only on retained tiles. Rejected tiles leave
  all online-softmax state unchanged, as in the paper.
- The launch uses four warps and two pipeline stages. At this shape the kernel
  consumes 255 registers/thread and 98,320 bytes of shared memory, which
  permits two resident CTAs per H100 SM. Eight warps increased pressure and
  reduced throughput; a register cap cannot add a third CTA because shared
  memory is already limiting.

In an isolated `[3,4096,32,128]` BF16 attention-layer benchmark, the scheduled
kernel takes about 1.90 ms versus 2.19 ms for dense FlashAttention (1.15x).
The full model gains less because attention is only part of LLaDA's forward;
the projections, MLPs, normalization, and vocabulary head are unchanged.

Use `install_diffusion_blasst_kernel` around the denoising loop and call
`controller.set_remaining_mask_ratio(ratio)` before each model forward.
`ratio` may be a CUDA vector with one value per sequence, allowing requests at
different denoising stages to remain in the same batch.

```python
with install_diffusion_blasst_kernel(model) as controller:
    while masked.any():
        controller.set_remaining_mask_ratio(masked.float().mean().item())
        logits = model(input_ids=input_ids).logits
        # normal LLaDA token-transfer step
```

## Scope

The kernel supports inference-only BF16 MHA with head dimension 128, no
dropout, no causal mask, and no padding bias. It is wired only through LLaDA's
existing `flash_attn_func` call sites. The current Triton implementation is a
portable functional kernel, not the paper's SM90 warp-specialized/TMA kernel.
Like BLASST prefill, it loads V on the normal pipeline before the pruning
decision, then skips softmax and `P@V` for rejected tiles. The H100-tuned
launch uses four warps and two pipeline stages; other tested warp/stage
combinations were slower for this 128x64, head-dimension-128 shape.

Run the end-to-end sweep with:

```bash
conda run -n ljy_dlm python scripts/llada_blasst_kernel_benchmark.py \
  --context-length 4096 \
  --num-contexts 1 \
  --mask-ratios 0.15,0.5,0.9 \
  --lambdas 0.03 \
  --warmup 5 \
  --repeats 30 \
  --num-warps 4 \
  --pipeline-stages 2
```

Results are written to `blasst_llada_kernel_4096.json`.
