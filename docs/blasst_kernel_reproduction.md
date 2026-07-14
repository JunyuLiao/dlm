# BLASST Hopper kernel reproduction

This reproduction uses the authors' exact MLSys 2026 artifact and its pinned
TensorRT-LLM revisions rather than a functionally similar Triton kernel. That
distinction matters: the reported speedup depends on Hopper warpgroup MMA,
TMA, named barriers, warp votes/reduction atomics, and a phase-specific
pipeline schedule.

## Algorithm implemented by the kernels

For each query tile and KV tile, BLASST always computes the score tile
`S = QK^T`. The online softmax already computes the per-row local maximum
`m_local` and maintains a running maximum `m`. After updating
`m = max(m, m_local)`, the tile is pruned when every participating row agrees
that

```text
m_local - m < log(lambda)
```

The kernel threshold is length-calibrated. Its public input is the scale factor
`a`, and it evaluates `lambda = a / sequence_length`. Algorithm 2 fits
`a(s) = alpha * exp(beta * target_sparsity)` from calibration measurements, so
inference uses `lambda = alpha * exp(beta*s) / L`.

The first valid tile cannot be skipped because it establishes the running
maximum. Masked rows do not vote to prune a valid tile. Processing order is
permitted to change the running-maximum approximation; the released Hopper
kernels use the order chosen by their FlashAttention/XQA schedules.

## Hopper prefill design

Prefill is compute-bound. The SM90 FMHA kernel keeps the normal TMA prefetch of
both K and V and always executes BMM1 (`QK^T`). Two softmax warpgroups compute
the local/running maxima. Each warp votes on the predicate and a shared-memory
atomic/reduction coordinates the tile decision across the warpgroup. A pruned
tile skips exponentiation, row-sum work, packing, and BMM2 (`PV`). V is still
loaded because conditional V prefetch would disrupt the compute-bound
pipeline. The vote is overlapped with existing work, keeping the zero-sparsity
overhead small.

The pinned prefill implementation is TensorRT-LLM commit
`617440d385f091007daafe8bb7206e9a748bb84b`, primarily in
`cpp/kernels/fmha_v2/src/fmha/warpspec/compute.h` and
`cpp/kernels/fmha_v2/src/fmha/softmax.h`.

## Hopper decode design

Decode is memory-bound. A conventional pipeline prefetches V before BMM1 has
produced the pruning decision, which cannot save HBM traffic. The released XQA
kernel therefore batches several K loads and BMM1 operations first, retaining
their small score buffers. It then conditionally issues TMA V loads only for
accepted tiles. Pruned tiles also skip BMM2. Multiple X/V buffers, CTA barrier
pairs, and producer/consumer votes keep the K-first and conditional-V stages
pipelined without scoreboard bubbles.

The pinned decode implementation is TensorRT-LLM commit
`ae012977159bfa84e27b0305358d365fb97ee927`, primarily in
`cpp/kernels/xqa/mha_sm90.cu`. It supports the artifact's paged-KV, beam-width
one, non-speculative SM90 path.

## Reproduce on this H100

The driver clones the artifact at commit
`108bf9b3f29e29c5b5c055edc52f5c42491a4509`, initializes both pinned kernel
repositories, applies small host-build patches that disable unrelated NVRTC
tests and limit the generated prefill matrix to the two BF16/SM90 kernels under
test, builds separate dense and BLASST binaries, runs the authors'
correctness/performance sweep, and writes CSV results.

```bash
conda activate ljy_dlm
python scripts/blasst_hopper_reproduce.py
```

The phases can be run separately. The default prefill sweep uses five
representative points at the paper's 64K shape; the opt-in full sweep uses all
17 artifact factors at 16K and 64K.

```bash
python scripts/blasst_hopper_reproduce.py --phase decode
python scripts/blasst_hopper_reproduce.py --phase prefill
python scripts/blasst_hopper_reproduce.py --phase prefill --prefill-full-sweep
```

To reuse existing decode binaries:

```bash
python scripts/blasst_hopper_reproduce.py --skip-build
```

Results are written under `outputs/blasst_hopper_reproduction/`:

- `decode_dense.log`: unmodified XQA dense baseline;
- `decode_blasst.log`: threshold sweep with exact skipped/total block counts;
- `decode_bf16_h100.csv`: latency, achieved sparsity, and dense-relative
  speedup;
- `prefill_stats.log`: achieved skip ratios from the instrumented kernel;
- `prefill_perf.log`: non-instrumented fused-kernel timings;
- `prefill_bf16_h100.csv`: prefill latency, sparsity, and speedup.

The exact H100 measurements from this run are:

| Phase/configuration | Achieved skip | Dense | BLASST | Speedup |
|---|---:|---:|---:|---:|
| Prefill, batch 1, 64K | 49.24% | 227.132 ms | 188.729 ms | 1.20x |
| Prefill, batch 1, 64K | 79.52% | 227.132 ms | 155.863 ms | 1.46x |
| Prefill, batch 1, 64K | 96.19% | 227.132 ms | 145.407 ms | 1.56x |
| Decode, batch 64, 64K | 40.29% | 2.829 ms | 2.177 ms | 1.30x |
| Decode, batch 64, 64K | 79.89% | 2.829 ms | 1.675 ms | 1.69x |
| Decode, batch 64, 64K | 87.40% | 2.829 ms | 1.584 ms | 1.79x |

All released decode gtests passed across the complete threshold sweep. The
prefill executable was run with the artifact's `-skip-checks` option because
its dense numerical reference materializes the 64K score matrix; kernel-path
selection and skip counts were verified through the instrumented build.

The paper used an H200 for its Hopper table, while this machine is an H100
80GB. Kernel behavior and SM90 instructions are the same, but absolute latency
and the sparsity-to-speedup curve differ because HBM bandwidth, clocks, CUDA
toolchain, and the decode batch/sequence configuration are different. On this
H100, decode meets or exceeds the paper's relative-speedup curve, while
prefill is directionally comparable but lower: 1.20x versus the paper's 1.27x
at about 49.2% skipped, and 1.46x versus 1.64x at about 79.5% skipped. This is
evidence of successful kernel reproduction, not a claim of bit-for-bit H200
performance replication.

## Numerical oracle

`blasst/flash_attention.py` remains the readable Algorithm 1 implementation.
Its zero-threshold path is checked against dense PyTorch attention, and its
pruning test checks the reverse traversal and physical tile decision. It is
not used for timing because Python launches and materialized score tiles do not
represent the fused kernel.
