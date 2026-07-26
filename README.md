# Sparse attention for diffusion language models

The `sparse` branch tests whether BLASST-style attention-tile sparsity is
effective for diffusion language models with full bidirectional attention. It
contains the sparse LLaDA kernel, its numerical reference and calibration,
dense-relative evaluation, and the authors' Hopper artifact reproduction.

This is a mechanism-validation branch, not a production inference stack and
not a claim that the portable Triton kernel reproduces every architecture-
specific optimization in the BLASST CUDA kernel.

## Components

- `blasst/flash_attention.py`: readable tiled online-softmax oracle used for
  calibration and correctness checks.
- `blasst/triton_bidirectional.py`: fused BF16 Triton kernel for LLaDA's full
  bidirectional attention with physical 128x64 tile skipping.
- `scripts/llada_blasst_calibrate.py`: masked-token, lambda-zero-relative
  calibration on fixed 4096-token diffusion inputs.
- `scripts/llada_blasst_kernel_benchmark.py`: primary dense FlashAttention
  versus fused sparse-kernel evaluation.
- `scripts/llada_pre_qk_kernel_benchmark.py`: dense versus sparse versus
  conservative pre-QK evaluation, including a gate-disabled kernel ablation;
  see [`docs/pre_qk_kernel.md`](docs/pre_qk_kernel.md).
- `scripts/llada_blasst_vs_sparsed.py`: official `INV-WZQ/SparseD` artifact
  reproduction versus fused BLASST under the paper's matched LLaDA-1.5
  generation and long-context latency settings;
  see [`docs/llada_blasst_vs_sparsed.md`](docs/llada_blasst_vs_sparsed.md).
- `scripts/llada_blasst_benchmark.py`: unfused/reference-path benchmark.
- `scripts/blasst_hopper_reproduce.py`: build and evaluation driver for the
  authors' pinned SM90 prefill and decode artifacts.
- `tools/analyze_query_regrouping.py`: offline current/previous-step query-row
  grouping simulator; the H100 go/no-go analysis is in
  [`docs/query_regrouping.md`](docs/query_regrouping.md).
- `tools/analyze_cross_tile_exception_packing.py`: Phase-A active-row packing,
  V-reuse, extra-QK and metadata feasibility analysis; the semantics audit and
  Hopper stop decision are in
  [`docs/cross_tile_exception_packing.md`](docs/cross_tile_exception_packing.md).
- `eval/eval_active_voter_llada.py`: opt-in `--blasst-row-masking` quality
  evaluation for fully row-masked and output-only Active-Voter BLASST. The
  diagnostic Triton path preserves the selected recurrence but intentionally
  makes no packed-kernel speed claim.
- `tools/analyze_active_voter_contributions.py`: removed attention-mass,
  output-contribution, accumulation-bound, and training-free policy analysis;
  see [`docs/active_voter_blasst.md`](docs/active_voter_blasst.md).

## LLaDA experimental setup

| Item | Setting |
|---|---|
| Model | `GSAI-ML/LLaDA-8B-Instruct` |
| Attention | Full bidirectional MHA through the model's FlashAttention mode |
| Hardware / dtype | One NVIDIA H100 80GB, BF16 inference |
| Shape | Batch 3, sequence 4096, 32 heads, head dimension 128 |
| Diffusion states | One sequence each at 15%, 50%, and 90% remaining masks |
| Sparse tile | 128 query rows x 64 KV columns, reverse KV traversal |
| Dense latency baseline | Installed compiled `flash_attn_func` |
| Numerical reference | The fused sparse kernel at `lambda=0` |
| Fidelity metric | Argmax agreement at masked positions only |
| Lambda schedule | `1.0`, `0.3`, `0.03` for low-, mid-, and high-noise sequences |
| Timing | 5 warmups and 30 synchronized forwards; statistics collected separately |

Dense FlashAttention takes 470.71 ms in this setup. The heterogeneous
diffusion schedule takes 456.33 ms (1.032x), skipping 1,634,893 of 6,291,456
physical tiles (25.99%). Masked-token prediction agreement with the fused
`lambda=0` reference is 96.59%, 95.50%, and 96.71% at 15%, 50%, and 90%
remaining masks.

The intended takeaway is that BLASST's online-max criterion produces useful
physical sparsity in bidirectional diffusion attention. The full-model gain is
smaller than the isolated attention-kernel gain because projections, MLPs,
normalization, and the vocabulary head are unchanged.

## Run the primary evaluation

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

Results are written to `blasst_llada_kernel_4096.json`. The detailed algorithm,
kernel-only optimizations, measurement definitions, and limitations are in
[`docs/llada_blasst_kernel.md`](docs/llada_blasst_kernel.md). Reference-path
calibration is described in [`docs/blasst.md`](docs/blasst.md).

## Reproduce the authors' Hopper kernels

```bash
python scripts/blasst_hopper_reproduce.py
```

Use `--phase decode` or `--phase prefill` to select a kernel family. See
[`docs/blasst_kernel_reproduction.md`](docs/blasst_kernel_reproduction.md) for
the pinned revisions, build patches, setup, and H100 results.

## Pre-QK proxy feasibility framework

The opt-in research framework under `tracing/`, `analysis/`, `proxy/`, and
`eval/` tests whether physical tiles can be skipped before QK computation.
It keeps the fused kernel and all default attention behavior unchanged. Run
the dependency-free CPU smoke pipeline with:

```bash
bash scripts/run_proxy_smoke.sh
```

Collect sampled exact LLaDA traces on CUDA, then analyze/calibrate them with:

```bash
bash scripts/collect_proxy_traces.sh
bash scripts/run_proxy_analysis.sh artifacts/proxy_traces
bash scripts/run_proxy_score_eval.sh artifacts/proxy_calibration/policy_summary.json
```

The design, schema, counterfactual semantics, cost assumptions, and current
go/no-go status are in [`docs/proxy_blasst_feasibility.md`](docs/proxy_blasst_feasibility.md).

## Compare BLASST with official SparseD

Clone the pinned third-party artifact, then run the paper's batch-1
4k/8k/16k/32k/64k LLaDA-1.5 sweep:

```bash
git clone --depth 1 https://github.com/INV-WZQ/SparseD.git reference/SparseD
conda run -n ljy_dlm python scripts/llada_blasst_vs_sparsed.py
```

Results are written incrementally to
`outputs/llada_official_sparsed_vs_blasst.json`. The driver invokes the
authors' SparseD model and generation code directly; see
[`docs/llada_blasst_vs_sparsed.md`](docs/llada_blasst_vs_sparsed.md) for the
exact settings, the shorter smoke command, and the 64k/1024-step command.
