# Diffusion LM 动态 block sampling 实验

## `sparse` branch: diffusion-LM sparsity mechanism

This branch is an experimental testbed for answering one question: can
BLASST-style attention-tile sparsity be effective for a diffusion language
model with full bidirectional attention? It is not a reproduction of every
specialized BLASST kernel optimization and is not yet a production inference
stack. The branch contains:

- a readable PyTorch numerical oracle for the BLASST pruning rule;
- the authors' pinned Hopper artifact reproduction and build driver;
- a fused Triton implementation adapted to LLaDA's bidirectional attention;
- a diffusion-step-dependent lambda schedule and dense-relative evaluation;
- exact physical 2D tile counts, masked-token prediction agreement, and H100
  latency measurements.

The LLaDA experiment uses the following fixed setup:

| Item | Setting |
|---|---|
| Model | `GSAI-ML/LLaDA-8B-Instruct` |
| Attention | Full bidirectional MHA through the model's FlashAttention mode |
| Hardware / dtype | One NVIDIA H100 80GB, BF16 inference |
| Shape | Batch 3, sequence 4096, 32 heads, head dimension 128 |
| Diffusion states | One sequence each at 15%, 50%, and 90% remaining masks |
| Sparse tile | 128 query rows x 64 KV columns, reverse KV traversal |
| Dense reference | Installed compiled `flash_attn_func` |
| Fidelity reference | The same fused kernel with `lambda=0`, compared only at masked positions |
| Lambda schedule | `1.0`, `0.3`, `0.03` for low-, mid-, and high-noise sequences |
| Timing | 5 warmups and 30 synchronized full-model forwards; statistics collected separately |

With this setup, dense FlashAttention takes 470.71 ms. The heterogeneous
diffusion schedule takes 456.33 ms (1.032x speedup), skips 1,634,893 of
6,291,456 physical tiles (25.99%), and retains 96.59%, 95.50%, and 96.71%
masked-token prediction agreement with the fused `lambda=0` reference at 15%,
50%, and 90% masks. These numbers validate that the mechanism produces useful
sparsity in bidirectional diffusion attention; they do not imply that the
current portable Triton kernel reaches the paper's fully specialized-kernel
speedups.

Run the experiment with:

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

The detailed algorithm mapping, measurement definitions, kernel-only
optimizations, limitations, and results are in
[`docs/llada_blasst_kernel.md`](docs/llada_blasst_kernel.md). The exact Hopper
artifact reproduction is documented separately in
[`docs/blasst_kernel_reproduction.md`](docs/blasst_kernel_reproduction.md).

这个 repo 放的是一个轻量级实验脚手架，用来设计和画 Diffusion LM / DLM 的 serving 实验：batch size、dynamic sampling、一个 block 内 token 需要的 decoding step 数量不一样、以及 block size 16/32/64 对性能的影响。

重点分两条线：`scripts/run_h100_llada_experiment.sh` 是主实验，使用真实 LLaDA forward + 真实 batch size 记录数据；`scripts/run_dlm_experiment.sh` 只是 smoke/debug 用的模拟脚本，不作为最终实验结论。

## 快速开始

```bash
bash scripts/run_dlm_experiment.sh
# 每次运行默认带 UTC 时间戳，例如 outputs/prompt_difficulty_demo/20260603_123456

# 或者直接跑 Python：
python3 scripts/dlm_block_sampling_benchmark.py --mode simulate --prompt-file data/prompts_heterogeneous.jsonl --outdir outputs/demo
```

模拟脚本会根据 `data/prompts_heterogeneous.jsonl` 里的 prompt difficulty 生成不同难度 request，不是写死每个 block step；但它只用于检查图和 CSV 格式。它会输出 CSV 和四张诊断图：

1. `exp1_batch_speed_gap`：`x = batch size 2/4/8/16`，`y = 因 step 不均导致的同步等待 / waste ratio`。
2. `exp2_steps_vs_perf`：`x = 一个 block 里该 request 需要的 decoding steps`，`y = latency`，体现 step 数不一样如何变成性能差距。
3. `exp3_block_size`：比较 `block size = 16/32/64` 下的 mean latency / waste ratio。
4. `exp4_prompt_difficulty_steps`：检查 easy/medium/hard/extreme prompt 是否真的产生了不同 step 数。

另外会写 `prompt_block_steps.csv`，这就是你问的“每个 prompt / request 的每个 block 需要多少 step”的表。真实 H100 脚本默认 `NUM_BLOCKS=4`，所以同一个 request 会有 `block_index=0..3` 多行。

详细实验设置、老师那两句话该怎么理解、以及 H100 上需要 log 什么字段，见 `docs/experiment_plan.md`。

## SparseD first-step accuracy/efficiency check

`scripts/sparsed_quick_eval.py` is a separate, lightweight dense-vs-SparseD
valuation harness. It imports the official SparseD repository as a backend and
records paired output fidelity, latency, generated-token throughput, and peak
GPU memory without attempting the paper's full benchmark suite. Setup, long-
context examples, output fields, and interpretation are in
`docs/sparsed_quick_eval.md`.


## A100 还是 H100

如果只能二选一，建议选 **H100**：这个实验主要看 batch size 变大、block size 变大、dynamic sampling 后的吞吐/latency 差异，H100 的算力、显存带宽和 BF16/Transformer Engine 余量更大，更适合跑 `bs=16`、`block_size=64`、多 trial 的完整 sweep。A100 可以先做小规模 smoke test，例如 `bs=2/4/8`、`block_size=16/32`，但如果要出最终图和 H100 serving 结论，就用 H100。

## H100 上跑真实模型

推荐先用 LLaDA-8B-Instruct 做 probe，因为它是开源 masked diffusion LM，比较适合测“一个 block 内不同 token 需要多少 step”。如果你想严格复现 block diffusion 论文里的 block-autoregressive 设定，可以再切到 BD3-LM；如果目标是效率优化，可以参考 DPad/Fast-dLLM 类代码。

真实模型探测脚本（主实验入口）：

```bash
python3 -m pip install -r requirements-h100.txt

BATCH_SIZES=1,2,4,8,16 \
BLOCK_SIZES=16,32,64 \
NUM_BLOCKS=4 \
MAX_STEPS_PER_BLOCK=64 \
ACCEPTANCE_POLICY=confidence_cutoff \
CONFIDENCE_THRESHOLD=0.95 \
MASK_TOKEN_ID=126336 \
DEVICE_MAP=none \
bash scripts/run_h100_llada_experiment.sh
# 每次运行默认带 UTC 时间戳，例如 outputs/h100_llada/20260603_123456
```

真实 runner 做的是：prompt batching -> append masks -> real LLaDA forward -> confidence-cutoff dynamic unmasking -> 写真实 CSV -> 用真实 CSV 画图。它不是先 `bs=1` 跑完再离线模拟 batch size，也不是把 total steps 平均分给 blocks。输出里最重要的是：

- `block_steps.csv`：一行一个 request/block，含真实 `steps_used`, `latency_ms`, `mean_confidence`, `min_confidence`。
- `batch_block_latency.csv`：一行一个真实 batch/block，含真实 batch 同步 cost、最慢 request step、waste token-steps。
- `per_request_rows.csv`：完整分析表；H100 主路径只保留实际跑出来的 `real_llada_confidence_cutoff` 结果，不再默认输出未真实执行的 dynamic/oracle 曲线。

prompt 要故意混合简单翻译、代码、数学、长摘要、SQL、推理等不同复杂度；否则一个 batch 里的 request 太像，step 分布不明显。当前默认 prompt 文件已经显式标了 `difficulty=easy/medium/hard/extreme`。

### Waste ratio 怎么理解

`waste ratio` 不是 confidence 小的 token 被丢掉的比例。它是 token-step 口径：`useful_token_steps=sum(token_steps)`，表示一个 request/block 里每个 token 被接受前实际经历的 forward 次数之和；`executed_token_steps=batch_finish_steps*block_size`，表示同步 batch 要等当前 batch/block 里最慢 request 时，dense forward 实际执行的 token-step 数。因此 `waste ratio=sum(executed_token_steps)/sum(useful_token_steps)` 衡量的是陪跑/straggler 浪费。


如果你遇到 `LLaDAModelLM object has no attribute all_tied_weights_keys`，更新后的 probe 已经在 `from_pretrained()` 期间加了兼容补丁（包括 `all_tied_weights_keys` 和新版 `tie_weights(missing_keys=...)` 兼容）；同时不要主动开 `DEVICE_MAP=auto`，默认 `DEVICE_MAP=none` 会先加载模型再 `model.to(cuda)`，单张 H100 可以放下 LLaDA-8B。


LLaDA 的 Hugging Face tokenizer 可能不暴露 `mask_token_id`；脚本默认使用官方推理代码常用的 `MASK_TOKEN_ID=126336`，如需覆盖可运行 `MASK_TOKEN_ID=<id> bash scripts/run_h100_llada_experiment.sh`。

## Request length / block steps probe

`llada_length_step_probe.py` isolates prompt-length effects. It creates 24
requests with exact token lengths from 50 to 2000, first runs each request at
batch size 1, and then reuses the same requests in three ordered batches of 8
(default maximum lengths: 300, 1100, and 2000).

```bash
conda run -n ljy_dlm python scripts/llada_length_step_probe.py \
  --outdir outputs/llada_length_steps \
  --block-size 32 \
  --num-blocks 4 \
  --max-steps 64 \
  --confidence-threshold 0.90
```

The output includes exact request and batch membership tables,
`exp1_step_latency.csv`, per-block and per-request experiment-1 summaries, and
`exp2_block_summary.csv`. The wide `exp1_request_summary.csv` shows all four
block step counts for each request; `exp2_batch_summary.csv` shows the eight
request IDs and lengths alongside all four synchronized batch/block step
counts. These batch step counts are not independently measured per request.

## BLASST Hopper kernel reproduction

The paper's exact pinned SM90 prefill/decode kernels and the BF16 H100
dense-versus-BLASST benchmark can be prepared, built, and run with:

```bash
python scripts/blasst_hopper_reproduce.py
```

Use `--phase decode` or `--phase prefill` to run one kernel family. The default
prefill run is the paper's 64K shape at five representative thresholds;
`--prefill-full-sweep` enables all 17 artifact thresholds at 16K and 64K.

See `docs/blasst_kernel_reproduction.md` for the full algorithm-to-kernel
mapping, pinned revisions, host-build patch, experiment configuration, and
result interpretation.
