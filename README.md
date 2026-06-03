# Diffusion LM 动态 block sampling 实验

这个 repo 放的是一个轻量级实验脚手架，用来设计和画 Diffusion LM / DLM 的 serving 实验：batch size、dynamic sampling、一个 block 内 token 需要的 decoding step 数量不一样、以及 block size 16/32/64 对性能的影响。

重点不是先做一个真实模型实现，而是先把实验口径定清楚：脚本默认用模拟数据验证图和指标；等你在 H100 上跑真实 DLM decoder 后，把真实 per-request/per-block CSV 喂给同一个脚本即可。

## 快速开始

```bash
bash scripts/run_dlm_experiment.sh

# 或者直接跑 Python：
python3 scripts/dlm_block_sampling_benchmark.py --mode simulate --prompt-file data/prompts_heterogeneous.jsonl --outdir outputs/demo
```

脚本会根据 `data/prompts_heterogeneous.jsonl` 里的 prompt difficulty 生成不同难度 request，不是写死每个 block step。它会输出 CSV 和四张诊断图：

1. `exp1_batch_speed_gap`：`x = batch size 2/4/8/16`，`y = 因 step 不均导致的同步等待 / waste ratio`。
2. `exp2_steps_vs_perf`：`x = 一个 block 里该 request 需要的 decoding steps`，`y = latency`，体现 step 数不一样如何变成性能差距。
3. `exp3_block_size`：比较 `block size = 16/32/64` 下的 mean latency / waste ratio。
4. `exp4_prompt_difficulty_steps`：检查 easy/medium/hard/extreme prompt 是否真的产生了不同 step 数。

另外会写 `prompt_block_steps.csv`，这就是你问的“每个 prompt / request 的每个 block 需要多少 step”的表；当前默认每个 request probe 一个 block，所以 `block_index=0`，以后多 block 生成时这个字段可以直接扩展。

详细实验设置、老师那两句话该怎么理解、以及 H100 上需要 log 什么字段，见 `docs/experiment_plan.md`。


## A100 还是 H100

如果只能二选一，建议选 **H100**：这个实验主要看 batch size 变大、block size 变大、dynamic sampling 后的吞吐/latency 差异，H100 的算力、显存带宽和 BF16/Transformer Engine 余量更大，更适合跑 `bs=16`、`block_size=64`、多 trial 的完整 sweep。A100 可以先做小规模 smoke test，例如 `bs=2/4/8`、`block_size=16/32`，但如果要出最终图和 H100 serving 结论，就用 H100。

## H100 上跑真实模型

推荐先用 LLaDA-8B-Instruct 做 probe，因为它是开源 masked diffusion LM，比较适合测“一个 block 内不同 token 需要多少 step”。如果你想严格复现 block diffusion 论文里的 block-autoregressive 设定，可以再切到 BD3-LM；如果目标是效率优化，可以参考 DPad/Fast-dLLM 类代码。

真实模型探测脚本：

```bash
python3 -m pip install -r requirements-h100.txt

bash scripts/run_h100_llada_experiment.sh

# 等价于：先运行 scripts/llada_block_step_probe.py 生成真实模型 CSV，
# 再运行 scripts/dlm_block_sampling_benchmark.py --mode plot-csv 画图。
# 默认 NUM_BLOCKS=4，block k 的 step 来自模型 confidence，并以前面已生成 block 为上下文。
```

prompt 要故意混合简单翻译、代码、数学、长摘要、SQL、推理等不同复杂度；否则一个 batch 里的 request 太像，step 分布不明显。当前默认 prompt 文件已经显式标了 `difficulty=easy/medium/hard/extreme`。


如果你遇到 `LLaDAModelLM object has no attribute all_tied_weights_keys`，更新后的 probe 已经在 `from_pretrained()` 期间加了兼容补丁；同时不要主动开 `DEVICE_MAP=auto`，默认 `DEVICE_MAP=none` 会先加载模型再 `model.to(cuda)`，单张 H100 可以放下 LLaDA-8B。
