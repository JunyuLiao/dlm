# v20 今日直接计时初报（screen002）

2026-09-27 13:05 UTC。两台 H100 各自完成整套同 GPU 对照；每类两题分配在两台机器。下面是前 4 次完整 decoder forward 的直接 CUDA-stream span 之和除以 4，仅为捕获状态的平均直接成本；包含投影、注意力、MLP/MoE、logits，未包含 sampler。AIME/LongBench 分别覆盖两个真实 canvas（0/8 与 0/4）；RULER 为 canvas0。不是 request 时间除以调用数，也不是自然答案的加速比。

## GLOBAL、P0、Hopper 的初步实测

| GPU | 数据集 | native ms/forward | M1/native | R2/native | R3/native | B_A8/native | 最大 bracket 漂移 |
|---|---|---:|---:|---:|---:|---:|---:|
| mpk | ruler4k | 139.32 | 1.032 | 1.018 | 1.017 | 1.008 | 0.49% |
| mpk | aime26 | 126.94 | 1.046 | 1.035 | 1.036 | 1.030 | 0.66% |
| mpk | longbench_v2 | 147.13 | 1.007 | 0.969 | 0.968 | 0.943 | 2.47% |
| dllm | ruler4k | 129.64 | 1.044 | 1.024 | 1.021 | 1.008 | 1.05% |
| dllm | aime26 | 119.38 | 1.038 | 1.027 | 1.027 | 1.021 | 0.93% |
| dllm | longbench_v2 | 149.74 | 0.972 | 0.939 | 0.933 | 0.906 | 0.30% |

比值均为 method/native，<1 更快；每格只在本 GPU 内配对。多个 canvas 的比值用几何平均，绝对 ms 是本 GPU 状态平均，未跨机拼接。全部 260 行无新 JIT；括号漂移最大 2.48%。完整明细随后附 per_forward 文件。

当前结论：R2/R3 减少 M1 重决策成本，但 GLOBAL 的收益主要出现在 LongBench，且仍慢于更简单的 B_A8。AIME/RULER 尚无单次 forward 优势；全层 M1/R2/R3 在三类任务均较慢。不能据此声称质量保持或整请求加速。

## 冻结的后续选择（不看答案）

按已提交的 cost_selection_rule，选择 GLOBAL_ONLY_NATIVE_LOCAL 做基本答案对照；全层路径作为已测负结果保留。P1 在 LongBench 有约 0.5%–1% 的额外成本下降，但其他数据集不稳定，保留 P0，不继续找阈值。

consumer 比较仅用 P0 的 R2 与 B，同题同状态，先各类等权，再两台等权。ALL 的 Triton/Hopper=1.0130，因此 ALL 选 Hopper；GLOBAL 的比值为 0.9996876，因此按预定规则选 Triton，但差异仅约 0.03%，不足以声称 consumer 性能优势。两台分别 1.0006874 与 0.9986879，方向不一致。选定 consumer 后须完成同 QKV 数值检查及较长完整序列、whole denoising step 才放行答案阶段。

## 尚未完成

- 前 84 次正式 first/warm：0/84，不能提供 accuracy、调用数或 E2E 结论。
- 较长（最多 16 次、以 native 停止为上限）和 whole denoising step：下一阶段。
- G75L30 native-Q128 当代移植已测 N4，尚待数值与完整 adaptive 对照；不等同旧 vLLM grouping。旧 adaptive 小面板有证据但缺直接 forward 和合格 decode-only 时间，见 held_bitmap_evidence.md。
- 原始输出/失败保存在不可变私有结果；screen001 的配置身份失败已保留，没有重试任何正式答案。

直接计时与自然 generation 分开运行，避免每 forward 同步干扰 E2E。decode-only 若没有已核验的 prefill 边界，将保留 N/A。
