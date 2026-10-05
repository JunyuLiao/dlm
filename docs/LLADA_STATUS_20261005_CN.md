# LLaDA 系列：迁移现状、为什么看 2.2、下一步（2026-10-05）

## 一句话结论

- **LLaDA2.1-mini：** 移植已完成，并验证正确，但这个模型没有加速空间。原因有两个：上下文最长 32K，attention 占比只有 5–10%；我们的稀疏路径在 SGLang 里只能跑 eager，比官方 FULL CUDA graph 慢约 6 倍。所以停在了"不值得跑稀疏预览"这一步。
- **LLaDA2.2-mini（2026-09-05 发布）：** 是现有开源 DLM 里最合适的第二个模型，已经实测 profile。
  - 每次 decode forward 中前缀 attention 的占比，batch 1 时 16–34%（32K–122K），batch 4 时最高 55%，比 2.1 好得多。
  - 但在 SGLang 当前实现里，端到端被 prefill 卡住：prompt 每 32 个 token 跑一次 forward，122K 的 prompt 需要 3,822 次 prefill forward，W = 52 s，其中 decode 只占约 2 s。
  - 稀疏路径还需要先能被 FULL graph 捕获。
- **建议：** 论文主模型仍然是 DiffusionGemma。LLaDA2.2 等 DiffusionGemma 方法定稿（约 10-14）后，作为"泛化到第二个模型"的补充，具体范围见第 6 节。

## 1. 为什么要看其他 DLM

- 论文需要说明方法不只适用于 DiffusionGemma。
- 我们只加速前缀 attention。所以一个模型值不值得做，取决于前缀 attention 在每次 forward 里占多少时间（Amdahl）。
- 理想的每步加速约为 `1 / (1 − s + s/15)`，其中 s 是前缀 attention 的占比。

## 2. LLaDA2.1-mini 迁移：做到哪一步、为什么停

分支：`research/llada21-sparse-port-20261003`，结果在 `results/llada21_20261003/README.md`。

**做了什么：**
- 写了不依赖具体几何的选择核心：M1-DP、MAGE、temporal。
- 写了 SGLang FlashInfer adapter，以 SGLang plugin 形式接入，不改安装文件。
- MAGE / 方法的配置、CPU 测试 18 个。

**正确性（S1，通过）：**
- all-kept 的逐 KV head 路径与官方 dense 的 prefix attention 逐位相同：输出 o 和 LSE 的差都是 0，覆盖 2,240 / 5,660 次调用。
- 前 4 步 logits 相同，最终输出相同。

**官方 dense 基线（S2）：**
- 最快且正确的是 SGLang FlashInfer + FULL decode graph（官方 cookbook 默认）。
- LongBench-v2 8K / 16K / 32K 上 W = 4.39 / 7.04 / 17.89 s。
- 对比：eager 慢 4.5–4.8 倍；FA3 eager 直接崩溃；Triton eager 输出不同。

**为什么停：**
- FULL graph 下每次 forward 约 6 ms，其中随前缀长度变化的部分在 32K 以内只有 0.2–0.6 ms（5–10%）。2.1 的上下文上限就是 32K，再长也测不了。
- 我们的稀疏路径无法被 SGLang 的可断开 decode graph 捕获（实测），只能跑 eager：每次 forward 约 35 ms，adapter 再加 2.4–2.9 ms。这样根本赢不了官方 FULL 的 6 ms。
- 结论：在 2.1 上即使去掉全部前缀 attention，上限也只有几个百分点，所以没有跑稀疏预览（S3），没有浪费 GPU。

## 3. 为什么选 LLaDA2.2-mini

分支：`research/dlm-model-headroom-20261004`（文献和配置调研 + 解析估算）、`research/llada22-headroom-20261004`（实测）。

| 模型 | 问题 / 优点 |
|---|---|
| LLaDA2.1-mini | 上限 32K，前缀占比 5–10%，无空间 |
| **LLaDA2.2-mini** | **原生 128K**（先 64K 训 300B token，再 128K 训 200B）；几何与 2.1 相同，移植可复用；**block routing**：每个 32-token block 最多用 256 个专家中的 48 个，每次 forward 读的权重从约 21 GB 降到约 7 GB，attention 占比随之上升；Apache-2.0；够新 |
| I-DLM-8B / 32B | 最长 40K，扩散训练只用了 4K；因果、每次 forward 只有 7 行，本质上更像自回归稀疏（Quest / DSA 那一类），我们"每个 canvas 观测一次再复用"无从摊销；32B 单卡放不下 32K KV |
| Nemotron-Labs-Diffusion-8B | 解析占比高（64K 约 36%），但只训到 16K，没有长上下文评测；SGLang 支持的 PR 未合并 |
| SDAR / TraDo / DreamReasoner / Fast-dLLM v2 | 上限 32K |
| UltraLLaDA / LongLLaDA / Dream | 整段重算、没有精确 KV cache，不是可部署的场景 |

## 4. LLaDA2.2-mini 实测（dlm2，H100，2026-10-05，约 25 GPU 分钟）

**设定：**
- 官方启动方式：SGLang 0.5.21，加上 block routing PR #31768 和插入/删除解码 PR #31773（两个都未合并，用 overlay 加载，不改环境）。
- FlashInfer，FULL decode graph，JointThreshold Speed 默认参数。
- 输出 512 token，LongBench-v2 prompt。
- 自检：CUPTI 能看到 graph 内的 kernel；block routing 与 HF 参考一致。

**前缀 attention 占 decode forward kernel 时间的比例：**

| batch | 4K | 32K | 64K | 122K |
|---|---|---|---|---|
| 1 | 8.1% | 16.4% | 23.8% | 33.8% |
| 4 | 9.0% | 27.7% | 41.0% | 55.2% |
| 8 | 10.8% | 36.7% | 52.2% | KV 池放不下 |

**其他数据：**
- batch 1 时每次 decode forward 4.96 / 5.74 / 6.27 / 7.19 ms，其中前缀 attention 0.39 / 0.87 / 1.40 / 2.29 ms，与独立 kernel 基准一致。
- 调研阶段的解析估算（64K 约 21–28%，128K 约 35–43%）在 batch 1 下得到验证。
- 关掉 block routing 时，64K 占比 21.5%（开着是 23.8%）。
- 每个 decode block 11.5–14.9 步；用 InDel 解码是 9.3–12.1 步。

**关键问题，端到端被 prefill 卡住：**
- SGLang 的 dLLM 路径对 prompt 也按 32-token block 逐个 forward。122K 的 prompt 有 3,822 次 prefill forward，而 decode 只有 225 次，W = 52 s 里 decode 只占约 2 s。
- 512 token 输出下，就算 decode 无限快，端到端也几乎不变。
- 要让我们的方法在端到端上显现，需要满足其一：长输出（thinking）、并发（batch 4–8）、或者把 prefill 改成按块大批量处理。最后一项是推理栈的问题，不是我们的贡献。

## 5. 把方法真正迁到 LLaDA2.2 还缺什么

1. **稀疏路径能被 FULL CUDA graph 捕获：** 这是最大的工作量，预计几天。不做的话 eager 一定输给官方 FULL。
2. **adapter 从 2.1 改到 2.2：** 几何相同，需要适配 qk-norm、block routing、InDel 解码，工作量小。
3. **负载设计：** 长输出任务，或者 batch 4 / 8 的服务场景，否则测不出端到端收益。
4. **SGLang PR 未合并：** 我们用 overlay，论文里需要写明版本和 PR。
5. **准确率评测：** 在 LLaDA2.2 上用我们的数据集子集，按同样的官方评分。

## 6. 建议和时间

- **现在：** 不在 LLaDA2.2 上投入 GPU，全力做 DiffusionGemma 的算法（MLSys 截稿 10-30）。
- **DiffusionGemma 方法定稿后（约 10-14），二选一：**
  - **A. 只做准确率可迁移性 + 占比分析（1–2 天）：** 在 LLaDA2.2 上用 eager 跑我们的选择，证明准确率守得住，再配上面的占比表，说明"batch ≥ 4、64K+ 时空间更大"。诚实地不报端到端。
  - **B. 完整迁移（约 1 周）：** FULL graph 捕获加长输出或并发负载，报每步和端到端速度。时间紧，风险较高。
- 我倾向 A。B 只在 10-14 之后 DiffusionGemma 主结果已经稳定、还有余量时再做。

## 7. 文件位置

- **LLaDA2.1 移植和结果：** `research/llada21-sparse-port-20261003`（`results/llada21_20261003/README.md`、`docs/LLADA21_PORT_AUDIT_20261003.md`）。
- **其他 DLM 调研：** `research/dlm-model-headroom-20261004`（`docs/DLM_MODEL_HEADROOM_20261004.md`）。
- **LLaDA2.2 实测：** `research/llada22-headroom-20261004`（`docs/LLADA22_HEADROOM_PLAN_20261004.md` 末尾的 Results 节，`results/llada22_20261004/out/summary.md`）。
- **dlm2 上的作业目录：** `/home/exouser/dyh/llada22_20261004/`，模型已下载并校验，32.5 GB。
