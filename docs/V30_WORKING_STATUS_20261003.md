# 当前工作短记

更新：2026-10-03 04:40（US Central，UTC−5）。本页集中说明做了什么、卡在哪里、接下来做什么；历史证据继续保留在账本。

**当前已按用户要求暂停。** 04:34 等 AIME/HumanEval 全部生成完毕后停止 LongBench；三卡空闲，六个后续协调器已停止，不会自动续跑。

## 现在能下的结论

- **还没有证明 main 或新增 Junyu 变体在最强 vLLM dense 上取得稳定端到端收益。** HF 上的精度证据保留，历史 HF 加速比不能当成相对原生 vLLM 的加速比。
- 先前“sparse 一定比 native 慢”的概括不准确。已有小样本中，64K/96K 的 S/N 比值分别约 0.902/0.864；64K 的 W 却约 1.084，伴随 N 约 1.197。这说明必须同时看单步摊销和步数，不能只看删块率。它们是预览，不是显著性结论。[原始长度核算](../results/v30_20261003/closed_length_accounting001/README.md)
- W 是整次请求耗时，包含 prefill；S 是生成阶段耗时；N 是实际 denoising forward 数。S/N 是摊销成本，固定输入的 forward 微测另做。

## 做了什么

| 工作 | 当前事实 | 证据 / 位置 |
|---|---|---|
| 正式多题、多 seed 面板 | LongBench、AIME、HumanEval 各 8 个 seed，共 96 个 worker；完成 75 个：AIME 32/32、HumanEval 32/32、LongBench 11/32；另中断 1 个 LongBench worker，尚未最终评分 | [暂停记录及 6,857 条脱敏数值](../results/v30_20261003/pause_after_short001/README.md) |
| dense 轨迹差异检查 | 已读实际 sampler/graph/adapter；新增无钩子、native 钩子、仅 metadata 同步的对照，以及独立首步状态诊断；已冻结；队列现已停止 | [既有队列回执](../results/v30_20261003/campaign_queue001/status.json) |
| M3+carry0 的权重对照 | temporal、全 1、前一步置信度三种配置已实现，保留 DP/R6/A64/Q128；全题、8 seed 的 spec 已有，资格检查队列现已停止 | [实现](../experiments/numerical_qk_reuse/v30_sensitivity.py) |
| Junyu 原方法 | 固定分支 b890ff494；原 CPU 公式 31 项通过，CUDA kernel 已编译；原 kernel 的 29 项 GPU 检查队列现已停止，尚未执行 | [原 kernel 构建记录](../results/v30_20261003/junyu_build002) |
| **不带 M3 的 Junyu 新对照** | 新增 vLLM adapter 源码：裸 peer dense、value 观测全保留、独立 value-aware、叠加前一步置信度、叠加前一步 temporal；均不安装 M3/DP/R6/carry0 | [新实现](../experiments/numerical_qk_reuse/v30_peer_standalone.py) |
| 本次实际测试 | 新 adapter 的 10 项 CPU 测试全部通过；CUDA 未初始化，GPU 秒数 0。覆盖原生层选择、张量排布、时序、全保留、异常清理和回执拒绝错误路径 | [逐项测试与源码指纹](../results/v30_20261003/standalone_cpu002/README.md) |

**新增独立 adapter 目前只完成源码和 CPU 测试，尚未接入可启动的端到端 runner，也没有其 GPU/E2E 结果。** 原 kernel 检查和先前 M3 权重/诊断队列也已停止；独立 adapter 从未启动。

## 有什么问题

1. **dense 为什么走出不同步数仍未定位。** 已确认 native hook 每步有 metadata 的 GPU→CPU 读取，官方 sampler 使用默认 CUDA RNG；这两点都不能直接证明根因。首步 tensor/RNG 检查用于定位，侵入式 trace 不用来报速度。
2. **高 sparse 率不等于同幅度整体加速。** GLOBAL 只是整步的一部分，还要计入 LOCAL、MoE、搬运、投影、选块和合并。Junyu 的 fresh value 算法每步仍计算当前 QK，并借助投影 V 估计影响；能跳过的工作与我们的缓存选块并不相同。必须实测各段，不能先承诺收益。
3. **Junyu 当前 uniform Gaussian32 的正式校准阈值没有提交。** README 的 0 是占位；已公开的旧 RULER8K rank32 GLOBAL 阈值为 −4.000799179077148、−2.956143856048584。可作为明确标注的历史阈值迁移候选，不能冒称当前配置已校准或一定达到 50%/75% 稀疏。当前未绑定有限阈值。
4. **独立 Junyu 消费者是他的 SM90 kernel，不是 FA4。** 原生 vLLM dense 仍是 FA4。需要两种全保留对照，把换 kernel 和执行观测的代价单独量出来；不能把换 kernel 的收益都归因于选块。
5. **Haowei regroup 和新模型尚无新的合格收益结论。** regroup 的变体必须算上搬运和恢复顺序开销；新模型不能由 Gemma 的结果外推。

## 用户明确恢复之后，按顺序

1. 复用已完成结果，用新目录重跑中断的 b2_native 和其余 20 个 LongBench worker，组成新 family 后严格打分；随后执行已冻结的开销分解、dense 等价性和工程优化诊断，确认是否存在代码、状态或 RNG 使用问题。
2. 让原 peer kernel 通过 GPU 数值资格检查，再完成独立 adapter 的固定源码装载、真实 paged-KV/GLOBAL-only 数值测试和端到端 runner。CPU fake-router 测试不能替代这些检查。
3. 冻结独立 Junyu 对照：原生默认 FA4 dense、匹配 PIECEWISE dense、裸 peer dense、value 全保留、value-only、value+C、value+T；另外保留 M3+carry0 的对应参考。有限阈值先走独立开发校准或明确命名的历史阈值迁移，不根据最终题集调参。
4. 资格通过后，再在多题 × 8 seed 上比较 LongBench 32K/64K/96K，同时做完整短题 AIME/HumanEval 检查。所有臂同机器/部署/题目/seed，计时中不产生新图；报告 W、S、S/N、N、答对数、实测删块率和按题目聚类的 95% CI。
5. 接着评估 regroup 自然顺序对照、分组及恢复代价，并推进新模型 GPU 冒烟。只扩展通过正确性和执行路径检查的候选。

原始记录已在三台机器和本地私有目录双份保存；旧队列状态文件保留，不能据其 armed/active 字段判断当前仍在运行。恢复时见暂停记录，不能直接重启旧 supervisor。本页保持简短；代码、逐项回执、脱敏数值/trace 随提交保存，原始题目、生成文本与 tensor/RNG trace 留在私有目录。
