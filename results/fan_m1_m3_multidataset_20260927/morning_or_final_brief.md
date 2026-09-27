# v20 最终简报：完整 forward 有局部收益，M3 整体门槛未通过

2026-09-27，生产代码 `00a2c4d`，原生 adaptive stopping 全程保留。700 次主面板及 100 次 G75/L30 独立参考均已完成并评分；400 次首次输出与 400 次 warm 重复全部收齐，warm 全部通过一致性检查。加上资格验证共 844 次 generation，GPU 进程累计 21256.635 秒（5.905 小时，限额 8 小时）。最后 GPU 阶段于 16:56:14 UTC 结束；17:10 UTC 左右只读检查两台 H100 均无计算进程。

**结论：Fan 要求的 M1→周期调用 M1 的 M3 已真实实现、验证和测量。本轮 A8 / P0 / GLOBAL 五层配置未证明 M3 相对 native、fresh-T 和简单 matched B 的增量优势，不能作为已成立的论文贡献。** 这是这一冻结配置的负结果，不是整个方法方向不可行的证明。

## 方法是否正确执行

数值锚点 A8 与决策间隔 R1/R2/R3 分离，锚点重置决策年龄；R3 在 call8 锚定后下次决策为 call11。D 阶段用历史数值 QK 与当前投影 V 调用 M1，H 阶段复用 bitmap；最终输出使用保留块的当前 QK/PV。公共 fast-T、native legality、实际 QKV 数值检查、物理 QK/PV 计数及同代码双机桥接均已验证。没有跳过 Q/K/V 线性投影，没有使用旧最终 attention 权重。

主面板统一 GLOBAL_ONLY_NATIVE_LOCAL / P0 / Triton，五层 GLOBAL 路由、25 层 LOCAL 保持 native。此选择发生在答案产生前。ALL 路径和 P1 也已做成本筛查；没有根据评分结果选 R 或重新拟合阈值。

## 单次完整成本与 adaptive 总成本

下表所有比值均为 method/reference，低于 1 更快。直接 forward 为真实状态重放的完整 decoder 调用，包含方法维护开销；E2E 为同题、同 seed、同 GPU 的 accepted warm 请求配对比值几何平均。

| 数据集 | R2/R3 直接 forward 相对 native，mpk | R2/R3 直接 forward，dllm | R2/R3 请求 E2E/native | native/T/R2/R3 正确数 |
|---|---|---|---|---|
| RULER4K | 1.020 / 1.019 | 1.025 / 1.026 | 1.075 / 1.048 | 均 24/26，官方 macro 92.31% |
| AIME26 | 1.029 / 1.027 | 1.023 / 1.020 | 1.096 / 1.135 | 5 / 8 / 6 / 5，分母 12，严格 EOS |
| LongBench-v2 | 0.958 / 0.950 | 0.918 / 0.912 | 1.111 / 1.031 | 6 / 6 / 6 / 5，分母 12 |

LongBench 的完整 forward 便宜约 4%–9%，完整 denoising-step 同样改善；但 R2/R3 调用数从 native 的 2360 增到 2842/2514，分别多 20.4%/6.5%。相对 fresh-T，R2/R3 的 E2E 为 1.224/1.136；相对 B_A8 为 1.111/1.031。简单 B 的直接 forward 也更便宜。因此局部计算节省没有变成完整请求的增量收益。早期 84 次中的两题 R3 乐观点估计没有延续到完整面板，原表保留。

实际物理稀疏度依赖状态：例如 mpk/P0/R2，AIME 早/晚 canvas 的 GLOBAL QK/PV 跳过分别为 0.8%/1.1% 和 13.6%/16.0%，LongBench 早 canvas 为 49.8%/57.5%。这些分母仅覆盖五层 GLOBAL，不是全模型；A8 数值年龄及真实 A/D/H 序列见 [计数与状态证据](selected_physical.md)。LongBench 为未截断 10–20K prompt；AIME 的晚状态为真实可达 prefix，不强制延长 native 轨迹。

质量门槛也不能宣布通过：AIME native/T/R2/R3 输出封顶分别 5/4/5/6 次，R3 另有一个 task-correct-at-cap，单列而未改算严格成功；LongBench R3 少一个正确答案。样本只有 13/6/6 个已暴露开发题、两个 seed，不能由不显著差异推断质量等价或非劣。

## 用户追加的 G75/L30 adaptive 参考

旧代码确实有 adaptive first-bitmap 证据，但旧 LongBench 仅 8 题、cap1024、旧 vLLM grouping，缺少合格的直接完整 forward 和去初始 prefill 时间。本轮单独命名 **G75L30_nativeQ128**，是 ALL 层新移植，不能冒充旧 grouping 等价。

| 数据集 | G75 / native 正确数 | G75 / native 总 decoder 调用 | G75 E2E/native |
|---|---|---|---:|
| RULER | 24/26 / 24/26；macro 均 92.31% | 106 / 99 | 1.093 |
| AIME | 8/12 / 5/12，严格 EOS | 3367 / 3600 | 0.978 |
| LongBench | 3/12 / 6/12 | 2838 / 2360 | 1.067 |

G75 的 LongBench 直接 forward/native 为 mpk 0.960、dllm 0.920，但调用增加且正确数下降；AIME 直接 forward 为 1.083/1.092，少调用使请求点估计略快，并非单次成本下降。G75 没有提供跨任务质量与时间同时成立的正结果。其请求在主面板之后执行，虽然保持同 GPU、同题、同 seed 配对，仍存在跨阶段时间漂移局限。[独立参考完整统计](historical_panel_report.md)保留每 host 秒数、canvas 分布、cap、错误和精确配对分解。

## 计时边界、失败与缺项

- 逐 forward 同步计时与干净生成分开；请求 E2E 包含初始 prefill、后续 commit、sampler 和输出，排除模型加载。没有用请求时间/调用数冒充直接 forward，也没有跨机拼绝对时间。
- 首次 prefill 结束到生成结束的 CUDA event 跨度已测，并通过开关一致性检查；它含 CPU 发射等待与后续 commit，**不是纯 decode 墙钟、GPU-active 或 TBT**。这些更细边界保持 N/A。
- RULER 原定 N16 自然不可达，32 个缺失序列条目保留，另做实际 N4 合格补测。早期 screen001 配置失败及两次 CPU 构建失败保留；没有重试失败或不确定的 scored 请求。主面板及 G75 无执行失败、无缺失 scored cell。
- 每 canvas 工作、阶段、输出 token、封顶和物理计数均有表。针对自然完整轨迹逐状态重新定价、完整 GPU-active component trace 未新增，不能将捕获状态价格当成整条轨迹直接测量。M2、A4、新 early-phase 规则 NOT_RUN。
- Fan 共享 speaker notes 已核对；指定 ASR transcript 无法取得，保留为缺失佐证。英文三页与中文讲稿仅本地草稿，未发送或修改共享 PPT。

**下一项有根据的工作：** 先利用已保存的配对轨迹定位 LongBench 中 R2 增加 canvas/call 的位置，区分数值历史、bitmap 持有和答案长度的贡献，再决定是否值得另立冻结验证。今天不追加 GPU 参数矩阵，不挑早期正例、不改 stopping 救结果。

[今日完整汇报](fan_today.md) · [主面板逐方法统计](adaptive_panel_report.md) · [三页草稿](fan_slide_drafts.md) · [资源终账](resource_final.json)

所有报告与可运行检查点本地提交。自动审批两次拒绝向既有 GitHub 远端推送，要求用户明确授权 `coconight01/dlm_test` 的 `research/fan-m1-m3-cost-refresh-20260927` 分支；该确认仍待回复，因此推送尚未完成。
