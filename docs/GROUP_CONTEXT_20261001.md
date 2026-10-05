# 群聊研究背景核查

2026-10-01 23:10，US Central（UTC−5）。审阅基点 `073d87ece`。
用户已明确授权只读搜索指定四人研究群；按群成员定位并限定该群检索、阅读相关回复。
本文件仅保存脱敏的方法背景，不保存原始聊天记录、联系信息或附件。未发送消息、编辑共享材料或运行 GPU；GPU 秒数 **0**。

## V 维度与数据集

- 9 月 18 日同学报告：RULER4K、约 70% 稀疏时 full-dimensional 为 88.0%，Gaussian-32 为 87.2%，
  BLASST 为 79.9%；RULER8K、约 75% 稀疏时分别为 49.8%、47.4%、3.3%。这些是同学侧报告，
  缺少该消息对应的完整部署/seed/计时契约，不是本分支的冻结面板结果。
- 仓库引用的 RULER8K mass-only 为 21.3%。**BLASST 与 mass-only 是不同控制臂**，不能把群里
  BLASST 的 3.3% 写成 mass-only，也不能拿上述值与 v27 的历史 QK、GLOBAL-only 底座直接比较。
- 这支持 V 方向信息可能重要，但 rank-32 已保留大部分收益，不能据此认定完整 512 维必要。
  同学也明确描述 AIME/LongBench 的 rank 结果不稳定、非单调，并计划研究动态维度；“也许不需要 V”
  仍是待验证假说。本分支的 E8/E9/E11 阴性不外推到 RULER。
- 指定群内 HumanEval 搜索没有命中。共享 megakernel PPT 可提取文字提及该任务，但图像表格尚未核验。
  不能据此声称 HumanEval 尚无人跑过，或已有完整结果；本分支仍未注册其数据集/执行 scorer。

## 去噪步数与整合限制

- 9 月 21 日同学报告：约 70% 稀疏时 value-aware 的步数约为 dense 的 4 倍，约 50% 时增加约 5%；
  初步分析发现从首步开始置信度降低、top-1 分歧与接受位置变少。消息未完整列出任务/seed/输出预算，
  因此这是机制线索，不能替代本分支的配对 W/S/N/答对数或聚类 CI。
- 保护稳定位置、条件 dense 校正、query sensitivity 等与同学已有方向重叠，继续标为合作候选。
  改接受阈值会改变原生停止规则，不能混入标准稀疏臂；若试验，必须另命名并重新冻结协议。
- 9 月 4 日回复已澄清旧 S/TS 的时序问题：S 使用**同一 forward 前一 Q8 tile**，对 forward 前决定的
  regroup 不可用；来自前一步的 T 或前一步空间平移信号才可能提前提供。这是历史信号定义的限制，
  不是本次发现 v27 实际路径已使用不可用信号。这里的 S/T 是旧空间/时间信号，不是当前性能指标 S。
- 9 月 9–10 日 Haowei 描述其 JAX/XLA 路径逐层 launch 的 CPU 开销由 XLA 消除，余下 GPU launch
  约 2–3 微秒，认为其设置下开销较小。这不测量当前 PyTorch/FA4 底座；megakernel 收益必须实测剩余瓶颈。
- 群里未找到 merge 消息，A/B 的确切 refs 仍未知。9 月 19 日提出集成观察真实端到端性能，并不构成
  本分支已完成集成的证据。收到 refs 后先核对信号可用时点、GLOBAL/LOCAL 范围、QK 新鲜度、V selector
  维度、原生停止规则、投影实际成本与回执，再拟定同底座整合比较；当前不吸收或合并同学代码。

## 群里分享的相关论文

以下已核对原论文页面/实验设置，属于相关工作，不是本分支结果。

- [HERALD](https://arxiv.org/html/2606.21633v2)：块级选 KV、CPU 选择与 GPU 去噪流水重叠。
  §VI-A 使用 LLaDA 2.0-mini/SDAR，准确率评估允许置信度驱动的步数变化，性能评估固定 T=20。
  最大吞吐量还按各方法可用 batch 取峰值，不能作为当前 batch-1、原生自适应停止的 W 加速证据。
- [PRR](https://arxiv.org/html/2606.30389v1)：预测并重用选择集合，将遗漏块用 online-softmax 状态补回。
  §5.1 使用 AR 模型、Quest/InfLLM-v2、H100 TP=2；其参考是串行 DSA，不能直接移植为 FA4 dense 对比结论。
- [SSV](https://arxiv.org/html/2605.19893v2)：稀疏 speculative verification 的跨 query 复用、融合及调度。
  §6 优化接受 token 数与每步耗时，并显式区分精度类别；这与 DiffusionGemma 原生去噪仍需分开评价。
- [BRISK-DLM](https://arxiv.org/abs/2609.33390)：训练风险/收益加权与 prefix-conditioned corrector，
  针对 proposal-verification 的 verified progress。群内已指出 prefix-condition 范式并不自动推广到所有 DLM；
  本次仅核对摘要，未核验附在群里的 PDF 与公开版本是否完全一致。

下一步实验顺序不变：先完成 map drift 的逐层/eligible-prefix 诊断，再做 RULER 的匹配 V 消融。
HumanEval 用作代码质量检验；完整 LongBench-v2 的上下文/显存覆盖工程遵循 `INTAKE_AUDIT_20261001.md`。
