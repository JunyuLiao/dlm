# 新附件核对：value direction 与 C gate（2026-10-02，UTC−5）

来源为用户本次提供的 10 页 `sparse_dllm (1).pdf`，标题为
“Value-direction-aware block skipping with query-sensitivity protection”。
已提取全文并逐页渲染核对。附件保留在私有本地位置，不上传 PDF、截图或正文副本。
它补上了此前审计缺少的 §4 方法正文；不能继续写“最新公式未取得”。
以下同行数据均为附件报告值，未取得完整运行回执进行独立复算。

## 1. 对此前判断的修正

- **C gate 的状态更新公式与本分支 HF 实现一致。** 第 9–10 页公式 (1)–(7) 可逐式对应
  `experiments/value_direction_hopper/query_adaptive.py` 的 `enable_cgate`、`begin` 与
  `observe_logits`：gamma_q=0.65、tau=2.5、beta=3，q 初始为 1、r 初始为 0，首 call
  权重为 4；发生 flip 时 r 归零，否则只在 accepted 时递增。
- `renoised = not accepted` 有当前 HF sampler 源码依据：renoise mask 是 accepted mask
  的补集。observe 在 acceptance 后读取该 mask，下一 call 才使用更新权重，没有发现时间错位。
  实现使用温度处理后的 top-1 概率；附件没有明确温度前/后的约定，所以这一点仍是复现边界。
- prefix/tail 都通过加 `log(s)` 实现乘法加权，但本分支风险还除以 DP reference，
  并非附件的绝对 retained-state 风险。R6 held call 虽更新状态，实际仍消费旧图；
  **carry_first 的首 call 也不会因新 canvas 权重 s=4 而重判保护。**
- **公式一致不等于完整方法已经复现。** P17 将 gate 接到 historical-QK、dense-prefix、
  GLOBAL-only、R6、carry-first 上；附件 §3 是 current-QK、按已保留块递推的输出风险，
  并涉及 GLOBAL/LOCAL 与可稀疏 canvas。P17 未显示收益，不足以否定这套完整方法。
- **V 的负结果只能限定到我们的已测设置。** 附件提供了 RULER 高稀疏度下较强的 V 方向证据。
  这不改写本分支 R17 的负结果，但排除了“V 普遍无用”的泛化说法。

验证：本次重跑既有 `tests/test_v27_cgate.py`，3 项 CPU 测试通过。
没有新算法、GPU 实验或配置变更；冻结 V18b 的 `8704cd072` 部署保持不变。
vLLM 尚无 accepted-token mask，上一轮添加的 unsupported-gate guard 仍然必要。

## 2. 新证据强在哪里，不能推出什么

| 附件证据 | 支持的判断 | 限制 |
|---|---|---|
| 表 5，RULER4K、约 70% 稀疏：full-centered 87.96，BLASST 79.90；报告差值 CI [4.36,12.18] pp | 此工作点的路由准则有区分度 | dense 89.92，Gaussian32 87.21；仍未证明无损。70% 是依据 sweep 选择的工作点，需独立确认。 |
| 表 8，RULER8K、约 75% 稀疏：Gaussian32 47.38，mass-only 21.27；报告差值 CI [18.01,34.31] pp | 在该高稀疏退化区间，V 方向与投影维度确有价值 | dense 89.23，所有这些稀疏臂均严重退化，不满足当前无损目标。 |
| 表 9–10，同 QKV、同 retained history 的算子与反事实诊断 | 比只比较分叉生成轨迹的 retained mass 更能隔离投影误差 | 390 个校准快照来自 step 0，不覆盖所有步骤；较低算子误差不保证较高最终准确率。 |
| 表 11，30 题 × 3 seeds：C gate 54.44%，Gaussian32 51.11%，dense 46.67% | 有值得验证的正向点估计 | 分别相当于 49/90、46/90、42/90；缺逐题配对数据与题聚类区间，不能据 pooled 比例宣称显著或非劣。 |

表 11 的平均每 canvas 步数：C gate 18.63，Gaussian32 19.12，dense 14.97。
C gate 相对 Gaussian32 约少 2.6%，但仍比 dense 多约 24.4%。
若仅作相同 canvas 数、相同非生成成本下的粗略收支估算，每步成本需低于 dense 的
14.97/18.63≈0.804，才能抵消这一步数差。这里不是实际 W、S/N 或配对速度测量，
不能把平均每 canvas 步数直接替代完整请求 forward 数。

附件 §2.7 明确没有硬件端到端速度结论。其 §3 的拟议 kernel 在 QK、exp 和低维 PZ
之后决策，主要跳过完整 PV，不能跳过 QK；§2.7 的 “pre-QK” 措辞与此流程不一致，
应以 §3 的显式依赖顺序为准。不能将物理 tile 稀疏率直接当成同幅度 attention 或请求加速。

## 3. 需要补齐的复现与公平性信息

1. **表 1 与表 11 的基线来源。** §2 说明 seed 42、全部 30 道 AIME。表 1 dense=17/30，
   表 11 seed 42 dense=14/30；Gaussian32 对应也为 15/30 与 16/30。可能来自不同部署、
   配置或运行数值，但附件没有解释。不能混池，也不能据此直接认定代码错误；需要各自 manifest。
2. **相同总稀疏率不足以匹配预算。** 表 7 已披露 BLASST 的 LOCAL/GLOBAL 分配与其他方法
   不同。表 11 只列总稀疏率，仍需分层、prefix/canvas、逐步实际执行量及选择器成本。
3. **RULER 输出预算会影响 dense。** §2.7 披露 8K 的 dense VT 十题均耗尽 30-token 预算并
   得零分；事后排除 VT 后 dense 为 96.67%。公开披露是必要的，但正式对照应统一修正预算后
   重新运行全部臂，不能事后只挑有利任务。RULER 百分比也不能未经核对就转成 strict-correct 题数。
4. **选择与推断分离。** 70% 工作点、最佳 rank 和 current-best gate 均有选择过程；未调整多重
   比较的区间是该实验的报告证据，不等于跨工作点、跨任务的确认。增加 seed 时仍以独立题为聚类。
5. **实现边界。** 拟议 retained-state kernel 要单独测试无有效 KV、首个有效块、全跳过以及
   drop 时不提交 softmax 分母等情况。附件没有完整 kernel 回执，本次不推断这些边界已正确实现。

## 4. 对研究路线的实际影响

保留当前 vLLM 面板优先级。下一步若验证合作方法，应把“current-QK 的 V 路由”和“C gate”
拆成可辨识因子，在同一官方 serving 底座、相同保护范围和已独立校准的分层预算上比较。
先复核默认预算与 dense，再测 W/S/N、实际执行 tiles、选择器时间和答对数；保持 matched
native-hook/all-kept 参考。不能只复用 P17 的同阈值比较便宣布复现完成。

更可能产生额外贡献的问题是：如何在足够便宜的观察与重判成本下，把敏感 query 的保护转化为
可执行的 tile 工作量减少，同时控制 native stopping 的步数膨胀。全 tile 取 max 可能让少数
敏感 query 保护整个 tile，需要先测“敏感行比例 → 实际保留 tile → N → W”的链条。
这只是待检验假设，和组员 query-sensitivity、已有图复用工作有重叠；标为合作候选，不吸收代码。
新实验方向需另开分支，先 spec/freeze；本次仅更正来源与解释。
