# 变体与公平性审计（2026-10-02，US Central UTC−5）

审计底座：本分支 `7d9195047`；正式 V18b 的冻结部署仍是 `8704cd072`。
CHW 只读对象：`chw/value_aware`，`6f7279c1625f7fa53bacb233b96fb69a385df8c1`。
没有合并、修改或吸收组员代码，没有发送 Slack 或修改共享 PPT。

结论：没有证实 main、q64、q64c、regroup、c01 的核心数学、时钟或路由错误，
也没有发现要求中断当前 V18b 的新问题。这是静态路径、已有回执及有限 CPU 测试的审计，
不是对所有 GPU 数值和所有输入的正确性证明。确认的问题主要是历史统计声明、测量边界、
消融范围，以及未支持配置缺少提前拒绝。历史原始结果不改写；本页给出勘误。

## 1. 本分支：已确认的问题与范围

| 项目 | 证据与更正 | 对当前 V18b 的影响 |
|---|---|---|
| 精度“不降/显著更高”的措辞过强 | E13–E15 的 cell 级 McNemar 把同一题的多个 seed 作为配对单位；它不等于题级推断。保留正确数，旧 p 值只作探索性统计，不能证明非劣性。见下节。 | 正式 summary 已按题聚类；仍不能自动宣布非劣性。 |
| HF “零 CUDA capture”没有相应计数证据 | `v27_substrate.py` 读取 Dynamo `stats.unique_graphs`，`v21_run.py` 记录其增量。历史 `Wc` 是排除该计数非零的配对，不能把它称为已验证零 CUDA capture。 | V18b 另有实际 CUDA capture 和 backend/inductor 检查。没有证据表明旧速度必然受 capture 污染。 |
| HF W 的边界比完整请求窄 | `runner.py` 先进入 `_runtime`，随后计 `generate`，结束计时后才 finish/退出 runtime；安装和清理没有纳入 W。漏计大小未测。 | V18b 将 adapter begin/end 纳入 W；不包含 HTTP/网络时间。 |
| mass-only 并非完全 V-free | `v27_dense_prefix.py` 的 mass 分支只替换 prefix top-k 排名，随后仍调用使用 V 风险的 `_dp_tail`；`integration.py` 仍构建 V 投影和 mu summary。低 rank 张量也补零到 32 维。 | main 不变。历史消融支持“prefix 排名的 V 项未显示优势”，不支持“所有 V 依赖/开销都无用或已移除”。 |
| vLLM adapter 不支持 C gate / density gate | `on_sample` 向观察函数传 `accepted=None`，两类 gate 分别需要 `accepted.to()` / `.sum()`。新构造检查提前明确拒绝，新增 6 个 CPU 单元测试。 | 冻结 main 不启用这些 gate；历史 HF P17 也不受影响。新 guard 只用于后续部署。 |
| R17 92K main 回执笔误 | 原 `ruler_long_panel_r17/receipts.md` 写 main/dense=91/91 却标 +1/−0；私有评分表重新聚合为 **+0/−0**。 | 正确数 91/91 不变。原产物保留，本页与账本追加勘误。 |
| “仅请求边界同步”须限定对象 | V18 计数 tracker 不添加逐步 GPU 操作，但 adapter 每步对设备张量 `.tolist()` 仍触发 D2H。 | 成本已由 native-hook/all-kept/main 三臂支付；这是底座优化候选，不是方法创新。 |

额外核对：R6 观察 call 1 后在 call 7、13…重判；q64 细化原 128 行图；q64c 保守保护
新增 prefix 边界与 canvas；regroup 在原 128 行块内重排并逆散射；c01 复用上一 canvas
输出而仍建立观察 summaries。未发现与定义相矛盾的路径。q64 的 `records()` 仍报告
Q128 决策图，不能当作 Q64 实际执行量；E14 另外核对了真实 64 行 list-build 回执。

## 2. 历史准确率：保留计数，降低结论强度

E13–E15 main/dense 汇总正确数分别为 267/274、228/238、93/76。
18 个 seed 是同一组题的重复，不是把独立题数增加 18 倍。按题重采样的准确率差区间约为：

| 长度 | 题数 × seed | main − dense，百分点 | 题聚类 bootstrap 95% CI，百分点 |
|---|---:|---:|---:|
| 32K | 24 × 18 | −1.62 | [−6.02, +2.55] |
| 64K | 24 × 18 | −2.31 | [−6.02, +0.93] |
| 96K | 11 × 18 | +8.59 | [+0.51, +18.18] |

96K 的 11 题精确配对 sign-flip 检验 p=0.125，而旧 cell McNemar p≈0.012。
题 bootstrap 区间与 sign-flip 检验使用不同假设/构造，有限题数下不能互相替代。
不能挑选更有利的检验宣称稳健显著，更不能把“未检出下降”写成已经证明“不降”。
32K/64K 区间仍允许数个百分点下降。要做非劣性结论，需要事先约定可接受差值并有足够独立题目；
严格零损失不是有限样本中所有“不显著”结果的默认解释。

可复现聚合脚本与 toy 单元测试见 `scripts/v27_historical_accuracy_audit.py`。
公开聚合见 `results/m1_m2_m3_frontier_v27_20260929/variant_fairness_audit_20261002/historical_accuracy.json`。
固定 seed 7、20,000 次题级重采样；公开输出只含计数和统计量，私有题标识、生成文本与标准答案均不发布。
该统计复核不改变旧计时或正确数。历史其他重复 seed 面板的 cell p 值同样不能自动解释为题级证据。

## 3. CHW：实现所做的工作与可比性

以下均是固定提交上的只读发现，不是对组员全部实验的否定。

- `_modules.py::_global_sparse_attn` 先计算完整 QK，再按当前 QK 和 value table 排名，
  固定 K 选择后 gather scores/V 并计算 PV。它主要减少 PV 工作，不能与同时跳过 QK/PV 的
  FA4 稀疏 consumer 混称同一 kernel 优化。
- GLOBAL 默认 dense 是 JAX einsum；代码说明此前 GLOBAL BLASST 有未写 KV 的 mask 问题。
  `bench_global_einsum_skip.py` 的局部比例未计完整 criterion 成本。它们不是最快官方
  serving 的端到端速度证据；已修复的 block-id/slot-id 等历史错误不应再报成当前错误。
- 实际工作由 fixed-K cut 决定，而不是输出的 `may_skip` 诊断位图。比较“稀疏度”必须数
  真正执行的 tiles，并计入选择、gather、重排和逆重排。
- step 0 的 value table 全零，代码用 scalar fallback 排名；对 query-group/GQA 取 max 后
  可能大量并列，稳定排序会选旧的前 K 块。生产代码注释已承认此风险，尚无 dense bootstrap
  gate。不能仅凭注释推断真实质量损失；应在独立 dense-bootstrap 对照中测量。
- 本提交 `verify_global_sparse_local.py` 的 numpy `_criterion` 仍直接除 raw RHS，
  没有生产路径的 nonpositive fallback；生产注释引用的 V4 函数在该测试文件中不存在。
  现有 full-keep float64 索引验证不能证明零 RHS 的生产行为或 bf16 整模型恒等。
- **已 CPU 复现的局部统计错误：** `gate_reorder_value.py::skipped_decisions` 对尾部补齐
  使用 `np.arange(pad)`，会重复真实行而不是引用新 padding 行。合成全可跳数据 T=65、
  1 个 block 时报告 128 个 row-block 单位，但实际只有 65 个；T=64/128/256 分别正常。
  当前常用 T=256 不受此例影响，不能据此推翻已有 regroup 数值。
- 最新 regroup 有 per-layer/per-head 路径；旧离线脚本的共享排列描述不能替代实际臂回执。
  本分支 regroup 已收为负结果，不能仅看到新代码就重新宣称有端到端收益。

## 4. PPT、Slack 文件与 C gate：不是同一协议

[megakernel 组会 PPT](https://docs.google.com/presentation/d/1Sr5zEgODnRXSQxlNON5ncgNfwc1rEKly__Clf_nvAnY/edit)
仍有旧的 T=1 与 strongest-dense 表述，当前仓库配置和 dense 更正优先。
PPT 的 regroup/JAX 局部计时与我们的 FA4/vLLM 不直接比较。

[9 月 16 日 sparse_dllm.pdf 消息](https://gaeaheadquarters.slack.com/archives/C0ARABBHG3W/p1789545076350899)
的三页材料已读：thinking OFF、单 seed 42、AIME 2048 / LB 4096 输出预算、LB 中间截断到
32K、GLOBAL+LOCAL、prefix+canvas 可稀疏。我们的 thinking ON、GLOBAL-only、canvas 保护、
自然长题协议不同。文件里的 T 是 token agreement 百分比，并非本分支 query sensitivity。
其校准题进入 headline 的情况有披露；相应 V 优势仍不足以构成独立质量证据。

[10 月 2 日更新消息](https://gaeaheadquarters.slack.com/archives/C0ARABBHG3W/p1790935917275729)
指向更新的 value-aware / query-sensitivity 章节。首次审计未取得正文；用户随后提供的
10 页附件已补齐 §4，现确认 HF C gate 状态公式 (1)–(7) 匹配，温度概率约定仍需区分。
R6 held call 不随每步新权重重判，carry_first 首步也继续消费旧图，因此仍非整套方法复现。
新增 RULER V 方向证据及表间基线差异见 `docs/PEER_PDF_UPDATE_20261002.md`。
P17 只在历史 DP/R6/carry-first 底座上替换 T，44 cells；并非组员 fresh-QK 整套方法复现。
同一个阈值不保证同实际密度，同实际密度也不保证同选择成本。现有结果足以维持“不采用此配置”，
不足以宣称“C gate 普遍无效”。相关方向保留合作归属，不复制组员实现。

## 5. 公平基础上争取收益：先验证，再决定下一分支

1. **当前 V18b 完成后再下速度结论。** 官方 dense 保留默认最快图路径；native-hook 和
   all-kept 给出匹配参考。所有臂同冻结部署、题目、预算、内存设置和机器。报告 W、S、S/N、
   实际 N、commit 数与正确数，使用题聚类区间。重复标签不是相同随机轨迹；对照子集每长度
   只有 4 题，也不能承担大样本精度结论。校准题与 held-out 题应分别披露，不能把新 seed
   当作全新的 held-out 题。
2. **先把通用执行优化做到匹配参考。** 优先测逐步 D2H、页表/metadata 复用和 split 合并成本。
   对适用的 controls/main 同样实现，保留原版 dense。这部分收益计为标准优化，不包装成
   选择算法的新颖性。每个新执行路径仍需测试、数值检查与回执，使用新冻结部署。
3. **最小机制实验比堆更多变体更有用。** 在同一最快 consumer 上比较 hold / 历史 QK+T /
   fresh QK，保持 bootstrap、保护范围和重判预算一致；再用独立校准集匹配分层实际密度。
   同时记录选择器时间与 N，防止每步变快但多走步吞掉收益。纯 V-free 若要研究，必须明确
   tail 与 observation 也如何处理，不能复用现有 mass-only 名字声称已经做过。
4. **潜在创新需针对尚未解释的机制。** 是否可以用廉价状态变化判断何时重观察，减少旧图导致的
   步数膨胀，同时把保护开销限定在真正敏感的阶段？这只是待检验假设，不是已成立贡献。
   SparseD/PulseCol 的图复用与周期选择、Focus-dLLM 的置信度方法已有重叠；与组员
   query-sensitivity 方向重叠的部分先标合作候选。只有相对这些匹配强参考的额外收益才有意义。
5. **取舍由数据决定。** 若 W 的区间跨 1，写“不显著”；若准确率区间仍允许实质下降，写
   “尚不能排除下降”。增加独立题目的价值通常高于只重复相同少量题的更多 seed。
   先扩大已冻结配置的检验，不在评测题上反复挑阈值来制造正结果。

相关原始论文链接与此前阅读记录见 `docs/VLLM_PANEL_V18_20261002.md`。
审计本身没有使用 GPU；正在运行的 V18b GPU 占用单独计入 campaign terminal receipts。

验证：新增 6 个 adapter guard 测试、18 个统计 toy 测试；与既有 V18 测试合跑，
共 188 项通过，20 个子测试通过。真实冻结配置 CPU rebind/validate 通过，方法字段不变。
历史 summary 生成器的术语同步更正，计算公式和已发布产物不改。
