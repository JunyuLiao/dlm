# 接手浏览与数据集可行性核查

2026-10-01 22:45，US Central（UTC−5）。审阅基点 `6b13fe178`，SSH 检查时三台 H100 均为 0% 利用率、0 MiB、无计算进程。
这轮没有加载模型权重、生成答案或启动 GPU worker，GPU 秒数 **0**。

## 公平性检查与修复

`scripts/v21_run.py` 已强制冻结的整格 host/GPU 分配，`scripts/v21_score.py` 已检查原始回执和执行身份。
但后续单独调用 `scripts/v27_fa4_panel_summary.py` 时，原实现有两个缺口：重复成功记录由后写的覆盖，
计时数据与打分 CSV 的连接没有复查 host/GPU、cell 和来源。漏传一个机器的 ledger 也可能悄悄缩小面板。

修复后，汇总拒绝重复 first（含先失败后成功的替换）、未打分或身份不符的成功记录、缺失的成功执行，
以及同格不同 host/GPU、substrate、protocol、模型 revision 或源码的组合。CUDA graph 计数未知时不纳入 Wc。
这只加强统计入口，不改变模型、稀疏算法、已有冻结协议或历史账本。

已核对 E4/E5/E6/E6b/E7/E8/E9/E10/E11 的 **8,826 次 first 执行**。同格配对与打分来源全部通过，
W/S/P/N/NC/T/SN/ST/Wc 点估计和答对数全部复现。E7、E8 各有已知的 3 次计时新图，仍应采用 Wc。
配对几何平均和按题目聚类的 bootstrap 算法未更改；本次没有重写历史 CSV 或据复算改动已发布 CI。
验证代码见 `tests/test_v27_fa4_panel_summary.py`，包括重复覆盖、跨机配对、缺失 ledger、未打分和未知图计数。
最终 **17 项 CPU 回归测试通过**（登记的 dlm2 解释器），最终校验器也已在上述九个面板全部通过。

多机只合并每格同机的 method/dense 比值。E5 最佳配置的 host 分层 W（每机每档 48 格）如下，作为诊断：

| 主机 | 32K W | 64K W |
|---|---:|---:|
| mpk | 0.8201 | 0.8416 |
| dllm | 0.9416 | 0.8938 |
| dlm2 | 0.9667 | 0.8260 |

这些主机对应不同题目/seed 组合，差异不能直接归因为硬件。下一轮按冻结整格分配、平衡题目和 seed、
平衡执行顺序，并报告 host 分层诊断。若要测机器效应，需要额外让一小组相同格在各机跑相同 dense/方法。
跨机重复不能当作独立的新题或精度样本。

## 记录同学关于 V 的线索

用户转述：**RULER 需要 V 保持维度，HumanEval 未知，AIME/LongBench 也许不需要**。这是待验证假说。
当前所有臂的实际注意力输出一直使用完整 V；rank-32/16/8/4 只作用于选块依据。`RankMaskedProjections`
仍保留 32 列存储形状，所以已有 rank 消融也不等于实际减少投影 kernel 的存储和计算规模。

仓库已引用的同学 RULER8K 结果，在约 75% 稀疏度下 full-V 为 49.8%、rank-32 为 47.4%、mass-only 为 21.3%。
这支持保留 V 方向信息，尚不能证明全部 512 维必要；不是本分支的新结果，也不能跨底座比较速度。
AIME/LongBench 的 E8/E9/E11 阴性仅限已测配置，不能外推到 RULER。

已只读查看共享 [megakernel PPT](https://docs.google.com/presentation/d/1Sr5zEgODnRXSQxlNON5ncgNfwc1rEKly__Clf_nvAnY/edit)
中可提取的文字：其中有 value-aware 对 BLASST 的观察、query sensitivity/temporal 保护及 HumanEval 后续计划。
“优于 BLASST”不等于“V 项优于 attention mass”；图像中的表格未作为已核验数字引用。
Slack 公开搜索没有找到相关 RULER 讨论，私有频道搜索需等待用户同意，不访问私信、不发送消息。

拟议下一轮保留 dense、mass-only、rank-32、完整 selector V，冻结匹配的 tile 预算。
先在同一 historical-QK、GLOBAL-only、当前底座上隔离任务差异；再把 fresh-QK 与 GLOBAL+LOCAL 设置单独测试，
避免把信息新鲜度或 LOCAL 稀疏归因到 V 维度。后者与同学工作重叠，属于合作候选。
32K/64K 现有 RULER pool 各只有 13 个样例，应扩充官方各子任务样本再下请求级结论。
完整 selector V 尚未接入 v27，新增时需要单测和 effective_method/计数器回执。

## 能否完成全部 LongBench-v2

[官方数据集](https://github.com/THUDM/LongBench) 共 **503 题**，原文最长达 200 万词。
现成的私有 build.log 已对所有题目使用固定模型 tokenizer 和本项目相同模板、thinking ON、无截断渲染。
本次只提取长度聚合数，没有输出或提交题目、答案、prompt token 或其哈希。
登记模型快照的 `config.json` 也已只读核验：文本 `max_position_embeddings = 262144`。

| 渲染后输入 tokens | 题数 |
|---|---:|
| <32,768 | 110 |
| 32,768–65,535 | 72 |
| 65,536–95,074 | 42 |
| 95,075–131,071 | 70 |
| 131,072–253,952 | 106 |
| 253,953–262,144 | 2 |
| >262,144 | 101 |
| 合计 | 503 |

输入最短/中位/最长为 **10,334 / 107,706 / 5,174,028 tokens**。
保留当前 8,192-token 输出预算，**400 题**在配置上下文以内。**224 题**不超过现有约 95K 的显存经验界限，
这只是长度筛选，不能保证所有题都可生成成功。当前已运行的 32K/64K/96K 面板并非完整 503 题测试。

有两条明确路线：

1. 原文可用范围：先做所有臂共用的 chunked prefill，检查与 dense 的数学语义和生成精度，再冻结上下文内
   400 题的全覆盖面板。保留完整原文，但报告其为上下文内子集，失败/OOM 逐项统计。
2. 全部 503 题：统一规定输入上限及截断或检索规则，所有臂使用完全相同处理。报告完整题目覆盖以及每档
   的截断数量，单独命名协议，不能混同为 503 题全部原文的原生推理结果。

Chunked prefill 降低峰值显存，不延长模型上下文；GLOBAL KV 的常驻显存与长输入注意力成本仍需测量。
在模型原生配置下，单靠更长 GPU 时间或三机并行不能让超出上下文的 101 题成为原文全覆盖。
三机当前是独立单卡 worker，没有模型并行底座。若实现新底座，dense 与稀疏均需在新底座重新测量，另开研究分支。
以上都是可行性分析，尚未启动完整数据集面板。

## HumanEval 规模与用途

[官方论文](https://arxiv.org/html/2107.03374v2)与[执行评分代码](https://github.com/openai/human-eval)包含
**164 道 Python 编程题**，按生成函数通过测试的情况评价，常用指标为 pass@1。
6 个 seed 对应 **984 次生成/方法**；dense + 3 个 selector 比较臂共 **3,936 次**。
这里的六次随机采样用于估计 pass@1 和配对差异，不自动等同于 pass@6。

HumanEval 适合作为代码质量泛化验证。其短上下文能否出现稀疏加速需实测，不能由长上下文结果推定。
v27 当前尚无 HumanEval 数据集注册、函数抽取与执行 scorer 契约；接入时应冻结原始题目、代码抽取、
输出预算、停止规则与隔离执行测试方式，并对脚本自测。先小样本查路径和抽取正确性，再覆盖全部 164 题。

## 接下来的优先顺序

1. 保留 HANDOFF 首项：测量 M3 在同一 canvas 内 call 2/8/14 的 map 变化，约 40 分钟、单机。
   现有 `v27_map_drift.py` 将各层变化合并统计，尚不输出每层/具体 call 对，也包含必留块。
   正式诊断前应补 eligible prefix 范围、逐层 call 对、翻转比例和新增/丢失 support，避免必留块掩盖变化。
   采样地图的诊断开销不进入正式计时比较。
2. RULER 的 V 信息匹配消融，优先于默认扩大已打平的 E10/E11。
3. HumanEval 质量验证和 LongBench 覆盖工程分别评估、各自冻结协议。

A/B 分支仍等待同学。当前未获得确切 refs，没有吸收同学源码或结果，也没有 merge/PR。
Mirage [MPK 原论文](https://arxiv.org/abs/2512.22219)讨论 persistent megakernel；对当前已使用 CUDA graph 的
DiffusionGemma 底座，需要先测剩余瓶颈。共享 PPT 的融合方向作为合作背景，尚无本分支的新颖性或收益证据。
