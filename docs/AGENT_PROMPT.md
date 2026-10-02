# 接手提示词（给继续这项研究的 AI）

复制下面整段作为新会话的第一条消息。状态细节都在仓库文档里，这里只放入口、目标和规则；仓库更新后这段一般不用改。

```text
你接手一个进行中的研究：在 DiffusionGemma-26B-A4B（Gemma 4 系列的块扩散 MoE 模型）上，对 5 个 GLOBAL 注意力层做块稀疏，
争取在不掉精度的前提下拿到端到端加速。请以仓库为准，不要依赖聊天记录。

【仓库与分支】
- 本地工作目录：E:/dlm/m3_output_numerics_20260927（git 命令加 -c safe.directory=*）
- 远端：coconight01/dlm_test，分支 research/m3-output-numerics-20260927（最新状态以 git log 为准）
- 阅读顺序：AGENTS.md → HANDOFF.md → docs/RESEARCH_CONTEXT.md → docs/DECISIONS.md → docs/RESULTS_LEDGER.md → STATE.json 的 current

【目标与评价标准】
- 核心是精度、性能、新颖性。基线必须是最强的官方 dense：FA4（vLLM fork）块稀疏接口全保留（D_fa4_allkept）。
- 速度同时报告端到端 W（含 prefill）和纯生成 S，以及每步 S/N、forward 数 N、答对数；配对几何平均 + 按题目聚类的 95% CI。
- 后续面板可以只跑较好的相关配置，无需每次重复原版 M1/M2/M3；历史对比保留。必须保留强 dense 和匹配的优化参考，所有变体如实命名，增量按"相对标准优化之上"计算。
- 请求级结论需要大样本（单个 seed 的步数比在 0.85–1.24 之间摆动）；小样本只能称为预览。
- 每个新变体：写单元测试；用回执（effective_method、计数器）证明预定路径确实执行；所有臂同底座、同部署、同机、同题、同 seed，计时中无新图。
- 宁要干净的负结果，不要硬凑正结果；不显著就写不显著，不过度声称。

【当前状态（详见 HANDOFF.md）】
- 最佳配置 M3 R6 DP −ln2 + carry_first：端到端 32K 0.907、64K 0.853、96K 0.822（纯生成 0.878 / 0.775 / 0.727），精度不降。
- AIME 没有加速空间（最好 1.005，精度持平）；已测试的 AIME/LongBench 中 V 项、M2 vs M3 没有显著优势；不能推广到 RULER。
- 同学提供的待验证线索：RULER 需要 V 维度信息，AIME/LongBench 也许不需要，HumanEval 未知。实际输出始终使用完整 V，这里讨论选块依据。A/B 分支仍等同学，拿到确切 refs 后再评估整合。
- 完整 LongBench-v2 共 503 题，按现有固定 tokenizer 统计，保留 8192 输出预算后 400 题在 262144 上下文以内；完整 503 原文不能只靠 chunked prefill。HumanEval 有 164 题，尚未接入 v27。
- GPU 空闲，没有任务在跑。下一步按 HANDOFF.md「Immediate next steps」的顺序。

【硬性规则】
- GPU 机器是用户自己的，不需要审批，预算不是停止条件，但要记录 GPU 秒数：
  dllm 149.165.159.64；mpk 149.165.151.254（写 /media/volume/dllm-1/dyh）；dlm2 149.165.168.28。
  只在各自 dyh 目录下创建或修改文件；他人目录只读；不改共享环境、包、模型和 JIT 缓存；
  解释器和环境变量用 E:/dlm/v20_private/hosts.json 与 E:/dlm/v27_lbfa4_env.json；每张卡只跑一个 worker，启动前确认 GPU 空闲。
- 面板流程：spec → freeze → deploy → bind → launch → score → summarize，命令见 HANDOFF.md「Operational notes」
  （协调脚本在 E:/dlm/，不在仓库）。被杀掉的 worker 用新的 run dir 重跑，打分时合并。
- 不上传任何凭据、prompt、prompt token 或哈希、标准答案、私有绝对路径；生成记录只发脱敏版。
  不改仓库可见性，不发 Slack 或邮件，不改共享幻灯片，不转发 ssh 密钥。
- 不提交 third_party/dinfer/assets/Wechat.JPG 和 ~$ 开头的锁文件；不 force push；不重置历史账本。
- 组内其他人的分支、结果、机器只读，只引用不吸收，重叠部分标为合作候选。
- 文档实时更新：新结果、新变体、新决策、运行状态变化，都在同一次提交里更新 HANDOFF.md、docs/ 和 STATE.json current，并立即 push。
  新的研究方向开新分支（推送，不合并，不开 PR）。
- 回复用中文；时间写用户本地时间（US Central，UTC−5）。
- 做组会 PPT：用 scripts/v27_build_deck_1001_compact.py 的方式（数字直接读 summary.csv，同时输出 markdown 源稿）；
  正文不标"Fan/我们"，只给外部论文标来源；方法和变体用白话讲清原版配置和每项改动；耗时拆分要把 GLOBAL、LOCAL 单列并带百分比。

【开始时】
1. 读完上述文档，用 git log 确认最新提交，ssh 检查三台机器 GPU 是否空闲。
2. 先用几句话复述你理解的现状、打算先做什么和预计耗时，再动手。
```
