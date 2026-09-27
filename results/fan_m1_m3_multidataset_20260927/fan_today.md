# Fan 今日汇报：M1 与周期调用 M1 的 M3

**14:04 UTC：直接计时、数值核验、两机桥接已完成；前84次正式答案对照正在运行，尚未评分；一次性CPU收集评分脚本已接续等待。** 原生 adaptive stopping、阈值、输出长度和seed未改。生产代码00a2c4d；本地可运行检查点8ded8b9。

M3 的实际刷新执行 M1：用历史数值QK与当前投影V重决策，保留块计算当前QK/PV输出。数值锚点A8与决策间隔R1/R2/R3分离；锚点重置决策年龄。未跳过Q/K/V线性投影。根据预先冻结的成本规则，答案面板选 **GLOBAL五层/P0/Triton**，其余25层保持native；全层路径已测且较慢。公共fast-T不是新贡献。

## 直接完整 decoder forward

每个比值均为同GPU的method/native，<1更快；每台机器分别列绝对时间，不跨机拼接。ms列为该GPU真实状态的直接完整调用均值，比值按状态几何平均。序列最长16次，以native实际停止为限。

| 数据集 / GPU | 实际调用数 | native ms | T/native | M1/native | R2/native | R3/native | B_A8/native |
|---|---|---:|---:|---:|---:|---:|---:|
| ruler4k / mpk | 4 | 139.31 | 1.014 | 1.029 | 1.020 | 1.019 | 1.010 |
| ruler4k / dllm | 4 | 130.36 | 1.005 | 1.042 | 1.025 | 1.026 | 1.016 |
| aime26 / mpk | 12,14 | 134.82 | 1.014 | 1.040 | 1.029 | 1.027 | 1.024 |
| aime26 / dllm | 7,8 | 120.14 | 1.010 | 1.031 | 1.023 | 1.020 | 1.017 |
| longbench_v2 / mpk | 16 | 161.99 | 0.989 | 0.989 | 0.958 | 0.950 | 0.932 |
| longbench_v2 / dllm | 16 | 158.17 | 0.972 | 0.949 | 0.918 | 0.912 | 0.886 |

**判断：** LongBench的R2/R3完整forward约便宜4%–9%，完整denoising-step也改善；AIME和RULER尚无forward优势。LongBench上简单B_A8仍更便宜，不能把native-relative收益当成M3的独有贡献。选定Triton的LongBench D_matched/native为mpk0.984、dllm0.972；R2/R3在较长序列低于它。consumer选择差距仅约0.03%，不宣传为backend优势。

数值误差遵守预先固定的v11容差，真实A/D/H与物理计数twin通过。例：mpk/P0/R2，AIME早canvas的GLOBAL QK/PV仅跳过0.8%/1.1%，晚canvas为13.6%/16.0%；LongBench早canvas为49.8%/57.5%。分母仅为被路由的五层，不能叫全模型跳过率。更高P1稀疏度未产生稳定跨任务成本收益，保留P0。

## 答案与整请求：下一道门槛

前84由每类两题、seed101、七方法×first/warm组成，各完整题目×seed组固定在一台GPU。首次输出用于质量，warm须匹配tokens/calls/终止/phase且无新JIT。会单列分数、EOS/未解析/封顶/失败、调用数/画布、输出长度、E2E及相对freshT/B比值。**当前不能声明质量保持、整请求加速或论文贡献。** 先发布84次结果，再在剩余预算内完成700计划。

逐forward计时与答案generation分开，避免同步干扰E2E。两事件计时开关已通过三方法、两机输出一致性检查；可报告首次prefill结束到生成结束的CUDA跨度（包括CPU等待和后续commit），不是decode-only墙钟时间/TBT。

G75/L30：旧adaptive小样本存在，但旧LB输出上限1024、grouping/runtime不同，且缺直接forward与合格decode-only统计。本轮native-Q128移植已完成直接计时/数值/桥接，单独100次答案扩展尚未开始，不冒充旧实现。

完整证据：[长序列与whole-step](selected001_forward.md)、[RULER原生N4](ruler_selected001_forward.md)、[物理稀疏与年龄](selected_physical.md)、[原始N4双scope筛选](per_forward.md)、[旧G75/L30证据](held_bitmap_evidence.md)、[三页英文草稿及讲稿](fan_slide_drafts.md)。RULER的N16保持missing，用原生4次补测，未强制多走step。

提交/推送：本地检查点已保存；自动审批阻止GitHubpush，正在等待对指定现有远端的明确确认。GPU工作和本地报告继续。
