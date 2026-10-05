# Fan 今日汇报：完整700次主面板与100次G75参考

**700/700执行、50/50完整有效配对、350/350 warm通过；主面板无执行失败。** RULER13题×2seed，AIME和LongBench各6题×2seed；均为已暴露开发集。生产00a2c4d、评分CP5。G75/L30单独100次参考也已完成评分，50次warm全部有效；两台GPU已空闲。

**当前判断：M1与周期调用M1的R2/R3已真实执行，但这一冻结配置未通过“比强dense及fresh-T更快、且答案质量可接受”的整体门槛。** LongBench的完整forward确实变便宜，native-adaptive增加的工作量抵消了收益。不能把早期两题R3快36%的结果当成完整结论，也不能据此断言整个方向不可行。

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

## 完整native-adaptive答案与E2E

时间均为同题同seed同GPU的method/native比值，再取几何平均；<1更快。质量为首次输出，AIME列严格EOS正确数；RULER列官方13任务macro。每个方法另有一次通过一致性校验的warm计时。没有跨机拼接绝对秒数。

| 方法 | RULER macro / E2E | AIME 正确/12 / E2E | LongBench 正确/12 / E2E |
|---|---|---|---|
| native | 92.31% / 1.000 | 5/12 / 1.000 | 6/12 / 1.000 |
| matched dense | 92.31% / 1.014 | 7/12 / 0.915 | 4/12 / 1.072 |
| fresh-T | 92.31% / 1.004 | 8/12 / 1.011 | 6/12 / 0.908 |
| M1 R1 | 92.31% / 1.054 | 7/12 / 1.042 | 6/12 / 1.025 |
| M3 R2 | 92.31% / 1.075 | 6/12 / 1.096 | 6/12 / 1.111 |
| M3 R3 | 92.31% / 1.048 | 5/12 / 1.135 | 5/12 / 1.031 |
| B_A8 | 92.31% / 1.068 | 7/12 / 1.057 | 4/12 / 1.001 |

- **LongBench：** native/fresh-T/R2/R3总decoder调用分别2360/2223/2842/2514。R2多20.4%调用，E2E慢11.1%；R3多6.5%调用，E2E慢3.1%。相对fresh-T，R2/R3的E2E比值为1.224/1.136；相对B为1.111/1.031。fresh-T点估计快9.2%，正确数同为6/12；探索性时间区间跨1，不能宣称已证明稳定加速。
- **AIME：** native/fresh-T/R2/R3严格正确5/8/6/5（分母12）；请求输出封顶5/4/5/6次。R3另有1个到cap时任务答案正确，按冻结口径单列，未改算严格成功。R2/R3相对fresh-T慢8.4%/12.2%，相对B慢3.6%/7.3%。不把封顶时间当完整成功解题加速。
- **RULER：** 所有方法macro92.31%，R2/R3的E2E慢7.5%/4.8%；短canvas没有足够刷新摊销空间。

调用总量是工作统计，不与几何时间比值相乘作精确分解。各GPU上的配对时间=调用数×摊销时间分解、每canvas分布、A/D/H、caps、错误/未解析和探索性区间见[完整表](adaptive_panel_report.md)与[JSON](adaptive_panel.json)。此样本不足以证明质量等价或非劣。

## G75/L30完整结果与边界

旧adaptive证据已核对：小样本、LBcap1024、旧grouping/runtime，缺合格直接forward/生成段计时。本轮G75L30_nativeQ128是明确标注的新移植，已直接计时/数值/桥接合格；冻结100次已完成，与同题同seed同GPU的native配对；参考运行晚于主面板，仍有跨阶段时间漂移局限。不能冒充旧实现等价。


| G75L30_nativeQ128 | 严格正确 / native | 调用数 / native | E2E/native |
|---|---|---|---:|
| RULER | 24/26 / 24/26，macro均92.31% | 106 / 99 | 1.093 |
| AIME | 8/12 / 5/12 | 3367 / 3600 | 0.978 |
| LongBench | 3/12 / 6/12 | 2838 / 2360 | 1.067 |

G75在LongBench直接forward/native为mpk0.960、dllm0.920，但额外调用及质量下降使整体无优势。AIME直接forward反而慢8%–9%，请求点估计略快来自工作量减少，不能称单次成本改善。RULER/AIME/LB输出cap分别2/4/0，LB另有1个canvas达到iteration cap。详见[独立参考统计](historical_panel_report.md)。

最终共844次generation（含44次资格验证）、5.905 GPU进程小时。完整结论与缺项见[最终简报](morning_or_final_brief.md)，不再追加GPU实验。

逐forward计时与干净generation分开；请求E2E含prefill和后续commit。另测首次prefill结束到生成结束的CUDA跨度（含CPU等待/commit），不是decode-only墙钟或TBT。RULER N16仍missing，以自然4次补测；M2/A4/新phase规则未运行。最终简报已交，不因本轮负结果开启新调参矩阵。

[早期84次](first84_report.md)保留不变；[三页英文草稿](fan_slide_drafts.md)随完整结果更新。所有运行检查点已本地提交。用户随后明确授权推送代码与可分享数据；800条生成记录见 [数据目录](generation_records/README.md)，发布状态见 [发布记录](publication.md)。
