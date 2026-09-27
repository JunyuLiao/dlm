# Fan 今日汇报：M1 与周期调用 M1 的 M3

**14:38 UTC：直接计时与前84次答案对照均已完成；84/84执行、6/6有效完整配对块、42/42 warm通过。** 原生adaptive停止、阈值、输出长度和seed未改。生产代码00a2c4d，评分CP5；后续继续同一冻结面板。

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

## 前84次：收益是否经得住adaptive

每类仅两题、seed101，所有七方法均保留；各题完整方法组固定同一GPU。下表E2E比值为同题同GPU比值的几何平均，method/reference<1更快；正确数均为首次输出。不是独立留出集，也不足以证明质量等价。

| 数据集 / 方法 | 正确/2 | decoder调用总数 | E2E/native | E2E/freshT | E2E/B |
|---|---:|---:|---:|---:|---:|
| RULER R2 | 1/2 | 8 | 1.013 | 1.006 | 1.006 |
| RULER R3 | 1/2 | 8 | 1.013 | 1.006 | 1.005 |
| AIME R2 | 1/2 | 651 | 0.980 | 0.968 | 1.170 |
| AIME R3 | 1/2 | 718 | 1.045 | 1.032 | 1.248 |
| LongBench R2 | 2/2 | 836 | 1.106 | 1.790 | 1.007 |
| LongBench R3 | 2/2 | 483 | 0.636 | 1.030 | 0.579 |

RULER是两任务子集：各方法官方分数均值0.5，不能称13任务macro；native也是8次调用。AIME所有方法均1/2，且各有一题到8192输出上限而未解析，不能把这一题的截断时间当作成功解题加速。native668次、freshT689次、B621次；R2/R3没有超过B。

LongBench的native/freshT/M1/R2/R3均2/2，D_matched和B均1/2（EOS错误）；未遇请求输出封顶。native707次、freshT474次、M1 700次、R2 836次、R3 483次。**R2更便宜的forward被额外调用吞掉，E2E反而慢约10.6%。R3相对native快约36.4%，但比freshT慢约3.0%，尚不支持M3的增量贡献。** 两题样本不作最佳R选择或质量非劣结论，后续固定协议不变。

全七方法、各主机秒数/调用数/每canvas分布、A/D/H、置信区间、EOS/封顶/未解析见[first84详细表](first84_report.md)和[评分JSON](adaptive_panel_first84.json)。全部首次输出和失败记录保留；这84次无执行失败或warm拒收。D_matched的LongBench有一个canvas达到48上限，单独记录。

逐forward计时与答案generation分开，避免同步干扰E2E。两事件计时开关已通过三方法、两机输出一致性检查；可报告首次prefill结束到生成结束的CUDA跨度（包括CPU等待和后续commit），不是decode-only墙钟时间/TBT。

G75/L30：旧adaptive小样本存在，但旧LB输出上限1024、grouping/runtime不同，且缺直接forward与合格decode-only统计。本轮native-Q128移植已完成直接计时/数值/桥接，单独100次答案扩展尚未开始，不冒充旧实现。

完整证据：[长序列与whole-step](selected001_forward.md)、[RULER原生N4](ruler_selected001_forward.md)、[物理稀疏与年龄](selected_physical.md)、[原始N4双scope筛选](per_forward.md)、[旧G75/L30证据](held_bitmap_evidence.md)、[三页英文草稿及讲稿](fan_slide_drafts.md)。RULER的N16保持missing，用原生4次补测，未强制多走step。

提交/推送：本地检查点已保存；自动审批阻止GitHubpush，正在等待对指定现有远端的明确确认。GPU工作和本地报告继续。
