# v21 冻结面板离线汇总（模板）

协议：`v21_zero_pruning_1ee6332c82377856`；类别：`zero_pruning_diagnostic`。已记录 12/12 次执行。

质量仅统计成功的首次生成；时间仅统计严格接受的 warm 请求。绝对时间按 GPU 主机分开；同 GPU 配对比值小于 1 表示请求耗时较少。

| 数据集 | 主机 | 方法 | 首次成功/计划 | 首次评分 | 严格正确 | warm 接受 | 平均 warm 请求秒 | 总调用 | 总 canvas |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| longbench_v2 | 149.165.151.254 | D_matched_legacy | 2/2 | 0 | N/A | N/A | N/A | 742 | 35 |
| longbench_v2 | 149.165.151.254 | D_matched_new_numeric | 2/2 | 0 | N/A | N/A | N/A | 518 | 26 |
| longbench_v2 | 149.165.151.254 | D_native | 2/2 | 0 | N/A | N/A | N/A | 564 | 30 |
| longbench_v2 | 149.165.159.64 | D_matched_legacy | 2/2 | 0 | N/A | N/A | N/A | 175 | 12 |
| longbench_v2 | 149.165.159.64 | D_matched_new_numeric | 2/2 | 0 | N/A | N/A | N/A | 248 | 17 |
| longbench_v2 | 149.165.159.64 | D_native | 2/2 | 0 | N/A | N/A | N/A | 196 | 14 |

请求的 CUDA 时间跨度从首次编码器 forward 结束至最终事件，包含主机间隙与后续工作，并非同步的纯解码墙钟。完整缺失/失败块与配对 bootstrap 描述性区间见 JSON。

此 12 次 CP1 自然诊断没有 warm 配对，质量及计时均不进入候选方法选择。
