# v21 冻结面板离线汇总（模板）

协议：`v21_bootstrap6_b8b2c59449fd0ebf`；类别：`bootstrap6`。已记录 96/240 次执行。

质量仅统计成功的首次生成；时间仅统计严格接受的 warm 请求。绝对时间按 GPU 主机分开；同 GPU 配对比值小于 1 表示请求耗时较少。

| 数据集 | 主机 | 方法 | 首次成功/计划 | 首次评分 | 严格正确 | warm 接受 | 平均 warm 请求秒 | 总调用 | 总 canvas |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| aime26 | 149.165.151.254 | B_native_bootstrap2_observe1 | 0/4 | 0 | 0 | 0 | N/A | 0 | 0 |
| aime26 | 149.165.151.254 | D_native | 0/4 | 0 | 0 | 0 | N/A | 0 | 0 |
| aime26 | 149.165.151.254 | M1_native_bootstrap2_observe1 | 0/4 | 0 | 0 | 0 | N/A | 0 | 0 |
| aime26 | 149.165.151.254 | M3_R3_A8_incumbent | 0/4 | 0 | 0 | 0 | N/A | 0 | 0 |
| aime26 | 149.165.151.254 | M3_native_bootstrap2_observe1 | 0/4 | 0 | 0 | 0 | N/A | 0 | 0 |
| aime26 | 149.165.151.254 | T_scope | 0/4 | 0 | 0 | 0 | N/A | 0 | 0 |
| aime26 | 149.165.159.64 | B_native_bootstrap2_observe1 | 0/4 | 0 | 0 | 0 | N/A | 0 | 0 |
| aime26 | 149.165.159.64 | D_native | 0/4 | 0 | 0 | 0 | N/A | 0 | 0 |
| aime26 | 149.165.159.64 | M1_native_bootstrap2_observe1 | 0/4 | 0 | 0 | 0 | N/A | 0 | 0 |
| aime26 | 149.165.159.64 | M3_R3_A8_incumbent | 0/4 | 0 | 0 | 0 | N/A | 0 | 0 |
| aime26 | 149.165.159.64 | M3_native_bootstrap2_observe1 | 0/4 | 0 | 0 | 0 | N/A | 0 | 0 |
| aime26 | 149.165.159.64 | T_scope | 0/4 | 0 | 0 | 0 | N/A | 0 | 0 |
| longbench_v2 | 149.165.151.254 | B_native_bootstrap2_observe1 | 4/6 | 4 | 2 | 4 | 38.811 | 885 | 50 |
| longbench_v2 | 149.165.151.254 | D_native | 4/6 | 4 | 3 | 4 | 43.611 | 922 | 52 |
| longbench_v2 | 149.165.151.254 | M1_native_bootstrap2_observe1 | 4/6 | 4 | 3 | 4 | 33.972 | 719 | 43 |
| longbench_v2 | 149.165.151.254 | M3_R3_A8_incumbent | 4/6 | 4 | 3 | 4 | 34.121 | 756 | 46 |
| longbench_v2 | 149.165.151.254 | M3_native_bootstrap2_observe1 | 4/6 | 4 | 3 | 4 | 37.634 | 833 | 50 |
| longbench_v2 | 149.165.151.254 | T_scope | 4/6 | 4 | 3 | 4 | 31.727 | 664 | 40 |
| longbench_v2 | 149.165.159.64 | B_native_bootstrap2_observe1 | 4/6 | 4 | 2 | 4 | 33.017 | 763 | 48 |
| longbench_v2 | 149.165.159.64 | D_native | 4/6 | 4 | 3 | 4 | 31.933 | 678 | 44 |
| longbench_v2 | 149.165.159.64 | M1_native_bootstrap2_observe1 | 4/6 | 4 | 2 | 4 | 34.769 | 763 | 48 |
| longbench_v2 | 149.165.159.64 | M3_R3_A8_incumbent | 4/6 | 4 | 2 | 4 | 31.961 | 734 | 41 |
| longbench_v2 | 149.165.159.64 | M3_native_bootstrap2_observe1 | 4/6 | 4 | 3 | 4 | 34.145 | 776 | 49 |
| longbench_v2 | 149.165.159.64 | T_scope | 4/6 | 4 | 3 | 4 | 33.263 | 726 | 43 |

请求的 CUDA 时间跨度从首次编码器 forward 结束至最终事件，包含主机间隙与后续工作，并非同步的纯解码墙钟。完整缺失/失败块与配对 bootstrap 描述性区间见 JSON。
