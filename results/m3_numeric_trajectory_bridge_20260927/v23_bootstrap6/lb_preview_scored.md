# v21 冻结面板离线汇总（模板）

协议：`v21_bootstrap6_b8b2c59449fd0ebf`；类别：`bootstrap6`。已记录 48/240 次执行。

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
| longbench_v2 | 149.165.151.254 | B_native_bootstrap2_observe1 | 2/6 | 2 | 0 | 2 | 32.117 | 355 | 19 |
| longbench_v2 | 149.165.151.254 | D_native | 2/6 | 2 | 0 | 2 | 33.878 | 356 | 20 |
| longbench_v2 | 149.165.151.254 | M1_native_bootstrap2_observe1 | 2/6 | 2 | 0 | 2 | 45.938 | 485 | 25 |
| longbench_v2 | 149.165.151.254 | M3_R3_A8_incumbent | 2/6 | 2 | 0 | 2 | 56.720 | 638 | 33 |
| longbench_v2 | 149.165.151.254 | M3_native_bootstrap2_observe1 | 2/6 | 2 | 0 | 2 | 29.217 | 311 | 18 |
| longbench_v2 | 149.165.151.254 | T_scope | 2/6 | 2 | 0 | 2 | 38.490 | 414 | 23 |
| longbench_v2 | 149.165.159.64 | B_native_bootstrap2_observe1 | 2/6 | 2 | 0 | 2 | 44.180 | 521 | 27 |
| longbench_v2 | 149.165.159.64 | D_native | 2/6 | 2 | 0 | 2 | 36.462 | 404 | 24 |
| longbench_v2 | 149.165.159.64 | M1_native_bootstrap2_observe1 | 2/6 | 2 | 0 | 2 | 59.243 | 656 | 31 |
| longbench_v2 | 149.165.159.64 | M3_R3_A8_incumbent | 2/6 | 2 | 0 | 2 | 41.393 | 476 | 25 |
| longbench_v2 | 149.165.159.64 | M3_native_bootstrap2_observe1 | 2/6 | 2 | 0 | 2 | 42.970 | 495 | 26 |
| longbench_v2 | 149.165.159.64 | T_scope | 2/6 | 2 | 0 | 2 | 37.283 | 419 | 22 |

请求的 CUDA 时间跨度从首次编码器 forward 结束至最终事件，包含主机间隙与后续工作，并非同步的纯解码墙钟。完整缺失/失败块与配对 bootstrap 描述性区间见 JSON。
