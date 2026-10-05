# v21 冻结面板离线汇总（模板）

协议：`v21_bootstrap6_ruler_4dbecd0c10841a11`；类别：`bootstrap6_ruler`。已记录 90/90 次执行。

质量仅统计成功的首次生成；时间仅统计严格接受的 warm 请求。绝对时间按 GPU 主机分开；同 GPU 配对比值小于 1 表示请求耗时较少。

| 数据集 | 主机 | 方法 | 首次成功/计划 | 首次评分 | 严格正确 | warm 接受 | 平均 warm 请求秒 | 总调用 | 总 canvas |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ruler4k | 149.165.151.254 | B_native_bootstrap2_observe1 | 7/7 | 7 | 6 | 1 | 1.687 | 34 | 7 |
| ruler4k | 149.165.151.254 | D_native | 7/7 | 7 | 6 | 1 | 1.397 | 31 | 7 |
| ruler4k | 149.165.151.254 | M1_native_bootstrap2_observe1 | 7/7 | 7 | 6 | 1 | 1.997 | 37 | 7 |
| ruler4k | 149.165.151.254 | M3_R3_A8_incumbent | 7/7 | 7 | 6 | 1 | 1.414 | 27 | 7 |
| ruler4k | 149.165.151.254 | M3_native_bootstrap2_observe1 | 7/7 | 7 | 6 | 1 | 1.411 | 33 | 7 |
| ruler4k | 149.165.151.254 | T_scope | 7/7 | 7 | 6 | 1 | 1.557 | 29 | 7 |
| ruler4k | 149.165.159.64 | B_native_bootstrap2_observe1 | 6/6 | 6 | 6 | 1 | 0.756 | 19 | 6 |
| ruler4k | 149.165.159.64 | D_native | 6/6 | 6 | 6 | 1 | 0.745 | 18 | 6 |
| ruler4k | 149.165.159.64 | M1_native_bootstrap2_observe1 | 6/6 | 6 | 6 | 1 | 0.760 | 18 | 6 |
| ruler4k | 149.165.159.64 | M3_R3_A8_incumbent | 6/6 | 6 | 6 | 1 | 0.752 | 19 | 6 |
| ruler4k | 149.165.159.64 | M3_native_bootstrap2_observe1 | 6/6 | 6 | 6 | 1 | 0.757 | 19 | 6 |
| ruler4k | 149.165.159.64 | T_scope | 6/6 | 6 | 6 | 1 | 0.753 | 19 | 6 |

请求的 CUDA 时间跨度从首次编码器 forward 结束至最终事件，包含主机间隙与后续工作，并非同步的纯解码墙钟。完整缺失/失败块与配对 bootstrap 描述性区间见 JSON。
