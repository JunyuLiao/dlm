# Plot guide

数据来源：`real LLaDA H100 data`。H100 真实实验只消费 LLaDA real probe 写出的 CSV；模拟模式只用于 smoke/debug。

- `real LLaDA confidence-cutoff batch`: H100 主实验结果；每一步按 confidence 从高到低接受达到阈值的前缀 token，所以不同 block/request 可以用不同步数。
- `real LLaDA fixed top-k batch`: 官方固定步数 top-k unmasking 对照；如果 `max_steps_per_block > block_size`，`steps_used` 很可能等于 `block_size`，不适合作为动态步数主图。
- `simulated sync batch` / `simulated dynamic token skip`: 只会出现在 simulator 输出里，用于 debug 图表，不作为 H100 真实实验结论。

## 输出文件

- `per_request_rows.csv`: 完整分析表；H100 真实 runner 中只包含真实 `real_llada_*` 行，用于画真实 latency/step 图。
- `prompt_block_steps.csv`: 从完整表整理出的 per request/block step 表。
- H100 真实模型 runner 还会写 `../block_steps.csv`: 一行一个 prompt/request/block，字段包括 `steps_used`, `mean_confidence`, `min_confidence`。

## 关键指标定义

- `token_step`: 一个 token 直到被接受为止经历了多少次 denoising forward；例如第 5 步才被接受，就是 5 token-steps。
- `useful_token_steps = sum(token_steps)`: 一个 request/block 内所有 token 的完成步数求和。这个求和是在统计每个 token 实际经历过的 forward 次数，不是说这些步骤按 token 顺序串行执行。
- `executed_token_steps = batch_finish_steps * block_size`: 同步 batch 口径下，这个 request/block 被 dense forward 陪跑的 token-step 数；`batch_finish_steps` 是同一 batch 当前 block 里最慢 request 的步数。
- `waste ratio = sum(executed_token_steps) / sum(useful_token_steps)`: 衡量同步 batch / block 内 step 不均带来的陪跑浪费；它不是低 confidence token 被丢弃的比例。

## 图怎么读

1. `exp1_batch_speed_gap.png`: batch size 越大，sync 的 waste ratio 通常越高，说明容易被 hardest request 拖住；真实 LLaDA 曲线展示实际 batch/block 的同步 cost。
2. `exp2_steps_vs_perf.png`: 只画真实/动态行，横轴是这个 block 实际用了多少 denoising steps，纵轴是 latency。
3. `exp3_block_size_latency.png`: block size 16/32/64 对 mean latency 的影响。
4. `exp4_prompt_difficulty_steps.png`: 不同 difficulty 的 prompt 平均实际 block steps；真实模型下规律不一定单调，要以 `block_steps.csv` 为准。

当前默认 NUM_BLOCKS=1；每个 request 会有多个 `block_index`。
