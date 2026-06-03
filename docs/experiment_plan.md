# Diffusion LM / DLM 动态 block sampling 实验设计

## 0. 这次实验到底要体现什么

你老师说的重点可以理解成：DLM 不是传统 AR LM 一个 token 一个 token 地生成，而是一个 block 里多个 token/位置一起 denoise；但是同一个 block 里，不同 token/位置达到收敛、被接受、或者不再需要更新的 step 数量不一定一样。

所以实验要体现两层“不均匀”：

1. **block 内部不均匀**：同一个 request 的同一个 block 里，有些 token 2 步就够，有些 token 8 步才够。
2. **batch 内部不均匀**：同一个 serving batch 里，不同 request 的 block 难度不同，某些 request 会成为 straggler。

如果实现是同步的，大家通常会被最难的 token / request 拖住；如果 dynamic sampling / dynamic scheduling 做得好，就能跳过已经完成的 token 或 request，从而减少 wasted computation。

## 1. 实验 1：`x = batch size 2/4/8/16`，`y = 速度差异`

### 推荐设置

- 固定：模型、prompt 集、最大生成长度、block size，例如先用 `block_size = 32`。
- 横轴：`batch size = 2, 4, 8, 16`。
- 纵轴：不要只写“速度差异”，建议画下面至少一个指标：
  - `waste_ratio = executed_token_steps / useful_token_steps`。
  - `straggler_ratio = max(request_latency) / mean(request_latency)`。
  - `p95_latency_ms - p50_latency_ms`。

### 为什么 batch size 越大差异越明显

batch 越大，越容易混进一个“很难”的 request 或 block。同步实现里，一个 batch 往往要等最慢的 request/block；所以 easy request 即使早就够了，也会继续陪跑，`waste_ratio` 和 latency gap 会变大。

## 2. 实验 2：`x = number decoding step needed`，`y` 应该是什么

这里有两个画法，取决于老师到底想看“8 个 request 分别多少 step”，还是想看“step 数量不同带来的性能差距”。

### 画法 A：如果老师说“8 个 request 分别多少 decoding step”

最清楚的图是：

- 横轴：`request_id = req0 ... req7`。
- 纵轴：`block_finish_steps = max(token_steps_in_this_block)`。
- 每个柱子或点表示一个 request 的当前 block 需要多少 denoising/refinement step 才完成。

这张图主要证明：一个 batch 里的 8 个 request 难度不同。

### 画法 B：如果横轴必须是 `number decoding step needed`

那纵轴就应该放“性能后果”：

- 横轴：`block_finish_steps`，也就是一个 request 的这个 block 中最难 token 需要的 step 数。
- 纵轴：推荐用 `latency_ms`，也可以用：
  - `normalized_slowdown = observed_latency / ideal_latency_if_served_alone`。
  - `wasted_token_steps = executed_token_steps - useful_token_steps`。
  - `block_internal_waste_ratio = block_size * block_finish_steps / useful_token_steps`。

最终推荐图：scatter plot，`x = block_finish_steps`，`y = latency_ms`，颜色区分 `sync` / `dynamic`，也可以用 marker 区分 block size。

## 3. 实验 3：block size 16/32/64

### 推荐设置

- 横轴：`block size = 16, 32, 64`。
- 纵轴：`mean_latency_ms`、`throughput_tokens_per_s` 或 `waste_ratio`。
- 曲线：`sync` vs `dynamic`。

### 预期解释

不要预设结论，真实结果要靠 H100 测出来。但一般可以这样分析：

- block size 大：一次 forward 能并行更多 token，但 block 内 token step 差异可能更大，容易浪费。
- block size 小：每个 block 内差异可能小一些，但 block 切换次数更多，调度/启动开销更明显。
- dynamic sampling 如果能跳过完成 token 或完成 request，应该能降低 `executed_token_steps / useful_token_steps`。

## 4. H100 真实实验怎么 log

每个 measured batch 建议至少跑 30 次，正式计时前 warm up 10 个 batch。每条 per-request/per-block row 记录：

```text
batch_size,block_size,scheduler,trial,request_id,block_index,
steps_needed,steps_executed,useful_token_steps,executed_token_steps,
token_step_min,token_step_mean,token_step_max,block_internal_waste_ratio,
latency_ms,tokens_generated
```

字段含义：

- `steps_needed`：这个 request 的这个 block 完成需要多少 step，通常等于 block 内 token step 的最大值。
- `steps_executed`：调度器实际让这个 request/block 跑了多少 step；同步 batch 下可能等于 batch 内最大 step。
- `useful_token_steps`：block 内所有 token 真正需要的 step 总和，例如 `sum(token_steps)`。
- `executed_token_steps`：实际执行的 token-step 数量；同步实现约等于 `steps_executed * block_size`。
- `block_internal_waste_ratio`：只看一个 block 内部，因为 token step 不同导致的浪费。
- `latency_ms`：该 request/block 的端到端耗时或模型 forward 耗时；两种都可以，但要在论文/图注里写清楚。

## 5. 代码怎么接真实 DLM

现在的脚本默认 `--mode simulate`，里面会模拟“一个 block 里 token 的 step 数量不一样”。真实跑 H100 时，你只需要把 decoder 的日志整理成上面的 CSV，然后执行：

```bash
python3 scripts/dlm_block_sampling_benchmark.py --mode plot-csv --input-csv your_real_run.csv --outdir outputs/h100_run
```

如果你的 DLM stopping criterion 是 confidence threshold、mask count 变成 0、token accepted，或者 residual 小于阈值，都可以；关键是统一定义 `token_steps` 和 `steps_needed`，并且 sync / dynamic 两种 scheduler 用同一个定义。


## 6. 怎么模拟“一个 block 内 step 不一样”

最推荐的模拟方法不是随便给 request 一个难度标签，而是给 **block 内每个 token/位置** 一个自己的 `token_step`：

```text
token_steps = [2, 2, 3, 1, 8, 4, 2, ...]
steps_needed = max(token_steps)
useful_token_steps = sum(token_steps)
sync_executed_token_steps = block_size * max(token_steps)
```

这样才能表达 DLM 的关键现象：同一个 block 里有的 token 很早就确定了，有的 token 要多次 refinement 才稳定。当前 `scripts/dlm_block_sampling_benchmark.py` 的 `sample_token_steps()` 就是这样做的：每个 token 单独采样 step，然后用 `max(token_steps)` 作为这个 block 的完成 step。

如果要让 batch 里的差异更明显，需要构造不同复杂程度的 requests。建议 prompt 集混合：

- 简单翻译 / 改写：通常 token confidence 较快变高。
- 算术、数学证明、逻辑推理：中间 token 更容易反复变化。
- 代码 / SQL：局部语法 token 可能早确定，但变量名、条件、缩进附近可能更慢。
- 长摘要 / 长回答：后面位置更不确定，block 内 variance 往往更大。

## 7. H100 上用什么模型

建议顺序：

1. **LLaDA-8B-Instruct**：优先。它是 masked diffusion LM，官方 repo 提供 `generate()` / `chat.py`，单张 H100 跑 8B 级别 probe 比较现实。适合先把 `token_steps`、`steps_needed`、`waste_ratio` 测清楚。
2. **BD3-LM / Block Diffusion**：如果你要和 block diffusion 论文概念完全对齐，再用它。它明确把序列分成 blocks，并在 block 内做 discrete diffusion；但工程上可能不如 LLaDA 直接。
3. **DPad / Fast-dLLM 类实现**：如果重点是 dynamic sampling / suffix pruning 的加速效果，可以作为优化 baseline 或参考实现。

## 8. H100 真实 probe 代码逻辑

`llada_block_step_probe.py` 的逻辑是：

1. 对每个 prompt，在末尾追加 `block_size` 个 `[MASK]`。
2. 每一步 forward 后，看 block 内每个 masked token 的最大 softmax probability。
3. 如果某个 token 的 confidence 超过阈值，就认为它在当前 step 完成，记录 `token_step = step`。
4. 如果某一步没有任何 token 超过阈值，就强制接受当前 confidence 最高的一个 token，避免死循环。
5. 一个 request 的 `steps_needed = max(token_steps)`；block 内浪费看 `block_size * max(token_steps) / sum(token_steps)`。
6. 输出 CSV 后，再用 `dlm_block_sampling_benchmark.py --mode plot-csv` 画图。

注意：当前 H100 主实验默认不再输出 `dynamic_oracle`。如果以后要研究理论上界，可以单独做离线分析；主结果只看真实 `real_llada` 行。


## 9. 当前代码到底是按 prompt 还是写死 step

当前模拟代码 **不是写死每个 block 需要多少 step**。流程是：

1. 从 `data/prompts_heterogeneous.jsonl` 读取 prompt 和 `difficulty`。
2. 每个 batch 按 `easy -> medium -> hard -> extreme` 混合 request，保证 batch 内难度差距大。
3. `sample_token_steps(block_size, prompt, rng)` 根据 prompt difficulty 采样每个 token 的 step：
   - `easy`：scale 小、方差小，token 通常很快完成。
   - `medium`：中等 step。
   - `hard`：scale 大、方差大，部分 token 会拖尾。
   - `extreme`：scale 最大、方差最大，最容易成为 straggler。
4. 一个 block 的 `steps_needed = max(token_steps)`，不是手动指定。
5. 同步调度 `sync` 用 batch 内最大 step；`dynamic` 按还没完成的 token 计算 active token 数。

所以实验里“复杂程度不同的 req”已经加进去了。你可以直接改 prompt 文件里的 `difficulty` 或 prompt 文本来放大/缩小差距。

## 10. 完整运行命令

最简单：

```bash
bash scripts/run_dlm_experiment.sh
```

这个 bash 会跑：

```bash
python3 scripts/dlm_block_sampling_benchmark.py \
  --mode simulate \
  --prompt-file data/prompts_heterogeneous.jsonl \
  --batch-sizes 2 4 8 16 \
  --block-sizes 16 32 64 \
  --schedulers sync dynamic \
  --trials 50 \
  --outdir outputs/prompt_difficulty_demo
```

输出：

- `outputs/prompt_difficulty_demo/per_request_rows.csv`
- `outputs/prompt_difficulty_demo/summary.csv`
- `outputs/prompt_difficulty_demo/difficulty_summary.csv`
- `exp1_batch_speed_gap.png`
- `exp2_steps_vs_perf.png`
- `exp3_block_size_latency.png`
- `exp4_prompt_difficulty_steps.png`

如果要 H100 真实模型 probe，再跑 README 里的 `scripts/llada_block_step_probe.py` 命令。


## 11. A100 和 H100 二选一

建议：**最终实验选 H100**。理由：

- 你关心的是 serving batch size 2/4/8/16、block size 16/32/64、dynamic sampling 的速度差异；这些设置会放大显存带宽、矩阵计算和调度 overhead 的影响。
- H100 对 BF16/FP8/Transformer Engine 的支持和吞吐更强，更容易把 `bs=16 + block_size=64` 跑稳。
- A100 适合做代码 smoke test 和小规模 sanity check；如果用 A100 出图，最好把结论写成 “A100 上的趋势”，不要直接说 H100 serving。

推荐流程：

1. A100：先跑 `--trials 2 --batch-sizes 2 4 --block-sizes 16 32`，检查 CSV 字段、图、显存。
2. H100：跑完整 sweep：`bs=2/4/8/16`、`block_size=16/32/64`、`trials>=30`。
3. 论文/汇报图：优先用 H100 结果；A100 结果可以放 appendix 或作为硬件对比。

## 12. “根据 prompt 难度模拟 step” 和真实 DLM confidence 是什么关系

这里要分清两条线：

### 12.1 模拟实验线

`scripts/dlm_block_sampling_benchmark.py --mode simulate` **没有加载真实 DLM**，所以它不会真的看模型 confidence。它做的是可控模拟：

- prompt 文件里写 `difficulty=easy/medium/hard/extreme`。
- `sample_token_steps()` 根据 difficulty 的 scale/sigma，为 block 内每个 token 采样一个完成 step。
- 这样可以人为制造明显差距，快速验证图、指标、batch straggler 逻辑是否合理。

这条线适合：先定实验图怎么画、CSV 怎么存、指标怎么算。

### 12.2 真实 DLM probe 线

`scripts/llada_block_step_probe.py` 才是用现成 masked diffusion LM 的 probe。它会：

1. 加载 LLaDA 这类 masked diffusion LM。
2. 在 prompt 后面追加一个 `[MASK]` block。
3. 每个 denoising step forward 一次。
4. 对 block 内每个 masked token 取 softmax 最大概率作为 confidence。
5. confidence 超过阈值的 token 认为完成，记录这个 token 的 `token_step`。
6. 一个 block 的 `steps_needed=max(token_steps)`。

所以：**真实 DLM 版本会根据 confidence 自动决定每个 token/block 的 step 个数；模拟版本只是用 prompt difficulty 近似生成 token_steps。**

### 12.3 最建议你实际采用的实验组合

- 先跑模拟：确认图和指标。
- 再在 H100 上跑 LLaDA probe：拿真实 confidence-based `token_steps`。
- 最终汇报时说清楚：模拟用于设计实验，H100 LLaDA probe 用于真实 DLM 测量。


## 13. 能不能记录每个 prompt、每个 block 需要多少 step

可以，而且建议记录。现在脚本会额外输出：

```text
outputs/.../prompt_block_steps.csv
```

这张表比完整 `per_request_rows.csv` 更好读，核心字段是：

```text
trial,batch_size,block_size,scheduler,request_id,prompt_id,prompt_difficulty,
block_index,steps_needed,steps_executed,token_step_min,token_step_mean,
token_step_max,useful_token_steps,executed_token_steps,latency_ms
```

怎么看：

- `prompt_id/request_id`：是哪条 prompt / request。
- `block_index`：第几个 block；当前默认 probe 一个 block，所以是 `0`。
- `steps_needed`：这个 prompt 的这个 block 需要多少 step，通常是 block 内最难 token 的 `max(token_steps)`。
- `token_step_min/mean/max`：这个 block 内 token step 分布，能看出同一个 block 内 step 是否不均。
- `steps_executed`：调度器实际执行了多少 step；sync 下可能被 batch 内 hardest request 拉高。

如果后面实现真正多 block 生成，只要每生成一个 block append 一行，把 `block_index=0,1,2,...` 填进去即可，画图和 summary 都可以继续复用。


## 14. 真实模型统一 H100 跑法

老师要的真实模型版本，直接跑：

```bash
python3 -m pip install -r requirements-h100.txt
bash scripts/run_h100_llada_experiment.sh
```

默认配置：

- 模型：`GSAI-ML/LLaDA-8B-Instruct`。
- batch size：`2 4 8 16`。
- block size：`16 32 64`。
- `NUM_BLOCKS=4`，也就是每个 prompt 连续生成 4 个 block。
- `MAX_STEPS=64`，每个 block 最多 denoise 64 step。
- `CONFIDENCE_THRESHOLD=0.90`，token confidence 到阈值就认为完成。

关键点：这不是模拟 difficulty。`llada_block_step_probe.py` 会真的跑模型：

1. 对原始 prompt tokenize。
2. append 第 0 个 `[MASK] * block_size`。
3. 每个 denoising step forward 一次，用 softmax 最大概率当 token confidence。
4. confidence 过阈值的 token 写回 `input_ids`，并记录这个 token 在第几步完成。
5. 第 0 个 block 完成后，append 第 1 个 `[MASK] * block_size`；此时上下文已经包含第 0 个 block 的生成结果。
6. 重复到 `NUM_BLOCKS-1`。

所以每行 `prompt_block_steps.csv` 的 `steps_needed` 来自：

```text
当前 prompt + 前面已经生成的 blocks + 当前 block 内 token confidence
```

如果要先小跑确认显存：

```bash
BATCH_SIZES="2" BLOCK_SIZES="16" TRIALS=1 NUM_BLOCKS=2 bash scripts/run_h100_llada_experiment.sh
```

如果 H100 显存足够，再跑默认完整 sweep。


## 15. LLaDA 加载时报 `all_tied_weights_keys` 怎么办

如果报错：

```text
AttributeError: 'LLaDAModelLM' object has no attribute 'all_tied_weights_keys'
```

原因通常是较新的 `transformers` 会访问 remote-code 模型类没有实现的 `all_tied_weights_keys` 属性；`device_map=auto` 会触发，某些版本即使不用 `auto` 也会在 finalize loading 时触发。脚本现在会在 `from_pretrained()` 期间临时补一个兼容属性，把 LLaDA 的 `_tied_weights_keys` 转成 transformers 期望的 dict-like `all_tied_weights_keys`。

当前脚本默认就是单卡加载，并带兼容补丁：

```bash
DEVICE_MAP=none bash scripts/run_h100_llada_experiment.sh
```

如果你明确要多 GPU shard，才尝试：

```bash
DEVICE_MAP=auto bash scripts/run_h100_llada_experiment.sh
```

但 LLaDA 这个 remote model class 在部分 transformers 版本下可能不支持 `auto`。


## 16. LLaDA 加载时报 `tie_weights() got an unexpected keyword argument` 怎么办

如果报错：

```text
TypeError: LLaDAModelLM.tie_weights() got an unexpected keyword argument 'missing_keys'
```

原因也是新版 `transformers` 和 LLaDA remote-code 模型类接口不完全一致：新版 transformers 会调用 `tie_weights(missing_keys=..., recompute_mapping=False)`，但 LLaDA 的 `tie_weights()` 可能不接受这些 keyword。

当前 `llada_block_step_probe.py` 已经在 `from_pretrained()` 外层加了兼容 patch：如果 remote model 的 `tie_weights` 不接受这些新版 keyword，会自动退回调用老接口 `tie_weights()`。所以更新代码后直接重新跑：

```bash
bash scripts/run_h100_llada_experiment.sh
```


## 17. tokenizer 没有 mask token 怎么办

如果报错：

```text
ValueError: tokenizer 没有 mask token
```

LLaDA 的 Hugging Face tokenizer 可能没有把 mask token 注册成 `tokenizer.mask_token_id`，但官方推理代码通常使用固定 mask id `126336`。当前脚本已经默认 fallback 到 `126336`，H100 runner 也显式设置：

```bash
MASK_TOKEN_ID=126336 bash scripts/run_h100_llada_experiment.sh
```

如果你换了别的 diffusion LM，需要用对应模型的 mask token id 覆盖：

```bash
MASK_TOKEN_ID=<your_mask_id> bash scripts/run_h100_llada_experiment.sh
```


## 18. 确认真实模型实验不是固定 steps / block

`llada_block_step_probe.py` 没有采用官方 LLaDA 那种把 total `steps` 平均分给每个 block 的固定预算逻辑。真实 H100 probe 的逻辑是：

1. `NUM_BLOCKS=4` 时，每个 request 会顺序 probe 4 个 block。
2. 第 `k` 个 block 是在 prompt + 已完成的 `0..k-1` blocks 后面 append `[MASK] * block_size`。
3. 当前 block 每一步 forward 后，计算每个 masked token 的 confidence。
4. confidence 达阈值的 token 被 unmask；如果没有 token 达阈值，会接受当前 confidence 最高的一个 token，避免无限等待。
5. 当前 request 的当前 block 的 `steps_used=max(token_steps)`，因此不同 prompt、不同 block 可以有不同 `steps_used`。
6. 最多只受 `MAX_STEPS` 截断，不会把 total steps 均分到 blocks。

真实模型会额外输出紧凑表：

```text
outputs/h100_llada/block_steps.csv
```

字段正是老师要看的 block-level 口径：

```text
request_id,prompt_id,difficulty,trial,batch_size,block_index,block_size,
steps_used,latency_ms,mean_confidence,min_confidence,num_tokens
```

`per_request_rows.csv` 现在用于保存真实 `real_llada` 行并画 latency/step 图；`block_steps.csv` 则是一行一个 prompt/request/block，更适合直接汇报“每个 block 用了多少 denoising steps”。


## 19. 输出目录按时间戳保存不同版本

两个统一入口现在都会默认使用 UTC 时间戳目录，避免覆盖之前结果：

```bash
bash scripts/run_dlm_experiment.sh
# -> outputs/prompt_difficulty_demo/YYYYMMDD_HHMMSS

bash scripts/run_h100_llada_experiment.sh
# -> outputs/h100_llada/YYYYMMDD_HHMMSS
```

如果你想固定某次实验名，可以手动传：

```bash
RUN_ID=bs_sweep_v1 bash scripts/run_h100_llada_experiment.sh
```

如果想完全指定目录，也可以传：

```bash
OUTDIR=outputs/my_manual_run bash scripts/run_h100_llada_experiment.sh
```

## 12. 当前代码的主实验路径：真实 batched LLaDA，不用离线模拟 step

现在最终 H100 实验统一跑 `scripts/run_h100_llada_experiment.sh`。这个入口会把 `BATCH_SIZE` / `BATCH_SIZES` 传给 `scripts/llada_block_step_probe.py`，然后真实地按 batch tokenize prompts、append mask block、调用 LLaDA forward，并用 LLaDA 官方 top-k/low-confidence 规则决定每一步接受哪些 token。模拟脚本 `scripts/run_dlm_experiment.sh` 只用于检查 CSV/plot pipeline，不作为最终实验数据。

推荐命令：

```bash
BATCH_SIZES=1,2,4,8,16 \
BLOCK_SIZES=16,32,64 \
NUM_BLOCKS=4 \
MAX_STEPS_PER_BLOCK=64 \
ACCEPTANCE_POLICY=topk \
CONFIDENCE_THRESHOLD=0.95 \
MASK_TOKEN_ID=126336 \
DEVICE_MAP=none \
bash scripts/run_h100_llada_experiment.sh
```

真实 probe 的循环是：

1. 每个 batch 一次 tokenize + padding。
2. 对 `block_index = 0..NUM_BLOCKS-1`：只 append 当前 block 的 mask token。
3. 在当前 block 内迭代真实 LLaDA forward；每一步只更新当前 block 里还没 finalized 的 token。
4. 某个 request 当前 block 先完成时，记录它自己的 `steps_used`；整个 batch 继续跑到所有 request 当前 block 都完成或达到 `MAX_STEPS_PER_BLOCK`。
5. 当前 block 结束后，再 append 下一个 block，因此后续 block 的 step 会依赖 prompt 和前面真实生成出来的内容。

真实输出：

- `block_steps.csv`：一行一个 request/block，字段包括 `run_id`, `batch_size`, `batch_id`, `request_id`, `prompt_id`, `difficulty`, `block_index`, `steps_used`, `latency_ms`, `mean_confidence`, `min_confidence`。
- `batch_block_latency.csv`：一行一个 batch/block，字段包括 `batch_block_steps`, `max_request_steps_used`, `min_request_steps_used`, `mean_request_steps_used`, `latency_ms`, `waste_token_steps`。
- `per_request_rows.csv`：用于复用画图脚本；H100 主路径只包含真实 `real_llada` 行，不再默认混入未真实执行的 dynamic/oracle 曲线。

## 13. LLaDA confidence / acceptance 规则修正

H100 主实验默认 `ACCEPTANCE_POLICY=topk`，尽量贴近官方 LLaDA `generate.py` 的采样方式：每一步先对当前 block 中仍为 mask 的位置计算预测 token 的 confidence，然后按线性 schedule 决定本 step 要 unmask 多少个 token，并选择 confidence 最高的 top-k 位置写回。也就是说，官方流程不是“所有超过固定 threshold 的 token 都接受”，而是“每步必须接受 schedule 指定数量的最高置信 token；其他位置继续保持 mask 到后续 step”。

`CONFIDENCE_THRESHOLD` 仍保留给 `ACCEPTANCE_POLICY=threshold` 这个实验模式，但 H100 runner 默认不会使用它。最终图默认只画真实 `real_llada` 行，不再默认混入未真实执行的 dynamic/oracle 曲线。
