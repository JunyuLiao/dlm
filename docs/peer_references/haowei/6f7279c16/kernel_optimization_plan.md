> Peer reference snapshot, author/source: Haowei, `chw/value_aware` at `6f7279c1625f7fa53bacb233b96fb69a385df8c1`.
> Original: https://github.com/coconight01/dlm_test/blob/6f7279c1625f7fa53bacb233b96fb69a385df8c1/kernel_optimization_plan.md
> Relative document links point to the pinned original; private absolute paths are redacted.
> Peer claims and historical instructions are quoted reference material, not our verified results, contribution, or current execution protocol.

b

# Skip-Attention CUDA 优化大纲

目标卡:H100 (sm_90 / sm_90a)。当前基线:[blasst_minimal.cu](https://github.com/coconight01/dlm_test/blob/6f7279c1625f7fa53bacb233b96fb69a385df8c1/blasst_minimal.cu)(正确性骨架,scalar 实现)。

三件事的依赖关系:

```
Task 1 (q_ids 映射)  ──┬──►  Task 2 (regroup + FFN 并行 + 端到端对比)
                       └──►  Task 3 (kernel 内部 HBM 优化 + 与 FA 单独对比)
                                   │
                                   └──►  回填 Task 2 的最终端到端数字
```

- Task 1 是 Task 2/3 的前置(真实 regroup 布局要求 q_ids 间接寻址)。
- Task 2 先用**朴素 kernel**跑通 pipeline + 拿到第一个端到端数字(预期比 FA 慢,只验证机制)。
- Task 3 做重活(把 kernel 往 FA2/FA3 方向重写),完成后**回填** Task 2 的端到端对比,才有最终结论。

---

## Task 1:q_ids 映射(小,前置)

**目标**:regroup 会把 Q token 打散,group 里的 Q 不再连续。让 kernel 支持"每个 group 引用一组 Q token 下标"。

**改动点**:

1. 数据布局新增 `q_ids`(与 `block_ids` 同款 CSR):`q_ids[gid * max_q + i]` = group g 的第 i 个 Q token 的全局下标。
   - regroup 强制每组 token 数相等 = `GQ`,所以 `max_q = GQ` 固定,`q_count` 可省;但建议先带上 `q_count` 防 regroup 实现留出不等长的余地。
2. kernel 里把 `q_base = gid * GQ` 改成间接寻址:
   ```cuda
   // load Q tile
   int i = idx / D_, d = idx % D_;
   int qrow = q_ids[gid * max_q + i];
   Qs[i][d] = Q[qrow * D_ + d];
   ```

   其余(online softmax、block 遍历)不变。
3. 出口写回同样用 `q_ids`: `O[q_ids[gid*max_q + i] * D_ + d] = ...`。

**验证**:main() 里生成一个乱序 permutation 作为 q_ids,参考实现改成同样按 q_ids 寻址,比对 `PASS`。

**交付**:kernel 支持任意 token→group 映射;`blasst_minimal.cu` 回归仍 `PASS`。

---

## Task 2:regroup(爬山法)+ FFN 并行 + 端到端对比

### 2.0 核心设计决策(先定这个,再动代码)

**决策 A:block_max 从哪来?**

- **A1(推荐,BLASST 论文路线):预测 block_max** —— 用上一步去噪的 block_max 预测当前步(diffusion 相邻步注意力图几乎不变,EMA 或直接复用即可)。这样"skip 集在整个 forward 开始前已知",regroup 可以完全提前跑,与 FFN 并行天然成立。
- A2:在线算当前步 block_max → regroup 只能等 Q/K 就绪后再跑,重叠窗口变小,且"先算 block_max 再跳过"本身就损失了一部分收益。

**决策 B:regroup 跑在哪?**

- **B1(推荐先做):CPU 侧,独立 stream** —— 爬山法是迭代/串行的(swap + 多 restart),天然不适合并行成 GPU kernel;而推理时 CPU 基本空闲。用 pinned memory + 非阻塞拷贝 + event 同步,把 CPU 侧的 regroup 藏在 GPU 的 FFN 时间里。
- B2(进阶):把 regroup 下沉成 CUDA kernel 跑在独立 stream。仅当 B1 测出 CPU 成为瓶颈才做。

**决策 C:爬山法太慢怎么办?**

- 爬山法 `_regroup_swap_layer` 是 O(restarts × iters × T × K × n_blocks),离线可跑,但运行时可能 > FFN 时间。
- 先**实测**爬山法 wall-time vs FFN wall-time:
  - 爬山法 ≤ FFN → 直接用它(重叠后免费)。
  - 爬山法 > FFN → 运行时退回 `_regroup_greedy_layer`(O(T·K·n_blocks)),爬山法结果作为质量上界/离线分析用。
- 关键前提:**n_blocks 很小**(canvas/block_size,如 8 或 4),group 数 K 也不大,所以即便爬山法,单层开销也可能可控。这是"能不能和 FFN 并行"的决定性数字,先量它。

### 2.1 把 regroup 接进 pipeline

1. 从 `analyze_sparsity.py` 复用 `_regroup_swap_layer` / `_regroup_greedy_layer`(离线 NumPy 版),先做**功能正确性**验证:输入预测 block_max → 输出 group→(q_ids, block_ids) CSR。
2. 做一个**运行时版本**的 regroup(先 C++/Python + 独立 stream,见决策 B1),产出 CSR 后通过非阻塞 copy 送进 GPU。
3. 双 stream + event 编排:
   ```
   Stream A(主模型): ... FFN[L-1] → attention[L](等待 E_L) → FFN[L] ...
   Stream B(regroup): regroup[L] → record E_L (供 Stream A 的 attention[L] 等待)
   ```

   Stream B 提前一层,attention 的等待 ≈ 0。

### 2.2 集成点(替换 flash attention)

- **IDLM**:`IDLM/models/dit.py` 的两处 attention —— `flash_attn_varlen_qkvpacked_func`(line 288, causal 分支)和 `regular_attention_multi_headed` 里的 `F.scaled_dot_product_attention`(line 137)。
- **DiffusionGemma(真正目标)**:定位 gemma 源码里对应的 attention 调用点(在 `gemma/` 下),同样替换。
- 落地方式:把 CUDA kernel 包成 **torch extension**(`torch.utils.cpp_extension`),暴露 `skip_attention(q, k, v, block_ids, q_ids, ...)`,在 dit.py / gemma 里一行替换原 flash_attn 调用。
- 待确认:端到端基准主跑 DiffusionGemma(生产模型),IDLM 作为小模型快速试验台。

### 2.3 端到端基准

- 同一份模型权重 + 同一输入,跑两遍:(i) 原 flash attention;(ii) skip attention(此刻先用朴素 kernel)。
- 用 `cuda.Event` 测 wall-clock;输出质量**不计入指标**(用户明确)。
- 仍然跑一次数值正确性 spot-check(对比参考),只为确保 pipeline 没产生垃圾输出,不做质量优化。
- 预期:第一版数字**比 FA 慢**(朴素 kernel 没 tensor core)。这是预期内结果,验证的是机制与重叠,不是速度。

**验收**:pipeline 端到端跑通;regroup 时间被 FFN 隐藏(可从 ncu 或时间线确认);存在一个端到端 ms 数字(哪怕慢,作为后续 Task 3 的对照基线)。

---

## Task 3:kernel 内部 HBM 优化 + 与 FA 单独对比

### 3.0 立场(避免走弯路)

- **稠密(sparsity≈0)时 FA3 在 H100 上基本必胜** —— 它是 SOTA 稠密 kernel(WGMMA + warp spec + TMA)。skip-attention 有额外的 CSR 间接寻址 + block_max 开销,做同样的活一定更慢。
- **skip-attention 唯一能赢的地方:sparsity 足够高**,因为它是按"保留块数"付账,FA 是按 O(N²d) 付账。
- 所以 Task 3 的目标不是"全面打赢 FA",而是:**(a) 把每个保留块的单位成本逼近 FA,(b) 测出 crossover 稀疏度**,然后看 diffusion 实际 sparsity(他们数据里 union 常 ~2.5 块,sparsity 很高)落在 crossover 哪一侧。

### 3.1 优化子步骤(按影响排序,每步都跑正确性回归)

1. **Tensor core(WGMMA,sm_90a)**:把 scalar 点积换 tensor-core matmul。这是当前 kernel 到 FA 之间最大的鸿沟,单此一步可能差 ~100×。→ 编译目标从 `sm_90` 升 `sm_90a`。
2. **cp.async → TMA + 双/三缓冲**:K/V tile 异步加载,盖住访存延迟。
3. **Warp specialization**:producer warp(TMA 取数)/ consumer warp(WGMMA 算)。
4. **共享内存布局防 bank conflict**:D=256 时走 swizzled layout(对应之前记的 SBH3D 思路),消 bank conflict。
5. **向量化 load(float4)+ 访存合并**:CSR 间接寻址要保证 K/V 段内的访存是连续的、128-bit 对齐的。
6. **Occupancy / 寄存器调优**:按 H100 的 SM 资源(228KB shared、65536 regs/SM)定 warp 数。

### 3.2 独立微基准(与 FA 单独对比)

- 固定一个 group 的 Q tile + 它的 block 列表,分别测 skip kernel 与 `flash_attn` 库在同一 shape 下的延迟。
- **扫 sparsity**:kept_fraction ∈ {1.0, 0.75, 0.5, 0.25, 0.125, 0.05},或按真实 DiffusionGemma block_max 分布 + 不同阈值 λ 扫。
- 指标:cuda.Event 延迟;用 `ncu`(Nsight Compute)看 **achieved HBM BW / TFLOPs / 内存受限 vs 计算受限**,确认"是否满载"。
- 产出:**crossover 曲线**(sparsity → skip/FA 延迟比),标出 skip 开始反超的 kept_fraction。

### 3.3 回填

- 用优化后的 kernel 重跑 Task 2.3 的端到端对比,得到最终结论:修改后 vs 修改前的 forward 快慢。

**验收**:在 diffusion 真实 sparsity 下,skip kernel 单层延迟 ≤ FA 的 X%(X 由实测 crossover 定);`ncu` 报告 HBM 利用率接近 FA 同 shape 的水平。

---

## 里程碑汇总

| #  | 里程碑                     | 交付          | 通过标准                |
| -- | -------------------------- | ------------- | ----------------------- |
| M1 | q_ids 映射                 | Task 1        | 乱序 q_ids 下`PASS`   |
| M2 | regroup 接 pipeline + 重叠 | Task 2.1      | regroup 被 FFN 隐藏     |
| M3 | 端到端第一版               | Task 2.3      | 有 ms 数字(可慢)        |
| M4 | 内核 tensor-core 化        | Task 3.1(1)   | 正确性回归`PASS`      |
| M5 | TMA+warp-spec              | Task 3.1(2-6) | `PASS` + ncu BW 上升  |
| M6 | crossover 曲线             | Task 3.2      | 找到反超 sparsity       |
| M7 | 端到端最终对比             | Task 3.3      | skip vs FA 最终快慢结论 |

## 风险 / 待确认

1. **n_blocks 到底多大** —— 决定 win 上限和爬山法开销。需先确认实际 KV 长度(canvas_length / block_size)。若 n_blocks 只有个位数,每块跳过收益是离散大台阶,regroup 的优化空间可能被稀释。
2. **爬山法 wall-time vs FFN wall-time** —— 决定决策 C 走哪条。先量,别急着下沉 CUDA。
3. **block_max 预测精度** —— A1 的预测误差会直接变成"跳错的块",虽然质量不计指标,但会影响速度收益的上下界。需要一个便宜且足够准的预测(EMA/直接复用上一步)。
4. **torch extension 集成成本** —— 把 CUDA 包成 torch op 并替换 gemma/IDLM 的 attention,是纯工程但容易踩坑的活,提前留时间。

---

## 当前进度（2026-08-30）

### 已完成

- **Task 1（q_ids 映射）**：kernel 已支持 q_ids / block_ids / skip_flags 间接寻址。
- **Task 3.1 kernel 主体**：WGMMA(RS) + split-K + 寄存器 softmax + D=256 与 **D=512（GLOBAL 层）** 双支持（`HEAD_DIM` 模板 + `grid.z=HALVES` 头半拆分，绕过 255 reg/thread 上限）。dense/skip 两路数值正确（max_err ~2e-5）。
- **torch extension**（`skip_attn_ext.cu`）：`skip_attention(q,k,v,q_ids,skip_flags,split)` 已可被 Python 直接调用，D=256/512 双分派。
- **regroup 算法（Python）**：`analyze_sparsity.py` 的 contiguous/sorted/swap/greedy（block_max 离线路径）+ `regroup_csr.py` 的 `skip_matrix_to_csr` 桥（**直接吃 teammate 的 skip 矩阵** → q_ids+skip_flags，mask 版 contiguous/sorted/swap/greedy）+ `block_max_to_csr`（离线路径）+ skip-aware 参考实现。T/NUM_KV/NBLK 已全部形状驱动（真实模型 cache=2048 → NBLK=64）。
- **regroup CPU-offload 脚手架**（`regroup/regroup_scheduler.py`）：专用 side stream + pinned staging + 非阻塞 H2D + record()/wait() 已就绪（但**未接真实模型**）。
- **regroup 功能正确性测试**（`regroup/test_regroup_correctness.py`）：block_max → CSR → kernel == 参考，闭环 PASS。
- **JAX XLA-FFI 封装**（`jax_ffi/`）：`skip_attn_ffi.cu`（FFI handler）+ `build.py`（nvcc→`.so`）+ `skip_attn.py`（`jax.ffi` 注册 + 纯 JAX `quantize_tf32`，K/V 在 JAX 侧截断、Q 交 kernel）+ `test_skip_attn_ffi.py`（eager+jit 相对误差判 PASS）。已写待 H100 编译验证。
- **T3（gemma4 集成）**：`skip_attn.py::skip_attention_gemma4` 适配函数（BTNH→head-major 转置 + bf16→fp32 + `q*=16` 抵消 kernel `SCALE` + 调 FFI + 转回 bf16）；`_modules.py::Attention.__call__` 切点（`use_skip_attention` 开关包住 logits→softmax→PV 段，默认关、老路径逐 bit 不变）；`Block.__call__`/`_apply_attention`/`call_with_self_conditioning` 三层参数穿透（全默认 `None`，dense 即默认）；`test_gemma4_integration.py`（dense/skip/jit 三判）。GLOBAL 与 LOCAL 均 PASS（dense_err ~1.5e-2 为 bf16 输出舍入，skip 不引入额外误差，jit 与 eager bit 级一致）。

### 未实现（集成目标剩余工作，按依赖排序）

> 约束（2026-08-30 更新）：**block_max / 每个 token 能 skip 哪些块，由 teammate 负责并保证正确**，直接以 `(block_size, seq_len)` 矩阵形式传进来，我只需消费它做 regroup。**不做 IDLM**，只做 DiffusionGemma（JAX）。

| #  | todo                                      | 依赖  | 说明                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| -- | ----------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| T1 | regroup 桥接改为吃 teammate 的 skip 矩阵  | —    | ✅ 完成。`skip_matrix_to_csr(skip_matrix)` 直接消费 skip 矩阵 `[n_blocks, seq_len]`（1/True=该 token 可 skip 该 KV 块，匹配 kernel skip_flags 约定），mask 版 4 策略（contiguous/sorted/swap/greedy），`skip_flags[g,b]=AND 成员`、`union=OR 成员`；T/NUM_KV/NBLK 全部形状驱动（真实 cache=2048 → NBLK=64）。`test_regroup_correctness.py --mode skip_matrix` 闭环验证                                                                                        |
| T2 | JAX XLA-FFI 封装 kernel                   | T1    | ✅ 完成（代码已写，待 H100 验证）。`jax_ffi/skip_attn_ffi.cu`（XLA FFI handler，复用 `blasst_wgmma_kernel.cuh`，导出符号 `skip_attention`）+ `build.py`（nvcc 编 `.so`）+ `skip_attn.py`（`jax.ffi` 注册 + 纯 JAX `quantize_tf32`）+ `test_skip_attn_ffi.py`（eager+jit 相对误差判 PASS）。**注意**：`ffi.h` 头 API 随 jaxlib 版本可能有差异（`PlatformStream`/`Buffer.dimensions`/`Ffi::Bind`），若 H100 编译报错需按本机 jaxlib 头微调 |
| T3 | gemma4`Attention.__call__` 替换         | T2    | ✅ 完成。`skip_attention_gemma4` 适配 + `Attention.__call__` 切点 + 三层参数穿透 + dense/skip/jit 测试，GLOBAL(16H/2HK/D512) 与 LOCAL(16H/8HK/D256) 均 H100 PASS。详见下方「T3 备注」                                                                                                                                                                                                                                                                       |
| T4 | FFN 并行编排（JAX host 重叠）             | T2    | 🚧 进行中（骨架已落地，见「T4 备注」）。一步滞后 + `io_callback(ordered=False)`，host(CPU) regroup 与 device forward 重叠。剩：H100 端到端验证（io_callback 语义 + 端到端脚本）、量重叠收益 |
| T5 | sliding/causal 硬 mask（kernel 语义缺口） | —    | 当前 kernel 无 mask（全注意力）。LOCAL_SLIDING 层(25/30)的滑窗/attn_mask 是**硬 mask**（被 mask 的分母归零），与 soft-skip（分母保留 exp 质量）语义不同，kernel 需补硬 mask 档                                                                                                                                                                                                                                                                                     |
| T6 | 决策 C 量测                               | T1    | 实测爬山法 vs greedy vs FFN wall-time，定运行时用哪个                                                                                                                                                                                                                                                                                                                                                                                                                    |
| T7 | 端到端基准（Task 2.3）                    | T3/T4 | 原 attention vs skip 两遍，测 wall-clock，拿第一个端到端 ms 数字                                                                                                                                                                                                                                                                                                                                                                                                         |
| T8 | Task 3 回填                               | T7    | crossover 曲线（skip 反超 FA 的稀疏度）+ 优化后最终端到端对比                                                                                                                                                                                                                                                                                                                                                                                                            |

### 距离判断

Kernel 本体（含 GLOBAL 层）已就绪；block_max 预测由 teammate 负责（我吃现成 skip 矩阵），IDLM 不做。T1（regroup 桥接吃 skip 矩阵）、T2（JAX XLA-FFI 封装）、T3（gemma4 集成）均完成并 H100 验证 PASS。剩余距离：host 侧 regroup 与 device FFN 重叠（T4）、LOCAL_SLIDING 硬 mask 这个 kernel 语义缺口（T5）、端到端基准（T7）、Task 3 回填（T8）。关键路径 T4→T7（先拿第一个端到端数字），T5 是 LOCAL 层正确性的硬前提，T8 收口。

---

## T3 备注（gemma4 集成详解，2026-08-31）

> 给不懂 gemma4 内部的人看的通俗版：T3 把「算注意力」这件事从 JAX 的一串 einsum 换成 H100 tensor-core 上、能按 teammate 给的 skip 决定省掉部分 KV 计算的 kernel，并把两边的话（布局 / 精度 / scale）都对上。

### 0. gemma4 的 attention 在算什么

`Attention.__call__` 的核心是一串矩阵乘法：

1. `x` → 投影成 **Q（query）** / **K（key）** / **V（value）**，各自过 RMSNorm + RoPE。
2. `logits = Q · Kᵀ` —— 当前 token 与过去每个 token 的「相关度」分数。
3. `probs = softmax(logits)` —— 归一化成 0~1、和=1 的概率。
4. `encoded = probs · V` —— 按概率加权汇总过去 token 的信息。

gemma4 两个关键设定：(a) **RMSNorm 已控好 Q/K 幅度，所以不再除以 sqrt(d)，scale=1**；(b) **GQA**：16 个 query head 共享 2（GLOBAL）或 8（LOCAL）个 KV head。

### 1. 四个「语言不通」的地方（T3 就是做翻译）

| 维度 | gemma4 现状 | kernel 要的 | 翻译方式 |
| --- | --- | --- | --- |
| 布局 | `[B,T,N,H]` | `[H,T,D]`（head 在最前） | 转置 |
| dtype | bf16 | fp32（内部 tf32） | bf16→fp32 |
| scale | 1（不除） | kernel 硬编码 `SCALE=1/16` | JAX 侧先 ×16 抵消 |
| 层 | GLOBAL(D=512) / LOCAL(D=256) 混排 | 认 head_dim 256/512 | 从形状自动推断（512 走 `grid.z=HALVES`） |

### 2. T3 做的 5 件事

1. **适配函数 `skip_attention_gemma4`**（`jax_ffi/skip_attn.py`）：进去前翻译成 kernel 的话（转置 + bf16→fp32 + ×16），出来再翻译回 gemma4 的话（转置 + 转回 bf16）。
2. **切点**（`_modules.py::Attention.__call__`）：把 logits→softmax→PV 段包进 `if use_skip_attention and cache is not None:`，走 FFI；否则原 einsum 原封不动（默认关，老路径逐 bit 不变）。
3. **三层参数穿透**：`q_ids`/`skip_flags` 从 `call_with_self_conditioning` → `_apply_attention` → `Block.__call__` → `Attention.__call__`，全带默认 `None`，不传即 dense。
4. **dense 测试**：不 skip 全量算，对比 fp32 参考——验证翻译（转置/scale/dtype）对。
5. **skip 测试**：喂真实 `q_ids`/`skip_flags`，断言 skip 不引入超出 dense 基线的额外误差。

### 3. 两个最值得记的「拍板」决定

- **`q *= 16` 抵消 kernel 的 `SCALE=1/16`**：kernel 默认 head_dim=256 要做标准缩放（1/sqrt(256)=1/16），但 gemma4 scale=1。JAX 侧先乘 16，kernel 再乘 1/16，净效果=1；16=2⁴，乘 16 只是挪指数位，**精确不丢精度**。
- **bf16 喂 tf32 kernel 是「精确」而非降精度**：尾数位数 bf16(8) ⊆ tf32(10) ⊆ fp32(23)，所以 bf16→tf32 无损；kernel 内部还 fp32 累加，**比原 bf16 einsum（中间结果舍回 bf16）更准**。

### 4. 关键边界（T3 明确接受 / 不做）

- **safe-skip vs 硬 mask**：safe-skip 的 softmax 分母仍含所有 block（只跳 V 加载），硬 mask 是 logits→-inf、分母分子都归零。**T3 只做 safe-skip**，gemma4 自带的 attn_mask/sliding 是硬 mask，留 T5。
- **满 cache 才严格精确**：缓存未满时 padding 是零初始化 K/V，只稀释分母、不注入垃圾，是有界近似，T3 接受。
- **B=1**：`skip_attention_gemma4` 只支持 batch=1（diffusion 推理即 B=1）。
- 不做：prefill（`cache is None` + causal）、B>1、位置滑窗 per-token 精确、T4 host 重叠。

### 5. 验证状态

- ✅ GLOBAL（16H/2HK/D512）与 LOCAL（16H/8HK/D256）均 PASS。
- dense_err ~1.5e-2（= bf16 输出舍入量级，预期内）；skip_err == dense_err（skip 零额外误差）；jit 与 eager bit 级一致（模型 `__call__` 是 jitted 的，这是真实路径）。
- ⚠️ 测试环境无 torch → skip 桥走了 contiguous 回退；真实管道需 torch 跑 `regroup_csr.skip_matrix_to_csr`（greedy，非平凡 q_ids）。
- 风险提示：`call_with_self_conditioning` 上的 `@flatten_unflatten_batch_dim()` 会把参数当 pytree 遍历，`skip_attn_inputs`（list of `(q_ids[T], skip_flags[K*nblk])`）端到端时是否被它正确往返尚未实测，T3 验证走的是 adapter 级（不经该装饰器）。

---

## T4 备注（FFN 并行编排骨架，2026-09-01）

> 目标：把「skip 决策 + regroup」从 device 拿到 host(CPU) 跑，和 device 的 forward 重叠。一步滞后：第 t 步的注意力用第 t-1 步输出预测的 skip；第 0 步 dense。

### 0. 为什么是「一步滞后」

skip_matrix 是**每个 denoising step 一个**（一次 forward = 一个单位），预测要用**上一步**的 query/结果。但去噪循环是 `jax.lax.while_loop`（不是 Python 循环），没有 Python 级 step 边界，所以只能在 carry 里带 `(q_ids, skip_flags)`：body_fn 末尾算出下一步的、下一步开头消费 —— 这就是软件流水的一步滞后。经用户拍板，预测输入**先只用 canvas 和/或 sc_embeddings**，predictor 用桩函数占位。

### 1. 做了什么（4 个文件）

| 文件 | 动作 |
| --- | --- |
| `gemma/gemma/diffusion/_skip_regroup.py`（新） | numpy-only host 回调：`predict_skip_matrix_stub`（桩，占位给 teammate）+ `skip_matrix_to_csr_np`（greedy，镜像 `regroup/regroup_csr.py`，不依赖 torch）+ `predict_and_regroup`（io_callback 入口） |
| `gemma/gemma/diffusion/_sampler.py` | `_WhileLoopCarry` 加 `q_ids`/`skip_flags`；`sample_step` 加 `skip_attn_inputs` 透传；`sample_next_canvas` 里 step0=dense（identity q_ids + 全零 skip_flags），body_fn 末尾 `io_callback(ordered=False)` 从上一步 `sc_embeddings`+冻结 `canvas` 预测下一步 |
| `gemma/gm/nn/gemma4/_transformer.py` | `_apply_attention` 的 `skip_attn_inputs` 从 per-layer Sequence 改成单个 `(q_ids, skip_flags)`，广播到所有层 |
| `gemma/diffusion/_transformer.py` | `call_with_self_conditioning` 同名参数同步改成单 pair |

### 2. 关键拍板

- **单 pair 而非 per-layer 序列**：teammate 只给**一个** `[n_blocks, seq_len]` 矩阵，且所有层 n_blocks 相同，天然共享 → `skip_attn_inputs` 退化为单个 `(q_ids, skip_flags)`，广播到全部 30 层（比 T3 的 per-layer 序列更贴约束、更简单）。
- **numpy-only 不 import torch**：host 进程可能没装 torch，regroup 在这里重实现 greedy（与 `regroup_csr.py` 保持同步），不 import 它。
- **step0 = dense**：identity `q_ids` + 全零 `skip_flags`，等价不 skip，与后续 step 共用同一套 carry 结构。

### 3. 还没做 / 风险

- ✅ **`use_skip_attention` 已接进 config**（2026-09-01）：`TransformerConfig` 加 `use_skip_attention: bool = False` + `split: int = 8`，`_transformer.py::setup` 里传进每个 `Block`。`sample_next_canvas` 的 `skip_attn_enabled` 也改成 `cache is not None and self.model.config.use_skip_attention`（否则 tiny config 的 `_sampler_test.py` 会因 `T%GQ≠0` 崩）。
- ✅ **`@flatten_unflatten_batch_dim()` 已修**（2026-09-01）：`_flatten_batch_dim` 加 guard——只 reshape「前导维 == batch_shape」的 `jax.Array`，`q_ids[T]`/`skip_flags[K*nblk]` 这类无 batch 维的数组原样透传。原 try/except 的「Inconsistent batch shape」报错路径随之移除（`typechecked` 先于该装饰器运行，形状不符会被更早拦住，不会静默漏过）。对现有带 batch 的 arg 行为逐 bit 不变。
- **`jax.experimental.io_callback` 的 API 未在本机验证**（本机无 jax）。若 0.11 改了名（如只剩 `pure_callback`），H100 报错会指向 body_fn 那行，改起来是局部替换。
- **重叠收益未量**：callback 读的是 `out.sc_embeddings`（上一步 forward 的最后产物），严格卡在上一步/下一步之间，重叠靠 `ordered=False` 让设备不等 host，真实收益归 T7 的 H100 测量。

### 4. 验证方式（怎么跑）

- **纯 numpy 部分**（本机可跑）：`_skip_regroup.py` 自测 —— q_ids 是 0..T-1 的置换、skip_flags 与分组语义一致、桩函数端到端跑通。
- **FFI 层**（不变，H100）：`python jax_ffi/test_gemma4_integration.py`。
- **sampler 端到端**（H100）：`gemma/gemma/diffusion/_sampler_test.py` 的 `test_sample_next_canvas` / `test_sample_next_canvas_while_loop_matches_for_loop` 用 tiny config 跑通 while_loop；但要真正走 FFI 仍需先接 `use_skip_attention` config。

---

## 端到端对比（kernel+regroup vs flash）剩余 TODO（2026-09-01）

> 目标：真跑一次「我们的 skip kernel + host regroup」vs「flash attention」，拿到扣掉 regroup 之后的净收益。**不拆成单层/regroup/flash 三段加减**——必须端到端跑通再对比。

| # | todo | 状态 | 说明 |
| -- | -- | -- | -- |
| E1 | 修 `flatten_unflatten_batch_dim` | ✅ 本次完成 | `skip_attn_inputs` 无 batch 维被错误 reshape 崩溃，已加 guard 透传；单测 `test_flatten_batch_dim_passthrough_non_batched_arrays`（`_jax_utils_test.py`，需 jax，H100 跑） |
| E2 | 加 flash 参考 | ✅ 已验证 | `jax_ffi/flash_ref.py::flash_attention_gemma4`（BTNH + GQA `repeat` + `scale=1` + 按 head_dim 分派 impl）。`test_flash_ref.py` 两档 PASS：LOCAL(256,cudnn) err=0.0094、GLOBAL(512,xla) err=0.0141（均为 bf16 输出舍入量级，非错位） |
| E3 | 验证 `io_callback`（jax 0.11） | ✅ 已验证 | `jax_ffi/test_io_callback.py`：eager/jit/while_loop 三档全 PASS。`ordered=False` 在 jax 0.11 可用，host numpy 返回 int32 被接住 → T4 重叠设计成立 |
| E4 | 端到端对比脚本 | ✅ 已验证（GLOBAL 1.02× / LOCAL 0.939×） | flash 已接入模型（`use_flash_attention` 贯穿 config→Block→Attention）。`jax_ffi/bench_skip_vs_flash.py`：单层 DiffusionGemma 跑 `sample_next_canvas`（真实 while_loop + io_callback regroup）skip vs flash 两遍计时，T=256/S=2048/steps=5，真实 26B 注意力形状。关键看点：`ordered=False` regroup 是否真被 FFN 隐藏 |
| E5 | H100 环境 | 🟡 前置 | 编 `.so`、`jax_ffi` 可 import（E2/E3/E4 的物理前提） |

依赖：E1→（E5）→ E2/E3/E4。E1、E2、E3、E4 已闭环；E5 已就绪（`.so` 已编、`jax_ffi` 可 import）。

关键发现：cuDNN FlashAttention head_dim 上限 256 → gemma4 GLOBAL(512) 无 flash 基线，回退 xla fused attention。「flash vs 我们 kernel」只对 LOCAL(256, 25/30 层)成立；GLOBAL(512) 是 kernel 的差异化点（`grid.z=HALVES` 劈半，标准 flash 做不了）。

### E4 结果（2026-09-02）

GLOBAL（16H/2HK/D512）单层 DiffusionGemma 端到端，5 步 while_loop + `io_callback(ordered=False)` regroup，T=256/S=2048：

```
[skip+regroup]  1999.57 ms/loop  (min 1910.42 ms)
[flash       ]  1960.25 ms/loop  (min 1892.99 ms)
[delta       ] skip/flash = 1.020  -> flash faster by 2.0%
```

**结论**：

| attn | flash 基线 | skip+regroup | flash | skip/flash |
| -- | -- | -- | -- | -- |
| GLOBAL 16H/2HK/D512 | xla fused attention | 1999.57 ms | 1960.25 ms | 1.020（慢 2%） |
| LOCAL 16H/8HK/D256 | cudnn flash | 1935.17 ms | 2060.30 ms | 0.939（快 6.1%） |

**LOCAL（25/30 层，真 cudnn flash 基线）skip 快 6.1%；GLOBAL（5/30 层，xla-fused 基线）skip 慢 2%。**

⚠️ 但 skip ratio 未知（随机初始化单层）。若 ratio≈0，两组数测的是「dense kernel vs flash」而非 skip 收益：LOCAL 6.1% = kernel 在 D256 dense 就比 cudnn flash 快；GLOBAL 2% = kernel 在 D512（`grid.z=HALVES` 劈半）比 xla-fused 慢。

**下一步**：
1. 给 bench 打印每步真实 skip ratio——决定上面两组数到底是不是 skip 收益（第一个要拿的数）。
2. GLOBAL D512 慢 2%：`grid.z=HALVES` 劈半路径有额外开销，是 kernel 优化点（T6）。
3. 若 ratio 高（skip 真发生了）→ LOCAL 6.1% 是净收益，可继续放大（T5 硬 mask / T6 优化）。

### 验证命令速查（T3 + E1/E2/E3）

> 前提：repo 根 `[private repository root]`；gemma 已 `pip install -e`；`jax_ffi/libskip_attn_ffi.so` 已编。**jax_ffi 下的三个测试必须先 `cd jax_ffi` 再跑**（脚本顶部 `import skip_attn` / `import flash_ref`，且 `build_if_missing` 用相对路径找 `.so`）。E1 是 pytest 且依赖 `kauldron` + `gemma` 包，须在 repo 根用 `python -m pytest`。

| 项 | 命令 | 验证什么 |
| -- | -- | -- |
| T3 | `cd jax_ffi && python test_gemma4_integration.py` | GLOBAL 16H/2HK/D512：`skip_attention_gemma4` dense+skip+jit vs BTNH fp32 参考 |
| T3 | `cd jax_ffi && python test_gemma4_integration.py --num_kv_heads 8 --head_dim 256` | LOCAL 16H/8HK/D256 同上 |
| E2 | `cd jax_ffi && python test_flash_ref.py` | `flash_attention_gemma4` 两档（LOCAL 256 cudnn / GLOBAL 512 xla）vs fp32 参考 |
| E3 | `cd jax_ffi && python test_io_callback.py` | `io_callback(ordered=False)`：eager / jit / while_loop 三档 |
| E1 | `python -m pytest gemma/gemma/gm/utils/_jax_utils_test.py -v` | `flatten_unflatten_batch_dim`：带 batch 归一 + 无 batch 透传（含 `test_flatten_batch_dim_passthrough_non_batched_arrays`） |

一键连跑（jax_ffi 三个）：

```bash
cd jax_ffi
python test_gemma4_integration.py \
  && python test_gemma4_integration.py --num_kv_heads 8 --head_dim 256 \
  && python test_flash_ref.py \
  && python test_io_callback.py
```

---

## FA3 迁移里程碑（2026-09-06）

> 里程碑：LOCAL 层（D=256）skip-attention **首次在 kernel-level 反超 flash**。

`bench_attn_only.py --attn_type local`（T=256/S=2048，16H/8HK，50 iters）：

```
[skip attn]    0.202 ms  (min 0.191 ms)
[flash    ]    0.209 ms  (min 0.198 ms)
[delta    ] skip attn/flash = 0.968  -> skip attn faster by 3.2%
```

**为什么是里程碑**：LOCAL 层从 BLASST WGMMA kernel 的 **2.8× 落后** → FA3 迁移（dense-first → safe-skip 注入 → overlap+cluster 恢复）→ **host 预计算位图**（把 on-device 的 `_blasst_skip_to_fa3` 转换挪到 host，消掉曾把 skip_gemma4 拖到 0.352ms 的临界路径开销），首次做到 skip 路径（0.202ms）比 cuDNN flash（0.209ms）**快 3.2%**。FFI 层未动、`.so` 未重编（纯 python/host 侧改动）。

**诚实的边界**（别把 3.2% 读成「skip 的稀疏收益兑现了」）：
- 这是 `bench_attn_only.py` = **单层 attention-op** 计时，skip 输入（q_ids/skip_flags/fa3_bitmap）在 host 预计算、**不计入计时**；不是端到端 `bench_skip_vs_flash.py`。
- skip ratio：token-level 0.536、group-level 0.469，但 FA3 的 128×80 tile 粒度把有效 tile-level skip 压到 ~2-6%（「关键发现 1」skip 崩塌）。所以这 3.2% 本质是 **FA3-dense 比 cuDNN flash 快**，skip 位图只贡献一个很小的尾巴。
- 换言之：LOCAL 层的胜利点是「换 kernel 到 FA3」+「转换下 host」，不是「skip 省了 47% 的 PV」。真正吃 skip 稀疏收益的是 GLOBAL 层（BLASST 保留 32×32 粒度）——另一条线。

**下一步（端到端收口）**：
1. 端到端 `bench_skip_vs_flash.py` 复测 LOCAL（此前 E4 的 LOCAL 0.939 是 BLASST 时代；现在 FA3 + host 位图应更优）。
2. 若端到端也反超，回填 M7（端到端最终对比）。

