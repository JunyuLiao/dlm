# DiffusionGemma 长上下文稀疏 attention：进度总结（更新于 2026-10-05 04:30 UTC / 10-04 23:30 UTC−5）

- 分支：`research/v31-progress-aware-20261004`（github.com/coconight01/dlm_test）
- 详细研究日志：`docs/V31_BASELINE_ROOT_CAUSE_20261003.md`（末尾各节）
- 论文提纲：`docs/PAPER_OUTLINE_MLSYS27.md`
- 结果表：`results/v31_20261003/panels/`
- 目标：MLSys 2027（10-30 截稿）。三项都要够：性能（端到端）、准确率、novelty。

## 1. 设定

- **模型和推理栈：**
  - DiffusionGemma-26B-A4B，vLLM 0.30，FA4 SM90，单张 H100。
  - 只稀疏 5 个 GLOBAL 层的前缀 attention；25 个 LOCAL 层（窗口 1024）、prefill、canvas 本身都保持 dense。
  - 解码用官方采样器和**官方自适应停止**，不固定步数。
- **评测：**
  - 官方 pool 和官方评分器：RULER（v33 确认池 / v34 探索池）、LongBench-v2 `0shot_think`（thinking 打开，输出上限 16K，120K 截断）、OpenAI MRCR、GraphWalks；短任务用 AIME26、HumanEval。
- **筛选协议（S1）：**
  - 每个候选在 4 个长数据集的小子集上跑：RULER 78 题、LongBench 32 题、MRCR 24 题、GraphWalks 12 题，seed 1–2；LongBench / GraphWalks / MRCR 正在补 seed 3–4。
  - 两台机器分片，每个 cell 的所有 arm 都在同一台机器上跑。
  - LongBench 和 HumanEval 留出集从不参与筛选，留给最终的预注册确认。
- **指标：**
  - 官方准确率；
  - 端到端 W、纯生成 S；
  - 总步数 N、每个 canvas 的步数 N/C、输出长度 T、每次 forward 的耗时 S/N。
  - CI 是按题目聚类的 bootstrap。

## 2. 方法（当前版本）

- **lean 均衡选择：**
  - 每个 canvas 第 1 步做一次精确观测（FA4 kernel 内顺便写出每个 tile 的注意力质量）。
  - 每个（query head × 128 行块）单元按"最差行"的前缀质量份额取 top-k，**k 相同**，保证 CTA 均衡。
  - 选择在 canvas 内保持；下一个 canvas 的第 1 次调用沿用上一份选择（carry）。
  - 预算：4096 或 8192 个 token。
- **按进度在 canvas 中途重新观测：** 采样器已接受比例达到 0.5（或者用 C gate"已定程度"作时钟）时重选一次，不加权。粘性版给已选 tile 加分，减少扰动。
- **两级候选池：**
  - 第 1 步的观测同时圈出 4k 个候选 tile。
  - 重选时在 FA4 block-sparse kernel 里只观测候选池（候选 tile 作为 mask block，观测钩子写出 z），用 S=2 split 均衡 CTA。

## 3. 主要结果

### 3.0 最新：最终候选"粘性进度重选"（10-05 04:30 UTC，seed 1–2，两台机器）

方法：lean 8192；canvas 中途已接受比例达到 0.5 时重选一次；粘性（已选 tile 加 1.386 的 log 份额）。

| 相对 dense | RULER v34 | MRCR | GraphWalks | LongBench think |
|---|---|---|---|---|
| 准确率差 | +0.61 [−0.58, +2.28] | −0.010 [−0.110, +0.066] | −0.005 [−0.124, +0.129] | +0.047 [−0.094, +0.188] |
| 总步数 N | 0.993 | 0.964 | 0.938 | 0.928 |
| 每 canvas 步数 N/C | 0.993 | 1.029 | 0.875 | 0.997 |
| 端到端 W（官方 dense） | 1.012 | 0.974 | 0.901 | 0.856 [0.737, 0.995] |

- 四个数据集准确率都与 dense 持平；GraphWalks 是第一次不掉点（其他 lean 变体约 −0.10）。总步数都不多于 dense。
- **加 LOCAL 修复后（LongBench think，都加修复）：** 相对修复后 dense，W 0.806 [0.60, 1.03]，每步耗时 0.748；lean 8192 为 W 0.799 [0.66, 0.96]、每步 0.698。
- **弱点：** RULER 每个 canvas 约 5 步就收敛，观测开销摊不开，每步比 dense 慢约 13%（端到端约 1.01，因为 prefill 为主）。计划用候选池重观测降低这部分开销。
- seed 3–4、AIME26 / HumanEval 在 sc2 队列中。


### 3.1 和 MAGE 比（stage A 预注册确认，RULER v33，585 个 cell）

| | 相对 dense | cwe |
|---|---|---|
| dense | — | 84.0 |
| MAGE 4096 | −2.46 [−3.56, −1.38] | 56.0 |
| lean 4096 | −1.06 [−2.12, −0.01] | 77.6 |

- **lean − MAGE = +1.40 [+0.57, +2.21]，p = 0.005**，每步开销相同。
- MRCR：dense 0.290，lean 0.225，MAGE 0.205。
- GraphWalks 32K：dense 0.264，lean 0.238，MAGE 0.178。

### 3.2 S1 准确率（官方评分，两台机器 × 2 个 seed，相对 dense）

| | RULER v34 | LongBench think | MRCR | GraphWalks |
|---|---|---|---|---|
| lean 4096 | −0.16 | +1.6 | **−0.111 [−0.21, −0.02]** | −0.10 |
| lean 8192 | **+1.15**（cwe 84.2 vs 85.8） | +1.6 | −0.046 | −0.10 |
| lean 4096 + 进度重选（不加权，T0） | +0.10 | 0.0 | −0.028 | −0.096 |
| lean 4096 + 进度重选 + C gate 行权重（T1） | −0.06 | −6.3（n.s.） | −0.021；vs lean **+0.091 [+0.031, +0.167]** | −0.044 |
| T1，重选预算降到 2048（T2） | −0.90 | +1.6 | −0.099 | — |

**两种失效机制：**
- 预算不够：cwe 这类聚合任务，加到 8192 能补，重选补不了。
- 选择过时：MRCR / GraphWalks，重选能补，8192 补不了。MRCR 上 lean 的回答被截短（191 vs 285 token）。

### 3.3 速度

**每次 forward 的 decode 时间（相对 dense）：**
- lean 4096：64–96K 约 0.82，128K 约 0.75–0.77，LongBench think 0.78–0.80。
- 8192 只多约 1%。

**all-kept 基线**（同一个 block-sparse kernel，tile 全保留）：
- 与 dense 只差 1–2%；
- 稀疏本身每次 forward 省 16–18%（64–96K）、23–26%（128K）。

**端到端 W（正常生成，相对 dense；越小越快）：**

| | LongBench think | GraphWalks | MRCR | RULER |
|---|---|---|---|---|
| lean 4096 | **0.79** [0.68, 0.90] | 0.96 | 0.97 | 1.01 |
| lean 8192 | 0.84 | 0.97 | 0.99 | 1.01 |
| T0 | **0.75** [0.65, 0.85] | 1.16 | 0.97 | 1.02 |
| T1 | 0.88 | 1.27 | 0.97 | 1.01 |

- 只有长输出任务有明显的端到端收益。RULER / MRCR 是 128K prefill 加短回答，时间以 prefill 为主（我们不加速 prefill）。
- **kernel 几何（nsys）：**
  - FA4 dense 和 block-sparse 都是每个 SM 1 个 CTA、128 个 CTA、一波跑完，SM 覆盖 97%。
  - 均衡预算达到线性缩放的 85–90%；同样平均保留比例下，负载偏斜慢 3.5–6 倍。

### 3.4 步数（核心指标）

**固定 canvas（每个 canvas 从同一参考前缀和噪声出发，只比 attention 的影响；mpk 上噪声下限为 0；LongBench 64K/96K，583 个 canvas）：**

| | 每个 canvas 的步数比 | 与 dense 的 token 一致率 |
|---|---|---|
| dense | 1.000 | 1.000 |
| lean 4096 | 1.026 [1.000, 1.052] | 0.216 |
| lean + 固定第 4 步重选 | 1.027 | 0.219 |
| **lean + 进度重选（不加权）** | **1.013** [0.992, 1.033] | **0.222** |
| lean + 进度重选 + C gate 行权重 | 1.033 | 0.218 |

**正常生成的总步数 N 和每 canvas 步数 N/C（相对 dense）：**

| | LongBench N（N/C） | GraphWalks N（N/C） | MRCR N（N/C） |
|---|---|---|---|
| lean 4096 | 0.88（1.04） | 1.09（1.03） | 1.03（**1.19**） |
| lean 8192 | 0.93（0.99） | 1.08（0.97） | 1.05（1.14） |
| T0 | 0.76（0.95） | **1.24**（0.89） | 0.98（1.06） |
| T1 | 0.95（1.00） | **1.45**（**1.23**） | 0.98（1.06） |

**结论：**
- 每个 canvas 的步数基本可控：lean 约 +2.6%，进度重选后约 +1%。
- 按进度触发优于固定步：固定第 4 步时长思考 canvas 只接受了约 20% 的行，重选白选。
- C gate 行权重有害：已接受的行上下文被抽走，GraphWalks N/C +23%。
- 总步数的主要波动来自输出长度（canvas 数），在 GraphWalks 上最大。正在用加 seed 和粘性重选确认。

### 3.5 其他结论

- **C gate 作为采样器停止规则（每行连续 3 步稳定被接受就结束 canvas）：**
  - 518 个 canvas 上从未触发，官方收敛总是更早，已放弃。
  - 它会改变解码本身，本来也不放进主比较。
- **预算随进度递减（2048）：** 丢掉 MRCR 收益，也不省时间，已否决。预算只往上加：8192，或按覆盖度自适应。
- **候选池重观测 GPU 校验通过：**
  - z 与 dense 观测逐位一致；输出误差在 bf16 舍入范围内。
  - 128K：0.96 ms vs dense 观测 3.22 ms；64K：0.94 vs 1.64 ms。
- **意外发现：LOCAL（滑窗）attention 在长上下文下异常贵（nsys，dense 官方路径）。**
  - 每次调用：32K 223 µs，64K 43 µs，128K 826 µs。
  - 128K 时 25 个 LOCAL 调用约 20.7 ms/forward，比 5 个 GLOBAL（16.6 ms）还多。
  - 窗口只有 1024，应与长度无关（64K 正常）。
- **根因已确认并修复（10-05 03:30 UTC）：**
  - vLLM 0.30 的 FA4 SM90 kernel 中，dynamic_causal 分支对双向序列（canvas）把 KV 块范围重置为全上下文，丢掉了 LOCAL kernel 已算好的窗口范围。
  - 窗外块全被 mask，所以输出正确，只是白扫一遍。
  - 修复只改两行（producer 和 consumer 各一行，只对 LOCAL kernel 生效），用 `FA4_LOCAL_FIX=1` 开启。
  - GPU 校验（mpk）：16K–131K 共 12 个长度，与原调用逐位一致；每次调用固定 0.105 ms，原来是 0.48–3.19 ms。
  - GLOBAL 层和因果（prefill）序列不受影响。
  - 语义说明：vLLM 的 LOCAL 窗口是按 query 对称滑动的 (1023, 1023)，与 HF 的 flash-attention 路径一致；HF 默认 sdpa 路径让所有 canvas 行看同样的前缀 1023 个 token，两者对最后几行最多差 255 个最老的 token。这是原有差异，所有 arm 相同，论文中说明即可。
- **FX 端到端（进行中）：** LongBench think S1 上跑 dense / lean 8192 / 粘性候选三个臂，都加修复。sc2 暂停约 1 小时后自动恢复。预期 token 与未修复时完全相同，只有每步耗时变化。报告时同时给出"官方原样"和"修复后 dense"两个基线，主结论用更强的修复后基线。

## 4. 正在跑 / 下一步

- **sc2（两台机器，约 10 小时）：**
  - LongBench / GraphWalks / MRCR 的 seed 3–4（dense、lean 4096 / 8192）；
  - 最终候选"8192 + 进度重选（不加权 / 粘性）"及其 seed 3–4；
  - 同预算消融：固定第 4 步、C gate 行权重、C gate 已定程度作时钟、候选池多次刷新、覆盖度自适应预算、jcgate 权重。
- **dlm2 GPU 空档：**
  - LOCAL 窗口 attention 微基准（约 10 分钟）；
  - LLaDA2.2-mini dense 时间拆分（2026-09 发布，原生 128K + block routing，attention 占比估算 35–43% @128K，约 35 分钟）。
- **之后：** 最终方法定型 → 留出集预注册确认（RULER v33、LongBench 留出 343 题、MRCR / GraphWalks 未用题、HumanEval 留出 124 题，多 seed）→ 速度扩展到 256K → 写作。

## 5. 和组内工作的关系（只读，合作部分单独标注）

- **Junyu 的报告（value-direction routing + Cgate，组内 PDF）：** 评的是短上下文（AIME / RULER 4–8K / LongBench ≤32K）、按比例稀疏 50–70% 下的选块准确率。
  - 不报运行时间。
  - 步数仍膨胀：AIME 50% 稀疏时 dense 15.0、BLASST 28.0、Cgate 18.6 步/canvas。
  - 我们是长上下文、固定绝对预算、实测端到端，步数 +1–3%，两者互补。
- **C gate / query sensitivity 是 Junyu 的思路：**
  - 在我们这里作为进度时钟（何时重看）有用；
  - 作为重选行权重有害（数据见上）。
  - 相关部分按合作候选标注。
- **Fan 的 M1–M3：** 周期刷新（M3）对应我们的"固定步重选"对照组，sc2 中有同预算对比。
