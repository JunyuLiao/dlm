"""Build the 2026-09-30 group-meeting deck (pptx): one experiment per slide, table or figure + short text on the
slide, details in the speaker notes. Numbers are copied from the pushed panel summaries (sources in the notes).

Writes results/m1_m2_m3_frontier_v27_20260929/ppt_sample/dlm_sparse_attention_20260930.pptx.
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parents[1] / 'results/m1_m2_m3_frontier_v27_20260929/ppt_sample'
FONT = 'Microsoft YaHei'
GRAY = RGBColor(0x55, 0x55, 0x55)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def text(slide, x, y, w, h, lines, size=14, bold=False, color=None, bullet=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ('• ' if bullet else '') + line
        p.space_after = Pt(4)
        for r in p.runs:
            r.font.size, r.font.bold, r.font.name = Pt(size), bold, FONT
            if color is not None:
                r.font.color.rgb = color
    return box


def table(slide, x, y, w, rows, widths, size=11, bold_rows=()):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(0.4 * len(rows)))
    t = shape.table
    for j, cw in enumerate(widths):
        t.columns[j].width = Emu(int(Inches(w) * cw / sum(widths)))
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size, r.font.name = Pt(size), FONT
                    r.font.bold = i == 0 or i in bold_rows
    return t


def slide(title, config, notes):
    s = prs.slides.add_slide(BLANK)
    text(s, 0.5, 0.3, 12.3, 0.7, [title], size=24, bold=True)
    if config:
        text(s, 0.5, 0.95, 12.3, 0.6, [config], size=12, color=GRAY)
    s.notes_slide.notes_text_frame.text = notes.strip()
    return s


# ---------------------------------------------------------------- 0 title
s = prs.slides.add_slide(BLANK)
text(s, 0.8, 2.4, 11.5, 1.0, ['DiffusionGemma-26B-A4B 上的块稀疏注意力：本周结果'], size=32, bold=True)
text(s, 0.8, 3.5, 11.5, 1.0, ['2026-09-30 · 长上下文端到端 · 与 SparseD 正面对比 · AIME 稀疏度'], size=18, color=GRAY)
s.notes_slide.notes_text_frame.text = ('所有数字来自已推送的结果：仓库 coconight01/dlm_test，分支 research/m3-output-numerics-20260927，'
                                       '目录 results/m1_m2_m3_frontier_v27_20260929/。完整说明见该目录的 progress_20260930.md。')

# ---------------------------------------------------------------- 1 methods and setup
s = slide('方法与实验设置',
          '模型 DiffusionGemma-26B-A4B（MoE）· 自带 adaptive stopping，默认配置（置信度 0.005、每个 canvas 最多 48 步）· 每个 canvas 256 token · '
          '只稀疏 5 个 GLOBAL 层 · dense 基线 = FlashAttention-4',
          """
【生成配置】全部使用模型 generation_config.json 的默认值：confidence_threshold 0.005、stability_threshold 1、max_denoising_steps 48（每个 canvas 最多 48 步）、EntropyBoundSampler entropy_bound 0.1、t_max 0.8、t_min 0.4，即 DiffusionGemma 自带的 adaptive stopping。只改了两处：max_new_tokens 从默认 256 改为 8192（32 个 canvas），并开启 thinking。每个 canvas 的步数由模型自己决定，所以"稀疏导致步数增加"会直接体现在端到端时间里。

【注意力结构】30 层 decoder：5 层 GLOBAL（第 5/11/17/23/29 层，看全部上下文，head_dim 512，16 个 query head / 2 个 KV head），25 层 LOCAL（滑动窗口 1024，保持 dense）。只有 GLOBAL 随上下文变贵，所以所有方法只稀疏 GLOBAL 层。
【块的定义】query 每 128 个一组，key 每 64 个一组（FA4 kernel 的 tile）；"跳过"= 跳过某个 (query 块, key 块) 组合。canvas 自己的 key 块总是保留。

【各方法的具体做法】
- dense：FA4（vLLM 的 flash-attn cute 构建）的 block-sparse 接口保留全部块；与 FA4 dense 逐位一致，且快 1–2%。HF 默认的 SDPA 在 head_dim 512 下端到端慢 1.6–2.2 倍，不用它当基线。FA2/FA3 不支持 head_dim 512；FlashInfer 与 FA4 持平。
- 风险（M1 的选块依据）：对每个 query 行、每个 key 块，估计"跳过这个块，注意力输出会变多少"：risk = log(‖α_j (μ_j − P)‖ / ref) + log T。α_j 是该块的注意力质量占比，μ_j 是块内按注意力加权的 V 投影（投影到 32 维）均值，P 是保留状态下的输出估计，ref 是参考范数，T 是 query 敏感度。一个块的风险取 128 个 query 行里的最大值。注意力分数（QK）来自观测步，V 用当前步的。
- 原版 M1（R1 A8）：每一步都按上面的风险从前往后逐块决策，风险低于阈值的块跳过；每 8 步重新观测一次完整 QK。
- 原版 M2c（R1 A8）：同 M1，但 μ_j 用块内 V 的均值（pooled compact），不做注意力加权投影。
- 原版 M3（R3 A8）：同 M1，但每 3 步决策一次，中间沿用上一次的块选择。
- 所有方法的注意力输出都用当前步的 QK 和原始 V 精确计算，选块只决定跳过哪些块。
- B：每个 canvas 第 1 步 dense，第 2 步 dense 并同时观测、按风险阈值选块，之后整个 canvas 沿用这张选择（hold），不再更新。
- M3 R6 DP −ln2（本周选定的配置）：①每 6 步决策一次（R6）；②每个 canvas 只观测一次，观测与第 2 步的 dense 计算融合在同一个 kernel 里（A64）；③DP：风险里的 P 用"所有更早的块都保留"的 dense 前缀状态，而不是逐块累积的保留状态，所有块可以并行判断，一次决策 0.27 ms/层（原版逐块顺序决策 2.3–7.9 ms/层）；④选块放在副 CUDA 流上异步执行；⑤阈值从默认对数阈值 −3.18 下调 ln2 到 −3.874（更保守，多保留块）。
- 风险 top-k XX%（固定稀疏度变体，本周新加）：同 M3 R6 DP 的风险与节奏，但不卡阈值，而是每个（head, query 块）按风险排序，只保留风险最高的 (1 − XX%) 的 prefix 块。原版和阈值版都没有"目标稀疏度"这个旋钮，加它是为了和 SparseD、同学在同一稀疏度下比较。
- SparseD 移植（ICLR 2026，标明是移植）：每个 canvas 前 1 步（s1）或前 10 步（s10）dense；在观测步上算完整注意力概率，按每个 128×64 块做平均池化，每个（head, query 块）保留分数最高的 XX% 的 prefix 块；观测步本身输出仍是 dense，之后沿用到 canvas 结束。

【公平性】所有稀疏方法跑同一个 FA4 kernel，与 dense 的区别只在跳过了哪些块；同一格（题目 × seed）的所有方法在同一块 GPU 上跑，比较都在机器内配对；每个面板在生成前冻结题目、seed、方法和机器分配。底座（所有方法相同）：decoder 分段 CUDA graph（vLLM 式），GLOBAL 注意力在图外执行，prefill 用 FA4，每步不再重新拼接 KV。

【统计口径】格 = 一个（题目, seed）组合，每格只取第一次输出。W = 请求墙钟时间比（含 prefill），S = 去掉 prompt prefill 后的解码时间比，均为方法 / dense，< 1 表示更快；配对几何平均，95% 置信区间按题目聚类 bootstrap（4000 次）。答对差用配对符号检验。计时期间捕获新 CUDA graph 的请求（首次遇到新形状，多出几秒到几十秒）从速度比中剔除。跨机器生成不能逐位复现，同一方法的答对数跨机器约差 ±3/36。
""")
table(s, 0.4, 1.55, 12.5, [
    ['方法', '多久观测一次注意力分数', '多久重新决定跳哪些块', '给块打分的依据', '跳多少', '工程优化'],
    ['dense（基线）', '—', '—', '—', '不跳', '—'],
    ['原版 M1', '每 8 步（单独算一遍）', '每步', '风险（QK + V 投影）', '风险 < 阈值就跳', '无'],
    ['原版 M2c', '每 8 步', '每步', '风险（V 用块内均值）', '风险 < 阈值就跳', '无'],
    ['原版 M3', '每 8 步', '每 3 步', '风险（同 M1）', '风险 < 阈值就跳', '无'],
    ['B', '每个 canvas 1 次', '每个 canvas 1 次', '风险（同 M1）', '风险 < 阈值就跳', '融合观测、异步'],
    ['M3 R6 DP −ln2（选定）', '每个 canvas 1 次', '每 6 步', '风险（DP 并行算法）', '风险 < 更低的阈值', '融合观测、异步'],
    ['风险 top-k XX%', '每个 canvas 1 次', '每 6 步', '风险（DP 并行算法）', '固定跳 XX%（按风险排序）', '融合观测、异步'],
    ['SparseD 移植', '每个 canvas 1 次（第 1 或第 10 步后）', '每个 canvas 1 次', '平均池化的注意力分数', '固定跳 XX%（按分数排序）', '无'],
], [2.4, 2.4, 2.0, 2.1, 2.1, 1.5], size=11)
text(s, 0.4, 5.75, 12.5, 1.5, [
    '所有稀疏方法跑同一个 FA4 kernel，与 dense 只差跳过了哪些块；同一格（题目 × seed）的所有方法在同一块 GPU 上配对比较。',
    '输出始终用当前步的 QK 和原始 V 精确计算；"跳过"只是不算这些块。名词解释见下一页。'], size=13, bullet=True)

# ---------------------------------------------------------------- 1b glossary
s = slide('名词解释：观测、决策、融合观测、异步选块、−ln2、风险、top-k', None, """
【观测】为了知道哪些块可以跳，需要先看一眼完整的注意力分数（QK）。这一步要算完整注意力，所以贵。原版每 8 步单独算一次；我们每个 canvas 只做 1 次。
【决策】根据最近一次观测的分数（以及当前的 V），决定这一步跳哪些块。原版 M1 每步都决策，而且是从前往后一个块一个块地判断（每次 2.3–7.9 ms/层，比 dense 注意力本身还贵）。
【融合观测】每个 canvas 的第 2 步本来就要做一次 dense 计算；把"算观测统计"塞进这次 dense 计算的同一个 kernel 里，就不用为观测单独再算一遍完整注意力。
【异步选块】把选块的计算放到 GPU 的第二条工作队列（CUDA 流）上，与后面几层的计算同时进行；主计算不用停下来等它，用的时候（下一步）已经算好。选出的块与同步计算完全相同，只是省了等待时间。
【DP（dense 前缀）】原版判断第 j 个块的风险时，要知道前面哪些块已经被保留，所以只能一个个顺序判断。DP 改成"假设前面的块全都保留"来算，于是所有块可以同时判断，一次决策只要 0.27 ms/层。代价是决策和原版不完全相同，所以单独命名。
【风险】对每个块估计"跳过它，注意力输出会变多少"：risk = log(‖α_j(μ_j − P)‖ / ref) + log T。α_j 是这个块占的注意力质量，μ_j 是块内 V（投影到 32 维）的加权均值，P 是当前输出的估计，ref 是参考范数，T 是 query 的敏感度。一个块的风险取 128 个 query 行里最大的那个。
【阈值与 −ln2】风险低于阈值的块就跳过。默认对数阈值 −3.18；−ln2 表示再降低 ln2（约 0.69），变成 −3.874，也就是允许的输出变化减半，于是更多块被保留、更保守。阈值扫描里 −ln2 端到端最快，更激进的阈值会让去噪步数增加，反而更慢。
【风险 top-k】不用阈值，而是在每个（head, 128 个 query）里把块按风险排序，只保留风险最高的 (1 − XX%)，所以稀疏度固定为 XX%。排序只对约 100（AIME）到 1000（64K）个数做，每 6 步一次，放在副流上；AIME 上它的每步成本是 dense 的 0.985–0.99，与同样要排序的 SparseD 一样，排序不是瓶颈。
【公平性】融合观测和异步选块是工程优化。与 dense 比较公平（dense 不需要选块）；与原版 M1–M3 比较时，我们的加速同时来自算法改动和工程优化，没有拆开；SparseD 移植没有这两项，对它不利（见 SparseD 那一页）。
""")
table(s, 0.4, 1.3, 12.5, [
    ['名词', '一句话解释'],
    ['观测', '先算一遍完整注意力分数（QK），作为选块依据；贵，所以要少做'],
    ['决策', '根据最近的观测，决定这一步跳哪些块'],
    ['融合观测', '把观测塞进本来就要做的那次 dense 计算的同一个 kernel，不额外再算一遍'],
    ['异步选块', '选块放到 GPU 第二条工作队列上，和后面几层同时算，主计算不等它；选出的块不变'],
    ['DP（dense 前缀）', '假设前面的块都保留来算风险，所有块可同时判断：一次决策 0.27 ms/层（原版逐块 2.3–7.9 ms）'],
    ['风险', '估计"跳过这个块，注意力输出会变多少"，用观测的 QK + 当前 V（投影到 32 维）'],
    ['阈值、−ln2', '风险低于阈值就跳；−ln2 = 阈值再降 ln2，允许的输出变化减半，更保守、跳得更少'],
    ['风险 top-k XX%', '不用阈值，按风险排序只保留最高的 (1 − XX%)，稀疏度固定；排序很小，每步成本与 SparseD 相同'],
], [2.3, 10.2], size=13)

# ---------------------------------------------------------------- 2 main result
s = slide('长上下文：64K 端到端快约 13%，精度不降',
          'LongBench-v2 自然长度 32K 档（28–40K token）/ 64K 档（56–76K token）· 各 12 题 × 3 seed = 每档 36 格/方法 · '
          'dense = FA4 · 同格同机配对 · 生成前冻结',
          """
【数据】LongBench-v2 官方题（固定版本），按 NeMo 评测模板渲染、thinking 开启后的真实 token 数分档，不截断、不预筛长度：32K 档 = [28000, 40000) token，64K 档 = [56000, 76000) token。每档按 sha256(id) 排序取前 24 题，其中前 12 题为正式题（本页），另 12 题为开发题（只用于调参，不评分）。评分：NeMo LongBench 多选评分器（四选一）。
【设置】seed 101/202/303；每格只取第一次输出；两块 H100（dllm、mpk），同一格的 8 个方法在同一块 GPU 上；每个方法计时前先做两个 prompt 的预热；底座 piecewise_v3。协议 v27_final_lb_f1（生成前冻结）。
【读表】W = 请求墙钟时间比（含 prefill），S = 去掉 prefill 后的解码时间比，均为方法 / dense，< 1 更快；方括号为 95% 置信区间（按题目聚类 bootstrap）。答对 = strict correct 的格数。
【结论细节】
- M3 R6 DP −ln2：64K 请求 0.869 [0.805, 0.933]，解码 0.783；答对 23 对 18（+5/−0，p = 0.06，只说"不降"，不宣称提升）。B：0.864 [0.825, 0.905]。
- 32K：最好 0.902，置信区间跨过 1，不显著。
- 原版 M1/M2c/M3 在两档都更慢（1.12–1.49 倍；原版 M3 在 32K/64K 上置信区间刚好跨过 1）。原因见下一页。
- 每个 canvas 的平均步数：M3 R6 DP −ln2 为 dense 的 0.95，B 为 1.04。
- 加 32K 长度门限（key 数 < 32768 时直接走 dense）后，10–19K 的 LongBench 标准池与 dense 完全一致（同样的 token），额外开销约 0.5%；64K 的收益全部保留。
【复现】64K 速度复现了 5 次：阈值扫描 0.868、变体面板 0.872、最终面板 0.869、SparseD 对比面板 0.865、新机器 dlm2 上 0.821。前 4 次在同一批机器上生成的 token 完全相同，所以答对数也相同，这不是精度的独立复现；dlm2 上 token 不同，答对 18 对 21，在噪声内（跨机器同一方法的答对数约差 ±3/36）。
【出处】final_panel_f1/（summary_lb.csv、cells_lb.csv、summary_lb_sensitivity.csv）。
""")
table(s, 0.5, 1.6, 12.3, [
    ['方法', '64K 请求 W [95% CI]', '64K 解码 S', '64K 答对（dense 18）', '32K 请求 W [95% CI]', '32K 答对（dense 28）'],
    ['原版 M1 / M2c / M3', '1.27 / 1.49 / 1.12', '1.41 / 1.79 / 1.18', '20 / 16 / 21', '1.33 / 1.33 / 1.15', '25 / 27 / 29'],
    ['B', '0.864 [0.825, 0.905]', '0.788', '20', '0.935 [0.862, 1.016]', '31'],
    ['M3 R6 DP −ln2', '0.869 [0.805, 0.933]', '0.783', '23', '0.902 [0.780, 1.015]', '31'],
], [2.6, 2.2, 1.4, 1.9, 2.2, 1.9], size=13, bold_rows=(3,))
text(s, 0.5, 4.0, 12.3, 2.5, [
    '64K：请求时间减少约 13%，去掉 prefill 后解码减少约 22%，精度不降；速度在 5 个面板里复现。',
    '32K：不显著。原版 M1/M2c/M3 在两档都比 dense 慢（选块开销大于省下的注意力）。',
    '加 32K 长度门限后，短上下文与 dense 完全一致，可直接部署。'], size=15, bullet=True)

# ---------------------------------------------------------------- 3 three levels
s = slide('为什么端到端只有约 1.15 倍：加速在三层上怎么缩小',
          '64K · 同一个 FA4 kernel · kernel 级（单次 GLOBAL 调用）→ 注意力模块级（真实请求中逐次计时）→ 端到端（最终面板）',
          """
【kernel 级】scripts/v27_kernel_bench.py，kernel_bench/kernel_bench.jsonl。mpk H100，torch 2.12.1，FA4 = vLLM flash-attn cute 构建。形状：一次 GLOBAL 调用，16 个 query head × 256 个 canvas query，2 个 KV head，head_dim 512，bf16，key 数 = 上下文 + 256，双向无 mask。随机保留块（canvas 块始终保留），CUDA events，10 次预热后取 50 次中位数；每个长度都核对了全保留与 FA4 dense 逐位一致。64K、保留 10% 时快 7.3 倍；8K→128K 时从 3.3 倍增到 8.3 倍。
【注意力模块级：64K 的 prompt 来自哪里】scripts/v27_observe_profile.py，kernel_bench/module_profile_64k.jsonl。在变体面板（协议 v27_long_lb_variants_v5）上，各跑 LongBench-v2 32K 档和 64K 档正式题里的第 1 个请求：32K 为 34,879 个 prompt token，64K 为 71,516 个 prompt token，都是官方题按自然长度分档、不截断。用 CUDA events 包住每一次 GLOBAL 注意力调用及其各部分（dense 首步、融合观测、选块决策、稀疏调用）。每层调用的平均耗时：B 0.68 ms，M3 R6 DP −ln2 1.09 ms，同一请求里 FA4 dense 调用的中位数 2.95 ms。B 和 M3 的观测选块在副流上执行，若按串行计入作为上界，B 为 3.3 倍。
【端到端】取自最终面板（上一页）：请求 W 与解码 S 的倒数。
【14% 与 13% 怎么对上】14% 是 GLOBAL 注意力占整条 dense 请求的比例（单个 64K 请求的分解），但它占一次 decoder forward 的 35%（44.2 ms 里 15.4 ms）。实测：①每步解码成本降到 0.846（阈值扫描，−ln2），即每步省 15%；②每个 canvas 的去噪步数降到 0.952 [0.918, 0.991]，少约 5%，每少一步就省一整次 forward（不只是注意力）；③两者相乘，解码时间降到 0.783（少 22%）；④prefill 不变，解码约占 dense 请求的 57%，0.57 × 22% ≈ 12.5%，与实测请求少 13%（0.869）一致。注意：步数减少是实测的平均效应（B 反而多 4%），不能保证每次都有；只看注意力本身，能省的是 35% × 约 0.6 的每步时间。
【时间构成】substrate/time_breakdown.json：一个 LongBench-v2 64K 档请求（74,999 个 prompt token），dense，底座 piecewise_v2，总 10.2 s：prefill 43%，decoder forward 43%（其中 GLOBAL 注意力 14%），其余为采样、停止判断、canvas 追加与主机开销。一次 forward 里 MoE 专家矩阵乘约占 40%。
【原版为什么在模块层就慢】逐块顺序选块每次 2.3–7.9 ms/层，比 dense 注意力本身（1.5–3 ms/层）还贵；每 8 步还要重算一次完整 QK（6–11 ms/层）。我们的配置去掉了这些开销：每 canvas 一次融合观测、DP 并行决策（0.27 ms/层）、副流异步、决策之间沿用。
【剩余空间】M3 R6 DP −ln2 的一条 64K 请求约 7.8 s，其中 prefill 约 4.35 s（没有改动），剩下的 GLOBAL 注意力只有约 0.34 s：即使把注意力全部省掉，请求最多再快约 4%。
【口径】论文里三层都要报；只报 kernel 会把端到端效果夸大约 6 倍。
""")
s.shapes.add_picture(str(HERE / 'fig_three_levels.png'), Inches(0.5), Inches(1.5), width=Inches(8.4))
text(s, 9.1, 1.6, 3.9, 5.6, [
    'kernel 快 7.3 倍 → 注意力模块 2.7–4.4 倍 → 端到端请求约 1.15 倍（时间少 13%）。',
    '注意力占 dense 请求 14%，为什么能少 13%？它占一次 forward 的 35%：每步省 15%，去噪步数又少约 5% → 解码少 22%；解码占请求 57% → 请求少约 13%。prefill（43%）没有改动。',
    '原版在模块层就比 dense 慢：逐块选块比注意力本身还贵。'], size=13, bullet=True)

# ---------------------------------------------------------------- 4 SparseD
s = slide('与最接近的已有工作 SparseD 正面对比：速度上撞车',
          'LongBench-v2 32K/64K（与上页同题）× 3 seed = 每档 36 格/方法 · SparseD（ICLR 2026）移植到同一个 FA4 kernel 与底座 · 504 次运行',
          """
【SparseD 原论文】扩散 LLM 的稀疏注意力：前 20% 的去噪步用完整注意力；之后按块平均池化的注意力分数一次性选出每个 head 的稀疏模式（长上下文保留 30%、块 128），之后一直复用；FlexAttention 实现，LLaDA-1.5 / Dream-7B，固定 32–1024 步。
【移植是否正确】选块算法与论文描述一致（v20_controls.sparsed_bitmap，有单元测试）：在观测步上算完整注意力概率，按每个 128×64 块平均池化，每个（head, query 块）保留分数最高的 XX% 的 prefix 块；观测步的输出仍是 dense，下一步起才用新模式，之后沿用到 canvas 结束。收据核对：每个 canvas 每层观测一次，其余步沿用，保留比例为 0.1/0.2/0.3；504 条请求全部成功，计时中没有捕获新图。
【与原实现的差异（已标明是移植）】①块用 FA4 的 128×64 tile（原论文块 128）；②canvas 自己的 key 块始终保留（原论文对生成 token 的 key 也按同一比例选；64K 下只占约 0.4%）；③原论文的"前 20% 步"是固定步数下的比例，我们的步数是自适应的，所以给了两种：s10 = 48 步上限的 20%，s1 = 前 1 步（与 B 对齐，每个 canvas 实际是 2 步 dense）；④kernel 用 FA4 而不是 FlexAttention。
【工程优化是否公平】选块方法相同，但工程上 SparseD 吃亏：我们的观测与 dense 计算融合、选块在副流异步执行；SparseD 移植在观测步上实际算出完整 QK 分数矩阵，再 softmax、池化、排序，全部同步在主流上，另外再算一次 dense 输出。这部分每个 canvas 每层只做一次。实测每步成本"我们 / SparseD 保留 10%"为 0.993 [0.979, 1.007]，几乎没有差别。这个不对等只会让 SparseD 显得更慢，也就是偏向我们；我们仍然没有显著快过它，所以"速度撞车"的结论只会更稳。
【稀疏度可比】我们的 M3 R6 DP −ln2 在 64K 开发题上约保留 12% 的块，与 SparseD 保留 10–20% 的两档相当。
【判断标准（事先定的）】同等速度下精度不优于 SparseD、同等精度下也不更快，就按撞车处理。速度：撞车。精度：64K 上我们 23、SparseD 20–21、dense 18；32K 上 SparseD 保留 20% 与论文默认都是 24（dense 28，+4/−8、+1/−5，不显著）。都在噪声内。
【SparseD 论文默认在自适应停止下失效】前 10 步 dense，而一个 canvas 通常只走 10–12 步，所以一条请求的 330 次调用里有 300 次是 dense，请求时间 1.005。这只是"已有方法在自适应停止下需要调参"的观察，不是我们的新方法。
【运行说明】dllm 那一半因早先误杀的 worker 留下没有结束标记的启动记录（没跑出任何一格），续跑保护拒绝重跑；用同一冻结协议、同一部署在新目录重跑后与 mpk 那一半合并评分。协议 v27_long_lb_sparsed_c1；数据 sparsed_panel_c1/。
""")
table(s, 0.5, 1.6, 12.3, [
    ['方法', '64K 请求 W [95% CI]', '64K 每步成本', '64K 步数', '64K 答对（dense 18）', '32K 请求 W'],
    ['M3 R6 DP −ln2', '0.865 [0.804, 0.929]', '0.836', '0.940', '23', '0.912'],
    ['B', '0.860 [0.819, 0.903]', '0.795', '0.991', '20', '0.931'],
    ['SparseD 保留 10%，前 1 步 dense', '0.905 [0.816, 0.997]', '0.842', '1.017', '21', '0.916'],
    ['SparseD 保留 20%，前 1 步 dense', '0.926 [0.816, 1.036]', '0.864', '1.041', '21', '1.018'],
    ['SparseD 论文默认（保留 30%，前 10 步 dense）', '1.005 [0.897, 1.096]', '1.000', '1.032', '18', '1.025'],
], [3.6, 2.2, 1.5, 1.2, 2.0, 1.6], size=12)
text(s, 0.5, 4.3, 12.3, 3.0, [
    '直接配对（我们 / SparseD 保留 10%）：请求 0.957 [0.872, 1.072]，每步成本 0.993：没有显著差别，速度上按撞车处理。',
    '公平性：移植的选块算法与论文一致，但 SparseD 没有用融合观测和异步选块，对它不利。即便如此我们也没有显著更快，所以"撞车"成立；但不能据此说我们超过了 SparseD，要声称超过必须先给它同样的工程优化再比。',
    'SparseD 论文默认的"前 20% 步 dense"在自适应停止下基本等于 dense（每个 canvas 只走 10–12 步）。'],
    size=14, bullet=True)

# ---------------------------------------------------------------- 5 AIME
s = slide('AIME：稀疏度对精度和速度的影响',
          'AIME26 全部 30 题 · 目标稀疏度 30/40/50%（3 seed = 90 格）与 70/80%（2 seed = 60 格）· 风险 top-k vs SparseD 移植 · dense = FA4 · 同格同机配对',
          """
【数字】（相对同一面板的 dense；实际稀疏度 / 答对 / 请求 W / 去噪步数）
- 我们（风险 top-k）：24% / 49 对 49（90 格）/ 1.076 / 1.087；32% / 48 / 1.086 / 1.106；41% / 48 / 1.062 / 1.078；57% / 28 对 34（60 格）/ 1.196 / 1.231；66% / 23 对 34（+3/−14，p ≈ 0.013）/ 1.206 / 1.269。
- SparseD 移植：23% / 48 / 0.988 / 0.978；32% / 47 / 1.034 / 1.045；40% / 47 / 1.021 / 1.025；56% / 29 / 1.081 / 1.107；65% / 29 / 1.075 / 1.094。
- 每步成本（摊销）两者都在 0.95–1.01。
【定义】实际稀疏度按构造计算：被跳过的 GLOBAL key 块占全部 GLOBAL 调用的比例；canvas 块总保留、每个 canvas 第一步 dense，所以低于目标值。两种选块的实际稀疏度对齐，是同等稀疏度下的比较。风险 top-k 见"方法与实验设置"页：原版和阈值版没有目标稀疏度旋钮（阈值版在 AIME 上只稀疏 2–12%），所以加了按风险排序取 top-k 的变体。
【为什么 AIME 没有提速空间（前三条实测）】①8K key 时 FA4 dense 一次 GLOBAL 调用 0.44 ms，保留 50% 只快 1.7 倍；②在 CUDA graph 底座上用真实状态重放整步、块选择直接给定（不计选块成本）：17K key（AIME 最长上下文的 2 倍）时保留 10% 每步也只省 9%；③本页实测每步成本 0.95–1.01，而步数随稀疏度增加 8–27%；④对照：64K 上每步省约 16%，才换来约 13% 的请求加速。AIME 提示约 80–380 token、生成最多 8192 token，GLOBAL 的 key 数最多约 8.4K。
【最终 AIME 面板】（120 格，选定配置 M3 R6 DP −ln2，阈值版）：答对 65 对 65；请求 1.008 [0.957, 1.063]（剔除计时中捕获新图的配对；早先汇报的 0.971 被 dense 的两条异常请求拉低，已更正）。
【与同学和上周的结果】我们上周在 HF 原生 eager 路径（SDPA、无 CUDA graph、含 LOCAL 跳块）上测到 AIME 1.10–1.22×，换成 FA4 + CUDA graph 后消失。同学 9/30 的 AIME 结果（value_aware 18.4 s/task、37/90；dense 19.6 s/task、44/90）在他们自己的执行栈上（JAX/FA3，生成上限 2048 token），PPT 没写 dense 用的注意力 kernel，两边不能直接比，需要向同学确认。
【负结果】高稀疏度下风险 top-k 不如 SparseD，原因还没分析（猜测：每 6 步才更新一次的风险估计在高稀疏度下不够准，待查）。
【收据核对】topk 各档每 6 步做一次 DP 选块，每次选块都生成 FA4 块表，每个 canvas 第一步 dense；SparseD 每个 canvas 每层观测一次；每个方法最多 2 条请求在计时中捕获新图，已从速度比中剔除。协议 v27_aime_sparsity_t1、v27_aime_sparsity_hi；数据 aime_sparsity_panel_t1/、aime_sparsity_panel_hi/。
""")
table(s, 0.4, 1.55, 12.5, [
    ['实际稀疏度', '答对：风险 top-k', '答对：SparseD', '答对：dense', '请求时间：风险 top-k', '请求时间：SparseD', '步数：风险 top-k', '步数：SparseD'],
    ['约 23–24%', '49', '48', '49 / 90', '1.08', '0.99', '1.09', '0.98'],
    ['约 32%', '48', '47', '49 / 90', '1.09', '1.03', '1.11', '1.05'],
    ['约 40–41%', '48', '47', '49 / 90', '1.06', '1.02', '1.08', '1.03'],
    ['约 56–57%', '28', '29', '34 / 60', '1.20', '1.08', '1.23', '1.11'],
    ['约 65–66%', '23（显著掉分）', '29', '34 / 60', '1.21', '1.08', '1.27', '1.09'],
], [1.5, 1.7, 1.4, 1.2, 1.9, 1.7, 1.6, 1.5], size=12, bold_rows=(5,))
text(s, 0.4, 4.25, 12.5, 3.0, [
    '读法：请求时间、步数都是相对同一面板的 dense（1 = 一样，> 1 = 更慢/更多）。两种选块每步的计算成本都只有 dense 的 0.95–1.01。',
    '精度：实际稀疏度 41% 以下两种选块都不掉分；66% 时我们的选块显著掉分（23 对 34，p ≈ 0.013），SparseD 只降到 29。',
    '速度：没有一档变快。每步几乎省不下时间（原因见下一页），而稀疏度越高去噪步数越多，所以都变慢；我们的选块多出的步数更多。'],
    size=14, bullet=True)

# ---------------------------------------------------------------- 5b AIME time composition
s = slide('AIME 为什么没有提速空间：一步的时间花在哪里',
          'AIME 上下文最多约 8.4K key（多数步 2–6K）· 估算 = 32K 实测的每步时间构成，只把与上下文有关的部分换成 AIME 长度的实测 kernel 时间',
          """
【方法】一次去噪步里，MoE、矩阵乘、LOCAL 注意力（窗口 1024）、采样器的成本只取决于 canvas 的 256 个 token，与上下文长度无关；只有 GLOBAL 注意力随上下文变。所以用 32K 请求实测的每步构成（substrate/time_breakdown.json：profiler 的 kernel 时间表 C 部分 + 整步重放 D 部分，底座 piecewise_v2），去掉 v3 已删除的 KV 拼接，再把 GLOBAL 注意力换成 AIME 长度下的 kernel 实测时间（kernel_bench：8K key 0.44 ms/层，约 4K key 时约 0.28 ms/层，5 层合计 1.4–2.2 ms）。这是估算，不是在 AIME 上直接测的。
【数字】32K 实测（ms/步）：MoE 专家矩阵乘 12.8 + MoE 路由 1.2、其他矩阵乘（投影、dense MLP、router、lm_head）3.5、norm/逐元素/拷贝/softmax 3.2、未归类 1.5、LOCAL 注意力 0.8、GLOBAL 注意力 7.2、KV 拼接 2.2（v3 已去掉）、采样与停止判断等 forward 之外的时间 4.6。AIME 估算：GLOBAL 换成 1.4 ms，去掉拼接，合计约 29 ms/步。
【实测对照】AIME 面板里每步的摊销时间是 36–94 ms（按题目不同，所有方法一起变），比 29 ms 还多，多出的部分与注意力无关，所以 GLOBAL 的实际占比更低（≤ 4–5%）。
【能省多少】保留 50% 时 FA4 在 8K 只快 1.7 倍，即省掉 GLOBAL 时间的约 40%：约 0.6 ms/步，约占一步的 2%。再减去观测和选块的开销，几乎不剩；实测每步成本 0.95–1.01，与此一致。
【对照 64K】一次 forward 44.2 ms 里 GLOBAL 注意力 15.4 ms（35%），所以那里能省下 15% 的每步时间。
【上周为什么有 AIME 提升】上周的数字（AIME 生成 1.10–1.22×）在三个条件下测得：①dense 是 HF 原生 eager 路径——注意力用 PyTorch SDPA（head_dim 512 下 8K 时每层 2.56 ms，是 FA4 的 5.8 倍），且没有 CUDA graph；②FIXED16：每个 canvas 固定 16 步，没有"稀疏导致步数增加"的代价；③包含 LOCAL 跳块，在 eager 路径上有收益，在 CUDA graph 底座上 LOCAL 合计只有约 0.8 ms，跳了也省不下。换成 FA4 + CUDA graph + 原生自适应停止后，这个提升消失。结论：AIME 上的加速取决于基线；用最强的官方基线时没有空间。
【同学的结果】同学 9/30 的 AIME 结果在他们自己的执行栈上（JAX/FA3，生成上限 2048 token），PPT 没写 dense 用的注意力 kernel，不能直接与我们比较，需要向同学确认。
""")
table(s, 0.4, 1.55, 7.6, [
    ['一步里的部分', '32K 实测（ms）', 'AIME 估算（ms）', 'AIME 占比'],
    ['MoE（专家矩阵乘 + 路由）', '14.0', '14.0', '约 48%'],
    ['其他矩阵乘（投影、MLP、lm_head 等）', '3.5', '3.5', '约 12%'],
    ['采样、停止判断等 forward 之外', '4.6', '4.6', '约 16%'],
    ['norm、逐元素、拷贝等', '4.7', '4.7', '约 16%'],
    ['LOCAL 注意力（25 层）', '0.8', '0.8', '约 3%'],
    ['GLOBAL 注意力（5 层，可稀疏的部分）', '7.2', '1.4–2.2', '约 5–7%'],
    ['合计', '34.8（另有 KV 拼接 2.2，v3 已去掉）', '约 29–30', '100%'],
], [3.4, 1.5, 1.5, 1.2], size=12, bold_rows=(6,))
text(s, 8.3, 1.55, 4.7, 5.6, [
    '可稀疏的只有 GLOBAL 注意力，在 AIME 上只占一步约 5–7%（估算）；实测每步 36–94 ms，实际占比更低。',
    '保留 50% 时只省得下约一步的 2%，还要减去选块开销：所以每步几乎不变。',
    '上周的 AIME 提升（1.10–1.22×）的基线不同：HF eager + SDPA（注意力慢 5.8 倍、无 CUDA graph）、固定 16 步、含 LOCAL 跳块。'], size=13, bullet=True)

# ---------------------------------------------------------------- 6 M2 + longer context
s = slide('补充：M2 的变体与更长的上下文',
          'M2 变体：LongBench-v2 32K/64K × 3 seed，新机器 dlm2 · 96K：自然长度 84–104K 的 12 题 × 2 seed · 128K：112–144K · 单卡 H100 80GB',
          """
【M2 变体】M2c R6 DP −ln2 = 均值 V 用 M2c 的 pooled-compact 近似，其余与 M3 R6 DP −ln2 完全相同（每 6 步决策、每 canvas 一次融合观测、dense 前缀风险、副流异步、阈值 −ln2）。收据核对：pooled 均值构建次数等于 DP 选块次数，M3 那一组为 0；计时中没有捕获新图。64K：M2c R6 DP −ln2 0.847 [0.744, 0.937]，同面板 M3 0.821 [0.729, 0.902]，原版 M2c 1.439（更慢）。32K 都不显著；精度都在噪声内（64K 答对 19 / 18 对 dense 21）。协议 v27_long_lb_m2_c1，数据 m2_panel_c1/。
【96K】84–104K 的 12 题里，提示 ≤ 95,074 token 的 6 题所有方法都能跑；≥ 98,282 token 的 6 题连 dense 都在预填充时 OOM。能跑的 12 格：M3 R6 DP −ln2 请求 0.948 [0.863, 1.024]（不显著），每步成本 0.818 [0.715, 0.981]，但步数多 18%；B 1.042；原版 M1/M2c/M3 1.28/1.57/1.36。只有 12 格，只作参考。协议 v27_long_lb96k_c2，数据 lb96k_panel_c2/。
【128K】第一题上所有方法（包括 dense）都在单次预填充时 OOM（已分配 76.7 GiB 时再申请 1.38 GiB 失败），之后停止。更长的上下文需要分块预填充或多卡，对 dense 和稀疏都一样。
""")
table(s, 0.5, 1.6, 12.3, [
    ['实验', '方法', '请求 W [95% CI]', '说明'],
    ['M2 变体，64K', 'M2c R6 DP −ln2', '0.847 [0.744, 0.937]', '原版 M2c 为 1.44（更慢）'],
    ['M2 变体，64K', 'M3 R6 DP −ln2（同面板）', '0.821 [0.729, 0.902]', '与 M2c 在置信区间内重叠'],
    ['96K（能跑的 12 格）', 'M3 R6 DP −ln2', '0.948 [0.863, 1.024]', '每步成本 0.818，但步数 +18%'],
    ['128K', '所有方法，包括 dense', '—', '单卡预填充 OOM'],
], [2.6, 3.2, 2.6, 3.9], size=13)
text(s, 0.5, 4.0, 12.3, 2.5, [
    'M2 放进同样的配置后，64K 上同样有端到端提速。',
    '单卡 H100 上 dense 能跑的上限约 95–98K token，所以端到端最优目前停在 64K。'], size=15, bullet=True)

# ---------------------------------------------------------------- 7 conclusion
s = slide('结论与下一步', None, """
能说的结论有证据支持；要如实说的三条都在前面各页有数据。下一步第 1 条是高稀疏度下的负结果；第 3 条涉及同学的工作，属于合作内容，需要商量。详细进度与名词表见 progress_20260930.md；逐条生成记录（脱敏）在 generation_records_v27_0930/。
""")
text(s, 0.5, 1.4, 12.3, 5.5, [
    '能说的：在最快的官方 dense 基线（FA4）上，64K 端到端快约 13%，精度不降；原版 M1/M2c/M3 都更慢。',
    '要如实说的：速度上与 SparseD 撞车；AIME 上任何注意力稀疏都没有提速空间；高稀疏度下我们的选块不如 SparseD。',
    '下一步 1：查清高稀疏度下风险 top-k 为什么不如 SparseD。',
    '下一步 2：更长上下文要先解决单卡预填充的显存（分块预填充或多卡）。',
    '下一步 3：与同学的 query 自适应保护能否结合（合作内容）。'], size=18, bullet=True)

out = HERE / 'dlm_sparse_attention_20260930_v2.pptx'
prs.save(out)
print('saved', out, len(prs.slides), 'slides')
