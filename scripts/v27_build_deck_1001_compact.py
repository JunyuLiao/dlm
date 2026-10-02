"""Build the compact 2026-10-01 group-meeting deck (cover + 5 pages): baseline, 32K/64K variants, 96K and the
step-count check, AIME cost structure, and the V-term controls. Panel ratios are read from the pushed summary.csv
files (not hand-copied); static numbers (kernel timings, time breakdown, step-count check) cite their files in the
speaker notes.

Writes results/m1_m2_m3_frontier_v27_20260929/ppt_sample/dlm_sparse_attention_20261001_compact_v6.pptx.
"""
import csv
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

R = Path(__file__).resolve().parents[1] / 'results/m1_m2_m3_frontier_v27_20260929'
HERE = R / 'ppt_sample'
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
        p.space_after = Pt(3)
        for r in p.runs:
            r.font.size, r.font.bold, r.font.name = Pt(size), bold, FONT
            if color is not None:
                r.font.color.rgb = color
    return box


def table(slide, x, y, w, rows, widths, size=11, bold_rows=()):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(0.3 * len(rows)))
    t = shape.table
    for j, cw in enumerate(widths):
        t.columns[j].width = Emu(int(Inches(w) * cw / sum(widths)))
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(0.3)
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size, r.font.name = Pt(size), FONT
                    r.font.bold = i == 0 or i in bold_rows
    return t


def rows_of(panel):
    p = R / panel / 'summary.csv'
    return list(csv.DictReader(p.open(encoding='utf-8'))) if p.exists() else []


def get(panel, arm, dataset=None):
    for r in rows_of(panel):
        if r['arm'] == arm and (dataset is None or r['dataset'] == dataset):
            return r
    return None


def ci(r, k):
    return r[k + '_ci'].replace(',', ', ')


def f3(x):
    return f'{float(x):.3f}'


def cols(r):
    """W [CI], S, per-step S/N, N, correct -- read from the pushed summary.csv."""
    if r is None:
        return ['待出', '', '', '', '']
    return [f'{f3(r["W"])} {ci(r, "W")}', f3(r['S']), f3(r['SN']), f'{float(r["N"]):.2f}', r['correct']]


MD = []


def md_table(rows):
    out = ['| ' + ' | '.join(c.replace('|', '/') for c in rows[0]) + ' |', '|' + '---|' * len(rows[0])]
    out += ['| ' + ' | '.join(c.replace('|', '/') for c in r) + ' |' for r in rows[1:]]
    return '\n'.join(out)


def page(title, config, blocks, notes):
    MD.append(f'## {title}\n\n**【正文顶部】** {config}\n')
    for kind, *args in blocks:
        if kind == 'table':
            MD.append(md_table(args[0]) + '\n')
        elif kind == 'label':
            MD.append(f'**{args[0]}**\n')
        else:
            MD.append('\n'.join('- ' + x for x in args[0]) + '\n')
    MD.append('**【Speaker notes】**\n\n' + notes.strip() + '\n\n---\n')
    s = prs.slides.add_slide(BLANK)
    text(s, 0.4, 0.2, 12.5, 0.6, [title], size=22, bold=True)
    text(s, 0.4, 0.8, 12.5, 0.6, [config], size=11, color=GRAY)
    y = 1.4
    for kind, *args in blocks:
        if kind == 'table':
            rows, widths, size, bold, *height = args
            table(s, 0.35, y, 12.6, rows, widths, size=size, bold_rows=bold)
            y += (height[0] if height else 0.3 * len(rows)) + 0.3
        elif kind == 'label':
            text(s, 0.35, y, 12.6, 0.35, [args[0]], size=12, bold=True)
            y += 0.35
        else:
            lines, size = args
            text(s, 0.35, y, 12.6, 0.3 * len(lines) + 0.3, lines, size=size, bullet=True)
            y += 0.3 * len(lines) + 0.2
    s.notes_slide.notes_text_frame.text = notes.strip()


D = 'D_fa4_allkept'
M3 = 'M3_R6_A64_fused_dp_async_m1ln2_fa4'
M3C = 'M3_R6_A64_fused_dp_async_m1ln2_c0_fa4'
M2C = 'M2c_R6_A64_fused_dp_async_m1ln2_c0_fa4'
B = 'B_A64_fused_rp_async_fa4'
BC = 'B_A64_fused_rp_async_c0_fa4'
L32, L64 = 'longbench_v2_32k', 'longbench_v2_64k'
WIDE = [3.7, 2.1, 0.72, 0.72, 0.6, 0.85, 2.1, 0.72, 0.72, 0.6, 0.85]


def cover():
    s = prs.slides.add_slide(BLANK)
    text(s, 0.8, 2.3, 11.5, 1.0, ['DiffusionGemma-26B-A4B 块稀疏注意力：基线、方法、长上下文结果、选块对照与 AIME 耗时'], size=30, bold=True)
    text(s, 0.8, 3.6, 11.5, 1.4, ['2026-10-01 · 所有比值 = 方法 / dense（FA4），< 1 表示更快',
                                  'W = 端到端（含 prefill）· S = 纯生成（不含 prefill）· 每步 = S/N（摊销）· N = forward 数 · 答对 = strict correct 格数',
                                  '95% CI 按题目聚类 bootstrap · 每格 = 一个（题目, seed），同格所有方法同机'], size=15, color=GRAY)
    s.notes_slide.notes_text_frame.text = ('表中面板数字由 scripts/v27_build_deck_1001_compact.py 直接读取已推送的 summary.csv；'
                                           '仓库 coconight01/dlm_test，分支 research/m3-output-numerics-20260927，'
                                           '目录 results/m1_m2_m3_frontier_v27_20260929/。')


def p1_baseline():
    page('1. Dense 基线：FA4 块稀疏接口、全部块保留（官方实现中最快，与 FA4 dense 逐位相同）',
         'GLOBAL 层真实几何：256 个 canvas query、16 Q 头 / 2 KV 头、head_dim 512、双向、bf16、单张 H100；同一进程 CUDA event 中位数（ms / 每层调用）',
         [('table', [['实现（官方 / 大型开源）', '16,640 keys', '32,768', '65,536', '说明'],
                     ['FA4 块稀疏接口，全部块保留（采用）', '0.756', '1.391', '2.672', '最快；与 FA4 dense 逐位相同'],
                     ['FA4 dense（vLLM 官方 fork，CuTe DSL）', '0.782', '1.462', '2.851', 'DiffusionGemma 技术报告使用的 kernel'],
                     ['FlashInfer 0.6.18（FA2 后端）', '0.784', '1.467', '2.891', '与 FA4 持平'],
                     ['HF 默认路径：K/V 复制 8 倍 + SDPA mem-efficient', '1.64', '3.10', '6.16', '慢约 2.1 倍'],
                     ['FlexAttention（最佳 tile）', '–', '6.18', '20.8', '慢 4–7 倍'],
                     ['FA2 / FA3 / 上游 FA4 / cuDNN / SDPA flash', '–', '–', '–', 'SM90 上不支持 head_dim 512']],
           [4.4, 1.3, 1.3, 1.3, 4.3], 12, (1,)),
          ('bullets', ['dense 与所有稀疏方法用同一个 kernel：选块结果作为 keep map 传给 FA4 官方块稀疏接口，dense 与稀疏只差被跳过的块，收益不来自换代码路径。',
                       '执行底座也相同：decoder 整体 torch.compile + CUDA graph（在 GLOBAL 注意力处切分，vLLM 方式）；每个方法计时前单独预热，计时中不捕获新图。',
                       'FA4 块稀疏的执行时间几乎与保留块数成正比：64K 保留 10% 时 0.40 ms（dense 的 0.14 倍）。'], 13)],
         """
【来源】official_baseline/README.md（dense_sota_same_process.json：torch 2.13，同一进程，每个配置都与 FP32 对照）。
【出处】FA4 = vLLM 官方 flash-attention fork 中的 FlashAttention-4（CuTe DSL，BSD-3），53 个文件与官方 nightly wheel 的 RECORD 哈希逐字节一致。SM90 上 head_dim ≤ 512 只有 vLLM fork 支持（Dao-AILab 上游 SM90 上限 256），论文中需注明。
【为什么用全保留接口】比 FA4 dense 快 4–6%、输出逐位相同；另一条 dense 基线（FA4 plain dense）也保留在实验记录里。
【弃用的基线】我们自己的 64 行 Triton dense（慢 1.9 倍）、HF 默认路径（慢 2.1 倍）；早期相对它们的比值不再引用。
""")


def p_methods():
    page('2. 方法与变体：每个方法怎么选块，优化版具体改了什么',
         '只稀疏 5 个 GLOBAL 层在 canvas 上的注意力；25 个 LOCAL 层（滑窗 1024）、prompt prefill、canvas 追加都保持 dense · '
         '块 = 128 个 query × 64 个 key · 当前 canvas 的块和第一个块永远保留 · 输出只在保留的块上用当前 QK 和原始 V 精确计算（FA4），跳过的块完全不算',
         [('table', [['方法', '怎么选块', '原版配置'],
                     ['共同的打分：跳过风险', '对每个 key 块估计“把它跳过，注意力输出会偏多少”≈ 该块分到的注意力占比 × ‖该块的 V 均值 − 其余块给出的输出‖ ÷ V 的参考尺度。'
                      '一个 query 块（128 行）里最坏的那行风险也低于阈值，才跳过。query 敏感度 T = 1 + 3 × 该位置 argmax 是否还在变化的滑动平均（1–4），还在变的位置保留更多块', '—'],
                     ['M1', '用上面的风险选块。注意力来自“观测”：隔几步完整算一次 QK，中间步沿用；V 取块内按注意力加权的均值（投影到 32 维以省计算）', '每 8 步观测一次（A8），每步都重新选块（R1）'],
                     ['M2', '同 M1，但 V 取块内 64 个 token 的简单平均（不按注意力加权）', 'A8，R1'],
                     ['M3', '同 M1，但隔几步才重新选块，中间步直接沿用上次选出的块', 'A8，每 3 步重新选（R3）'],
                     ['B', '每个 canvas 只选一次块（第 1 步观测后选），整个 canvas 都用这一份，不再重新选', '—'],
                     ['SparseD（外部论文，移植）', '前几步 dense；之后按块的平均注意力分数保留固定比例的块，用到 canvas 结束', '—']],
           [2.4, 8.0, 2.2], 11, ()),
          ('label', 'M3 优化版 = M3 + 下面 5 项改动（第 4 页起简称 M3 优化版）；carry0 可再叠加'),
          ('table', [['改动', '原版怎么做 → 优化版怎么做'],
                     ['观测并入第 1 步（A64）', '原版每 8 步额外完整算一次 QK 来得到各块注意力（每层 6–11 ms，比 dense 注意力还贵）→ 每个 canvas 只观测一次，而且在第 1 步本来就要做的 dense 注意力里顺带算出，不额外跑'],
                     ['风险预先算好（DP）', '原版选块时逐块按顺序累计算风险（每层 2.3–7.9 ms）→ 观测时就把每个块的风险算好存起来，选块只是“风险 < 阈值？”的并行比较（每层 0.27 ms）'],
                     ['每 6 步重新选块（R6）+ 异步', '原版每 3 步重新选 → 每 6 步一次；选块放在另一条 GPU 流上与主计算同时跑，不占主计算时间'],
                     ['阈值下调（−ln2）', '阈值是“跳不跳”的分界线；下调后风险要再小一半才跳，跳得更少、更保守，换来步数不增加'],
                     ['carry0', '每个 canvas 的第 0 步原本 dense → 直接用上一个 canvas 最后的块图（新进来的块全保留），每个 canvas 少一次 dense 步'],
                     ['试过、未采用', '短上下文门控（key < 2K 直接 dense，AIME 用）；第 2 步才观测；每 3 步选块；输出稳定后整段转 dense；块图跨 4 个 canvas 复用（步数增加、掉点）；保留全部生成部分']],
           [2.6, 10.0], 10, ())],
         """
【每个 canvas 的步骤】第 0 步 dense；第 1 步 dense + 观测；第 2 步起按块图稀疏。carry0 把第 0 步也变成稀疏。
【配置名对照】原版：M1_R1_A8、M2c_R1_A8、M3_R3_A8（均经 FA4 执行）。优化版：M3_R6_A64_fused_dp_async_m1ln2（A64 = 每个 canvas 观测一次；fused = 观测并入第 1 步；dp = dense-prefix 风险；async = 副流选块；m1ln2 = 阈值下调 ln2）。carry0 = carry_first。完整说明见 docs/RESEARCH_CONTEXT.md §3。
【M2 的块均值】在 32 维投影空间里取块内简单平均（pooled_compact）。junyu 代码里的 value/vector_mean 是“块内 V 均值向量本身的范数”（不减当前输出），与这里的 M2 不同。
【保护】永远保留当前 canvas 的全部块与第一个块（attention sink）；carry0 在第 0 步保留新进来的块。没有使用 junyu 的 prefix_end / diagonal 保护。
""")


def p_flow():
    page('3. 一个 canvas 内具体怎么算（以 M3 优化版的一层 GLOBAL 注意力为例）',
         '关键：选块用的是“观测”时记下的旧分数；真正的注意力输出每一步都用当前的 Q、K、V，只在保留的块上精确计算 · 一个 canvas 约 14–18 步 · prefix 的 K、V 在一个 canvas 内不变',
         [('table', [['步骤', '做什么', '算 QK？', '乘 V？', '开销'],
                     ['第 0 步', 'dense 全量计算。用 carry0 时改为：直接沿用上一个 canvas 最后的块图，新进入 prefix 的块全部保留', '全部 / 保留块', '同左', 'dense / 稀疏'],
                     ['第 1 步：dense + 观测', '完整算一遍注意力（输出精确），顺带记下每个 query 行 r、每个 key 块 j（64 个 key）的两样东西：'
                      'z = 这一块分到的注意力总量（log Σ exp(q·k)，即“过去的 QK”）；μ = 块内 V 按这一行注意力加权的平均（V 先随机投影到 32 维）。'
                      '随即建“风险表”：按块顺序扫，风险 = log‖α ·（μ − 前面各块给出的输出）‖，α = 这一块在前 j 块中的注意力占比', '全部块', '全部块', '观测不额外跑'],
                     ['第 2 步选块，用于第 2–7 步；第 8 步重选，用于 8–13；第 14 步…（每 6 步）', '选块只读风险表：每个头、每 128 个 query 行一组、每个块，取 128 行里最坏的（风险 − log V 参考尺度），小于阈值就跳过。重选时不重新观测 QK、不重建 prefix 的风险表（prefix 的 K、V 在 canvas 内不变），只重算：当前 canvas 部分 V 的 32 维投影、每个位置当前的 query 敏感度 T，再重新比较一次', '不算', '只投影 canvas 部分', '0.27 ms/层，另一条 GPU 流上并行'],
                     ['第 2 步起每一步：真正的注意力', 'FA4 块稀疏接口：保留的块用当前这一步的 Q、K、V 正常算 QK → softmax（只在保留块上归一化）→ 乘完整 512 维原始 V；'
                      '跳过的块 QK 和乘 V 都不算。当前 canvas 的块和第一个块永远保留', '只算保留块', '只乘保留块', '省时间的地方']],
           [2.7, 7.1, 0.9, 1.1, 1.4], 10, (), 2.95),
          ('bullets', ['原版的区别：观测每 8 步额外完整算一次 QK（每层 6–11 ms）；风险是和“已经决定保留的块”的输出比较，所以只能逐块顺序现算（每层 2.3–7.9 ms），M1 每步算、M3 每 3 步算。'
                       '优化版改为和“前面全部块”比较，每块风险互不依赖，观测时一次算好、选块时并行比较（与原版 M1 的选块结果不完全相同，所以算变体）。',
                       '重选会明显改变块图（实测：相对第一次选块，44–71% 的保留块发生变化，Jaccard 0.59–0.70），因为 T 每步随“哪些位置还在变”更新。下一个 canvas 在第 1 步用新的 query 和更长的 prefix 重新观测、重建风险表。',
                       '“看不看 V”只体现在风险里的 μ：投影 32/16/8/4 维、M2 的块内简单平均，或不用 μ（只按注意力占比排序）。输出永远用原始 V。',
                       '更正：此前写的“query 敏感度取 1、重选几乎不改块图”是错的；T 在所有实验中都启用（β = 3，γ = 0.5）。'], 11)],
         """
【代码】风险表：experiments/numerical_qk_reuse/v27_dense_prefix.py 的 _dp_build（按块顺序扫，lognorm = log‖α(μ − 前缀输出)‖）；选块：_dp_decide（worst = max_rows(lognorm − log ref + log T) < THRESHOLD）；T 来自 router.query_sensitivity，在 numerical_qk_reuse 的所有实验中未设置（= 1）。观测并入第 1 步：fused_observe；每 6 步：decision_interval = 6；副流：async_route；阈值：全局 log 阈值 −3.874（下调 ln2 后）。
【M2】μ 换成块内投影 V 的简单平均（每个 KV 头、每块一个向量，所有行共用），其余完全相同（pooled_compact）。
【原版】M1_R1_A8 / M2c_R1_A8 / M3_R3_A8：每 8 步观测一次（完整 QK 记分数），风险按“已保留块”的状态顺序扫描计算。
""")


def trip(r):
    """'W [CI]', 'S / 每步 / N', correct for one length."""
    if r is None:
        return ['—', '—', '—']
    return [f'{f3(r["W"])} {ci(r, "W")}', f'{float(r["S"]):.3f} / {float(r["SN"]):.3f} / {float(r["N"]):.2f}', r['correct']]


def p_long():
    e4, e5, p96 = 'lb_confirm_panel_e4', 'lb_overhead_panel_e5', 'lb96k_pooled_e6_e6b'
    d = [get(e5, D, L32), get(e5, D, L64), get(p96, D)]
    hdr = ['方法', '32K 端到端 W [95% CI]', 'S / 每步 / N', '答对', '64K 端到端 W [95% CI]', 'S / 每步 / N', '答对',
           '96K 端到端 W [95% CI]', 'S / 每步 / N', '答对']
    rows = [hdr, ['dense（FA4，基线）'] + sum([['1', '1 / 1 / 1', x['correct']] for x in d], [])]
    spec = [('原版 M1', 'M1_R1_A8_fa4', e4, 'M1_R1_A8_fa4'), ('原版 M2', 'M2c_R1_A8_fa4', e4, 'M2c_R1_A8_fa4'),
            ('原版 M3', 'M3_R3_A8_fa4', e4, 'M3_R3_A8_fa4'), ('B', B, e5, B), ('M3 优化版', M3, e5, M3),
            ('M3 优化版 + carry0（当前最好）', M3C, e5, M3C), ('B + carry0', BC, e5, None)]
    for name, arm, panel, arm96 in spec:
        rows.append([name] + trip(get(panel, arm, L32)) + trip(get(panel, arm, L64)) + trip(get(p96, arm96) if arm96 else None))
    page('4. 长上下文 32K / 64K / 96K：各方法的端到端、纯生成、每步耗时、forward 数与精度',
         'LongBench-v2 · 32K、64K 各 24 题 × 6 个 seed = 每档 144 格/方法；96K = 单卡放得下的全部 11 道题 × 6 个 seed = 66 格 · '
         '比值 = 方法 / dense，< 1 更快 · S = 纯生成，每步 = S/N，N = forward 数',
         [('table', rows, [2.0, 1.6, 1.6, 0.5, 1.6, 1.6, 0.5, 1.6, 1.6, 0.5], 9, (7,), 0.3 * len(rows) + 0.25),
          ('bullets', ['M3 优化版 + carry0：端到端 32K 快 9%、64K 快 15%、96K 快 18%；纯生成快 12% / 22% / 27%；上下文越长每步省得越多（每步 0.93 / 0.82 / 0.75）；精度都不低于 dense。prefill 没有优化（所有方法约 1.00）。',
                       '原版 M1/M2/M3 在三档都比 dense 慢：选块和观测本身（每层每次 2.3–7.9 ms）比 dense 注意力（1.5–2.9 ms）还贵。',
                       '步数 N 的核查：按 seed 拆开看，单个 seed 的步数比在 0.85–1.24 之间摆动；9 个 seed 合并后 32K +2.4%、64K +1.5%（不显著）。所以“步数没有显著增加，最佳估计 +1–2%”，端到端结论需要大样本；每步耗时的结论各批次一致。'], 11)],
         """
【协议】32K/64K：原版 M1/M2/M3 来自 E4 v27_lb_confirm_e4_d3a7edb87a2b800a；B、M3 优化版、carry0 两行来自 E5 v27_lb_overhead_e5_1c93b97271a3a266（与 E4 同一批 288 格，dense/M3/B 输出与 E4 逐 token 相同）。96K：E6 + E6b 合并（v27_lb96k_confirm_e6_8ab3aa35d1874ca1、v27_lb96k_extend_e6b_d1b1c0c74a1dc69d），B + carry0 在 96K 没有跑。
【实际稀疏度】主线（M3 优化版 −ln2）在稀疏调用里实际跳过的块：32K 约 79%、64K 约 88%（fidelity_v6 诊断）。
【步数核查】32K 同 24 题、同底座：seed 101/202/303（来自 E3）= 1.23/0.89/1.24，404–909（E4）= 0.85/0.98/1.12/0.99/1.00/1.00；84 种 3/6 分组中 13 种的组间差距不小于实际分组。64K 合并（traj_t1 v4 + E4 v5）1.015 [0.948, 1.085]。
【单次 forward】每步 = S/N 是摊销值；同一请求里单次 forward 的直接计时见第 5 页。
""")


def p_select():
    e8, e9, e10 = 'aime_vterm_panel_e8', 'lb64_vterm_panel_e9', 'lb_m2opt_panel_e10'

    def cell(panel, arm, key):
        r = get(panel, arm) if arm else None
        return '—' if r is None else (r['correct'] if key == 'acc' else f3(r['W']))
    e11 = 'lb64_vterm_hi_panel_e11'
    V = [('投影 V 32 维（M1/M3 的做法）', 'M3_R6_A64_fused_dp_async_topk70_fa4', 'M3_R6_A64_fused_dp_async_topk88_c0_fa4', 'M3_R6_A64_fused_dp_async_topk95_c0_fa4'),
         ('投影 V 16 维', 'M3_R6_A64_fused_dp_async_topk70_r16_fa4', None, None),
         ('投影 V 8 维', 'M3_R6_A64_fused_dp_async_topk70_r8_fa4', 'M3_R6_A64_fused_dp_async_topk88_r8_c0_fa4', None),
         ('投影 V 4 维', 'M3_R6_A64_fused_dp_async_topk70_r4_fa4', None, None),
         ('块内简单平均 V（M2 的做法）', 'M2c_R6_A64_fused_dp_async_topk70_fa4', 'M2c_R6_A64_fused_dp_async_topk88_c0_fa4', 'M2c_R6_A64_fused_dp_async_topk95_c0_fa4'),
         ('不看 V（只看注意力）', 'M3_R6_A64_fused_dp_async_topk70_mass_fa4', 'M3_R6_A64_fused_dp_async_topk88_mass_c0_fa4', 'M3_R6_A64_fused_dp_async_topk95_mass_c0_fa4'),
         ('SparseD 70%（外部论文，移植）', 'SparseD_s70_skip1_fa4', None, None)]
    tab = [['选块时 V 怎么用', f'64K 跳 88%：答对（dense {get(e9, D)["correct"]}）', '端到端 W',
            f'64K 跳 95%：答对（dense {get(e11, D)["correct"]}，预览）', '端到端 W',
            f'AIME 跳 70%：答对（dense {get(e8, D)["correct"]}）', '端到端 W']]
    for name, a8, a9, a11 in V:
        tab.append([name, cell(e9, a9, 'acc'), cell(e9, a9, 'w'), cell(e11, a11, 'acc'), cell(e11, a11, 'w'),
                    cell(e8, a8, 'acc'), cell(e8, a8, 'w')])
    d32, d64 = get(e10, D, L32), get(e10, D, L64)
    tab2 = [['当前工作点（阈值 −ln2 + carry0，其余优化全相同）', '32K 端到端 W [95% CI]', 'S / 每步 / N',
             f'答对({d32["correct"]})' if d32 else '答对', '64K 端到端 W [95% CI]', 'S / 每步 / N',
             f'答对({d64["correct"]})' if d64 else '答对']]
    for name, arm in [('M3 优化版（投影 V 32 维）', M3C), ('M2 同样优化（块内简单平均 V）', M2C)]:
        r32, r64 = get(e10, arm, L32), get(e10, arm, L64)
        tab2.append([name] + (trip(r32) if r32 else ['运行中', '', '']) + (trip(r64) if r64 else ['运行中', '', '']))
    page('5. 选块要不要看 V、M2 还是 M3：同稀疏度下没有可分辨的差别',
         '上表：固定稀疏度，只换风险里的 V 项。64K = M3 优化版 + carry0，跳 88%（= 主线工作点，96 格/组）与跳 95%（预览，48 格/组）；'
         'AIME = M3 优化版、固定保留 30%（主线阈值在 AIME 只跳约 4%），180 格/组 · 下表：主线阈值下 M2 与 M3 只差 V 的取法（32K/64K 各 24 题 × 4 seed）',
         [('table', tab, [3.4, 1.9, 1.0, 2.2, 1.0, 1.9, 1.0], 10, ()),
          ('table', tab2, [3.6, 1.9, 1.6, 0.8, 1.9, 1.6, 0.8], 10, ()),
          ('bullets', ['64K（工作点 88%）：任何 V 项都不比“不看 V”更准（50–54 vs 51，全部不显著），速度也无显著差别（不看 V / 32 维 0.960 [0.905, 1.018]）。',
                       '64K 跳到 95%（预览，48 格）：各组精度都不低于 dense，仍无 V 优势（不看 V / 32 维 1.004 [0.916, 1.099]）；每步 0.76，但步数略增，端到端约 0.89，不比 88% 工作点好。',
                       'AIME（强制 70%）：不同 V 项确实选出不同的块（答对 86–94），但最好的是 4 维或不看 V；此稀疏度下步数多 6–13%，都比 dense 慢。',
                       '主线阈值下 M2（简单平均）/ M3（投影 32 维）直接比：32K 0.987 [0.914, 1.063]、64K 0.983 [0.940, 1.024]，每步耗时相同，答对差异不显著——打平。',
                       '结论：“看 V 选块”和“M2 vs M3”都不能作为贡献，选块可以只看注意力。'], 11)],
         """
【协议】E9 v27_lb64_vterm_e9_5b125c63817898f6（576/576 成功；dense 与 −ln2 + carry0 两组与 E5 逐 token 相同）；E8 v27_aime_vterm_e8_bd293196ff730851（1,440/1,440 成功）；E10 v27_lb_m2opt_e10_c965792c1cb6542d（576 次运行，M2 与 M3 只差 mu_mode）。每次运行的配置记录都核对过。
【AIME 详细】对 dense 99：32 维 87（p = 0.08）、16 维 86（p = 0.035）、8 维 87（p = 0.036）、4 维 94、简单平均 92、不看 V 91、SparseD 94。
【64K 详细】对 dense 51：32 维 53、8 维 50、简单平均 54、不看 V 54；−ln2 版 49。
【实际稀疏度】AIME 固定保留 30% 的 prefix 块，算上 canvas 块和 dense 步约跳 57% 的 GLOBAL 块；64K 固定保留 12% ≈ 跳 88%。
【组内对照】junyu 在 RULER8K（当前 QK 选块，GLOBAL 和 LOCAL 都稀疏）75% 稀疏时 32 维 47.4% vs 只看注意力 21.3%；50% 时都约 92%。只引用。
""")


def tb_load(name):
    p = R / 'time_breakdown_v5' / f'{name}_dense_first.jsonl'
    if not p.exists():
        return None
    out = {}
    for line in p.open(encoding='utf-8'):
        if line.strip().startswith('{'):
            r = json.loads(line)
            out.setdefault(r['part'], []).append(r)
    return out


def tb_split(tb):
    """Compiled-forward kernel categories (ms) of one captured dense decoder call, plus step and sampler."""
    c, dpart = tb['C_compiled_forward_kernels'][0], tb['D_step_replay'][0]
    cat = dict(global_attn=0.0, local_attn=0.0, moe=0.0, route=0.0, gemm=0.0)
    for k in c['top']:
        n = k['kernel']
        if 'flash_attn' in n or 'FlashAttentionForward' in n:
            cat['global_attn'] += k['ms']
        elif 'sdpa' in n and 'cudnn' in n:
            cat['local_attn'] += k['ms']
        elif 'cutlass' in n and 'Gemm' in n:
            cat['moe'] += k['ms']
        elif 'radixSort' in n or 'gatherTopK' in n or 'bitonicSort' in n:
            cat['route'] += k['ms']
        elif 'nvjet' in n:
            cat['gemm'] += k['ms']
    fwd, step = dpart['forward_ms'], dpart['step_ms']
    cat['other'] = fwd - sum(cat.values())
    cat['sampler'] = dpart['step_minus_forward_ms']
    return step, fwd, cat, tb['B_forward_composition_eager_ms'][0].get('global_prefix_keys')


def p_aime():
    names = [('aime26', 'AIME'), ('longbench_v2_32k', 'LongBench 32K'), ('longbench_v2_64k', 'LongBench 64K')]
    data = {k: tb_load(k) for k, _ in names}
    split = {k: tb_split(v) if v else None for k, v in data.items()}

    def ms(k, key):
        sp = split[k]
        if sp is None:
            return '测量中'
        step, fwd, cat, _ = sp
        v = {'step': step, 'fwd': fwd}.get(key, cat.get(key))
        return f'{v:.1f} ms（{100 * v / step:.0f}%）' if key != 'step' else f'{v:.1f} ms（100%）'

    def keys(k):
        sp = split[k]
        return '测量中' if sp is None else f'{sp[3] + 256:,}'

    def fwd_pair(k):
        a = data[k]
        if not a:
            return '测量中'
        arms = {r['arm']: r['parts']['decoder_forward']['median_ms'] for r in a.get('A_request', [])}
        dv = arms.get(D)
        mv = next((v for kk, v in arms.items() if kk != D), None)
        return f'{dv:.1f} → {mv:.1f} ms（{mv / dv:.2f}）' if dv and mv else '—'
    best = get('aime_confirm_panel_e7', 'M3_R6_A64_fused_dp_async_m1ln2_c0_gate2k_fa4')
    rows = [['（dense，同一底座，各取一次真实 forward）'] + [n for _, n in names],
            ['这次 forward 的 key 数'] + [keys(k) for k, _ in names],
            ['单步总耗时'] + [ms(k, 'step') for k, _ in names],
            ['  5 层 GLOBAL 注意力'] + [ms(k, 'global_attn') for k, _ in names],
            ['  25 层 LOCAL 注意力（滑窗 1024）'] + [ms(k, 'local_attn') for k, _ in names],
            ['  MoE 专家矩阵乘（每步读约 46 GB 权重）'] + [ms(k, 'moe') for k, _ in names],
            ['  MoE 路由（排序 / top-k）'] + [ms(k, 'route') for k, _ in names],
            ['  其他矩阵乘（投影、dense MLP、lm_head 等）'] + [ms(k, 'gemm') for k, _ in names],
            ['  其余（norm、逐元素、拷贝等）'] + [ms(k, 'other') for k, _ in names],
            ['  采样 + 停止判断（26 万词表）'] + [ms(k, 'sampler') for k, _ in names],
            ['主线实际跳过的 GLOBAL 块（稀疏调用中）', '约 4%', '约 79%', '约 88%'],
            ['单次 forward 中位数：dense → 最好变体（同一请求）'] + [fwd_pair(k) for k, _ in names],
            ['最好变体：端到端 W / 答对 vs dense',
             f'{f3(best["W"])} / {best["correct"]} vs {best["base_correct"]}' if best else '', '0.907 / 92 vs 86', '0.853 / 76 vs 75']]
    page('6. AIME：精度不降，但没有加速空间——每步耗时拆开看（与 LongBench 对比）',
         'AIME26 30 题 × 6 个 seed = 180 格/方法（E7）· 耗时：同一底座（v5，CUDA graph）下各取一次真实 decoder 调用，按 kernel 归类；百分比 = 占单步总耗时',
         [('table', rows, [4.4, 2.75, 2.75, 2.75], 10, ()),
          ('bullets', ['AIME 上 GLOBAL 注意力只占一步的约 2%（key 少），LOCAL 约 3%；全部省掉也快不了多少，而选块是每步的固定开销，正好抵消。',
                       '主线阈值在 AIME 只跳约 4% 的块（key 少，每个块风险都不低）；强行跳 70% 就掉点、变慢（第 5 页）。',
                       '长上下文时 GLOBAL 注意力的占比随 key 数增长，这是稀疏能省时间的地方；MoE 和采样各上下文基本不变。'], 11)],
         """
【耗时测量】scripts/v27_time_breakdown.py，底座 piecewise_v5：AIME 用 E7 的冻结配置（第 0 题、第 55 次 decoder 调用）；32K/64K 用 E5 的冻结配置（第 0 题、第 5 次调用）。C 部分 = 编译后 forward 的 CUDA graph 重放中各 kernel 的设备时间（前 30 个 kernel 归类，其余计入“其余”）；D 部分 = 整个去噪步重放（forward + 采样 + 停止判断）。数据：time_breakdown_v5/*_dense_first.jsonl。
【单次 forward】A 部分：同一请求、各自轨迹下 decoder forward 的中位数（AIME 最好变体 = M3 优化版 + carry0 + 2K 门控；32K/64K = M3 优化版 + carry0）。
【AIME 结果】E7 v27_aime_confirm_e7_0515fa3125588ff0：最好变体 W 1.005 [0.967, 1.042]，答对 99 vs 99；M3 优化版 1.033；原版 M1/M2/M3 1.089/1.115/1.069；各方法精度差异都不显著。
【实际稀疏度】fidelity_v6（诊断，每类 2 个请求）：主线在稀疏调用中保留块比例 AIME 0.96、32K 0.21、64K 0.12。
""")


def build(out_name='dlm_sparse_attention_20261001_compact_v7.pptx', md_only=False):
    cover()
    p1_baseline()
    p_methods()
    p_flow()
    p_long()
    p_select()
    p_aime()
    head = ('# 组会材料源稿（2026-10-01，压缩版）：DiffusionGemma-26B-A4B 块稀疏注意力\n\n'
            '由 scripts/v27_build_deck_1001_compact.py 与 pptx 同时生成；表中面板数字直接读取已推送的 summary.csv。'
            'pptx：ppt_sample/' + out_name + '。\n\n---\n')
    (R / 'weekly_slides_20261001_compact.md').write_text(head + '\n'.join(MD), encoding='utf-8', newline='\n')
    if md_only:
        return R / 'weekly_slides_20261001_compact.md'
    HERE.mkdir(parents=True, exist_ok=True)
    prs.save(HERE / out_name)
    return HERE / out_name


if __name__ == '__main__':
    import sys
    print(build(md_only='--md-only' in sys.argv))
