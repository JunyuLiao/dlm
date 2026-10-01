"""Build the 2026-10-01 group-meeting deck (pptx): one experiment per slide, a table + a few lines on the slide,
details in the speaker notes. Numbers are copied from the pushed panel summaries; the markdown source with the same
content and sources is results/m1_m2_m3_frontier_v27_20260929/weekly_slides_20261001.md.

Writes results/m1_m2_m3_frontier_v27_20260929/ppt_sample/dlm_sparse_attention_20261001.pptx.
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


def table(slide, x, y, w, rows, widths, size=12, bold_rows=()):
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


def slide(title, config, rows, widths, points, notes, bold_rows=(), size=12):
    s = prs.slides.add_slide(BLANK)
    text(s, 0.5, 0.3, 12.3, 0.7, [title], size=24, bold=True)
    text(s, 0.5, 0.95, 12.3, 0.6, [config], size=12, color=GRAY)
    table(s, 0.4, 1.6, 12.5, rows, widths, size=size, bold_rows=bold_rows)
    text(s, 0.4, 1.75 + 0.4 * len(rows), 12.5, 2.5, points, size=14, bullet=True)
    s.notes_slide.notes_text_frame.text = notes.strip()
    return s


SLIDES = []   # (title, config, rows, widths, points, notes, bold_rows)

SLIDES.append((
    '大样本确认（E4）：64K 端到端快 12%，32K 快 6%，精度不降，步数不增加',
    'LongBench-v2 32K 档（28–40K token）/ 64K 档（56–76K token）各 24 题 × 6 个从未用过的 seed（404–909）= 每档 144 格/方法 · '
    'dense = FA4（全保留）· 底座 piecewise_v5 · 三台 H100，同格同机 · 生成前冻结',
    [['方法', '64K 端到端 W [95% CI]', '64K 纯生成 S', '64K 答对（dense 75）', '32K 端到端 W [95% CI]', '32K 纯生成 S', '32K 答对（dense 86）'],
     ['M3 R6 DP −ln2', '0.879 [0.820, 0.926]', '0.807', '77', '0.940 [0.897, 0.980]', '0.921', '86'],
     ['B', '0.876 [0.812, 0.935]', '0.805', '81', '0.941 [0.882, 1.003]', '0.918', '88'],
     ['M3，第 2 步才观测（obs2）', '0.884 [0.845, 0.921]', '0.820', '77', '0.925 [0.860, 0.994]', '0.899', '92'],
     ['M3，每 3 步决策（R3）', '0.889 [0.816, 0.950]', '0.823', '74', '0.937 [0.886, 0.988]', '0.917', '88'],
     ['原版 M1 / M2c / M3', '1.258 / 1.341 / 1.093', '—', '74 / 85 / 80', '1.251 / 1.293 / 1.060', '—', '94 / 90 / 94']],
    [2.6, 2.0, 1.2, 1.4, 2.0, 1.2, 1.4],
    ['W = 端到端请求时间（含 prefill）；S = 纯生成时间（去掉 prefill）；prefill 所有方法都约 1.00（我们没有改 prefill）。',
     '6 个新 seed 下每 canvas 步数 0.97–1.03、输出长度 0.95–1.01：此前 3 个 seed 时看到的"步数/输出变多"是轨迹噪声。',
     '原版 M1/M2c/M3 在两档都比 dense 慢（1.06–1.34 倍）：选块与观测的开销超过省下的注意力。'],
    """
【协议】v27_lb_confirm_e4_d3a7edb87a2b800a（specs/v27_lb_confirm_e4.json），部署提交 9e47ddb9d；2304/2304 次运行成功；计时中没有一次捕获新 CUDA graph；每格的所有方法在同一台机器上（dllm、mpk、dlm2）。
【数据】LongBench-v2 官方题，按 NeMo 模板渲染、thinking 开启后的真实 token 数分档，不截断。每档 24 题：按 sha256(id) 前 12 题为正式题（曾用于挑 −ln2 阈值），后 12 题为留出题。seed 404–909 此前从未在 LongBench 上用过。
【口径】比值 = 方法 / dense，配对几何平均，95% 置信区间按题目聚类 bootstrap（4000 次）。答对 = strict correct 格数。
【收据核对】obs2 的 dense 引导次数正好是观测次数的 2 倍；R3 实际决策间隔为 3；−ln2 各臂全局阈值 −3.874。
【只看留出题】（按总时间之比，口径与正表略不同）64K M3 0.85 [0.73, 0.96]（显著）；32K M3 0.96 [0.91, 1.01]（不显著），obs2 0.95 [0.91, 0.98]。
【为什么之前的结论被推翻】9/21 的 dense 对照：同一 dense 换一次模型加载，去噪步数比值中位数就差 1.17 倍；3 个 seed × 12 题时步数比值的噪声足以造成 ±20% 的假象。以后请求级结论只用大样本面板。
【数据路径】results/m1_m2_m3_frontier_v27_20260929/lb_confirm_panel_e4/summary.md
""", (1,)))

SLIDES.append((
    '去掉每个 canvas 的一个 dense 步（carry0）：64K 端到端快 15%，32K 快 9%',
    '与 E4 完全相同的 288 格（同题、同 seed、同机、同底座 v5）· dense = FA4',
    [['方法', '64K 端到端 W [95% CI]', '64K 纯生成 S', '64K 答对（dense 75）', '32K 端到端 W [95% CI]', '32K 纯生成 S', '32K 答对（dense 86）'],
     ['M3 R6 DP −ln2（上一页）', '0.879 [0.818, 0.927]', '0.807', '77', '0.942 [0.900, 0.984]', '0.920', '86'],
     ['M3 R6 DP −ln2 + carry0', '0.853 [0.802, 0.895]', '0.775', '76', '0.907 [0.858, 0.952]', '0.878', '92'],
     ['B + carry0', '0.858 [0.800, 0.912]', '0.774', '80', '0.931 [0.883, 0.980]', '0.907', '86'],
     ['M3 + stable1 门控', '0.891', '0.820', '75', '0.953', '0.939', '81'],
     ['B + stable1 门控', '0.889', '0.826', '79', '0.938', '0.920', '88']],
    [2.6, 2.0, 1.2, 1.4, 2.0, 1.2, 1.4],
    ['carry0：每个 canvas 第 0 步改用上一个 canvas 最后的选块图（新增块全部保留），第 1 步照常 dense + 观测。',
     '与不带 carry0 的 M3 直接配对：64K 再快 2.9%（0.971 [0.939, 1.001]），32K 再快 3.7%（0.963 [0.914, 1.011]）；精度不降。',
     'stable1（输出稳定后转 dense）：更慢，32K 少对 5 题，不采用。'],
    """
【协议】v27_lb_overhead_e5_1c93b97271a3a266（specs/v27_lb_overhead_e5.json），部署提交 758723513；2016/2016 次运行成功，计时中无新 CUDA graph；dense、M3、B 三臂与 E4 逐 token 相同（各 288/288），本页与上一页可直接对照。
【收据核对】carry0 每个请求只在 canvas 0 做 dense 引导（5 层 × 1 次），其余 canvas 第 0 步都消费了上一 canvas 的图（carried_first_calls = 5 ×（canvas 数 − 1））。stable1 在 225/288 个 M3 请求里触发，共把 12,810 次 GLOBAL 调用转成 dense。
【与 E1 的区别】E1 的跨 canvas 复用（K=4）整个 canvas 都用旧图，步数变多、掉点；carry0 只在第 0 步用旧图，第 1 步就刷新，每 canvas 步数 64K 0.98、32K 1.00。
【直接配对精度】不一致格：64K 7（只 carry0 对）/ 8（只 M3 对）；32K 15 / 9。
【数据路径】results/m1_m2_m3_frontier_v27_20260929/lb_overhead_panel_e5/summary.md
""", (2,)))

SLIDES.append((
    '96K：每步省 24%，端到端快 18–26%（6 道题，置信区间宽）',
    'LongBench-v2 96K 档中单卡放得下的 6 道题（prompt ≤ 约 95K token）× 6 个 seed（404–909）= 36 格/方法 · dense = FA4 · 底座 v5 · 同格同机',
    [['方法', '端到端 W [95% CI]', '纯生成 S', '每步 S/N [95% CI]', '步数 N', '答对（dense 12）'],
     ['M3 R6 DP −ln2 + carry0', '0.744 [0.597, 0.903]', '0.619', '0.762 [0.747, 0.776]', '0.81', '16'],
     ['M3 R6 DP −ln2', '0.819 [0.690, 0.968]', '0.720', '0.762 [0.731, 0.785]', '0.95', '14'],
     ['B', '0.786 [0.641, 0.946]', '0.668', '0.743 [0.722, 0.760]', '0.90', '16'],
     ['原版 M1 / M2c / M3', '1.158 / 1.385 / 1.045', '—', '1.62 / 1.83 / 1.25', '—', '14 / 17 / 15']],
    [3.0, 2.3, 1.3, 2.4, 1.2, 1.8],
    ['上下文越长每步省得越多：32K 约 8%、64K 约 18–20%、96K 约 24–26%（每步的置信区间都很窄）。',
     '96K 端到端里有一部分来自步数少（N 0.81），6 道题时噪声很大；若步数不变，端到端预期约 0.85。精度不降。',
     '128K 在单卡上 prefill 就 OOM（dense 也一样），需要分块 prefill。'],
    """
【协议】v27_lb96k_confirm_e6_8ab3aa35d1874ca1（specs/v27_lb96k_confirm_e6.json）；252/252 次运行成功，计时中无新图，每格同机（dllm、mpk）；carry0 收据：每请求 5 次 dense 引导。
【数据】96K 档共 12 道题，另 6 道 prompt ≥ 98K token，在单卡 prefill 时所有方法（包括 dense）都 OOM；128K 档同样 OOM（76.7 GiB 已分配时再申请 1.38 GiB 失败）。
【单题差异】M3 + carry0 / dense 的单格比值在 0.28–1.30 之间。
【数据路径】results/m1_m2_m3_frontier_v27_20260929/lb96k_confirm_panel_e6/summary.md
""", (1,)))

SLIDES.append((
    'AIME 大样本（E7）：精度不降，但没有速度收益（最好的变体持平）',
    'AIME26 30 题 × 6 个 seed（404–909）= 180 格/方法 · dense = FA4 · 底座 v5 · 三台 H100，同格同机 · 生成前冻结',
    [['方法', '答对（dense 99）', '多对 / 多错', '端到端 W [95% CI]', '纯生成 S', '每步 S/N'],
     ['M3 + carry0 + 2K 门控', '99', '+11 / −11', '1.005 [0.967, 1.042]', '1.005', '1.005'],
     ['M3 R6 DP −ln2', '96', '+16 / −19', '1.033 [0.977, 1.089]', '1.032', '1.010'],
     ['B + carry0 + 2K 门控', '93', '+9 / −15', '1.057 [1.023, 1.096]', '1.059', '1.032'],
     ['原版 M1 / M2c / M3', '97 / 92 / 99', '—', '1.089 / 1.115 / 1.069', '—', '1.08 / 1.10 / 1.03']],
    [3.0, 1.8, 1.6, 2.6, 1.3, 1.5],
    ['精度：所有方法与 dense 的差异都不显著（McNemar p ≥ 0.31）；主变体 99 = 99。',
     '速度：AIME 上下文短，GLOBAL 注意力只占一步的 4–7%，稀疏省下的时间被选块开销抵消，最好只是持平。',
     '原版 M1/M2c/M3 慢 7–12%。AIME 只作“精度不掉”的检查，不作加速场景。'],
    """
【协议】v27_aime_confirm_e7_0515fa3125588ff0（specs/v27_aime_confirm_e7.json），部署 v27_e7_ceefaf2；1,260/1,260 次运行成功，每格同机（dllm、mpk、dlm2），3 次计时中出现新图（剔除后主变体 Wc 1.016 [0.985, 1.047]）。
【收据核对】门控变体中约 31% 的 GLOBAL 调用 key < 2K，走原生 dense（74,845 次）；carry0 生效 13,115 次（M3）/ 13,470 次（B）；dense 引导只在 canvas 0（875 次）。
【数据路径】results/m1_m2_m3_frontier_v27_20260929/aime_confirm_panel_e7/summary.md、receipts.md
""", (1,)))

SLIDES.append((
    '时间都花在哪：为什么单请求下注意力稀疏的收益有上限',
    '64K dense 一个请求约 13.1 s，32K 约 9.7 s（piecewise_v3–v5 实测平均）· 每步约 45 ms（64K）/ 37 ms（32K）/ 31 ms（AIME）',
    [['部分（64K）', '时间', '占请求', '稀疏注意力能否省'],
     ['prompt prefill + 准备', '约 4.6 s', '35%', '否（不在本方法范围）'],
     ['解码：5 个 GLOBAL 注意力层', '约 2.8 s', '21%', '是：M3 降到约 1.2 s，加 carry0 后更低'],
     ['解码：MoE 专家', '约 2.4 s', '18%', '否（每步读约 46 GB 专家权重，已到 HBM 带宽上限）'],
     ['解码：采样（262K 词表）', '约 0.9 s', '7%', '否'],
     ['解码：其他（投影、LOCAL、norm、canvas 追加）', '约 2.4 s', '19%', '否']],
    [4.5, 1.5, 1.3, 5.2],
    ['即使注意力完全免费、步数不变，单请求上限也只有 64K 约 −21%、32K 约 −16%；我们已拿到 −15% / −9%。',
     'AIME（上下文 ≤ 约 8K）上 GLOBAL 注意力只占每步约 4–7%：注意力稀疏在 AIME 上天然没有多少空间。'],
    """
【每步构成】（32K dense，substrate/time_breakdown.json，v2 时期，KV 拼接已在 v3 去掉）forward 约 32 ms：MoE 专家矩阵乘 12.8 ms、GLOBAL 注意力 7.2 ms、其余矩阵乘 3.5 ms、LOCAL 0.8 ms、其他约 4 ms；另有采样约 4.6 ms。
【MoE】128 个专家、每 token 选 8 个；一步 256 个 canvas token × 8 = 2048 次分配，几乎所有专家都会被用到，所以每步要读全部专家权重（约 46 GB），12.8 ms 已接近 H100 的 HBM 带宽极限。
【GLOBAL 注意力】CUDA event 直接计时（kernel_bench/module_profile_*）：dense 每层 1.46 ms（32K）/ 2.95 ms（64K）；M3 R6 DP −ln2 摊到每步：32K 4.3 ms、64K 5.5 ms（dense 7.3 / 14.75 ms）。
【每步解码时间】（摊销，72 格平均）32K dense 37.3 ms、M3 34.7、B 33.5；64K dense 45.3、M3 37.2、B 36.1；AIME dense 30.7、M3 31.4。
【还能挖的】canvas 追加（encoder 与 decoder 共享权重，可复用最后的选块图）约 −1.4%（64K）；剩余观测/决策开销约 −1–2%。
""", ()))

SLIDES.append((
    '批量服务：注意力占比并不随 batch 变大（64K，单次 forward 测速）',
    '64K 真实状态（LongBench-v2 64K 第 1 题，71,772 个 key，第 6 次 decoder 调用）复制成 batch B · 整个 forward 用 CUDA graph 重放计时 · 合成保留图，不含选块开销',
    [['batch', 'dense forward', '保留 12%', '只留 canvas', 'prefix 注意力占比', '保留 12% 时每次 forward 省'],
     ['1', '54.4 ms', '41.8', '40.0', '26%', '23%'],
     ['2', '70.8', '57.6', '55.3', '22%', '19%'],
     ['4', '117.7', '93.7', '88.7', '25%', '20%'],
     ['8', 'OOM', '', '', '', '']],
    [1.2, 2.0, 1.6, 1.6, 2.4, 3.2],
    ['原以为批量服务里 MoE 权重读取被分摊、注意力会成为大头；实测 64K、batch ≤ 4 时占比不变，稀疏单步收益稳定约 20%。',
     '原因：batch=1 时 FA4（256 个 query）吃不满 GPU，batch 变大后注意力更高效；MoE 也从带宽受限变成算力受限。'],
    """
【脚本】scripts/v27_batch_step_bench.py（提交 02eead519），dlm2，E4 运行目录与部署；数据 results/m1_m2_m3_frontier_v27_20260929/batch_scaling/。
【口径】eager 的整次 decoder forward（含每次调用的 K/V 拼接，1.2/2.4/4.7 ms）一次性 CUDA graph 捕获后重放，CUDA event 取 15 次中位数；采样另测 2.5/4.8/9.4 ms。
【局限】只测了 64K 一个状态；batch 8 时复制的 64K 缓存放不下；合成保留图、不含选块开销；不是端到端，也不是真实服务。
""", ()))

SLIDES.append((
    '相关工作与新颖性现状',
    '2026-10-01 复核论文摘要/正文 · 组内工作只引用',
    [['工作', '做法', '设置', '与我们的关系'],
     ['SparseD（ICLR 2026）', '前若干步 dense，再按池化分数选块并复用', 'LLaDA/Dream，固定步数', '≈ B，撞车；同 kernel 正面对比速度无显著差别'],
     ['MAGE（2602.14209）', '块的第一步精确注意力选 top-k，整块复用；副流异步', '7–8B 块扩散，32 token 块 × 固定 32 步，128K 6.82×', '≈ B（撞车）；大倍数来自 dense 小模型 + 128K'],
     ['HERALD（2606.21633）', '批量块扩散服务：每块选一次 KV，CPU-GPU 协同取 KV', '解码吞吐 2.28×，5% KV 预算', '批量服务方向的直接相关工作'],
     ['PulseCol / Focus-dLLM / LoSA', '周期刷新 / 按置信度裁剪 / 稳定 token 复用', '固定步数或长度', '部分相关'],
     ['Lil / LessIsMore', '稀疏注意力导致推理输出变长', '自回归推理模型', '我们在大样本下没有观察到输出变长']],
    [2.6, 3.8, 3.0, 3.1],
    ['"每块选一次、复用"已被 SparseD / MAGE / HERALD 覆盖；"看 V 的选块更好"有待 E8 检验。',
     '站得住的：生产级 MoE dLLM + 原生自适应停止 + 最强官方基线 + 大样本协议下的端到端收益与精度，以及成本结构分析。'],
    """
【组内证据（不是我们的，只引用）】JunyuLiao 的 value-direction 选块维度扫描（分支 ljy/value_aware，提交 8b29f942d、47c47d9d7）：RULER8K 稀疏度约 75% 时，V 投影 32 维 47.4%、全维 49.8%、只看注意力质量 21.3%、2 维 2.5%；50% 时都约 92%。他的选块用当前 QK，与我们的历史 QK 不同。
【可能的论文方向】(1) 评测方法学：自适应停止下 dLLM 加速需要大样本（dense 对照方差、本次 E4 推翻 3 seed 结论）；(2) 看 V 选块（取决于 E8）；(3) 与同学的 query 敏感度方向合作。批量服务方向在 64K、batch ≤ 4 时没有放大收益（见上一页）。
""", ()))


def build(out_name='dlm_sparse_attention_20261001.pptx'):
    s = prs.slides.add_slide(BLANK)
    text(s, 0.8, 2.4, 11.5, 1.0, ['DiffusionGemma-26B-A4B 块稀疏注意力：大样本确认与剩余开销'], size=32, bold=True)
    text(s, 0.8, 3.5, 11.5, 1.0, ['2026-10-01 · 端到端（含 prefill）与纯生成时间同时报告 · dense 基线 = FlashAttention-4'],
         size=18, color=GRAY)
    s.notes_slide.notes_text_frame.text = ('所有数字来自已推送的结果：仓库 coconight01/dlm_test，分支 research/m3-output-numerics-20260927，'
                                           '目录 results/m1_m2_m3_frontier_v27_20260929/；源稿 weekly_slides_20261001.md。')
    for title, config, rows, widths, points, notes, bold in SLIDES:
        slide(title, config, rows, widths, points, notes, bold_rows=bold)
    HERE.mkdir(parents=True, exist_ok=True)
    prs.save(HERE / out_name)
    return HERE / out_name


if __name__ == '__main__':
    print(build())
