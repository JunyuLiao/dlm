"""Build the compact 2026-10-01 group-meeting deck (cover + 5 pages): baseline, 32K/64K variants, 96K and the
step-count check, AIME cost structure, and the V-term controls. Panel ratios are read from the pushed summary.csv
files (not hand-copied); static numbers (kernel timings, time breakdown, step-count check) cite their files in the
speaker notes.

Writes results/m1_m2_m3_frontier_v27_20260929/ppt_sample/dlm_sparse_attention_20261001_compact.pptx.
"""
import csv
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


def page(title, config, blocks, notes):
    s = prs.slides.add_slide(BLANK)
    text(s, 0.4, 0.2, 12.5, 0.6, [title], size=22, bold=True)
    text(s, 0.4, 0.8, 12.5, 0.6, [config], size=11, color=GRAY)
    y = 1.4
    for kind, *args in blocks:
        if kind == 'table':
            rows, widths, size, bold = args
            table(s, 0.35, y, 12.6, rows, widths, size=size, bold_rows=bold)
            y += 0.3 * len(rows) + 0.3
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
FAN = [('原版 M1（Fan）', 'M1_R1_A8_fa4'), ('原版 M2（Fan）', 'M2c_R1_A8_fa4'), ('原版 M3（Fan）', 'M3_R3_A8_fa4')]
L32, L64 = 'longbench_v2_32k', 'longbench_v2_64k'
WIDE = [3.7, 2.1, 0.72, 0.72, 0.6, 0.85, 2.1, 0.72, 0.72, 0.6, 0.85]


def cover():
    s = prs.slides.add_slide(BLANK)
    text(s, 0.8, 2.3, 11.5, 1.0, ['DiffusionGemma-26B-A4B 块稀疏注意力：基线、长上下文结果、AIME 与看 V 对照'], size=30, bold=True)
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


def p2_long():
    e4, e5, e10 = 'lb_confirm_panel_e4', 'lb_overhead_panel_e5', 'lb_m2opt_panel_e10'
    tab = [['方法', '32K 端到端 W [95% CI]', 'S', '每步', 'N', '答对(86)', '64K 端到端 W [95% CI]', 'S', '每步', 'N', '答对(75)']]
    for name, arm, panel in [*[(n, a, e4) for n, a in FAN], ('B（选一次、整段复用）', B, e5),
                             ('M3 优化版（R6 DP −ln2）', M3, e5), ('M3 优化版 + carry0（当前最好）', M3C, e5), ('B + carry0', BC, e5)]:
        tab.append([name] + cols(get(panel, arm, L32)) + cols(get(panel, arm, L64)))
    d32, d64 = get(e10, D, L32), get(e10, D, L64)
    tab2 = [['同样优化：M2 vs M3（E10，96 格/档）', '32K 端到端 W [95% CI]', 'S', '每步', 'N',
             f'答对({d32["correct"]})' if d32 else '答对', '64K 端到端 W [95% CI]', 'S', '每步', 'N',
             f'答对({d64["correct"]})' if d64 else '答对']]
    for name, arm in [('M3 优化版 + carry0（V = 投影 32 维）', M3C), ('M2 同样优化 + carry0（V = 块均值）', M2C)]:
        tab2.append([name] + cols(get(e10, arm, L32)) + cols(get(e10, arm, L64)))
    page('2. 32K / 64K：各变体的端到端、纯生成、每步耗时、forward 数与精度',
         'LongBench-v2 32K 档（28–40K token）、64K 档（56–76K token）各 24 题 × 6 个新 seed（404–909）= 每档 144 格/方法 · 生成前冻结方案',
         [('table', tab, WIDE, 10, (6,)),
          ('table', tab2, WIDE, 10, ()),
          ('bullets', ['原版 M1/M2/M3 都比 dense 慢：选块和观测每层每次 2.3–7.9 ms，比 dense 注意力本身（1.5–2.9 ms）还贵。',
                       '优化版：每个 canvas 一次融合观测、dense-prefix 风险、每 6 步决策、副流异步选块、FA4 执行，决策降到 0.27 ms/次；carry0 再去掉每个 canvas 的一次 dense 步。',
                       '精度都不低于 dense；prefill 所有方法约 1.00（没有动 prefill）。'], 12)],
         """
【协议】E4 v27_lb_confirm_e4_d3a7edb87a2b800a（原版 M1/M2/M3 取自这里）；E5 v27_lb_overhead_e5_1c93b97271a3a266（与 E4 同一批 288 格，dense/M3/B 输出与 E4 逐 token 相同）；E10 v27_lb_m2opt_e10_c965792c1cb6542d（M2 与 M3 只差 V 的摘要方式：块均值 vs 投影 V，其余优化完全一样）。
【口径】W、S、每步、N 都是 方法/dense 的配对几何平均；每步 = S/N 是摊销值。单次 forward 直接计时（64K 真实状态、CUDA graph 重放）：dense 54.4 ms，保留 12% 时 41.8 ms（−23%）。
【命名】Fan 的 M2 = M1 换成块均值 V（每步重新选）；M3 = M1 每 R 步才重新选。“M2 同样优化” = 块均值 V + 与 M3 优化版完全相同的流水线（每 6 步决策等），只为隔离 V 摘要的影响。
【M2 vs M3 其他证据】原版：M2 选块器更贵（64K 每层 9.5 vs 4.5 ms），更慢（1.34 vs 1.09）。固定 88% 稀疏度（E9，64K）：M2/M3 端到端 1.010 [0.945, 1.085]、答对 54 vs 53。AIME 70%（E8）：92 vs 87（p = 0.49）。都不显著。
【保护策略】始终保留：当前 canvas 的全部块、第一个块（attention sink）；carry0 在第 0 步保留刚并入 prefix 的块。没有用 Junyu 的 prefix_end / diagonal 保护（他自己的结论是不稳定：LongBench diagonal 步数 +58%，AIME beginning 掉点）。我们的 protect_output（保留全部生成部分）在 E3 测过，无显著收益，未采用。
【数据】lb_confirm_panel_e4/、lb_overhead_panel_e5/、lb_m2opt_panel_e10/ 的 summary.md。
""")


def p3_96k_steps():
    p96 = 'lb96k_pooled_e6_e6b'
    tab = [['方法', '96K 端到端 W [95% CI]', '纯生成 S', '每步', 'N', '答对(25)']]
    for name, arm in [*FAN, ('B', B), ('M3 优化版', M3), ('M3 优化版 + carry0', M3C)]:
        tab.append([name] + cols(get(p96, arm)))
    page('3. 96K，以及“步数会不会变多”的核查',
         '96K：LongBench-v2 84–104K 档题池中单卡放得下的全部 11 道题 × 6 个 seed = 66 格/方法（E6 + E6b）· 步数核查：32K 同 24 题、同底座 v5、9 个 seed',
         [('table', tab, [3.8, 2.8, 1.3, 1.1, 1.0, 1.2], 11, (6,)),
          ('label', '步数比 N（M3 优化版 / dense）按 seed 拆开（32K，24 题，同底座）'),
          ('table', [['seed', '101', '202', '303', '404', '505', '606', '707', '808', '909', '9 个合并 [95% CI]'],
                     ['N', '1.23', '0.89', '1.24', '0.85', '0.98', '1.12', '0.99', '1.00', '1.00', '1.024 [0.973, 1.076]']],
           [0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 2.6], 11, ()),
          ('bullets', ['单个 seed 的步数比在 0.85–1.24 之间摆动；之前 3 个 seed（101/202/303）偏高、6 个新 seed 偏低，都是同一分布的抽样（随机 3/6 分组中 15% 的差距不小于此）。',
                       '9 个 seed 合并：32K 步数 +2.4%、64K +1.5%，都不显著。应表述为“步数没有显著增加，最佳估计 +1–2%”；E4 的端到端可能因此偏乐观约 2%。每步耗时（S/N）的结论各批次一致、很稳。'], 12)],
         """
【96K】E6 v27_lb96k_confirm_e6_8ab3aa35d1874ca1 + E6b v27_lb96k_extend_e6b_d1b1c0c74a1dc69d，462/462 成功。单看 E6（6 题）步数 0.81，单看 E6b（5 题）1.19：96K 只引用合并结果。LongBench-v2 共 32 道题在 84–104K 档，14 道单卡放得下，其中 11 道在冻结题池里（全部用上）；更长的题 dense prefill 就 OOM，128K 需要分块 prefill。
【步数核查】每格步数比 = 方法 decoder 调用数 / dense 调用数，单格对数标准差 0.45（约 ×1.57）。seed 101/202/303 来自 E3（v5、同 24 题），404–909 来自 E4。84 种 3/6 分组中有 13 种的组间差距不小于实际分组。64K 合并用 traj_t1（v4，3 个 seed）+ E4（v5），N 1.015 [0.948, 1.085]。
【数据】lb96k_pooled_e6_e6b/summary.md、README.md；步数核查见 docs/RESULTS_LEDGER.md。
""")


def p4_aime():
    best = get('aime_confirm_panel_e7', 'M3_R6_A64_fused_dp_async_m1ln2_c0_gate2k_fa4')
    page('4. AIME：精度不降，但没有加速空间（与 LongBench 的每步耗时对比）',
         'AIME26 30 题 × 6 个 seed = 180 格/方法（E7）· 每步耗时为 dense 摊销值，分项来自模块/kernel 测速 · 同一模型、同一底座',
         [('table', [['', 'AIME', 'LongBench 32K', 'LongBench 64K'],
                     ['每次 forward 的 key 数（中位数）', '3.6K（prompt 约 160 token，其余是生成的推理）', '35.6K', '67.3K'],
                     ['dense 每步耗时（摊销）', '30.7 ms', '37.3 ms', '45.3 ms'],
                     ['  MoE 专家（每步读约 46 GB 权重，带宽上限）', '12.8 ms', '12.8 ms', '12.8 ms'],
                     ['  5 层 GLOBAL 注意力（dense）', '约 1.5 ms（≤ 5%，按 kernel 测速估计）', '7.3 ms（约 20%）', '14.8 ms（约 33%）'],
                     ['  采样（26 万词表）', '约 4.6 ms', '约 4.6 ms', '约 4.6 ms'],
                     ['  其余（投影、LOCAL、norm、canvas 追加等；余量）', '约 11.8 ms', '约 12.6 ms', '约 13.1 ms'],
                     ['M3 优化版每步耗时', '31.4 ms（略慢）', '34.7 ms', '37.2 ms'],
                     ['最好变体：端到端 W / 答对 vs dense',
                      f'{f3(best["W"])} {ci(best, "W")} / {best["correct"]} vs {best["base_correct"]}' if best else '',
                      '0.907 / 92 vs 86', '0.853 / 76 vs 75']],
           [4.4, 4.2, 2.0, 2.0], 11, ()),
          ('bullets', ['AIME 上注意力只占每步约 5%：全部省掉也只快约 5%，而选块的观测、决策、建块表是每步的固定开销，正好抵消。',
                       'key 少（中位数约 57 个块，其中 4 个 canvas 块必须保留）：70% 的 prefix 稀疏度只对应约 60% 的总块稀疏度；28% 的调用不到 2K key（门控后直接走 dense）。',
                       '原版 M1/M2/M3 在 AIME 上慢 7–12%；各方法精度与 dense 差异都不显著。AIME 只作为“精度不掉”的检查。'], 12)],
         """
【协议】E7 v27_aime_confirm_e7_0515fa3125588ff0，1,260/1,260 成功，每格同机。最好变体 = M3 优化版 + carry0 + 2K 门控（key < 2K 的调用走原生 dense，约 31% 的 GLOBAL 调用）；纯生成 1.005、每步 1.005。M3 优化版 1.033（答对 96）；原版 M1/M2/M3 端到端 1.089/1.115/1.069。
【耗时来源】LongBench 分项：substrate/time_breakdown.json、kernel_bench/module_profile_*.jsonl。AIME 的 GLOBAL 注意力是按 kernel 测速（8K key 每层 0.44 ms）插值的估计，AIME 没有单独的模块级 profile。每步耗时来自面板 S/N 摊销。
【AIME 长度】E7 dense：平均输出 5,695 token、22.6 个 canvas、每 canvas 13.1 步；key 数 p10 927、p90 7,342。
【数据】aime_confirm_panel_e7/summary.md、receipts.md。
""")


def p5_vterm():
    e8, e9 = 'aime_vterm_panel_e8', 'lb64_vterm_panel_e9'

    def cell(panel, arm, key):
        r = get(panel, arm) if arm else None
        return '—' if r is None else (r['correct'] if key == 'acc' else f3(r['W']))
    V = [('投影 V 32 维（M1/M3 的做法）', 'M3_R6_A64_fused_dp_async_topk70_fa4', 'M3_R6_A64_fused_dp_async_topk88_c0_fa4'),
         ('投影 V 16 维', 'M3_R6_A64_fused_dp_async_topk70_r16_fa4', None),
         ('投影 V 8 维', 'M3_R6_A64_fused_dp_async_topk70_r8_fa4', 'M3_R6_A64_fused_dp_async_topk88_r8_c0_fa4'),
         ('投影 V 4 维', 'M3_R6_A64_fused_dp_async_topk70_r4_fa4', None),
         ('块均值 V（M2 的做法）', 'M2c_R6_A64_fused_dp_async_topk70_fa4', 'M2c_R6_A64_fused_dp_async_topk88_c0_fa4'),
         ('不看 V（只看注意力质量）', 'M3_R6_A64_fused_dp_async_topk70_mass_fa4', 'M3_R6_A64_fused_dp_async_topk88_mass_c0_fa4'),
         ('SparseD 70%（移植，外部参照）', 'SparseD_s70_skip1_fa4', None)]
    tab = [['选块时用的 V', f'AIME 答对（dense {get(e8, D)["correct"]}）', 'AIME 端到端 W',
            f'64K 答对（dense {get(e9, D)["correct"]}）', '64K 端到端 W']]
    for name, a8, a9 in V:
        tab.append([name, cell(e8, a8, 'acc'), cell(e8, a8, 'w'), cell(e9, a9, 'acc'), cell(e9, a9, 'w')])
    page('5. 选块要不要看 V：同稀疏度下，看 V 没有比只看注意力质量更好',
         'AIME：M3 优化版固定保留 30% prefix 块（E8，180 格/组）· 64K：M3 优化版 + carry0 固定保留 12%（E9，24 题 × 4 seed = 96 格/组）· 只换风险里的 V 项',
         [('table', tab, [4.4, 2.2, 1.8, 2.2, 1.8], 12, ()),
          ('bullets', ['两处都没有任何 V 项显著好于“只看注意力质量”（AIME 相对 32 维 p ≥ 0.30；64K 全部 p ≥ 0.42）；64K 上只看质量与当前最好的 −ln2 版一样快（比值 0.999）。',
                       'AIME 并不是“没变化”：不同 V 项选出的块不同，答对数在 86–94 之间变化，但最好的是 4 维或不看 V；70% 稀疏时步数多 6–13%，所以都比 dense 慢。',
                       'AIME 的 70% prefix 稀疏度约等于 60% 的总块稀疏度（canvas 块必须保留）；64K 的 88% 约等于 88%。',
                       '结论：“看 V 选块”不能作为贡献；M2（块均值）与 M3（投影 V）在同样优化下也分不出差别（见第 2 页）。'], 12)],
         """
【协议】E8 v27_aime_vterm_e8_bd293196ff730851（1,440/1,440 成功，dense 与 E7 逐 token 相同）；E9 v27_lb64_vterm_e9_5b125c63817898f6（576/576 成功，dense 与 −ln2 + carry0 两臂与 E5 逐 token 相同）。每次运行的 effective_method 都核对过（k30 / k12、proj_rank、risk_value、mu_mode、carry0）。
【AIME 详细】对 dense 99：32 维 87（p = 0.08）、16 维 86（p = 0.035）、8 维 87（p = 0.036）、4 维 94、块均值 92、只看质量 91、SparseD 94。
【64K 详细】对 dense 51：32 维 53、8 维 50、块均值 54、只看质量 54、−ln2 版 49；只看质量 / 32 维 端到端 0.960 [0.905, 1.018]。
【组内对照】JunyuLiao 在 RULER8K（用当前 QK 选块）上发现 V 有用（稀疏 75% 时 32 维 47.4% vs 只看质量 21.3%）；我们用历史 QK，任务也不同。只引用。
【数据】aime_vterm_panel_e8/、lb64_vterm_panel_e9/ 的 summary.md、receipts.md。
""")


def build(out_name='dlm_sparse_attention_20261001_compact.pptx'):
    cover()
    p1_baseline()
    p2_long()
    p3_96k_steps()
    p4_aime()
    p5_vterm()
    HERE.mkdir(parents=True, exist_ok=True)
    prs.save(HERE / out_name)
    return HERE / out_name


if __name__ == '__main__':
    print(build())
