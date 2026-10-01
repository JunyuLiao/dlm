"""Figures for the 2026-09-30 group-meeting slide sample (numbers copied from the pushed panel summaries).

Writes results/m1_m2_m3_frontier_v27_20260929/ppt_sample/{fig_three_levels,fig_aime_sparsity}.png.
Sources: kernel_bench/README.md (kernel, module), final_panel_f1/summary_lb.csv (request W, decode S at 64K),
aime_sparsity_panel_t1/table.md and aime_sparsity_panel_hi/table.md (AIME target sparsity).
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
OUT = Path(__file__).resolve().parents[1] / 'results/m1_m2_m3_frontier_v27_20260929/ppt_sample'
OUT.mkdir(parents=True, exist_ok=True)

# ---- three levels at 64K, speedup over dense FA4 (> 1 is faster)
levels = ['kernel\n(保留约 10%)', '注意力模块\n(真实请求)', '解码 S\n(去掉 prefill)', '端到端请求 W']
series = {
    'B': [7.3, 4.4, 1 / 0.788, 1 / 0.864],
    'M3 R6 DP −ln2': [7.3, 2.7, 1 / 0.783, 1 / 0.869],
    'Fan M1': [7.3, 0.41, 1 / 1.413, 1 / 1.267],
    'Fan M2c': [7.3, 0.31, 1 / 1.788, 1 / 1.491],
    'Fan M3': [7.3, 0.66, 1 / 1.181, 1 / 1.120],
}
colors = ['#4C72B0', '#DD8452', '#9a9a9a', '#b5b5b5', '#cfcfcf']
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
x = np.arange(len(levels))
w = 0.16
for i, (name, vals) in enumerate(series.items()):
    bars = ax.bar(x + (i - 2) * w, vals, w, label=name, color=colors[i])
    for b, v in zip(bars, vals):
        if i < 2 or v < 1:
            ax.text(b.get_x() + b.get_width() / 2, v * 1.06, f'{v:.2f}×' if v < 2 else f'{v:.1f}×',
                    ha='center', va='bottom', fontsize=7)
ax.axhline(1, color='k', lw=0.8, ls='--')
ax.set_yscale('log')
ax.set_ylim(0.25, 13)
ax.set_yticks([0.25, 0.5, 1, 2, 4, 8])
ax.set_yticklabels(['0.25×', '0.5×', '1×', '2×', '4×', '8×'])
ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.set_ylabel('相对 dense FA4 的加速（> 1 更快）')
ax.set_title('64K：同一个 FA4 kernel 上，加速在三层上怎么缩小')
ax.legend(ncol=5, fontsize=8, loc='upper right', frameon=False)
fig.tight_layout()
fig.savefig(OUT / 'fig_three_levels.png')

# ---- AIME target sparsity: accuracy change and request time vs realized sparsity
ours_sp = [24, 32, 41, 57, 66]
sd_sp = [23, 32, 40, 56, 65]
# accuracy change in percentage points vs the paired dense of the same panel (90 cells for <= 41%, 60 cells above)
ours_dacc = [0 / 90 * 100, -1 / 90 * 100, -1 / 90 * 100, -6 / 60 * 100, -11 / 60 * 100]
sd_dacc = [-1 / 90 * 100, -2 / 90 * 100, -2 / 90 * 100, -5 / 60 * 100, -5 / 60 * 100]
ours_w = [1.076, 1.086, 1.062, 1.196, 1.206]
sd_w = [0.988, 1.034, 1.021, 1.081, 1.075]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=160)
a1.plot(ours_sp, ours_dacc, 'o-', color='#DD8452', label='我们：风险 top-k')
a1.plot(sd_sp, sd_dacc, 's-', color='#4C72B0', label='SparseD 移植')
a1.axhline(0, color='k', lw=0.8, ls='--')
a1.annotate('23 对 34\n(+3/−14, p≈0.013)', (66, ours_dacc[-1]), textcoords='offset points', xytext=(-95, -5), fontsize=8)
a1.set_xlabel('实际稀疏度（GLOBAL key 块）%')
a1.set_ylabel('答对率变化（百分点，相对同面板 dense）')
a1.set_title('AIME 精度 vs 稀疏度')
a1.legend(fontsize=8, frameon=False)
a2.plot(ours_sp, ours_w, 'o-', color='#DD8452', label='我们：风险 top-k')
a2.plot(sd_sp, sd_w, 's-', color='#4C72B0', label='SparseD 移植')
a2.axhline(1, color='k', lw=0.8, ls='--')
a2.set_xlabel('实际稀疏度（GLOBAL key 块）%')
a2.set_ylabel('请求时间 W（dense = 1，< 1 更快）')
a2.set_title('AIME 速度 vs 稀疏度（FA4 基线）')
a2.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / 'fig_aime_sparsity.png')
print('ok', sorted(p.name for p in OUT.iterdir()))
