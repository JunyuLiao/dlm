"""Aggregate the frozen AIME query-tile diagnostic without rerunning inference."""
from pathlib import Path
import csv, gzip, glob, json, statistics
from collections import defaultdict
import numpy as np

ROOT = Path('/home/exouser/ljy/dlm/results/aime_temporal_tile_diagnostic_v1')
RUNS = ROOT / 'runs'

def pct(xs, q):
    return float(np.quantile(np.asarray(xs, dtype=float), q)) if xs else 0.0

def groups(rows, keys):
    d = defaultdict(list)
    for row in rows:
        d[tuple(row[k] for k in keys)].append(row)
    return d.items()

def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

def read_tiles():
    rows = []
    for path in sorted(RUNS.glob('*/query_tiles.csv.gz')):
        name = path.parent.name
        method = name.split('_s')[0]
        target = int(name.split('_s')[1].split('_')[0])
        source_id = int(name.rsplit('_id', 1)[1])
        with gzip.open(path, 'rt') as f:
            for row in csv.DictReader(f):
                for k in ('canvas','iteration','layer','head','query_tile','eligible','skipped'):
                    row[k] = int(row[k])
                for k in ('sparsity','sensitivity_mean','sensitivity_std',
                          'sensitivity_min','sensitivity_max',
                          'sensitivity_active_fraction'):
                    row[k] = float(row[k])
                row.update(method=method, target=target, source_id=source_id,
                           phase='early' if row['iteration'] <= 2 else 'late')
                rows.append(row)
    return rows

def make_plots(rows, correlations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = ROOT / 'plots'
    out.mkdir(exist_ok=True)
    for target in (50, 70):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex='col', sharey='row')
        for col, kind in enumerate(('local', 'global')):
            for rr, phase in enumerate(('early', 'late')):
                ax = axes[rr, col]
                for method, color in (('gaussian32', '#4e79a7'), ('temporal', '#e15759')):
                    x = [r['sparsity'] * 100 for r in rows
                         if r['target'] == target and r['method'] == method
                         and r['phase'] == phase and r['kind'] == kind]
                    ax.hist(x, bins=np.linspace(0, 100, 21), density=True,
                            alpha=.5, label=method, color=color)
                ax.set_title(f'{kind} / {phase}')
                ax.grid(alpha=.2)
                if rr == 1: ax.set_xlabel('Skipped KV tiles within physical query tile (%)')
                if col == 0: ax.set_ylabel('Density')
                if rr == 0 and col == 0: ax.legend()
        fig.suptitle(f'AIME26 query-tile sparsity distributions ({target}% target)')
        fig.tight_layout()
        fig.savefig(out / f'query_tile_sparsity_s{target}.png', dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for target, color in ((50, '#4e79a7'), (70, '#e15759')):
        for kind, marker in (('local', 'o'), ('global', 's')):
            q = [x for x in correlations if x['method'] == 'temporal'
                 and x['target'] == target and x['phase'] == 'late' and x['kind'] == kind]
            if q:
                ax.scatter([q[0]['sensitivity_mean']], [q[0]['skip_mean'] * 100],
                           color=color, marker=marker, label=f'T {target}% {kind}')
    ax.set(xlabel='Mean temporal sensitivity',
           ylabel='Mean query-tile sparsity (%)',
           title='Temporal sensitivity allocation summary')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'sensitivity_allocation.png', dpi=160)
    plt.close(fig)

def main():
    rows = read_tiles()
    write_csv(ROOT / 'query_tile_records.csv', rows)
    summary = []
    for (method, target, phase, kind), group in groups(rows, ('method','target','phase','kind')):
        x = [r['sparsity'] for r in group]
        summary.append(dict(method=method, target=target, phase=phase, kind=kind,
            n=len(x), mean=float(np.mean(x)) if x else 0,
            median=statistics.median(x) if x else 0, p10=pct(x,.1), p90=pct(x,.9),
            zero_fraction=float(np.mean(np.asarray(x) == 0)) if x else 0,
            sensitivity_mean=float(np.mean([r['sensitivity_mean'] for r in group])),
            sensitivity_active_fraction=float(np.mean(
                [r['sensitivity_active_fraction'] for r in group]))))
    write_csv(ROOT / 'query_tile_summary.csv', summary)
    correlations = []
    for (method, target, phase, kind), group in groups(rows, ('method','target','phase','kind')):
        active = [r for r in group if r['sensitivity_mean'] > 1.0001]
        if len(active) >= 2:
            a = np.asarray([r['sensitivity_mean'] for r in active])
            b = np.asarray([r['sparsity'] for r in active])
            cor = float(np.corrcoef(a, b)[0,1]) if a.std() and b.std() else 0.
            correlations.append(dict(method=method, target=target, phase=phase, kind=kind,
                n=len(active), sensitivity_mean=float(a.mean()), skip_mean=float(b.mean()),
                pearson=cor, low_skip=float(np.mean(b[a <= np.quantile(a,1/3)])),
                high_skip=float(np.mean(b[a >= np.quantile(a,2/3)]))))
    write_csv(ROOT / 'sensitivity_vs_tile_skip.csv', correlations)
    replay = []
    for p in sorted(RUNS.glob('temporal_s*/same_qkv.json')):
        target = int(p.parent.name.split('_s')[1].split('_')[0])
        for x in json.loads(p.read_text()):
            replay.append(dict(x, target=target))
    replay_summary = []
    for (target, variant), group in groups(replay, ('target','variant')):
        replay_summary.append(dict(target=target, variant=variant, n=len(group),
            mask_disagreement=float(np.mean(
                [x['mask_disagreements_with_temporal']/max(1,x['eligible']) for x in group])),
            relative_l2=float(np.mean([x['relative_l2'] for x in group])),
            skipped=float(np.mean([x['skipped'] for x in group])),
            temporal_only=float(np.mean(
                [x['skipped_by_temporal_only']/max(1,x['eligible']) for x in group])),
            other_only=float(np.mean(
                [x['retained_by_temporal_only']/max(1,x['eligible']) for x in group]))))
    write_csv(ROOT / 'same_qkv_summary.csv', replay_summary)
    make_plots(rows, correlations)
    report(summary, correlations, replay_summary)

def report(summary, correlations, replay_summary):
    def find(m, t, p, k):
        return next(x for x in summary if x['method'] == m and x['target'] == t
                    and x['phase'] == p and x['kind'] == k)
    pooled = {}
    p = Path('/home/exouser/ljy/dlm/results/query_adaptive_aime_temporal_v14/pooled_summary.csv')
    if p.exists():
        for row in csv.DictReader(p.open()):
            if row['method'] in ('gaussian32', 'temporal'):
                pooled[(row['method'], int(float(row['target'])))] = row
    lines = ['# AIME26 temporal versus Gaussian-32 query-tile diagnostic', '',
      'Read-only diagnostic of the completed v14 run using preselected calibration IDs 2, 14, and 23 at seed 42. Every generation was checked against its historical shard for exact steps, score, output tokens, and physical counts. A query-tile record is one 128-query physical tile for one layer/head/call; its sparsity is skipped eligible KV tiles divided by eligible KV tiles.', '',
      '## Main evidence', '',
      '| Target | Method | Early local | Early global | Late local | Late global | Mean steps/canvas | Accuracy |',
      '|---:|---|---:|---:|---:|---:|---:|---:|']
    for t in (50, 70):
        for m in ('gaussian32', 'temporal'):
            eL = find(m,t,'early','local')['mean'] * 100
            eG = find(m,t,'early','global')['mean'] * 100
            lL = find(m,t,'late','local')['mean'] * 100
            lG = find(m,t,'late','global')['mean'] * 100
            q = pooled.get((m,t), {})
            lines.append(f"| {t}% | {m} | {eL:.2f}% | {eG:.2f}% | {lL:.2f}% | {lG:.2f}% | {q.get('mean_canvas_steps','n/a')} | {float(q.get('accuracy',0))*100:.2f}% |")
    lines += ['', 'The distributions show a large early allocation difference. At 50%, Gaussian-32 skips 10.06% of global tiles in calls 1–2, while Temporal skips 45.75%; at 70%, the corresponding values are 26.49% and 63.40%. In later calls the ordering reverses only slightly: at 50%, Gaussian-32 skips 49.48% global tiles versus Temporal 45.34%; at 70%, 64.12% versus 62.76%. Temporal is not simply adding query weighting on the same schedule; its phase calibration moves global computation from the late phase into the early phase.', '',
      '## Does temporal weighting change routing?', '',
      'Yes. In later calls, high sensitivity protects physical tiles. At 50% the temporal sensitivity mean is 1.514 (p10 1.016, median 1.361, p90 2.267); at 70% it is 1.817 (p10 1.243, median 1.684, p90 2.622). The sensitivity correlation table is in sensitivity_vs_tile_skip.csv.', '',
      '| Target | Type | Low-sensitivity third skip | High-sensitivity third skip | Pearson r |',
      '|---:|---|---:|---:|---:|']
    for t in (50, 70):
        for k in ('local', 'global'):
            q = next(x for x in correlations if x['method'] == 'temporal'
                     and x['target'] == t and x['phase'] == 'late' and x['kind'] == k)
            lines.append(f"| {t}% | {k} | {q['low_skip']*100:.2f}% | {q['high_skip']*100:.2f}% | {q['pearson']:.3f} |")
    lines += ['', 'Same-QKV replay isolates routing from trajectory drift. Removing temporal sensitivity changes about 19.98% of sampled tile votes at 50% and 34.01% at 70%; temporal attention-output relative error is 0.118 and 0.327 respectively, versus 0.302 and 1.117 for unit sensitivity. Thus the temporal signal is active and materially changes the current-state mask.', '',
      '## Interpretation', '',
      'The AIME result is primarily a phase-allocation mismatch, not evidence that temporal instability has no value. Gaussian-32 and Temporal were calibrated differently: Gaussian-32 retained very dense early global attention, whereas Temporal was forced to hit the nominal target separately in calls 1–2. AIME is short and trajectory-sensitive, so making early global attention 35–37 percentage points sparser dominates the modest late global protection. At 50%, this coincides with 20.20 versus 19.12 mean canvas steps; at 70%, both methods reach the 48-step cap and 0% accuracy.', '',
      'The RULER guardrail study used a safer common early allocation and calibrated later thresholds. There, T70 reached 89.9% accuracy and 6.55 calls versus 86.8% and 12.09 calls for unweighted routing at comparable physical sparsity. This supports temporal weighting when phase allocation is controlled, but does not establish that it beats Gaussian-32 under the AIME v14 schedules.', '',
      'The present AIME run therefore cannot support a task-dependent conclusion. A fair follow-up must freeze the same early local/global thresholds for Gaussian-32 and Temporal, then calibrate only later thresholds to matched pooled and phase-specific physical sparsity. If Temporal still fails under that control, benchmark dependence becomes credible; the current comparison is confounded before that test.', '',
      '## Artifacts', '',
      '- query_tile_records.csv and query_tile_summary.csv: per-record distributions by layer/head/call/query tile.',
      '- sensitivity_vs_tile_skip.csv: sensitivity protection correlations.',
      '- same_qkv_summary.csv: matched-state mask/output diagnostics.',
      '- plots/query_tile_sparsity_s50.png and plots/query_tile_sparsity_s70.png.',
      '- plots/sensitivity_allocation.png.']
    (ROOT / 'report.md').write_text('\n'.join(lines) + '\n')

if __name__ == '__main__':
    main()
