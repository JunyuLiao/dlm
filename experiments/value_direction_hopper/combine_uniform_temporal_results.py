"""Combine the completed Temporal v14 and uniform-baseline AIME reports."""
from __future__ import annotations
import csv, glob, json, math
from collections import defaultdict
from pathlib import Path

TEMP = Path('/home/exouser/ljy/dlm/results/query_adaptive_aime_temporal_v14')
UNIF = Path('/home/exouser/ljy/dlm/results/query_adaptive_aime_uniform_baselines_v1')
OUT = UNIF


def load(root, label):
    groups = defaultdict(list)
    for f in glob.glob(str(root / 'final/*/*.json')):
        try: d = json.loads(Path(f).read_text())
        except Exception: continue
        if d.get('status') == 'complete': groups[(label, d['condition'])].append(d)
    return groups


def counts(rows, phase=None):
    out = {k: {'eligible': 0, 'skipped': 0} for k in ('whole', 'global', 'local')}
    for r in rows:
        src = r.get('counts', {}) if phase is None else r.get('phase_counts', {}).get(phase, {})
        for k in out:
            for z in out[k]: out[k][z] += int(src.get(k, {}).get(z, 0))
    return out


def rates(c):
    return {k: 100*c[k]['skipped']/max(1, c[k]['eligible']) for k in c}


def stats(rows):
    allc = counts(rows)
    ec = {k: {z: counts(rows, 'call1')[k][z] + counts(rows, 'call2')[k][z]
              for z in ('eligible', 'skipped')} for k in allc}
    lc = counts(rows, 'late')
    canv = [int(c['iterations']) for r in rows for c in r.get('canvases', [])]
    phases = {'early': [], 'late': []}
    for r in rows:
        for x in r.get('step_records', []): phases['early' if int(x['iteration']) <= 2 else 'late'].append(x)
    def mean(p, key):
        v = [float(x[key]) for x in phases[p] if x.get(key) is not None]
        return sum(v)/len(v) if v else None
    method = 'dense' if rows[0]['condition'].startswith('dense') else rows[0]['method']
    target = 'dense' if method == 'dense' else rows[0]['condition'].split('_s')[1].split('_')[0]
    return dict(method=method, target=target, n=len(rows), accuracy=100*sum(float(r['score']) for r in rows)/len(rows),
        overall_sparsity=rates(allc)['whole'], global_sparsity=rates(allc)['global'], local_sparsity=rates(allc)['local'],
        mean_steps_canvas=sum(canv)/len(canv) if canv else None, median_steps_canvas=sorted(canv)[len(canv)//2] if canv else None,
        p90_steps_canvas=sorted(canv)[max(0, math.ceil(.9*len(canv))-1)] if canv else None,
        canvases=len(canv), cap_canvases=sum(x >= 48 for x in canv), e2e_seconds=sum(float(r.get('seconds', 0)) for r in rows),
        early_overall=rates(ec)['whole'], early_global=rates(ec)['global'], early_local=rates(ec)['local'],
        late_overall=rates(lc)['whole'], late_global=rates(lc)['global'], late_local=rates(lc)['local'],
        early_conf=mean('early','confidence_mean'), late_conf=mean('late','confidence_mean'),
        early_entropy=mean('early','processed_entropy_mean'), late_entropy=mean('late','processed_entropy_mean'),
        early_accepted=mean('early','accepted'), late_accepted=mean('late','accepted'),
        early_renoised=mean('early','renoised'), late_renoised=mean('late','renoised'),
        early_flips=mean('early','argmax_flips'), late_flips=mean('late','argmax_flips'))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def pct(x): return '—' if x is None else f'{x:.2f}%'
def num(x): return '—' if x is None else f'{x:.2f}'


def main():
    groups = {}; groups.update(load(TEMP, 'temporal_v14')); groups.update(load(UNIF, 'uniform'))
    rows = []
    # Dense is taken once from the configuration-matched Temporal v14 run.
    dense = [r for (label,c), rs in groups.items() if c.startswith('dense') for r in rs]
    rows.append(stats(dense))
    pooled = defaultdict(list)
    for (label,c), rs in sorted(groups.items()):
        if c.startswith('dense'): continue
        method = rs[0]['method']
        if label == 'temporal_v14' and method != 'temporal': continue
        target = c.split('_s')[1].split('_')[0]
        pooled[(method, target)].extend(rs)
    for rs in pooled.values():
        assert len(rs) == 90
        assert len({(r['id'], r['seed']) for r in rs}) == 90
        rows.append(stats(rs))
    rows.sort(key=lambda r: (0 if r['method']=='dense' else 1, r['method'], -1 if r['target']=='dense' else int(r['target'])))
    write_csv(OUT/'combined_summary.csv', rows)
    # Same-state attention-output diagnostic (36 calls/target; not full-run).
    diag = Path('/home/exouser/ljy/dlm/results/aime_temporal_tile_diagnostic_v1/same_qkv_summary.csv')
    if diag.exists():
        with diag.open() as f: drows = list(csv.DictReader(f))
        write_csv(OUT/'attention_error_diagnostic.csv', drows)
    lines = ['# Combined AIME26 results: Temporal, uniform Gaussian32, uniform BLASST', '',
        'Dense is the matched reference from Temporal v14. Temporal uses its previously frozen two-stage phase schedule; Gaussian32 and aggressive BLASST use the new one-stage thresholds copied identically to all denoising steps. Runtime is the sum of instrumented per-example generation timers across 30 problems and three seeds, excluding model loading and calibration. These are not warmed, interleaved speed benchmarks. Calibration overlaps six final questions. Search-incomplete policies are not evidence of unattainability; the coarse search failed to resolve intermediate thresholds. Residual local/global mismatches must be retained in comparisons.', '',
        '## Main results', '', '| Method | Target | Actual O/G/L sparsity | Accuracy | Mean steps/canvas | Canvases | E2E seconds |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['method']} | {r['target']} | {pct(r['overall_sparsity'])} / {pct(r['global_sparsity'])} / {pct(r['local_sparsity'])} | {r['accuracy']:.2f}% | {r['mean_steps_canvas']:.2f} | {r['canvases']} | {r['e2e_seconds']:.1f} |")
    lines += ['', '## Early (steps 1–2) versus later routing', '', '| Method | Target | Early O/G/L | Later O/G/L | Early confidence / entropy | Later confidence / entropy | Early accepted / flips | Later accepted / flips |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['method']=='dense': continue
        fields = [r['method'], r['target']]
        for phase in ('early', 'late'):
            fields.append(' / '.join(pct(r[f'{phase}_{k}']) for k in ('overall', 'global', 'local')))
        for keys in (('early_conf', 'early_entropy'), ('late_conf', 'late_entropy'),
                     ('early_accepted', 'early_flips'), ('late_accepted', 'late_flips')):
            fields.append(' / '.join(num(r[k]) for k in keys))
        lines.append('| ' + ' | '.join(fields) + ' |')
    lines += ['', '## Attention-output error', '', 'Final full-run shards did not record dense-versus-sparse attention-output error. The separate same-QKV diagnostic is available in `attention_error_diagnostic.csv`; it contains 36 replay calls per target for Temporal and a frozen Gaussian threshold control, not all final examples and not BLASST. Relative L2 error is reported there.']
    (OUT/'combined_results.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__': main()
