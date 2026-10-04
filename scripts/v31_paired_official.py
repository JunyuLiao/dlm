"""Paired comparison of arms against a reference arm on the OFFICIAL per-sample metric of LongBench-v2, AIME26,
HumanEval, MRCR or GraphWalks (aggregates only).

Inputs: the .official.json and .binding.json of the dataset's scorer (scripts/v31_score_*.py) and the planned cells.
- Cells: one explicit intersection over ALL arms in the file (not only those shown with --only); a cell outside the plan
  raises; a planned cell missing in an arm raises unless --allow-missing; per arm planned / present / dropped is printed.
- Comparability: per cell every arm must have run with identical rng_seed, budget, prompt_tokens, max_model_len, chunk,
  block_size, prompt and manifest sha256 (v31_official.check_comparable), else the tool refuses.
- Unit = item (dataset, index); its cells (seeds x repeats) are nested. Point estimate = mean over items of each item's
  mean over its cells (the official number: accuracy, avg@k, pass@1, mean ratio / F1). 95% CI of the paired difference:
  bootstrap over items with the cells resampled within each drawn item (4000 draws).
- Tests: boolean metrics (LongBench judge, AIME exact, HumanEval pass) -> exact McNemar on the discordant cells;
  continuous metrics (MRCR ratio, GraphWalks F1) -> exact sign test on the per-cell differences.
- Kind-specific: longbench also gives result.py's Overall / Easy / Hard / Short / Medium / Long per arm (mean over runs of
  unrounded run values) and their differences; mrcr / graphwalks are per bin (dataset), graphwalks also per problem type.
usage: python v31_paired_official.py {longbench,aime,humaneval,mrcr,graphwalks} MANIFEST_DIR REF_LABEL OFFICIAL.json
           BINDING.json --cells CELLS.json [--cells ...] [--repeats N] [--allow-missing] [--legacy-unbound]
           [--only SUBSTR,...] [--out FILE.md]
"""
import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_official as op  # noqa: E402

BOOLEAN = {'longbench': 'accuracy', 'aime': 'avg@k', 'humaneval': 'pass@1'}
CONTINUOUS = {'mrcr': 'mean ratio', 'graphwalks': 'mean F1'}


def item_of(key):
    ds, index, _, _ = key.split('|')
    return ds, int(index)


def groups(values, keys):
    """{item: [values of its cells]} over keys."""
    out = collections.defaultdict(list)
    for k in sorted(keys):
        out[item_of(k)].append(values[k])
    return out


def compare(arm, ref, keys, boolean, scale):
    """(arm metric, ref metric, diff, lo, hi, a, b, p) on keys: metric = nested mean x scale; (a, b) = McNemar
    discordant (arm only / ref only correct) or sign-test better / worse cells."""
    d = groups({k: scale * (float(arm[k]) - float(ref[k])) for k in keys}, keys)
    point, lo, hi = op.paired_bootstrap(d)
    m_arm = scale * op.nested_mean(groups({k: float(arm[k]) for k in keys}, keys))
    m_ref = scale * op.nested_mean(groups({k: float(ref[k]) for k in keys}, keys))
    if boolean:
        a = sum(bool(arm[k]) and not ref[k] for k in keys)
        b = sum(bool(ref[k]) and not arm[k] for k in keys)
    else:
        a = sum(arm[k] > ref[k] for k in keys)
        b = sum(arm[k] < ref[k] for k in keys)
    return m_arm, m_ref, point, lo, hi, a, b, op.binomial_two_sided(a, b)


def longbench_columns(values, keys, meta):
    """result.py per run (panel seed, repeat), unrounded, averaged over runs."""
    runs = collections.defaultdict(list)
    for k in keys:
        _, _, seed, rep = k.split('|')
        runs[(seed, rep)].append(dict(judge=bool(values[k]), **meta[item_of(k)]))
    per = [op.lb_result(v, digits=None) for v in runs.values()]
    return {c: (statistics.mean(r[c] for r in per) if all(r[c] is not None for r in per) else None) for c in op.LB_COLUMNS}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('kind', choices=sorted({**BOOLEAN, **CONTINUOUS}))
    p.add_argument('manifest_dir')
    p.add_argument('ref')
    p.add_argument('official')
    p.add_argument('binding')
    p.add_argument('--cells', action='append', required=True)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--allow-missing', action='store_true')
    p.add_argument('--legacy-unbound', action='store_true')
    p.add_argument('--only', default=None)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    official = json.loads(Path(a.official).read_text())
    binding = json.loads(Path(a.binding).read_text())
    if a.ref not in official:
        raise SystemExit(f'reference {a.ref} not in {a.official}')
    labels = sorted(official)
    plan = op.planned_cells(a.cells, a.repeats)
    common, coverage = op.common_cells({l: official[l] for l in labels}, plan, a.allow_missing)
    op.check_comparable(binding, labels, common, a.legacy_unbound)
    shown = [l for l in labels if l != a.ref and (a.only is None or any(s in l for s in a.only.split(',')))]
    datasets = sorted({k.split('|')[0] for k in common}, key=op.natural_key)
    fields = dict(longbench=('difficulty', 'length'), graphwalks=('problem_type',)).get(a.kind, ())
    rows = {ds: op.load_pool(a.manifest_dir, ds, fields)[0] for ds in datasets}
    meta = {(ds, i): {f: r.get(f) for f in fields} for ds, rs in rows.items() for i, r in enumerate(rs)}
    boolean = a.kind in BOOLEAN
    scale = 100.0 if boolean else 1.0
    metric = BOOLEAN.get(a.kind) or CONTINUOUS[a.kind]
    lines = [f'{a.kind}: {len(common)} common cells; reference {a.ref}; comparable settings checked on every cell', '']
    lines += ['coverage: ' + x for x in op.coverage_lines(coverage)] + ['']
    test = 'McNemar (arm only / ref only correct)' if boolean else 'sign test (better / worse cells)'
    head = ['arm', 'subset', metric, f'ref {metric}', 'diff [95% CI]', test, 'cells', 'items']
    lines += ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    subsets = []
    for ds in datasets:
        keys = {k for k in common if k.startswith(ds + '|')}
        subsets.append((ds, keys))
        if a.kind == 'graphwalks':
            for t in sorted({meta[item_of(k)]['problem_type'] for k in keys}):
                subsets.append((f'{ds} {t}', {k for k in keys if meta[item_of(k)]['problem_type'] == t}))
    if len(datasets) > 1 and boolean:
        subsets.append(('all', common))
    for label in shown:
        for name, keys in subsets:
            m_arm, m_ref, d, lo, hi, x, y, pv = compare(official[label], official[a.ref], keys, boolean, scale)
            fmt = '{:.1f}' if boolean else '{:.4f}'
            lines.append('| ' + ' | '.join([label, name, fmt.format(m_arm), fmt.format(m_ref),
                                             f'{d:+.2f} [{lo:+.2f}, {hi:+.2f}]' if boolean else f'{d:+.4f} [{lo:+.4f}, {hi:+.4f}]',
                                             f'{x}/{y} (p={pv:.3f})', str(len(keys)), str(len({item_of(k) for k in keys}))]) + ' |')
    if a.kind == 'longbench':
        lines += ['', "result.py columns (mean over runs of unrounded values; one decimal); diff = arm - ref", '',
                  '| ' + ' | '.join(['arm', 'dataset', *op.LB_COLUMNS]) + ' |', '|' + '---|' * (2 + len(op.LB_COLUMNS))]
        for ds in datasets:
            keys = {k for k in common if k.startswith(ds + '|')}
            ref_cols = longbench_columns(official[a.ref], keys, meta)
            lines.append('| ' + ' | '.join([a.ref, ds, *('-' if v is None else f'{v:.1f}' for v in ref_cols.values())]) + ' |')
            for label in shown:
                cols = longbench_columns(official[label], keys, meta)
                cells = [('-' if cols[c] is None else f'{cols[c]:.1f} ({cols[c] - ref_cols[c]:+.1f})') for c in op.LB_COLUMNS]
                lines.append('| ' + ' | '.join([label, ds, *cells]) + ' |')
    text = '\n'.join(lines) + '\n'
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
