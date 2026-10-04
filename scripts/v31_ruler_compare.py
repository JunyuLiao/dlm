"""Compare RULER arms on their common cells: totals per length, lost / gained vs a reference, exact McNemar p, by-task
counts, and the realized compute from the public records. Aggregates only (no ids, text or gold).
usage: python v31_ruler_compare.py MANIFEST_DIR REF_LABEL SCORES.json [...] [--alias NEW=A+B ...] [--public PUBLIC.jsonl ...]
  MANIFEST_DIR holds {dataset}_generation_manifest.json (task of each index; cell index = manifest position, checked
  for the v31 pool); labels are the keys of the score files (several files may hold parts of one label: merged).
  --alias joins labels run per length (e.g. a MAGE budget per length) into one arm.
Compute columns, over the arm's common cells (one record per cell; tile-weighted):
  sparse_kept  kept fraction of wholly-prefix tiles in the genuinely sparse GLOBAL calls;
  work         all GLOBAL calls (dense steps count every prefix tile) over the dense equivalent.
  Records before 2026-10-04 lack the exact fields; both are then reconstructed from the call counters, assuming one
  per-call prefix size per request (exact for single-canvas RULER answers).
"""
import json
import sys
from math import comb
from pathlib import Path

GROUPS = {'cwe': ('cwe',), 'fwe': ('fwe',), 'qa': ('qa_1', 'qa_2'), 'mv': ('niah_multivalue',), 'vt': ('vt',)}


def mcnemar(lost, gained):
    n = lost + gained
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(lost, gained) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compute(rec):
    """(sparse kept tiles, sparse tiles, GLOBAL work tiles, GLOBAL tiles) of one public record, or None."""
    a = (rec.get('receipts') or {}).get('adapter') or {}
    m = (rec.get('receipts') or {}).get('method') or {}
    if 'kept_prefix_tiles' not in a or not a.get('split_fa4_calls'):
        return None
    if 'global_prefix_work_fraction' in a:                           # exact receipts (2026-10-04 on)
        g = a['global_prefix_tiles']
        return (a.get('sparse_only_kept_tiles', 0), a.get('sparse_only_prefix_tiles', 0),
                a['global_prefix_work_fraction'] * g, g)
    per_call = a['sparse_prefix_tiles'] / a['split_fa4_calls']
    d = m.get('bootstrap_dense_calls', 0) if m else 0                 # dense calls routed through the split lists
    sparse_calls = a['split_fa4_calls'] - d
    sparse_kept = a['kept_prefix_tiles'] - d * per_call
    sparse_tiles = sparse_calls * per_call
    global_tiles = a['global_calls'] * per_call
    work = global_tiles - sparse_tiles + sparse_kept
    return sparse_kept, sparse_tiles, work, global_tiles


def main():
    args = sys.argv[1:]
    pub = args[args.index('--public') + 1:] if '--public' in args else []
    args = args[:args.index('--public')] if '--public' in args else args
    aliases = {}
    while '--alias' in args:
        i = args.index('--alias')
        new, parts = args[i + 1].split('=')
        aliases[new] = parts.split('+')
        del args[i:i + 2]
    man_dir, ref, score_files = Path(args[0]), args[1], args[2:]
    scores = {}
    for f in score_files:
        for label, cells in json.loads(Path(f).read_text()).items():
            scores.setdefault(label, {}).update(cells)
    for new, parts in aliases.items():
        scores[new] = {k: v for p in parts for k, v in scores.pop(p).items()}
    owner = {p: new for new, parts in aliases.items() for p in parts}
    task_of = {}
    for ds in sorted({k.split('|')[0] for v in scores.values() for k in v}):
        rows = json.loads((man_dir / f'{ds}_generation_manifest.json').read_text())
        task_of[ds] = [r['task'] for r in rows]
    common = set.intersection(*(set(v) for v in scores.values()))
    comp = {}
    for f in pub:
        label = Path(f).name.split('.jsonl')[0].split('_', 1)[1]
        label = owner.get(label, label)
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            key = f"{r['dataset']}|{r['index']}|{r['panel_seed']}|{r['repeat']}"
            c = compute(r)
            if key in common and c is not None:
                comp.setdefault(label, {})[key] = c                    # one record per cell (append-mode repeats)
    lengths = sorted({k.split('|')[0] for k in common})
    head = ['arm'] + lengths + ['total', 'lost/gained', 'McNemar p'] + list(GROUPS) + ['sparse_kept', 'work']
    print(f'common cells: {len(common)}\n')
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    r = scores[ref]
    for label in [ref] + sorted(x for x in scores if x != ref):
        s = scores[label]
        row = [label] + [str(sum(s[k] for k in common if k.startswith(ln + '|'))) for ln in lengths]
        row.append(f'{sum(s[k] for k in common)}/{len(common)}')
        lost = sum(r[k] and not s[k] for k in common)
        gained = sum(s[k] and not r[k] for k in common)
        row += ['-', '-'] if label == ref else [f'{lost}/{gained}', f'{mcnemar(lost, gained):.3f}']
        for tasks in GROUPS.values():
            ks = [k for k in common if task_of[k.split('|')[0]][int(k.split('|')[1])] in tasks]
            row.append(f'{sum(s[k] for k in ks)}/{len(ks)}')
        c = comp.get(label)
        if c and len(c) == len(common):
            t = [sum(x[i] for x in c.values()) for i in range(4)]
            row += [f'{t[0] / t[1]:.3f}' if t[1] else '-', f'{t[2] / t[3]:.3f}']
        else:
            row += ['-', '-']
        print('| ' + ' | '.join(row) + ' |')


if __name__ == '__main__':
    main()
