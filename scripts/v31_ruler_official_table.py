"""RULER arms side by side under BOTH scoring rules, on their common cells: the official RULER score (per-sample metric
0-100 with partial credit, mean per task, then mean over the 13 tasks; per length and overall), its cwe and no-cwe
parts, the strict all-correct count (stop / eos required) used for McNemar, the paired official-score difference to a
reference arm (mean per cell, 95% bootstrap CI over cells, better / worse cells, sign test), and the GLOBAL work.
Aggregates only. usage:
  python v31_ruler_official_table.py MANIFEST_DIR REF_LABEL RAW.json STRICT.json [--alias NEW=A+B ...]
      [--only SUBSTR,...] [--public PUBLIC.jsonl ...]
"""
import collections
import json
import random
import statistics
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v31_ruler_compare import compute  # noqa: E402


def sign_p(pos, neg):
    n = pos + neg
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n)


def main():
    args = sys.argv[1:]
    pub = args[args.index('--public') + 1:] if '--public' in args else []
    args = args[:args.index('--public')] if '--public' in args else args
    aliases, only = {}, None
    while '--alias' in args:
        i = args.index('--alias')
        new, parts = args[i + 1].split('=')
        aliases[new] = parts.split('+')
        del args[i:i + 2]
    if '--only' in args:
        i = args.index('--only')
        only = args[i + 1].split(',')
        del args[i:i + 2]
    man_dir, ref, raw_path, strict_path = Path(args[0]), args[1], args[2], args[3]
    raw, strict = json.loads(Path(raw_path).read_text()), json.loads(Path(strict_path).read_text())
    for new, parts in aliases.items():
        for d in (raw, strict):
            if all(p in d for p in parts):
                d[new] = {k: v for p in parts for k, v in d.pop(p).items()}
    labels = [l for l in raw if l == ref or only is None or any(s in l for s in only)]
    common = set.intersection(*(set(raw[l]) for l in labels))
    owner = {p: new for new, parts in aliases.items() for p in parts}
    work = collections.defaultdict(dict)
    for f in pub:
        label = Path(f).name.split('.jsonl')[0].split('_', 1)[1]
        label = owner.get(label, label)
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            key = f"{r['dataset']}|{r['index']}|{r['panel_seed']}|{r['repeat']}"
            c = compute(r)
            if key in common and c is not None:
                work[label][key] = c
    lengths = sorted({k.split('|')[0] for k in common})
    task_of = {ds: [r['task'] for r in json.loads((man_dir / f'{ds}_generation_manifest.json').read_text())]
               for ds in lengths}
    task = lambda k: task_of[k.split('|')[0]][int(k.split('|')[1])]

    def official(label, keys):
        per = collections.defaultdict(list)
        for k in keys:
            per[task(k)].append(raw[label][k])
        return statistics.mean(statistics.mean(v) for v in per.values()) if per else float('nan')

    print(f'common cells: {len(common)}; reference {ref}\n')
    head = (['arm'] + [f'{ln} official' for ln in lengths] +
            ['official', 'cwe', 'w/o cwe', 'strict correct', 'official diff vs ref [95% CI]', 'better/worse (p)', 'work'])
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    rng = random.Random(7)
    ref_keys = sorted(common)
    for label in [ref] + sorted(l for l in labels if l != ref):
        per_len = [official(label, [k for k in common if k.startswith(ln + '|')]) for ln in lengths]
        allk = sorted(common)
        tasks = sorted({task(k) for k in allk})
        per_task = {t: statistics.mean(raw[label][k] for k in allk if task(k) == t) for t in tasks}
        over = statistics.mean(per_task.values())
        nocwe = statistics.mean(v for t, v in per_task.items() if t != 'cwe')
        row = [label] + [f'{x:.1f}' for x in per_len] + [f'{over:.1f}', f"{per_task.get('cwe', float('nan')):.1f}",
                                                       f'{nocwe:.1f}', f"{sum(strict[label][k] for k in allk)}/{len(allk)}"]
        if label == ref:
            row += ['-', '-']
        else:
            d = [raw[label][k] - raw[ref][k] for k in ref_keys]
            boots = sorted(statistics.mean(rng.choice(d) for _ in d) for _ in range(4000))
            pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
            row += [f'{statistics.mean(d):+.2f} [{boots[100]:+.2f}, {boots[3899]:+.2f}]', f'{pos}/{neg} ({sign_p(pos, neg):.3f})']
        w = work.get(label)
        if w and len(w) == len(common):
            t = [sum(x[i] for x in w.values()) for i in range(4)]
            row.append(f'{t[2] / t[3]:.3f}')
        else:
            row.append('-')
        print('| ' + ' | '.join(row) + ' |')


if __name__ == '__main__':
    main()
