"""Direct paired comparison of two method arms (arm / base), optionally pooled over several panels.

Cells come from each panel's qualified scored.csv and worker ledgers through ``v27_fa4_panel_summary.read_cells``
(same provenance checks: one host per cell, identical substrate/protocol/source within a cell). Panels are pooled
by stacking their cells (seeds differ between panels, so no cell is counted twice); the bootstrap clusters by
question (dataset, id) across panels. Ratios are geometric means: W request wall, Wc W excluding pairs that captured
new graphs while timed, S decode span, P prefill, N decoder calls, NC steps per canvas, T tokens, SN amortized
per-call cost, ST time per token. Accuracy: paired counts and an exact two-sided McNemar p. Also per host.
usage: python -m scripts.v27_direct_compare OUT.md OUT.csv ARM BASE --panel NAME SCORED.csv LEDGER [LEDGER ...]
       [--panel NAME SCORED.csv LEDGER ...]
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict

from scripts.v27_fa4_panel_summary import cluster_ci, geo, read_cells

METRICS = (('W', lambda a, b: a['W'] / b['W']), ('S', lambda a, b: a['S'] / b['S']),
           ('P', lambda a, b: a['P'] / b['P']), ('N', lambda a, b: a['N'] / b['N']),
           ('NC', lambda a, b: (a['N'] / a['C']) / (b['N'] / b['C'])), ('T', lambda a, b: a['T'] / b['T']),
           ('SN', lambda a, b: (a['S'] / a['N']) / (b['S'] / b['N'])),
           ('ST', lambda a, b: (a['S'] / a['T']) / (b['S'] / b['T'])),
           ('Wc', lambda a, b: None if (a['new_graphs'] is None or b['new_graphs'] is None or a['new_graphs']
                                        or b['new_graphs']) else a['W'] / b['W']))
HOSTS = {'149.165.159.64': 'dllm', '149.165.151.254': 'mpk', '149.165.168.28': 'dlm2'}


def mcnemar(b, c):
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def parse(argv):
    out_md, out_csv, arm, base = argv[:4]
    panels, cur = [], None
    for tok in argv[4:]:
        if tok == '--panel':
            cur = []
            panels.append(cur)
        else:
            cur.append(tok)
    if not panels or any(len(p) < 3 for p in panels):
        raise SystemExit(__doc__)
    return out_md, out_csv, arm, base, [(p[0], p[1], p[2:]) for p in panels]


def summarize(pairs):
    row = dict(cells=len(pairs), arm_correct=sum(a['correct'] for _, a, _ in pairs),
               base_correct=sum(b['correct'] for _, _, b in pairs),
               arm_only=sum(a['correct'] and not b['correct'] for _, a, b in pairs),
               base_only=sum(b['correct'] and not a['correct'] for _, a, b in pairs))
    row['mcnemar_p'] = round(mcnemar(row['arm_only'], row['base_only']), 3)
    for name, f in METRICS:
        by_q = defaultdict(list)
        for k, a, b in pairs:
            try:
                v = f(a, b)
            except (TypeError, ZeroDivisionError):
                continue
            if v is not None:
                by_q[(k[0], k[1])].append(v)
        flat = [x for v in by_q.values() for x in v]
        lo, hi = cluster_ci(by_q)
        row[name] = round(geo(flat), 4) if flat else None
        row[name + '_ci'] = f'[{lo:.3f}, {hi:.3f}]' if lo else ''
    return row


def main(argv=None):
    out_md, out_csv, arm, base, panels = parse(argv or sys.argv[1:])
    pairs = defaultdict(list)            # dataset -> [(key, arm cell, base cell)]
    for name, scored, ledgers in panels:
        for key, v in read_cells(scored, ledgers).items():
            if arm in v and base in v:
                pairs[key[0]].append(((key[0], key[1], key[2], name), v[arm], v[base]))
    rows = []
    lines = [f'# {arm} / {base} (direct, paired)', '', 'Panels: ' + ', '.join(n for n, _, _ in panels), '',
             '| bin | host | cells | acc arm / base (+/-, McNemar p) | W [CI] | Wc | S [CI] | per-step S/N [CI] | N [CI] | '
             'steps/canvas N/C | T | P |', '|---|---|---:|---|---|---|---|---|---|---|---|---|']
    for dataset in sorted(pairs):
        groups = [('all', pairs[dataset])] + sorted(
            (HOSTS.get(h, h), [p for p in pairs[dataset] if p[1]['host'] == h])
            for h in {p[1]['host'] for p in pairs[dataset]})
        for host, ps in groups:
            r = summarize(ps)
            r.update(dataset=dataset, host=host, arm=arm, base=base)
            rows.append(r)
            lines.append(f"| {dataset} | {host} | {r['cells']} | {r['arm_correct']} / {r['base_correct']} "
                         f"(+{r['arm_only']}/-{r['base_only']}, p {r['mcnemar_p']}) | {r['W']} {r['W_ci']} | {r['Wc']} | "
                         f"{r['S']} {r['S_ci']} | {r['SN']} {r['SN_ci']} | {r['N']} {r['N_ci']} | {r['NC']} | {r['T']} | "
                         f"{r['P']} |")
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    open(out_md, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
