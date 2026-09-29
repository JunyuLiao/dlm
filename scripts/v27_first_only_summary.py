"""v27 first-only panel summary: scored.csv (quality) joined with per-cell first-receipt numbers
(decode span, wall, calls, canvases, tokens) by cell_id; paired geometric ratios vs a base arm with a
question-clustered bootstrap. No text is read. usage:
python -m scripts.v27_first_only_summary OUT.md OUT.csv SCORED.csv TIMING.csv [TIMING.csv ...]
"""
from __future__ import annotations

import csv
import math
import random
import statistics
import sys
from collections import defaultdict


def geo(xs):
    return math.exp(statistics.mean(math.log(x) for x in xs)) if xs else float('nan')


def cluster_ci(by_q, reps=4000, seed=7):
    qs = list(by_q)
    if len(qs) < 2:
        return None, None
    rng = random.Random(seed)
    boots = sorted(geo([x for q in rng.choices(qs, k=len(qs)) for x in by_q[q]]) for _ in range(reps))
    return boots[int(.025 * reps)], boots[int(.975 * reps)]


def main(argv=None):
    argv = argv or sys.argv[1:]
    md, out_csv, scored, timings = argv[0], argv[1], argv[2], argv[3:]
    quality = {r['cell_id']: r for r in csv.DictReader(open(scored, encoding='utf-8'))}
    cells = {}
    for path in timings:
        for r in csv.DictReader(open(path, encoding='utf-8')):
            r.update({k: quality[r['cell_id']][k] for k in ('task_correct', 'strict_correct', 'capped', 'parsed')})
            cells[r['cell_id']] = r
    failed = defaultdict(int)
    for r in quality.values():
        if r['first_status'] != 'success':
            failed[(r['dataset'], r['arm'])] += 1
    by = defaultdict(dict)
    for r in cells.values():
        by[(r['dataset'], r['id'], r['seed'])][r['arm']] = r
    arms = sorted({r['arm'] for r in cells.values()}, key=lambda a: (not a.startswith('D_'), a))
    rows, lines = [], []
    for dataset in sorted({k[0] for k in by}):
        lines += [f'### {dataset}', '',
                  '| arm | failed | correct | vs D_c64 +/- | vs D_native +/- | capped | decode span S [CI] | calls N [CI] | '
                  'S per call [CI] | wall W [CI] | tokens | canvases |',
                  '|---|---:|---:|---|---|---:|---|---|---|---|---:|---:|']
        for arm in arms:
            present = {k: v for k, v in by.items() if k[0] == dataset and arm in v}
            if not present:
                continue
            first = [v[arm] for v in present.values()]
            row = dict(dataset=dataset, arm=arm, cells=len(first), failed=failed[(dataset, arm)],
                       correct=sum(r['task_correct'] == 'True' for r in first),
                       capped=sum(r['capped'] == 'True' for r in first))
            for base, tag in (('D_c64', 'c64'), ('D_native', 'nat')):
                pairs = [(v[arm], v[base]) for v in present.values() if base in v]
                row[f'{tag}_better'] = sum(a['task_correct'] == 'True' and b['task_correct'] != 'True' for a, b in pairs)
                row[f'{tag}_worse'] = sum(a['task_correct'] != 'True' and b['task_correct'] == 'True' for a, b in pairs)
            ratios = defaultdict(lambda: defaultdict(list))
            for (d, q, s), v in present.items():
                if 'D_c64' not in v:
                    continue
                a, b = v[arm], v['D_c64']
                sa, sb = float(a['decode_span_s'] or 'nan'), float(b['decode_span_s'] or 'nan')
                na, nb = int(a['decoder_calls']), int(b['decoder_calls'])
                wa, wb = float(a['request_wall_s']), float(b['request_wall_s'])
                for key, val in (('S', sa / sb), ('N', na / nb), ('SN', (sa / na) / (sb / nb)), ('W', wa / wb)):
                    if val == val and val > 0:
                        ratios[key][q].append(val)
            for key in ('S', 'N', 'SN', 'W'):
                flat = [x for xs in ratios[key].values() for x in xs]
                lo, hi = cluster_ci(ratios[key])
                row[key] = round(geo(flat), 3) if flat else None
                row[key + '_ci'] = f'[{lo:.3f},{hi:.3f}]' if lo else ''
            row['tokens'] = round(statistics.mean(int(r['output_tokens'] or 0) for r in first))
            row['canvases'] = round(statistics.mean(int(r['canvases']) for r in first), 1)
            rows.append(row)
            lines.append(f"| {arm} | {row['failed']} | {row['correct']}/{row['cells']} | +{row['c64_better']}/-{row['c64_worse']} | "
                         f"+{row['nat_better']}/-{row['nat_worse']} | {row['capped']} | {row['S']} {row['S_ci']} | "
                         f"{row['N']} {row['N_ci']} | {row['SN']} {row['SN_ci']} | {row['W']} {row['W_ci']} | "
                         f"{row['tokens']} | {row['canvases']} |")
        lines.append('')
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(out_csv, 'w', newline='\n', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    open(md, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
