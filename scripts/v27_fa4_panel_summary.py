"""v27 FA4-panel summary (first-only panels): quality from scored.csv, per-cell numbers from the worker ledger's
redacted run events (no text): W = request wall (api_wall_s), S = decode span (phase_evidence
prefill_end_to_finish_gpu_s: first encoder forward end -> last CUDA event, i.e. EXCLUDES the initial prompt
prefill, which no method changes), P = W - S (prompt prefill plus host setup; a ratio of 1 confirms the methods
leave it alone), N = decoder calls, T = output tokens, C = canvases, N/C = decoder calls (denoising steps) per
canvas. Paired by (dataset, id, seed) against each base arm; geometric-mean ratios with
a question-clustered 95% bootstrap. S/N is an AMORTIZED per-call cost (not a direct per-forward price); S/T is decode
time per output token.
usage: python -m scripts.v27_fa4_panel_summary OUT.md OUT.csv SCORED.csv LEDGER.jsonl [LEDGER.jsonl ...]
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict

BASES = ('D_fa4_allkept', 'D_fa4', 'D_native')


def geo(xs):
    return math.exp(statistics.mean(math.log(x) for x in xs)) if xs else None


def cluster_ci(by_q, reps=4000, seed=7):
    qs = list(by_q)
    if len(qs) < 2:
        return None, None
    rng = random.Random(seed)
    boots = sorted(geo([x for q in rng.choices(qs, k=len(qs)) for x in by_q[q]]) for _ in range(reps))
    return boots[int(.025 * reps)], boots[int(.975 * reps) - 1]


def main(argv=None):
    argv = argv or sys.argv[1:]
    md, out_csv, scored, ledgers = argv[0], argv[1], argv[2], argv[3:]
    quality = {}
    for r in csv.DictReader(open(scored, encoding='utf-8')):
        if r.get('first_status') == 'success':
            quality[(r['dataset'], r['id'], r['seed'], r['arm'])] = r['strict_correct'] == 'True'
    cells = defaultdict(dict)
    for path in ledgers:
        for line in open(path, encoding='utf-8'):
            e = json.loads(line) if line.strip() else {}
            if e.get('event') != 'run' or e.get('role', 'attempt0') != 'attempt0' or not e.get('ok'):
                continue
            key = (e['dataset'], e['id'], str(e['seed']))
            span = (e.get('phase_evidence') or {}).get('prefill_end_to_finish_gpu_s')
            wall = e.get('api_wall_s') or e.get('outer_wall_s')
            cells[key][e['arm']] = dict(W=wall, S=span, P=(wall - span) if wall and span else None,
                                        N=e.get('decoder_calls'), T=e.get('output_tokens'),
                                        C=e.get('canvases') or len(e.get('per_canvas_calls') or []) or None,
                                        new_graphs=e.get('substrate_new_graphs'),
                                        correct=quality.get(key + (e['arm'],)))
    rows, lines = [], []
    for dataset in sorted({k[0] for k in cells}):
        arms = sorted({a for k, v in cells.items() if k[0] == dataset for a in v},
                      key=lambda a: (a not in BASES, BASES.index(a) if a in BASES else 0, a))
        lines += [f'### {dataset}', '']
        for base in [b for b in BASES if any(b in v for k, v in cells.items() if k[0] == dataset)]:
            lines += [f'#### paired vs {base}', '',
                      '| arm | cells | correct (base) | +/- vs base | W [CI] | decode S excl. prefill [CI] | prefill P [CI] | calls N [CI] | steps/canvas N/C [CI] | tokens T [CI] | S/N (amortized) [CI] | S per output token [CI] |',
                      '|---|---:|---|---|---|---|---|---|---|---|---|---|']
            for arm in arms:
                pairs = [(k, v[arm], v[base]) for k, v in cells.items() if k[0] == dataset and arm in v and base in v]
                if not pairs:
                    continue
                row = dict(dataset=dataset, base=base, arm=arm, cells=len(pairs),
                           correct=sum(bool(a['correct']) for _, a, _ in pairs),
                           base_correct=sum(bool(b['correct']) for _, _, b in pairs),
                           arm_only=sum(bool(a['correct']) and not b['correct'] for _, a, b in pairs),
                           base_only=sum(not a['correct'] and bool(b['correct']) for _, a, b in pairs),
                           timed_with_new_graphs=sum(bool(a['new_graphs']) or bool(b['new_graphs']) for _, a, b in pairs))
                for name, f in (('W', lambda a, b: a['W'] / b['W']), ('S', lambda a, b: a['S'] / b['S']),
                                ('P', lambda a, b: a['P'] / b['P']),
                                ('N', lambda a, b: a['N'] / b['N']),
                                ('NC', lambda a, b: (a['N'] / a['C']) / (b['N'] / b['C'])),
                                ('T', lambda a, b: a['T'] / b['T']),
                                ('SN', lambda a, b: (a['S'] / a['N']) / (b['S'] / b['N'])),
                                ('ST', lambda a, b: (a['S'] / a['T']) / (b['S'] / b['T']))):
                    by_q = defaultdict(list)
                    for k, a, b in pairs:
                        try:
                            by_q[k[1]].append(f(a, b))
                        except (TypeError, ZeroDivisionError):
                            pass
                    flat = [x for v in by_q.values() for x in v]
                    lo, hi = cluster_ci(by_q)
                    row[name] = round(geo(flat), 4) if flat else None
                    row[name + '_ci'] = f'[{lo:.3f},{hi:.3f}]' if lo else ''
                rows.append(row)
                lines.append(f"| {arm} | {row['cells']} | {row['correct']} ({row['base_correct']}) | "
                             f"+{row['arm_only']}/-{row['base_only']} | {row['W']} {row['W_ci']} | {row['S']} {row['S_ci']} | "
                             f"{row['P']} {row['P_ci']} | {row['N']} {row['N_ci']} | {row['NC']} {row['NC_ci']} | "
                             f"{row['T']} {row['T_ci']} | {row['SN']} {row['SN_ci']} | {row['ST']} {row['ST_ci']} |")
            lines.append('')
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    open(md, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
