"""v27 FA4-panel summary (first-only panels): quality from scored.csv, per-cell numbers from the worker ledger's
redacted run events (no text): W = HF generation-call wall (api_wall_s; setup/cleanup excluded), S = decode span (phase_evidence
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


def read_cells(scored, ledgers):
    """Read immutable first outputs, requiring the scorer's execution identity.

    The qualified scorer already checks frozen assignments. Recheck at this
    independent timing join so a wrong ledger list cannot silently mix hosts or
    replace an earlier output when killed workers are merged.
    """
    quality = {}
    with open(scored, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            key = (r['dataset'], r['id'], r['seed'], r['arm'])
            if key in quality:
                raise ValueError('duplicate scored first output')
            quality[key] = r
    cells = defaultdict(dict)
    seen = set()
    successful = set()
    for path in ledgers:
        identity = None
        with open(path, encoding='utf-8') as f:
            for line in f:
                e = json.loads(line) if line.strip() else {}
                if e.get('event') == 'start':
                    identity = e
                    continue
                if e.get('event') != 'run' or e.get('role', 'attempt0') != 'attempt0':
                    continue
                key = (e['dataset'], e['id'], str(e['seed']))
                qkey = key + (e['arm'],)
                if qkey in seen:
                    raise ValueError('duplicate ledger first output; merge disjoint executions only')
                seen.add(qkey)  # failed first outputs must not be replaced by successful retries
                if not e.get('ok'):
                    continue
                if not identity or any(not identity.get(k) for k in
                                       ('protocol_id', 'model_revision', 'source_hashes')):
                    raise ValueError('successful first output lacks worker provenance')
                if any(not e.get(k) or e[k] != identity.get(k) for k in ('host', 'gpu_uuid')):
                    raise ValueError('run host/GPU differs from worker provenance')
                q = quality.get(qkey)
                if (not q or q.get('first_status') != 'success' or
                        q.get('scored_first') != 'True' or q.get('strict_correct') not in ('True', 'False')):
                    raise ValueError('successful first output lacks a qualified score')
                if any(not e.get(k) or str(e[k]) != q.get(k) for k in ('host', 'gpu_uuid', 'cell_id')):
                    raise ValueError('timing output differs from scored execution identity')
                span = (e.get('phase_evidence') or {}).get('prefill_end_to_finish_gpu_s')
                wall = e.get('api_wall_s') or e.get('outer_wall_s')
                substrate = e.get('substrate') or {}
                substrate = substrate.get('substrate', 'eager') if isinstance(substrate, dict) else substrate
                if substrate != identity.get('substrate', 'eager'):
                    raise ValueError('run substrate differs from worker provenance')
                cells[key][e['arm']] = dict(
                    W=wall, S=span, P=(wall - span) if wall and span else None,
                    N=e.get('decoder_calls'), T=e.get('output_tokens'),
                    C=e.get('canvases') or len(e.get('per_canvas_calls') or []) or None,
                    new_graphs=e.get('substrate_new_graphs'), correct=q['strict_correct'] == 'True',
                    host=e['host'], gpu_uuid=e['gpu_uuid'], substrate=substrate,
                    provenance={k: identity.get(k) for k in
                                ('protocol_id', 'model_revision', 'source_hashes', 'substrate_source_sha256')})
                successful.add(qkey)
    qualified = {k for k, q in quality.items() if q.get('first_status') == 'success'}
    if successful != qualified:
        raise ValueError('timing ledger inventory differs from successful scored first outputs')
    for values in cells.values():
        reference = next(iter(values.values()))
        for value in values.values():
            if any(value[k] != reference[k] for k in ('host', 'gpu_uuid', 'substrate', 'provenance')):
                raise ValueError('cell arms differ in host/GPU, substrate, protocol, model or source')
    return cells


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
    cells = read_cells(scored, ledgers)
    rows, lines = [], []
    for dataset in sorted({k[0] for k in cells}):
        arms = sorted({a for k, v in cells.items() if k[0] == dataset for a in v},
                      key=lambda a: (a not in BASES, BASES.index(a) if a in BASES else 0, a))
        lines += [f'### {dataset}', '']
        for base in [b for b in BASES if any(b in v for k, v in cells.items() if k[0] == dataset)]:
            lines += [f'#### paired vs {base}', '',
                      '| arm | cells | correct (base) | +/- vs base | W [CI] | W excl. timed new-graph pairs [CI] | decode S excl. prefill [CI] | prefill P [CI] | calls N [CI] | steps/canvas N/C [CI] | tokens T [CI] | S/N (amortized) [CI] | S per output token [CI] |',
                      '|---|---:|---|---|---|---|---|---|---|---|---|---|---|']
            for arm in arms:
                pairs = [(k, v[arm], v[base]) for k, v in cells.items() if k[0] == dataset and arm in v and base in v]
                if not pairs:
                    continue
                row = dict(dataset=dataset, base=base, arm=arm, cells=len(pairs),
                           correct=sum(bool(a['correct']) for _, a, _ in pairs),
                           base_correct=sum(bool(b['correct']) for _, _, b in pairs),
                           arm_only=sum(bool(a['correct']) and not b['correct'] for _, a, b in pairs),
                           base_only=sum(not a['correct'] and bool(b['correct']) for _, a, b in pairs),
                           timed_with_new_graphs=sum(bool(a['new_graphs']) or bool(b['new_graphs']) for _, a, b in pairs),
                           timed_graph_counter_unknown=sum(a['new_graphs'] is None or b['new_graphs'] is None
                                                           for _, a, b in pairs))
                for name, f in (('W', lambda a, b: a['W'] / b['W']), ('S', lambda a, b: a['S'] / b['S']),
                                ('P', lambda a, b: a['P'] / b['P']),
                                ('N', lambda a, b: a['N'] / b['N']),
                                ('NC', lambda a, b: (a['N'] / a['C']) / (b['N'] / b['C'])),
                                ('T', lambda a, b: a['T'] / b['T']),
                                ('SN', lambda a, b: (a['S'] / a['N']) / (b['S'] / b['N'])),
                                ('ST', lambda a, b: (a['S'] / a['T']) / (b['S'] / b['T'])),
                                # sensitivity: W over pairs with zero timed Dynamo unique_graphs increments (not CUDA captures)
                                ('Wc', lambda a, b: None if (a['new_graphs'] is None or b['new_graphs'] is None or
                                                           a['new_graphs'] or b['new_graphs']) else a['W'] / b['W'])):
                    by_q = defaultdict(list)
                    for k, a, b in pairs:
                        try:
                            if f(a, b) is None:
                                continue
                            by_q[k[1]].append(f(a, b))
                        except (TypeError, ZeroDivisionError):
                            pass
                    flat = [x for v in by_q.values() for x in v]
                    lo, hi = cluster_ci(by_q)
                    row[name] = round(geo(flat), 4) if flat else None
                    row[name + '_ci'] = f'[{lo:.3f},{hi:.3f}]' if lo else ''
                rows.append(row)
                lines.append(f"| {arm} | {row['cells']} | {row['correct']} ({row['base_correct']}) | "
                             f"+{row['arm_only']}/-{row['base_only']} | {row['W']} {row['W_ci']} | {row['Wc']} {row['Wc_ci']} ({row['timed_with_new_graphs']} pairs with Dynamo new graphs, {row['timed_graph_counter_unknown']} unknown excl.) | {row['S']} {row['S_ci']} | "
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
