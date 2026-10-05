"""v27 Tier-1 policy screen: composed per-call cost for every frozen A/R/threshold point.

CONDITIONAL ESTIMATE, not an executed result:
- canvas lengths come from recorded native-adaptive generations (native arm only), so the
  candidate's own trajectory (different lengths / call counts) is NOT modelled;
- per-phase costs are directly timed complete model_forward calls on common captured states,
  expressed as ratios to native on the same state and GPU (v27 profiles), averaged over the
  states of a task;
- threshold dependence of H/D/A comes from the M3 R3/A8 threshold sweep; M2c's D is scaled
  by its measured P0 ratio to M1's D; B's H uses the same H(threshold).
Clock (origin 1): call0 B0, call1 BO, anchors A at 1+kA (<n), decision age resets at any
anchor, D when age >= R, else H; hold-only B never issues D.
usage: python -m scripts.v27_policy_screen OUT.csv RECORDS_DIR [...] --cost CSV [...] --thr CSV [...]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

SHIFTS = ('minus_ln2', 'P0', 'plus_ln2', 'plus_2ln2')


def phases(n, a, r, hold):
    if n <= 0:
        return {}
    out = defaultdict(int)
    out['B0'] += 1
    if n >= 2:
        out['BO'] += 1
    last = 1
    for i in range(2, n):
        if (i - 1) % a == 0:
            out['A'] += 1
            last = i
        elif not hold and i - last >= r:
            out['D'] += 1
            last = i
        else:
            out['H'] += 1
    return out


def lengths(dirs, arm='D_native'):
    by = defaultdict(list)
    for d in dirs:
        for path in Path(d).rglob('attempt00.json'):
            rec = json.loads(path.read_text(encoding='utf-8'))
            if rec.get('condition') not in ('native_dense',) and arm == 'D_native':
                continue
            dataset = 'longbench_v2' if 'longbench' in str(rec.get('id')) else \
                'aime26' if 'aime' in str(rec.get('id')) else 'ruler4k'
            by[dataset] += [int(c['decoder_calls']) for c in rec.get('per_canvas') or []]
    return by


def ratios(cost_csvs, thr_csvs):
    """Per dataset: phase -> threshold -> mean ratio to native over states."""
    per = defaultdict(lambda: defaultdict(list))
    def add(rows, label_of):
        states = defaultdict(dict)
        for row in rows:
            if row['boundary'] != 'model_forward':
                continue
            states[(row['host'], row['dataset'], row['target'], row['canvas'])][row['arm']] = row
        for (host, dataset, target, canvas), arms in states.items():
            native = float(arms['D_native']['native_ms'])
            for arm, row in arms.items():
                label = label_of(arm)
                if label is None:
                    continue
                for phase in ('B0', 'BO', 'A', 'D', 'H'):
                    value = row.get(f'{phase}_ms')
                    if value not in (None, ''):
                        per[dataset][(label, phase)].append(float(value) / native)
    for path in thr_csvs:
        add(list(csv.DictReader(open(path, encoding='utf-8'))),
            lambda arm: ('M3', arm.replace('M3_R3_A8_', '')) if arm.startswith('M3_R3_A8_') else None)
    for path in cost_csvs:
        add(list(csv.DictReader(open(path, encoding='utf-8'))),
            lambda arm: {'M1_R1_A8': ('M1', 'P0'), 'M2c_pool_R1_A8': ('M2c', 'P0'),
                         'M2_pool_R1_A8': ('M2ref', 'P0'), 'M3_R3_A8': ('M3cost', 'P0'),
                         'B_A8': ('B', 'P0')}.get(arm))
    return {d: {k: statistics.mean(v) for k, v in m.items()} for d, m in per.items()}


def cost(rat, family, shift, phase):
    m3 = rat.get((('M3', shift), phase))
    if phase in ('B0',):
        return rat.get((('M3', shift), 'B0'), 1.0)
    if family == 'M2c' and phase in ('D', 'A'):
        m1 = rat.get((('M1', 'P0'), phase))
        m2 = rat.get((('M2c', 'P0'), phase))
        return None if None in (m3, m1, m2) else m3 * m2 / m1
    return m3


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('records', nargs='+')
    p.add_argument('--cost', nargs='+', required=True)
    p.add_argument('--thr', nargs='+', required=True)
    a = p.parse_args(argv)
    lens = lengths(a.records)
    rat_all = ratios(a.cost, a.thr)
    points = []
    for shift in ('minus_ln2', 'P0', 'plus_ln2'):
        for aa in (8, 16, 64):
            for r in (1, 3, 6):
                points.append(('M1' if r == 1 else 'M3', aa, r, False, shift))
            points.append(('B', aa, None, True, shift))
            points.append(('M2c', aa, 1, False, shift))
    rows = []
    for dataset, ns in lens.items():
        rat = {tuple(k) if isinstance(k, tuple) else k: v for k, v in rat_all.get(dataset, {}).items()}
        if not rat:
            continue
        for family, aa, r, hold, shift in points:
            totals, calls, missing = defaultdict(int), 0, set()
            for n in ns:
                for phase, c in phases(n, aa, r or 999, hold).items():
                    totals[phase] += c
                calls += n
            est = 0.
            for phase, c in totals.items():
                value = cost(rat, family, shift, phase)
                if value is None:
                    missing.add(phase)
                    continue
                est += c * value
            rows.append(dict(dataset=dataset, family=family, A=aa, R=r if not hold else 'hold', threshold=shift,
                             canvases=len(ns), calls=calls, **{f'n_{k}': totals.get(k, 0) for k in ('B0', 'BO', 'A', 'D', 'H')},
                             est_per_call_over_native=round(est / calls, 4) if not missing else None,
                             missing_phase_costs=' '.join(sorted(missing)),
                             label='composed estimate on recorded native canvas lengths; not executed'))
    with open(a.out, 'w', newline='', encoding='utf-8') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == '__main__':
    for row in main():
        print(row['dataset'][:4], row['family'], 'A%s' % row['A'], 'R%s' % row['R'], row['threshold'][:9].ljust(9),
              row['est_per_call_over_native'], row['missing_phase_costs'])
