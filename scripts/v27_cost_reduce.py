"""Reduce v27 direct-cost profiles into per-arm, per-phase tables (no model, no GPU).

For each target/boundary/arm:
- sequence: per-call phase letters (0=B0, O=BO, A, D, H, N=native);
- per-phase median ms (median over repetitions of each directly timed call);
- mean ms per call over the reached N and its ratio to native on the same state;
- physical GLOBAL work by phase from the untimed counter twin: QK and PV skipped
  fraction of legal pairs (5 routed GLOBAL layers only; LOCAL layers are native);
- prepared-support floor (model_forward): each call receives its own frozen bitmap
  with no selector/cache/sketch work. Bootstrap calls run the all-kept consumer.

Selection overhead = D - H and observation overhead = A - D are derived per state
from medians of directly timed calls; they are not separate kernel timings.
usage: python -m scripts.v27_cost_reduce OUT.csv PROFILE.json [PROFILE.json ...]
"""
from __future__ import annotations

import csv
import json
import statistics
import sys

LETTER = {'native': 'N', 'B0': '0', 'BO': 'O', 'A': 'A', 'D': 'D', 'H': 'H', 'F': 'F', 'routed_other': '?'}


def classify(delta):
    if not delta:
        return 'native'
    for key, phase in (('bootstrap_dense_calls', 'B0'), ('bootstrap_observation_calls', 'BO'),
                       ('fresh_fused_calls', 'F'),
                       ('score_refresh_calls', 'A'), ('decision_refresh_calls', 'D'),
                       ('held_decision_calls', 'H')):
        if delta.get(key):
            return phase
    return 'routed_other' if delta.get('calls') or delta.get('attention_calls') else 'native'


def physical(diag, want='global'):
    out = {}
    phys = (diag or {}).get('physical') or {}
    for key, seg in (phys.get('by_phase_kind') or {}).items():
        phase, kind = key.split('/')
        if kind != want:
            continue
        whole = seg.get('whole') or {}
        legal = whole.get('eligible_pairs') or 0
        if legal:
            out[phase] = (whole.get('skipped_qk_pairs', 0) / legal, whole.get('skipped_pv_pairs', 0) / legal)
    return out


def rows_for(host, report):
    rows = []
    for key, target in report['targets'].items():
        for boundary, lengths in (target.get('boundaries') or {}).items():
            for label, entry in lengths.items():
                arms = entry['arms']
                native = arms.get('D_native')
                native_mean = (statistics.mean(native['summary']['per_call_event_median_ms'])
                               if native else None)
                c64 = arms.get('D_c64')   # strongest correctness-equivalent dense path
                c64_mean = (statistics.mean(c64['summary']['per_call_event_median_ms']) if c64 else None)
                for arm, data in arms.items():
                    summary = data['summary']
                    phases = [classify(d) for d in summary['phase_deltas']]
                    medians = summary['per_call_event_median_ms']
                    by = {}
                    for p, ms in zip(phases, medians):
                        by.setdefault(p, []).append(ms)
                    diag = (entry.get('diagnostic_replays') or {}).get(arm) or {}
                    work = physical(diag)
                    local = physical(diag, 'local')
                    floor = diag.get('prepared_support_floor') or {}
                    mean = statistics.mean(medians)
                    ph = {p: statistics.median(v) for p, v in by.items()}
                    row = dict(host=host, target=key.split('|canvas')[0].split('|')[-1][-12:],
                               dataset=target['dataset'], canvas=target['canvas'], boundary=boundary,
                               reached_calls=entry['reached_calls'], arm=arm,
                               sequence=''.join(LETTER[p] for p in phases),
                               mean_ms_per_call=round(mean, 3),
                               mean_over_native=round(mean / native_mean, 4) if native_mean else None,
                               mean_over_c64=round(mean / c64_mean, 4) if c64_mean else None,
                               total_ms=round(sum(medians), 2))
                    for p in ('native', 'B0', 'BO', 'A', 'D', 'H', 'F'):
                        row[f'{p}_ms'] = round(ph[p], 3) if p in ph else None
                        row[f'{p}_n'] = len(by.get(p, []))
                    row['select_overhead_D_minus_H_ms'] = (round(ph['D'] - ph['H'], 3)
                                                           if 'D' in ph and 'H' in ph else None)
                    row['observe_overhead_A_minus_D_ms'] = (round(ph['A'] - ph['D'], 3)
                                                            if 'A' in ph and 'D' in ph else None)
                    for p in ('A', 'D', 'H'):
                        row[f'{p}_local_qk_skip'] = round(local[p][0], 4) if p in local else None
                    for p in ('A', 'D', 'H'):
                        row[f'{p}_qk_skip'] = round(work[p][0], 4) if p in work else None
                        row[f'{p}_pv_skip'] = round(work[p][1], 4) if p in work else None
                    if floor.get('event_median_ms') is not None:
                        row['floor_total_ms'] = round(floor['event_median_ms'], 2)
                        row['floor_mean_ms_per_call'] = round(floor['event_median_ms'] / entry['reached_calls'], 3)
                        row['floor_over_native'] = (round(row['floor_mean_ms_per_call'] / native_mean, 4)
                                                    if native_mean else None)
                    rows.append(row)
    return rows


def main(argv=None):
    argv = argv or sys.argv[1:]
    out, paths = argv[0], argv[1:]
    rows = []
    for path in paths:
        host = 'mpk' if 'mpk' in path else 'dllm' if 'dllm' in path else '?'
        rows += rows_for(host, json.load(open(path, encoding='utf-8')))
    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 999)
    with open(out, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == '__main__':
    for r in main():
        print(r['host'], r['dataset'], r['canvas'], r['boundary'][:5], r['arm'][:16].ljust(16), r['sequence'][:24].ljust(24),
              r['mean_over_native'], 'H', r['H_ms'], 'D', r['D_ms'], 'A', r['A_ms'], 'N', r['native_ms'],
              'qkH', r['H_qk_skip'], 'pvH', r['H_pv_skip'], 'floor', r.get('floor_over_native'))
