"""v14 offline scoring + paired, question-clustered summaries of the CVM-T pilot (CPU only).

Reuses the v13 machinery (cells from the frozen schedule, warm acceptance, frozen
final-channel scorer, clustered bootstrap). Gold is read only here. Outputs:
  complete_request_results.json  meta, per-arm quality/caps/failures, pairs, parity vs v13
  complete_request_results.csv   one row per scheduled cell (every failure/cap kept)
Router counters (anchors / ordinary / native fallback / planner calls / metadata bytes)
are read from the private attempt-0 receipts (numbers only).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from scripts.v13_seed_runs import execution_key
from scripts.v13_summarize import aggregate, build_cells, score_attempt0

PAIRS = [('CVM_T', 'B8_P'), ('CVM_T', 'T_P'), ('CVM_T', 'D_native'), ('CVM_T', 'T_G_original'), ('B8_P', 'T_P'),
         ('T_P', 'T_G_original'), ('T_G_original', 'D_native'), ('B8_P', 'D_native'), ('T_P', 'D_native')]
V13_LEDGER = Path('/media/volume/dllm-1/dyh/numerical_qk_global_multiseed_20260925/ledger.jsonl')
V13_NAME = {'D_native': 'D_native', 'T_G_original': 'T_G'}


def counters(record):
    if not record or not record.get('private_receipt'):
        return {}
    c = json.loads(Path(record['private_receipt']).read_text()).get('counters') or {}
    return {k: c.get(k) for k in ('calls', 'anchors', 'ordinary', 'fallback_native', 'planner_calls', 'metadata_bytes')
            if k in c}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, nargs='+', required=True)
    p.add_argument('--ledger', type=Path, nargs='+', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=10000)
    a = p.parse_args()
    gold = {str(r['id']): str(r['expected']) for r in json.loads(a.manifest.read_text())}
    records, schedule, ids, seeds, protocols = {}, [], [], None, []
    for proto_path, ledger in zip(a.protocol, a.ledger):
        proto = json.loads(proto_path.read_text())
        protocols.append(proto['protocol_id'])
        if ledger.exists():
            for line in ledger.read_text().splitlines():
                e = json.loads(line) if line.strip() else {}
                if e.get('event') == 'run':
                    records[e['execution_key']] = e
        schedule += proto['schedule']
        ids += proto['ids']
        seeds = proto['seeds']
    quality = score_attempt0(records, gold)
    cells = build_cells(schedule, records, quality)
    dev = json.loads(a.protocol[0].read_text())['development_ids']
    summary = aggregate(cells, ids, seeds, dev, pairs=PAIRS, boot=a.bootstrap)
    # attempt-0 token parity vs the v13 cells of the same (arm, id, seed)
    v13 = {}
    for line in V13_LEDGER.read_text().splitlines():
        e = json.loads(line) if line.strip() else {}
        if e.get('event') == 'run' and e.get('role') == 'attempt0' and e.get('ok'):
            v13[(e['arm'], e['id'], e['seed'])] = e.get('completion_token_hash')
    parity = {}
    for arm, old in V13_NAME.items():
        rows = []
        for q in ids:
            for s in seeds:
                r = cells[(arm, q, s)]['attempt0'] or {}
                rows.append(dict(id=q, seed=s, identical=(r.get('completion_token_hash') is not None and
                                                         r.get('completion_token_hash') == v13.get((old, q, s)))))
        parity[arm] = dict(v13_arm=old, identical=sum(x['identical'] for x in rows), cells=len(rows), rows=rows)
    per_arm = {}
    for arm in sorted({k[0] for k in cells}):
        cs = [cells[(arm, q, s)] for q in ids for s in seeds]
        r0 = [c['attempt0'] or {} for c in cs]
        cnt = [counters(r) for r in r0]
        per_arm[arm] = dict(
            decoder_calls_total=sum(r.get('decoder_calls') or 0 for r in r0),
            canvases_total=sum(r.get('canvases') or 0 for r in r0),
            output_tokens_total=sum(r.get('output_tokens') or 0 for r in r0),
            calls_per_canvas=(sum(r.get('decoder_calls') or 0 for r in r0) / max(1, sum(r.get('canvases') or 0 for r in r0))),
            eos=sum(r.get('termination') == 'eos' for r in r0), caps=sum(r.get('termination') == 'length' for r in r0),
            eos_but_wrong_or_unparsed=sum(1 for c, r in zip(cs, r0) if r.get('termination') == 'eos'
                                          and not (c['quality'] or {}).get('correct')),
            failures=sum(1 for r in r0 if r and not r.get('ok')),
            warm_accepted=sum(c['time'] is not None for c in cs),
            warm_time_total_s=sum(c['time'] for c in cs if c['time'] is not None),
            attempt0_time_total_s=sum(r.get('api_wall_s') or 0 for r in r0),
            attempt0_minus_warm_s=[round((r.get('api_wall_s') or 0) - c['time'], 3) for c, r in zip(cs, r0) if c['time']],
            routed_global_calls=sum(x.get('calls') or 0 for x in cnt),
            anchors=sum(x.get('anchors') or 0 for x in cnt), ordinary=sum(x.get('ordinary') or 0 for x in cnt),
            native_fallback=sum(x.get('fallback_native') or 0 for x in cnt),
            planner_calls=sum(x.get('planner_calls') or 0 for x in cnt),
            peak_allocated_gib=max((r.get('peak_allocated_bytes') or 0) for r in r0) / 2 ** 30)
    planned = len(schedule)
    executed = sum(1 for e in schedule if execution_key(e) in records)
    meta = dict(protocols=protocols, questions=ids, seeds=seeds, planned_executions=planned, executed=executed,
                complete=executed == planned, scope='GLOBAL-only (layers 5,11,17,23,29); LOCAL native in all arms',
                time_definition='accepted single warm API wall (same cell, no new compilation, reproduces attempt 0)',
                note='development questions (examined in v13); small pilot: intervals descriptive, not noninferiority')
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'complete_request_results.json').write_text(json.dumps(
        dict(meta=meta, per_arm=per_arm, parity_vs_v13=parity, **summary), indent=2, sort_keys=True, default=str) + '\n')
    with (a.out / 'complete_request_results.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'id', 'seed', 'cell_id', 'ok', 'correct', 'parsed', 'capped', 'termination', 'output_tokens',
                    'decoder_calls', 'canvases', 'calls_per_canvas', 'token_hash16', 'attempt0_api_wall_s', 'warm_api_wall_s',
                    'timing_status', 'routed_global_calls', 'anchors', 'ordinary', 'native_fallback', 'planner_calls',
                    'metadata_bytes_end', 'peak_allocated_bytes', 'error'])
        for (arm, q, s), c in sorted(cells.items()):
            r, qq = c['attempt0'] or {}, c['quality'] or {}
            x = counters(r)
            calls, canv = r.get('decoder_calls'), r.get('canvases')
            w.writerow([arm, q, s, c['cell_id'], r.get('ok'), qq.get('correct'), qq.get('parsed'), qq.get('capped'),
                        r.get('termination'), r.get('output_tokens'), calls, canv,
                        round(calls / canv, 3) if calls and canv else None, (r.get('completion_token_hash') or '')[:16],
                        r.get('api_wall_s'), c['time'], c['timing_status'], x.get('calls'), x.get('anchors'),
                        x.get('ordinary'), x.get('fallback_native'), x.get('planner_calls'), x.get('metadata_bytes'),
                        r.get('peak_allocated_bytes'), r.get('error') or ''])
    print(json.dumps(dict(executed=executed, planned=planned,
                          quality={k: v['combined'] for k, v in summary['quality'].items()},
                          parity={k: f"{v['identical']}/{v['cells']}" for k, v in parity.items()}), indent=1))
    for name, pr in summary['pairs'].items():
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in pr.items()
                     if k in ('candidate_correct', 'reference_correct', 'geometric_time_ratio', 'summed_time_ratio',
                              'call_factor', 'amortized_time_per_call_factor', 'timing_complete')},
              pr['bootstrap'].get('geometric_time_ratio_95'))


if __name__ == '__main__':
    main()
