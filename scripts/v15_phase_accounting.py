"""v15 phase accounting on the scored LongBench-v2 panel (CPU only, after timing stopped).

Measured (from attempt-0 receipts): per-canvas decoder calls, canvases, GLOBAL-call phase counters
(anchors / ordinary / native fallback / planner calls) and metadata bytes at request end.
Counterfactual ESTIMATES (labeled; never interchangeable with measured request wall), using the
v14 teacher-forced LongBench step costs (lb/21 prefix 17911, lb/40 prefix 15720; same five arms):
  common_native_workload: every arm priced on the D_native per-canvas call histories of THIS panel
  realized_histories:     every arm priced on its OWN per-canvas call histories of this panel,
                          relative to D_native priced on D_native's histories
Canvas cost = sum of the profiled step medians by epoch position (v14_cost_budget.canvas_cost).
"""
import argparse
import json
from pathlib import Path

from scripts.v13_seed_runs import execution_key
from scripts.v14_cost_budget import canvas_cost

PROFILE_ARM = {'D_native': 'D_native', 'T_G_original': 'T_G', 'T_P': 'T_P', 'B8_P': 'B8_P', 'CVM_T': 'CVM_T'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--ledger', type=Path, required=True)
    ap.add_argument('--profiles', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    proto = json.loads(a.protocol.read_text())
    keys = {execution_key(e): e for e in proto['schedule'] if e['role'] == 'attempt0'}
    hist, phases = {arm: [] for arm in proto['arms']}, {arm: dict(anchors=0, ordinary=0, fallback_native=0, planner_calls=0,
                                                                  routed_calls=0, requests=0, metadata_bytes_end_max=0)
                                                       for arm in proto['arms']}
    for line in a.ledger.read_text().splitlines():
        e = json.loads(line) if line.strip() else {}
        if e.get('event') != 'run' or e.get('execution_key') not in keys or not e.get('ok'):
            continue
        rec = json.loads(Path(e['private_receipt']).read_text())
        arm = e['arm']
        hist[arm] += [int(c['decoder_calls']) for c in rec['per_canvas']]
        c = rec.get('counters') or {}
        ph = phases[arm]
        ph['requests'] += 1
        for k_out, k_in in (('anchors', 'anchors'), ('ordinary', 'ordinary'), ('fallback_native', 'fallback_native'),
                            ('planner_calls', 'planner_calls'), ('routed_calls', 'calls')):
            ph[k_out] += int(c.get(k_in) or 0)
        ph['metadata_bytes_end_max'] = max(ph['metadata_bytes_end_max'], int(c.get('metadata_bytes') or 0))
    for ph in phases.values():
        total = ph['anchors'] + ph['ordinary']
        ph['anchor_fraction_of_routed'] = ph['anchors'] / total if total else None
        ph['note'] = 'GLOBAL-layer call counts (5 per decoder call); metadata bytes are live at request end, not peak'
    estimates = {}
    for f in a.profiles:
        prof = json.loads(f.read_text())
        st = next(iter(prof['states'].values()))
        steps = {arm: st['arms'][PROFILE_ARM[arm]]['step_median_ms'] for arm in proto['arms']}
        d_common = sum(canvas_cost(steps['D_native'], n) for n in hist['D_native'])
        estimates[f"{prof['id']}@{st['absolute'][0]}"] = dict(
            common_native_workload={arm: sum(canvas_cost(steps[arm], n) for n in hist['D_native']) / d_common for arm in proto['arms']},
            realized_histories={arm: sum(canvas_cost(steps[arm], n) for n in hist[arm]) / d_common for arm in proto['arms']})
    out = dict(schema='v15_phase_accounting_v1', method=__doc__, measured=dict(
        per_canvas_call_histograms={arm: {str(k): h.count(k) for k in sorted(set(h))} for arm, h in hist.items()},
        canvases={arm: len(h) for arm, h in hist.items()}, calls={arm: sum(h) for arm, h in hist.items()},
        calls_per_canvas={arm: (sum(h) / len(h) if h else None) for arm, h in hist.items()}, global_phases=phases),
        counterfactual_estimates=estimates)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    for k, v in estimates.items():
        print(k, {x: {arm: round(c, 4) for arm, c in y.items()} for x, y in v.items()})
    print({arm: (round(p['anchor_fraction_of_routed'], 3) if p['anchor_fraction_of_routed'] else None) for arm, p in phases.items()})


if __name__ == '__main__':
    main()
