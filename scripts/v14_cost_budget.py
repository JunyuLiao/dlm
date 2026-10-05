"""v14 CP2: cost budget + phase-weighted complete-forward ratio from the profiler outputs.

Inputs: v14_forward_profile JSONs (each a teacher-forced replay of REAL consecutive
steps of one canvas, per arm) and the v13 D_native per-canvas decoder-call counts
(private receipts; only counts are read). Outputs (public, no text):
  cost_budget.csv            one row per (state, arm): measured GLOBAL call ms by phase,
                             saved attention vs added decision/observation, sequence c
  complete_forward_profile.json  per-state step medians + phase-weighted c

Phase weighting: a canvas with n native calls costs sum_k step_ms[pos(k)] where
pos(k)=k for k<len(sequence) and otherwise the same epoch position (k mod 8; 0 = A8
anchor, which is sequence step 8). Each measured state is weighted only with its own
complete-forward times (no cross-state composition of per-layer and model time).
The request-level figure maps each native canvas to the measured AIME state with the
nearest prefix; it is labeled an estimate composed from three measured states.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ARMS = ('D_native', 'T_G', 'T_P', 'B8_P', 'CVM_T')


def pos(k, n_seq, period=8):
    if k < n_seq:
        return k
    e = k % period
    return (8 if n_seq > 8 else 0) if e == 0 else e            # A8 anchor step (or the canvas-start anchor if absent)


def canvas_cost(step_ms, n):
    return sum(step_ms[pos(k, len(step_ms))] for k in range(n))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profiles', type=Path, nargs='+', required=True)
    p.add_argument('--v13-index', type=Path, required=True)
    p.add_argument('--dev-ids', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    idx = json.loads(a.v13_index.read_text())
    canvases = []                                           # (canvas_index, calls) for D_native dev ids, attempt0
    for r in idx['receipts']:
        if r['arm'] != 'D_native' or r['id'] not in a.dev_ids or not r['execution_key'].endswith(':attempt0:0'):
            continue
        rec = json.loads(Path(r['receipt']).read_text())
        canvases += [(c['canvas_index'], int(c['decoder_calls'])) for c in rec['per_canvas']]
    states, rows = {}, []
    for f in a.profiles:
        prof = json.loads(f.read_text())
        for cv, st in prof['states'].items():
            name = f"{prof['id']}:s{prof['seed']}:c{cv}"
            prefix = st['absolute'][0]
            per = {}
            for arm in ARMS:
                x = st['arms'][arm]
                calls = x['global_call_ms_by_step']
                gsum = {int(i): sum(v.values()) for i, v in calls.items()}
                phases = x['phases'] or None
                kind = []
                for i in range(st['steps']):
                    if phases is None:
                        kind.append('native' if arm == 'D_native' else 'fresh')
                    else:
                        ph = phases[i]
                        kind.append('fallback' if ph['fallback_native'] else 'anchor' if ph['anchors'] else 'ordinary')
                per[arm] = dict(step_ms=x['step_median_ms'], seq_ms=x['sequence_total_median_ms'], gsum=gsum, kind=kind,
                                identical=x['outputs_identical_across_reps'])
            d = per['D_native']
            n_calls = [c for _, c in canvases]
            weighted = {arm: sum(canvas_cost(per[arm]['step_ms'], n) for n in n_calls) /
                        sum(canvas_cost(d['step_ms'], n) for n in n_calls) for arm in ARMS}
            states[name] = dict(prefix_tokens=prefix, steps=st['steps'], cur_steps=st['cur_steps'],
                                benchmark='LongBench-v2 regime diagnostic' if prof['id'].startswith('lb/') else 'AIME26',
                                sequence_c={arm: per[arm]['seq_ms'] / d['seq_ms'] for arm in ARMS},
                                phase_weighted_c_native_canvas_mix=weighted,
                                step_median_ms={arm: per[arm]['step_ms'] for arm in ARMS},
                                global_attention_ms_by_step={arm: [per[arm]['gsum'].get(i) for i in range(st['steps'])] for arm in ARMS},
                                phase_by_step={arm: per[arm]['kind'] for arm in ARMS},
                                outputs_identical_across_reps={arm: per[arm]['identical'] for arm in ARMS})
            dn = statistics.median(v for v in d['gsum'].values())
            for arm in ARMS:
                x = per[arm]
                for phase in sorted(set(x['kind'])):
                    steps = [i for i, k in enumerate(x['kind']) if k == phase]
                    g = statistics.median(x['gsum'][i] for i in steps if i in x['gsum'])
                    dstep = statistics.median(d['step_ms'][i] for i in steps)
                    astep = statistics.median(x['step_ms'][i] for i in steps)
                    rows.append(dict(state=name, benchmark=states[name]['benchmark'], prefix_tokens=prefix, arm=arm, phase=phase,
                                     steps=len(steps), global_attention_ms_arm=round(g, 3), global_attention_ms_native=round(dn, 3),
                                     global_attention_delta_ms=round(g - dn, 3), step_ms_arm=round(astep, 3),
                                     step_ms_native_same_steps=round(dstep, 3), step_delta_ms=round(astep - dstep, 3),
                                     sequence_c=round(x['seq_ms'] / d['seq_ms'], 4),
                                     phase_weighted_c=round(weighted[arm], 4)))
    aime = {k: v for k, v in states.items() if v['benchmark'] == 'AIME26'}
    request = None
    if aime:
        num = {arm: 0. for arm in ARMS}
        for ci, n in canvases:
            prefix = 256 * ci + 200
            near = min(aime.values(), key=lambda s: abs(s['prefix_tokens'] - prefix))
            for arm in ARMS:
                num[arm] += canvas_cost(near['step_median_ms'][arm], n)
        request = dict(estimate={arm: num[arm] / num['D_native'] for arm in ARMS},
                       note='composed from the measured AIME states by nearest prefix (canvas prefix ~ 256*index+200); '
                            'native D canvas-length mix of the six development IDs x seeds 17/29 (v13 attempt0); '
                            'an ESTIMATE, not a measured request time')
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / 'cost_budget.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hist = {}
    for _, n in canvases:
        hist[n] = hist.get(n, 0) + 1
    (a.out / 'complete_forward_profile.json').write_text(json.dumps(dict(
        schema='v14_complete_forward_profile_summary_v1', arms=list(ARMS), states=states,
        native_canvas_call_histogram=dict(sorted(hist.items())), native_canvases=len(canvases),
        request_level_phase_weighted_estimate=request,
        method='teacher-forced replay of real consecutive denoising steps; per-step synchronized CUDA-event span of a '
               'complete denoising step (decoder + native sampler + T controller + method work); GLOBAL call timing '
               'from a separate in-situ repetition with per-call CUDA events; router state evolves naturally across '
               'the sequence (step 0 = new-canvas anchor, 8 = A8 anchor)'), indent=2, sort_keys=True) + '\n')
    for k, v in states.items():
        print(k, v['prefix_tokens'], {a_: round(c, 4) for a_, c in v['phase_weighted_c_native_canvas_mix'].items()})
    print('request estimate', request and {k: round(v, 4) for k, v in request['estimate'].items()})


if __name__ == '__main__':
    main()
