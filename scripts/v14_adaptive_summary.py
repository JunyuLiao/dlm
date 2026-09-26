"""v14: public adaptive_canvas_diagnostic.json from the gate-5.2 run (numbers and digests only)."""
import json
import sys
from pathlib import Path

ARMS = ('D_native', 'T_G', 'T_P', 'B8_P', 'CVM_T')


def main(src, dst):
    d = json.loads(Path(src).read_text())
    table, totals = {}, {a: dict(calls=0, restored=0, diverged=0, diverged_protected=0, stable_not_confident=0) for a in ARMS}
    for spec, st in d['states'].items():
        table[spec] = dict(prefix_tokens=st['prefix_tokens'][0] if isinstance(st['prefix_tokens'], list) else st['prefix_tokens'],
                           arms={})
        for arm in ARMS:
            s, steps = st['arms'][arm]['summary'], st['arms'][arm]['steps']
            div = [x.get('diverged_rows_vs_native') for x in steps]
            early = div[1] if len(div) > 1 and div[1] is not None else (div[0] if div and div[0] is not None else 0)
            table[spec]['arms'][arm] = dict(
                s, rows_diverged_at_step1=early,
                share_of_divergent_rows_already_diverged_at_step1=(early / s['rows_ever_diverged_from_native']
                                                                      if s['rows_ever_diverged_from_native'] else None),
                s_gt1_rows_by_step=[x['s_gt1_rows'] for x in steps], flips_by_step=[x['flips'] for x in steps],
                accepted_by_step=[x['accepted'] for x in steps], mean_entropy_by_step=[round(x['mean_entropy'], 5) for x in steps],
                stable_by_step=[x['stable'] for x in steps], confident_by_step=[x['confident'] for x in steps])
            t = totals[arm]
            t['calls'] += s['calls_to_finish']
            t['restored'] += s['restored_tiles']
            t['diverged'] += s['rows_ever_diverged_from_native'] or 0
            t['diverged_protected'] += s['diverged_rows_with_protection_available']
            t['stable_not_confident'] += s['steps_stable_not_confident']
    out = dict(schema='v14_adaptive_canvas_diagnostic_v1',
               method=('identical real canvas-start snapshots (prefix KV, noise canvas, CPU/CUDA RNG, native sampler/stop/'
                       'processors, controller); each arm runs the native loop to the native stable&confident stop or the '
                       '48-step cap; one trajectory per state and arm (descriptive, not a quality result); divergence = '
                       'argmax differs from the D_native run at the same step; protection available = s>1 for that row at '
                       'that step (causal: from completed iterations only; s=1 at iterations 1-2 by construction)'),
               states=table, totals_over_states=totals)
    Path(dst).write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    for arm, t in totals.items():
        print(arm, t)


if __name__ == '__main__':
    main(*sys.argv[1:])
