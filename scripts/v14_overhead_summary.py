"""v14: attribute the fixed per-step overhead of T-using arms (profile-only controls) + launch inventory.

Arms on the SAME teacher-forced real steps: D_native; N0 = GLOBAL-only binding with a
pass-through override (native SDPA on every call); C_T = exact fast-T controller observing
the native loop (no binding); N1 = both; T_G; CVM_T. Differences of step medians
(outside the GLOBAL calls = step delta - GLOBAL-call delta) are reported per state.
"""
import json
import statistics
import sys
from pathlib import Path


def main(out, *profiles):
    res = {}
    for f in profiles:
        d = json.loads(Path(f).read_text())
        for cv, st in d['states'].items():
            name = f"{d['id']}:s{d['seed']}:c{cv}"
            arms = st['arms']
            D = arms['D_native']
            dstep = statistics.median(D['step_median_ms'])
            dglob = statistics.median(sum(v.values()) for v in D['global_call_ms_by_step'].values())
            row = dict(prefix_tokens=st['absolute'][0], steps=st['steps'], arms={})
            for a, x in arms.items():
                step = statistics.median(x['step_median_ms'])
                glob = statistics.median(sum(v.values()) for v in x['global_call_ms_by_step'].values())
                inv = x.get('launch_inventory') or []
                row['arms'][a] = dict(
                    sequence_ms=x['sequence_total_median_ms'], sequence_c=x['sequence_total_median_ms'] / D['sequence_total_median_ms'],
                    median_step_ms=step, median_step_delta_ms=step - dstep, median_global_ms=glob,
                    median_global_delta_ms=glob - dglob, outside_global_delta_ms=(step - dstep) - (glob - dglob),
                    outputs_identical_across_reps=x['outputs_identical_across_reps'],
                    launch_inventory=[{k: v for k, v in s.items() if k != 'top'} for s in inv],
                    launch_top_first_ordinary=(inv[1]['top'] if len(inv) > 1 else None))
            res[name] = row
            print(name, row['prefix_tokens'])
            for a, r in row['arms'].items():
                inv = r['launch_inventory']
                print(f"  {a:15s} c={r['sequence_c']:.4f} step+{r['median_step_delta_ms']:.2f} glob+{r['median_global_delta_ms']:.2f} "
                      f"outside+{r['outside_global_delta_ms']:.2f} kernels/step={[s['total_kernels'] for s in inv][:3]} "
                      f"vd={[s['value_direction_v4_or_v5'] for s in inv][:10]} plan={[s['planner'] for s in inv][:10]} "
                      f"sup={[s['support_consumer'] for s in inv][:10]} sdpa={[s['native_sdpa_like'] for s in inv][:3]}")
    Path(out).write_text(json.dumps(dict(schema='v14_overhead_attribution_v1', method=__doc__, states=res), indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
