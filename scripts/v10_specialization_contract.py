"""v10 CP1: count compiled variants and real compile/load events per variant.

Run in a FRESH process with TRITON_CACHE_DIR set before import. Drives the
production entry points (anchor route+PV with summary STORE, ordinary
route_only with summary LOAD + preqk consumer) over model-shaped LOCAL and
GLOBAL geometries with the key lengths a real request produces, then an
unseen tail, then repeats. Records every in-memory miss via Triton's
cache_hook/compiled_hook with its duration and whether it produced a new
disk-cache entry.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('static', 'generic'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cache_dir = os.environ.get('TRITON_CACHE_DIR')
    if not cache_dir:
        raise SystemExit('set TRITON_CACHE_DIR before starting this process')
    import torch
    import triton
    from triton.runtime.jit import JITFunction
    from tests.test_v10_generic_kernels import GLOBAL, LOCAL, inputs, pipeline
    events = []
    t0 = time.perf_counter()

    def before(**kw):
        events.append(dict(fn=kw['fn'].name, t=time.perf_counter() - t0,
                           disk_entries_before=len(os.listdir(cache_dir)),
                           K=kw['compile']['constants'].get('K'), KDIV=kw['compile']['constants'].get('KDIV'),
                           PREFIX_TILES=kw['compile']['constants'].get('PREFIX_TILES')))
        return False

    def after(**kw):
        events[-1]['seconds'] = time.perf_counter() - t0 - events[-1]['t']
        events[-1]['new_disk_entries'] = len(os.listdir(cache_dir)) - events[-1]['disk_entries_before']
        return False

    JITFunction.cache_hook, JITFunction.compiled_hook = before, after
    # prompt ~109 tokens; each canvas adds 256. LOCAL saturates after the window crop.
    prompt = 109
    phases = dict(request=[prompt + 256 * c for c in range(0, 12)],
                  unseen_tail=[prompt + 256 * c + 37 for c in range(12, 16)] + [3000, 3002, 3004, 3008])
    rows = []
    for phase in ('request', 'unseen_tail', 'repeat'):
        lengths = phases['request'] if phase == 'repeat' else phases[phase]
        for prefix in lengths:
            for kind, geometry in (('local', LOCAL), ('global', GLOBAL)):
                n = len(events)
                start = time.perf_counter()
                pipeline(inputs(prefix, geometry, prefix), args.variant, geometry['threshold'])
                rows.append(dict(phase=phase, kind=kind, prefix=prefix, wall_s=time.perf_counter() - start,
                                 misses=len(events) - n,
                                 compile_s=sum(e.get('seconds', 0) for e in events[n:]),
                                 new_disk_entries=sum(e.get('new_disk_entries', 0) for e in events[n:])))
    JITFunction.cache_hook = JITFunction.compiled_hook = None
    by_phase = {}
    for phase in ('request', 'unseen_tail', 'repeat'):
        sel = [r for r in rows if r['phase'] == phase]
        by_phase[phase] = dict(calls=len(sel), misses=sum(r['misses'] for r in sel),
                               compile_or_load_s=sum(r['compile_s'] for r in sel),
                               new_disk_entries=sum(r['new_disk_entries'] for r in sel))
    report = dict(schema='v10_specialization_contract_v1', variant=args.variant, triton=triton.__version__,
                  triton_file=triton.__file__, torch=torch.__version__, cache_dir=cache_dir,
                  disk_entries_final=len(os.listdir(cache_dir)), by_phase=by_phase,
                  distinct_variants_total=len(events), rows=rows, events=events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(variant=args.variant, by_phase=by_phase, distinct=len(events))))


if __name__ == '__main__':
    main()
