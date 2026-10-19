"""Sanitized comparison of immutable first-prefill compiled-segment snapshots."""
import argparse
import json
from pathlib import Path

import torch

from scripts.v31_humaneval_repro_study import read, sha, write, base


FIELDS = ('output_hash', 'denoise_forwards', 'canvases', 'output_tokens')


def bound(leaf):
    marker = read(leaf / 'attempt_complete.json')
    config = read(leaf / 'config.json')
    assert config['variants'] == ['clean', 'observe', 'clean']
    assert marker['records'] == marker['expected'] == 26
    for rel, digest in marker['files_sha256'].items():
        if sha(leaf / rel) != digest:
            raise RuntimeError(f'artifact drift: {rel}')
    traces = list((leaf / 'private/segment').glob('*.pt'))
    if len(traces) != 1:
        raise RuntimeError('expected one question-zero first-prefill segment')
    records = base.jsonl(leaf / 'records.jsonl')
    n = records[8]['prompt_tokens']
    assert records[8]['index'] == 0
    within = sum(any(any(records[i][f] != records[i+j][f] for f in FIELDS)
                     for j in (1, 2)) for i in range(8, 26, 3))
    return config, records, torch.load(traces[0], map_location='cpu', weights_only=False), n, within


def difference(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise RuntimeError('incompatible native intermediate shapes/dtypes')
    neq = a != b
    count = int(neq.sum())
    if not count:
        return dict(different=0, elements=a.numel(), max_abs=0., mean_abs=0.)
    delta = (a.double() - b.double()).abs()
    return dict(different=count, elements=a.numel(), max_abs=float(delta.max()),
                mean_abs=float(delta.mean()))


def compare(left, right, out):
    if out.exists():
        raise RuntimeError('immutable comparison required')
    ca, ra, a, n, wa = bound(left)
    cb, rb, b, nb, wb = bound(right)
    if ca['worker_sha256'] != cb['worker_sha256'] or n != nb or len(a) != len(b):
        raise RuntimeError('observer source or native segment mismatch')
    if [(r['index'], r['rng_seed'], r['prompt_sha256']) for r in ra] != [
            (r['index'], r['rng_seed'], r['prompt_sha256']) for r in rb]:
        raise RuntimeError('request identity mismatch')
    entries = []
    for index, (x, y) in enumerate(zip(a, b)):
        if any(x[k] != y[k] for k in ('source_sha256', 'line', 'source')):
            raise RuntimeError('native generated code or call order mismatch')
        if x['values'].keys() != y['values'].keys():
            raise RuntimeError('snapshot coverage mismatch')
        values = {}
        for name, u in x['values'].items():
            v = y['values'][name]
            lhs, rhs = u['value'], v['value']
            # The generated segment can contain padded query rows. Compare the
            # valid prefix separately; padding differences alone are not causal.
            padded = lhs.ndim >= 2 and n <= lhs.shape[0] <= 512
            values[name] = dict(full=difference(lhs, rhs),
                valid=difference(lhs[:n], rhs[:n]) if padded else difference(lhs, rhs),
                valid_prefix_sliced=padded, shape=list(lhs.shape), dtype=str(lhs.dtype),
                strides=[u['stride'], v['stride']], alignment=[u['alignment'], v['alignment']])
        entries.append(dict(index=index, source=x['source'], line=x['line'],
            source_sha256=x['source_sha256'], values=values,
            launch_configs=[x.get('launch_configs'), y.get('launch_configs')]))
    first = next((x['index'] for x in entries
                  if any(v['valid']['different'] for v in x['values'].values())), None)
    result = dict(status='diagnostic_unqualified', left=left.name, right=right.name,
        within_engine_changed_groups={left.name:wa, right.name:wb}, unique_question_groups=6,
        observed_question=0, prompt_tokens=n, first_valid_difference_entry=first, entries=entries,
        output_differences={f:sum(u[f] != v[f] for u,v in zip(ra[8:], rb[8:])) for f in FIELDS},
        markers_sha256={p.name:sha(p/'attempt_complete.json') for p in (left, right)})
    write(out, result, True)
    print(json.dumps({k:v for k,v in result.items() if k != 'entries'}))
    if first is not None:
        e = entries[first]
        print(json.dumps(dict(source=e['source'], differences={k:v['valid'] for k,v in
            e['values'].items() if v['valid']['different']}, launch_configs=e['launch_configs'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('left', type=Path); p.add_argument('right', type=Path)
    p.add_argument('out', type=Path)
    args = p.parse_args()
    compare(args.left, args.right, args.out)
