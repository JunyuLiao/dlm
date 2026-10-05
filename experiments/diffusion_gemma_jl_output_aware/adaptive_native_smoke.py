"""Two-example guarded CUDA smoke for the adaptive router."""
import argparse
import json
from pathlib import Path
import torch

from dllm.models import create_adapter
from . import runner


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('results/diffusion_gemma_ruler4k_adaptive_v20'))
    p.add_argument('--source', type=Path, default=Path('results/diffusion_gemma_ruler4k_value_direction_s70_v19'))
    args = p.parse_args()
    out = dict(cuda_available=bool(torch.cuda.is_available()), passed=False, cases=[], error=None)
    if not torch.cuda.is_available():
        out['error'] = 'CUDA driver/device unavailable'
        (args.output/'native_smoke.json').write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2)); return
    try:
        setup = json.loads((args.source/'setup.json').read_text())
        adapter = create_adapter('diffusion_gemma', setup['model'], device='cuda',
            precision='bfloat16', revision=setup['revision']).load()
        rows = [dict(r, generation_budget=16) for r in setup['development'][:2]]
        config = dict(method='adaptive', family='gaussian', rank=64,
                      cascade_stages=(2, 8, 32, 64), projection_seed=1729,
                      interval_delta=1e-3)
        dense = []
        for row in rows:
            d = runner.generate(adapter, row, 'dense', {}, None)
            dense.append(d)
            for label, policy in (
                ('adaptive_unpruned', {k:dict(log_threshold=-1000., unpruned=True) for k in ('local','global')}),
                ('adaptive_pruned', {k:dict(log_threshold=-2.) for k in ('local','global')})):
                got = runner.generate(adapter, row, 'adaptive', config, policy)
                finite = bool(got.get('finite_calls', 0) == got.get('calls', got.get('finite_calls', 0)))
                records = got.get('records', [])
                skipped = sum(r.get('skipped', 0) for r in records)
                eligible = sum(r.get('eligible', 0) for r in records)
                out['cases'].append(dict(id=row['id'], label=label,
                    dense_completion_tokens=d['completion_tokens'],
                    completion_tokens=got['completion_tokens'],
                    completion_match=got['completion_tokens'] == d['completion_tokens'],
                    finite=finite, eligible=eligible, skipped=skipped,
                    sparsity=skipped/max(1, eligible),
                    work=got.get('work_accounting', {})))
        out['passed'] = all(x['finite'] and x['completion_match'] for x in out['cases'])
        del adapter
        torch.cuda.empty_cache()
    except Exception as exc:
        out['error'] = repr(exc)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'native_smoke.json').write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    if not out['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
