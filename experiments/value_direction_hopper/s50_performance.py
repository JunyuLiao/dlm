"""Clean, paired RULER4K130 timings; separate routing audit, frozen thresholds."""
import argparse
from contextlib import nullcontext
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch

from dllm.models import create_adapter
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _request, _set_context
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_ruler8k_jl import score
from .experiment import atomic, sha
from .integration import install
from .sparsity_steps import routing_counts

BASE = Path(__file__).resolve().parents[2]/'results'/'value_direction_sparsity_steps_v2'
METHODS = ['native_dense', 'kernel_dense', 'gaussian32_s50', 'blasst_s50', 'kernel_dense_native_mask']


def generate(adapter, row, method, cfg, projections, collect=False):
    policy = cfg['base']['policies'].get(method)
    if method.startswith('kernel_dense'):
        policy = {k: {'log_threshold': -float('inf')} for k in ('local', 'global')}
    ctx = nullcontext((None, None)) if method == 'native_dense' else install(
        adapter, cfg['library'], policy, mode='blasst' if method == 'blasst_s50' else 'value',
        torch_library=cfg['torch_library'], projections=projections, collect=collect, blasst_tma=True)
    with ctx as (binding, router):
        if binding:
            _set_context(binding, row)
            if method == 'kernel_dense_native_mask':
                def native_mask(*args, **kwargs):
                    # Native SDPA consumes the supplied mask without adding a window.
                    kwargs['sliding_window'] = None
                    return router(*args, **kwargs)
                binding.runtime.attention_override = native_mask
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = adapter.generate(_request(row))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        counts = routing_counts(router.records()) if collect and router else None
    return dict(id=row['id'], method=method, seconds=seconds,
                tokens=out.completion_tokens, steps=out.metadata['actual_denoising_step_count'],
                score=score(row, out.text), prediction=out.text, counts=counts,
                metadata=out.metadata, instrumented=collect)


def report(root):
    records = [json.loads(p.read_text()) for p in sorted((root/'timing').glob('*.json'))]
    cfg = json.loads((root/'configuration.json').read_text())
    expected = cfg['repeats'] * 130 * len(METHODS)
    if len(records) != expected:
        raise ValueError(f'Incomplete timings: {len(records)}/{expected}')
    summary = {}
    for method in METHODS:
        rows = [r for r in records if r['method'] == method]
        seconds = sum(r['seconds'] for r in rows)
        tokens = sum(len(r['tokens']) for r in rows)
        audit = [json.loads(p.read_text()) for p in (root/'audit'/method).glob('*.json')]
        if method in ('gaussian32_s50', 'blasst_s50') and len(audit) != 130:
            raise ValueError('Missing separate routing audit')
        eligible = sum(r['counts']['whole']['eligible'] for r in audit)
        skipped = sum(r['counts']['whole']['skipped'] for r in audit)
        summary[method] = dict(n=130, repeats=cfg['repeats'],
            mean_e2e_seconds=seconds/len(rows), total_seconds_per_repeat=seconds/cfg['repeats'],
            api_ttft_seconds=seconds/len(rows), amortized_tpt_ms=1000*seconds/tokens,
            mean_steps=float(np.mean([r['steps'] for r in rows])),
            accuracy=float(np.mean([r['score'] for r in rows])),
            actual_sparsity=skipped/eligible if eligible else 0., output_tokens_per_repeat=tokens/cfg['repeats'])
    baseline = summary['native_dense']['mean_e2e_seconds']
    for method, item in summary.items():
        item['speedup_vs_native'] = baseline/item['mean_e2e_seconds']
        paired = []
        for i in range(130):
            own = [r for r in records if r['method'] == method and r['index'] == i]
            native = [r for r in records if r['method'] == 'native_dense' and r['index'] == i]
            paired.append((np.mean([r['seconds'] for r in native]), np.mean([r['seconds'] for r in own])))
        data = np.array(paired)
        sample = np.random.default_rng(1729).integers(0, 130, size=(10000, 130))
        ratios = data[sample, 0].sum(1)/data[sample, 1].sum(1)
        item['speedup_ci95'] = np.quantile(ratios, [.025, .975]).tolist()
    atomic(root/'summary.json', summary)
    lines = ['# RULER4K130 target 50% performance', '',
        'Fresh synchronized adapter.generate wall time, batch 1, two paired interleaved repeats by default. Model load, warmup, binding setup, scoring and routing reduction excluded. Native decoding/stopper and frozen thresholds unchanged.', '',
        'The adapter returns a complete single canvas without streaming: API TTFT equals end-to-end latency. TPT is total wall time / actual returned tokens, including EOS; it is amortized latency, not inter-token delay. No first-denoising-step proxy is used.', '',
        '| Method | Actual sparsity | Steps | Accuracy | E2E / API TTFT (ms) | Amortized TPT (ms) | Native speedup |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m, r in summary.items():
        lines.append(f"| {m} | {r['actual_sparsity']:.2%} | {r['mean_steps']:.3f} | {r['accuracy']:.2%} | {1000*r['mean_e2e_seconds']:.2f} | {r['amortized_tpt_ms']:.3f} | {r['speedup_vs_native']:.3f}× |")
    lines += ['', 'kernel_dense_native_mask is a diagnostic: it omits the additional custom sliding-window constraint, retaining any supplied native mask. It does not change rounding or stopping. It is not substituted for the primary custom dense control.', '',
              'Routing counts are collected in a separate pass with exact token/step parity against the timed run. All prompts were previously examined; these are not fresh quality-validation examples.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run with host GPU access')
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base = json.loads((BASE/'configuration.json').read_text())
        rows = json.loads((BASE/'manifest.json').read_text())
        assert len(rows) == 130 and len({r['id'] for r in rows}) == 130
        sources = [Path(__file__), Path(__file__).with_name('integration.py'),
                   Path(__file__).with_name('cuda.py'), Path(__file__).with_name('projection.py'),
                   Path(__file__).with_name('masks.py'), args.library, args.torch_library,
                   BASE/'manifest.json', BASE/'configuration.json',
                   Path('src/dllm/models/adapters/diffusion_gemma.py')]
        cfg = dict(base=base, library=str(args.library.resolve()), torch_library=str(args.torch_library.resolve()),
                   repeats=args.repeats, methods=METHODS, gpu=torch.cuda.get_device_name(),
                   source_hashes={str(p.resolve()): sha(p) for p in sources})
        if (root/'configuration.json').exists():
            if json.loads((root/'configuration.json').read_text()) != cfg:
                raise ValueError('Changed configuration; use a new output root')
        else:
            atomic(root/'configuration.json', cfg)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        adapter = create_adapter('diffusion_gemma', base['model'], device='cuda',
                                 precision='bfloat16', revision=base['revision']).load()
        projections = Projections()
        warmed = set()
        done = 0
        for repeat in range(args.repeats):
            for index, row in enumerate(rows):
                offset = (index+repeat) % len(METHODS)
                for method in METHODS[offset:]+METHODS[:offset]:
                    path = root/'timing'/f'{repeat}_{index:03d}_{method}.json'
                    if path.exists():
                        continue
                    length = len(adapter.encode_prompt(row['prompt'], extra={'thinking': False}))
                    key = (method, length)
                    if key not in warmed:
                        generate(adapter, dict(row, generation_budget=1), method, cfg, projections)
                        warmed.add(key)
                    result = generate(adapter, row, method, cfg, projections)
                    if repeat:
                        old = json.loads((root/'timing'/f'0_{index:03d}_{method}.json').read_text())
                        if (old['tokens'], old['steps']) != (result['tokens'], result['steps']):
                            raise ValueError('Repeated generation is not deterministic')
                    result.update(repeat=repeat, index=index)
                    atomic(path, result)
                    done += 1
                    print(json.dumps(dict(stage='timing', repeat=repeat, index=index, method=method,
                                          seconds=result['seconds'], steps=result['steps'])), flush=True)
        for method in ('gaussian32_s50', 'blasst_s50'):
            for index, row in enumerate(rows):
                path = root/'audit'/method/f'{index:03d}.json'
                if path.exists():
                    continue
                result = generate(adapter, row, method, cfg, projections, collect=True)
                old = json.loads((root/'timing'/f'0_{index:03d}_{method}.json').read_text())
                if (old['tokens'], old['steps']) != (result['tokens'], result['steps']):
                    raise ValueError('Routing instrumentation changed generation')
                atomic(path, result)
                if index % 10 == 0:
                    print(json.dumps(dict(stage='routing_audit', method=method, index=index)), flush=True)
        report(root)
        atomic(root/'status.json', dict(status='complete'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--torch-library', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=2)
    args = p.parse_args()
    if args.repeats < 1:
        p.error('--repeats must be positive')
    run(args)
