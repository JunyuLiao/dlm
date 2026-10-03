"""Dense-only native-local/custom-global ablation; no sparse policy changes."""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from dllm.models import create_adapter
from dllm.attention.blasst.core import _attention_type
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _request, _set_context
from experiments.diffusion_gemma_jl_output_aware.projections import Projections
from experiments.diffusion_gemma_ruler8k_jl import score
from .experiment import atomic, sha
from .integration import install
from .s50_performance import generate as original_generate, BASE


def generate(adapter, row, method, cfg, projections):
    if method == 'native_dense':
        return original_generate(adapter, row, method, cfg, projections)
    thresholds = {k: {'log_threshold': -float('inf')} for k in ('local', 'global')}
    with install(adapter, cfg['library'], thresholds, torch_library=cfg['torch_library'],
                 projections=projections, collect=False) as (binding, router):
        _set_context(binding, row)
        def mixed(module, q, k, v, mask, **kwargs):
            if _attention_type(module, kwargs.get('sliding_window')) == 'local':
                return sdpa_attention_forward(module, q, k, v, mask, **kwargs)
            kwargs['sliding_window'] = None
            return router(module, q, k, v, mask, **kwargs)
        binding.runtime.attention_override = mixed
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = adapter.generate(_request(row))
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
    return dict(id=row['id'], method=method, seconds=seconds, tokens=out.completion_tokens,
                steps=out.metadata['actual_denoising_step_count'], score=score(row, out.text))


def run(root, primary):
    cfg = json.loads((primary/'configuration.json').read_text())
    root.mkdir(parents=True, exist_ok=False)
    atomic(root/'configuration.json', dict(primary=cfg, repeats=2,
           purpose='Dense only: native SDPA local, existing custom value kernel all-retained global',
           sources={str(p.resolve()): sha(p) for p in (Path(__file__), Path(__file__).with_name('integration.py'))}))
    rows = json.loads((BASE/'manifest.json').read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = cfg['base']
    adapter = create_adapter('diffusion_gemma', base['model'], device='cuda',
                             precision='bfloat16', revision=base['revision']).load()
    projections = Projections()
    records = []
    warmed = set()
    methods = ['native_dense', 'native_local_custom_global']
    for repeat in range(2):
        for i, row in enumerate(rows):
            for method in methods[::1 if (i+repeat)%2 else -1]:
                length = len(adapter.encode_prompt(row['prompt'], extra={'thinking': False}))
                if (method, length) not in warmed:
                    generate(adapter, dict(row, generation_budget=1), method, cfg, projections)
                    warmed.add((method, length))
                result = generate(adapter, row, method, cfg, projections)
                if repeat:
                    prior = next(r for r in records if r['repeat'] == 0 and r['id'] == row['id'] and r['method'] == method)
                    assert (prior['tokens'], prior['steps']) == (result['tokens'], result['steps'])
                result.update(repeat=repeat, index=i)
                records.append(result)
                atomic(root/'timing'/f'{repeat}_{i:03d}_{method}.json', result)
                print(json.dumps({k: result[k] for k in ('index', 'method', 'repeat', 'seconds', 'steps')}), flush=True)
    summary = {}
    for method in methods:
        rr = [r for r in records if r['method'] == method]
        summary[method] = dict(mean_seconds=sum(r['seconds'] for r in rr)/len(rr),
            mean_steps=sum(r['steps'] for r in rr)/len(rr), accuracy=sum(r['score'] for r in rr)/len(rr),
            tpt_ms=1000*sum(r['seconds'] for r in rr)/sum(len(r['tokens']) for r in rr))
    for item in summary.values():
        item['speedup_vs_native'] = summary['native_dense']['mean_seconds']/item['mean_seconds']
    atomic(root/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--primary', type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.primary)
