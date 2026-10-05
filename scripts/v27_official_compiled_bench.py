"""v27 substrate diagnostic: the official HF DiffusionGemma generate, eager (dynamic cache, the path every v21-v27
panel used) vs its own compiled path (cache_implementation='static' -> torch.compile(mode='reduce-overhead',
fullgraph=True) of the decoder forward, sampler and stopping criterion, i.e. CUDA graphs). Attention is the
official default (SDPA) in both; nothing of ours is installed. Per mode the request runs twice (the first run
absorbs compilation / graph capture) and the second run's per-denoising-step times are reported.
usage: python v27_official_compiled_bench.py MODEL MANIFEST ROW_INDEX BUDGET [MODE ...]
       MODE in eager_dynamic, compiled_static (default both)
"""
from __future__ import annotations

import json
import statistics
import sys
import time


def main(argv=None):
    argv = argv or sys.argv[1:]
    model_path, manifest, row_index, budget = argv[0], argv[1], int(argv[2]), int(argv[3])
    modes = argv[4:] or ['eager_dynamic', 'compiled_static']
    import torch
    from dllm.models import create_adapter
    row = json.load(open(manifest))[row_index]
    adapter = create_adapter('diffusion_gemma', model_path, device='cuda', precision='bfloat16').load()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = adapter.model
    inputs = adapter._prompt_tensor(row['prompt'], {'thinking': row.get('thinking', True)}, device=model.device)
    steps = []
    step_original = model._denoising_step

    def step(*a, **k):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        s.record()
        out = step_original(*a, **k)
        e.record()
        steps.append((s, e, (time.perf_counter() - t) * 1e3))
        return out
    model._denoising_step = step
    tokens = {}
    try:
        for mode in modes:
            extra = dict(cache_implementation='static') if mode == 'compiled_static' else {}
            for attempt in range(2):
                steps.clear()
                torch.cuda.synchronize()
                t = time.perf_counter()
                with torch.inference_mode():
                    out = model.generate(input_ids=inputs, max_new_tokens=budget, **extra)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t
                seq = out.sequences if hasattr(out, 'sequences') else out
                tokens[mode] = seq[0, inputs.shape[-1]:].tolist()
                span = [s.elapsed_time(e) for s, e, _ in steps]
                host = [h for _, _, h in steps]
                print(json.dumps(dict(mode=mode, attempt=attempt, torch=torch.__version__,
                                      prompt_tokens=int(inputs.shape[-1]), steps=len(steps),
                                      step_event_median_ms=round(statistics.median(span), 2) if span else None,
                                      step_host_median_ms=round(statistics.median(host), 2) if host else None,
                                      step_event_sum_s=round(sum(span) / 1e3, 3), wall_s=round(wall, 3),
                                      new_tokens=len(tokens[mode]),
                                      peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))), flush=True)
        if len(tokens) == 2:
            a, b = tokens.values()
            same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(json.dumps(dict(identical_prefix_tokens=same, lengths=[len(a), len(b)])), flush=True)
    finally:
        model._denoising_step = step_original


if __name__ == '__main__':
    main()
