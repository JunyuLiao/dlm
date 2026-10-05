"""v27 long-context memory probe: what stays resident after the prefill of one long request (GPU).

Loads the model, runs ONE request of a manifest row under the long-context prefill wrapper with
the D_c64 dense control, and reports memory_allocated at: after load, after the first encoder
forward (prefill), after generation, plus the largest live CUDA tensors after the prefill
(shape, dtype, GiB, a short owner hint). Prints only sizes; never prompt or output text.
usage: python -m scripts.v27_memory_probe MANIFEST ROW_INDEX MODEL_PATH REVISION [SEED]
"""
from __future__ import annotations

import gc
import json
import sys


def main(argv=None):
    argv = argv or sys.argv[1:]
    manifest, index, model_path, revision = argv[0], int(argv[1]), argv[2], argv[3]
    seed = int(argv[4]) if len(argv) > 4 else 101
    import torch
    from dllm.models import create_adapter
    from experiments.numerical_qk_reuse import v27_fast_dense as fast
    from dllm.models import GenerationRequest
    from experiments.numerical_qk_reuse.v27_long import prefill_dense64
    gib = lambda x: round(x / 2**30, 3)
    adapter = create_adapter('diffusion_gemma', model_path, device='cuda', precision='bfloat16',
                             revision=revision).load()
    torch.backends.cuda.matmul.allow_tf32 = False
    row = json.load(open(manifest))[index]
    report = dict(after_load=gib(torch.cuda.memory_allocated()), prompt_tokens=row['prompt_token_count'])
    marks = {}

    params = {p.data_ptr() for p in adapter.model.parameters()} | {b.data_ptr() for b in adapter.model.buffers()}
    report['parameter_gib'] = gib(sum(p.numel() * p.element_size() for p in adapter.model.parameters()))

    def largest(limit=14):
        seen = []
        for obj in gc.get_objects():
            try:
                if (torch.is_tensor(obj) and obj.is_cuda and obj.data_ptr() not in params
                        and obj.numel() * obj.element_size() > 2**26):
                    seen.append((obj.numel() * obj.element_size(), tuple(obj.shape), str(obj.dtype)))
            except Exception:
                pass
        seen.sort(reverse=True)
        return [dict(gib=gib(n), shape=s, dtype=d) for n, s, d in seen[:limit]]

    encoder = next(m for m in adapter.model.modules() if type(m).__name__ == 'DiffusionGemmaEncoderModel')

    def after_encoder(module, args, kwargs, output):
        if 'prefill' not in marks:
            torch.cuda.synchronize()
            marks['prefill'] = gib(torch.cuda.memory_allocated())
            marks['prefill_peak'] = gib(torch.cuda.max_memory_allocated())
            marks['largest_after_prefill'] = largest()
    handle = encoder.register_forward_hook(after_encoder, with_kwargs=True)
    decoders = [m for m in adapter.model.modules() if 'Decoder' in type(m).__name__ and m is not encoder]
    report['decoder_classes'] = sorted({type(m).__name__ for m in decoders})

    def before_decoder(module, args, kwargs):
        if 'prefill' in marks and 'first_decoder' not in marks:
            torch.cuda.synchronize()
            marks['first_decoder'] = gib(torch.cuda.memory_allocated())
            marks['largest_at_first_decoder'] = largest()
            by_shape, chains = {}, []
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.is_cuda and obj.data_ptr() not in params:
                        key = (tuple(obj.shape), str(obj.dtype))
                        n, c = by_shape.get(key, (0, 0))
                        by_shape[key] = (n + obj.numel() * obj.element_size(), c + 1)
                        if obj.dim() == 4 and obj.shape[-1] == obj.shape[-2] and obj.shape[-1] > 4096:
                            chain = []
                            for ref in gc.get_referrers(obj):
                                if ref is by_shape or isinstance(ref, list) and len(ref) > 1000:
                                    continue
                                item = type(ref).__name__
                                if isinstance(ref, dict):
                                    item += str(sorted(str(k) for k in ref.keys())[:12])
                                    item += ' <- ' + ','.join(type(r).__name__ for r in gc.get_referrers(ref)[:4])
                                chain.append(item[:300])
                            chains.append(chain[:6])
                except Exception:
                    pass
            top = sorted(by_shape.items(), key=lambda kv: -kv[1][0])[:18]
            marks['by_shape_at_first_decoder'] = [dict(shape=k[0], dtype=k[1], gib=gib(v[0]), count=v[1]) for k, v in top]
            marks['total_nonparam_gib'] = gib(sum(v[0] for v in by_shape.values()))
            marks['mask_referrers'] = chains
    handles = [m.register_forward_pre_hook(before_decoder, with_kwargs=True) for m in decoders[:1]]
    config = dict(control='D_c64', plugin=fast.PLUGIN)
    torch.cuda.reset_peak_memory_stats()
    with prefill_dense64(adapter.model, True), fast.install(adapter, config, fast.CONDITION):
        request = GenerationRequest(prompt=row['prompt'], max_new_tokens=row['generation_budget'], temperature=0.0,
                                    seed=seed, extra={'thinking': row['thinking']})
        output = adapter.generate(request)
    import hashlib
    report['completion_sha256'] = hashlib.sha256(json.dumps(list(output.completion_tokens)).encode()).hexdigest()
    report['row_id'] = row['id']
    handle.remove()
    for h in handles:
        h.remove()
    torch.cuda.synchronize()
    report.update(marks, after_generate=gib(torch.cuda.memory_allocated()),
                  generate_peak=gib(torch.cuda.max_memory_allocated()), largest_after_generate=largest())
    print(json.dumps(report))


if __name__ == '__main__':
    main()
