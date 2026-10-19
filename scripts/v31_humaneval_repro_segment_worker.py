"""Observe existing generated Python calls between native attention layers 0/1.

No recompilation, replacement operator, forward replay, or sampler change. Python
line tracing snapshots completed kernel arguments in the already loaded Inductor
wrapper. Only question zero's first prefill is observed; clean bookends qualify
whether the deferred snapshots alter its generation.
"""
import functools
import hashlib
import json
import linecache
import os
from pathlib import Path
import re
import runpy
import sys


def install(schedule, root):
    import torch
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    import vllm.model_executor.models.diffusion_gemma as dg
    state = dict(index=-1, active=False, segment=False, prefilled=False, entries=[])
    add, step = LLMEngine.add_request, LLMEngine.step
    forward = FlashAttentionImpl.forward
    finish = dg.DiffusionSampler._finish_prefills
    frames = {}
    previous_trace = None

    def capture(frame, lineno):
        source = linecache.getline(frame.f_code.co_filename, lineno).strip()
        if not any(x in source for x in ('.run(', 'extern_kernels.', 'torch.ops.vllm.')):
            return
        names = sorted(set(re.findall(r'\b(?:buf\d+|arg\d+_\d+)\b', source)))
        values = {}
        for name in names:
            x = frame.f_locals.get(name)
            if not isinstance(x, torch.Tensor):
                continue
            # Weight matrices are separately fingerprinted; retain activations,
            # norms and scalar inputs, including actual padded extents.
            if x.ndim > 1 and x.shape[0] > 512:
                continue
            values[name] = dict(value=x.detach().clone(), shape=list(x.shape),
                                stride=list(x.stride()), offset=x.storage_offset(),
                                alignment=x.data_ptr() % 256)
        state['entries'].append(dict(file=Path(frame.f_code.co_filename).name,
            source_sha256=hashlib.sha256(Path(frame.f_code.co_filename).read_bytes()).hexdigest(),
            line=lineno, source=source, values=values))
        if len(state['entries']) > 128:
            raise RuntimeError('bounded first-segment observer exceeded expected kernels')

    def trace(frame, event, arg):
        if not state['segment']:
            return None
        if event == 'call':
            if frame.f_code.co_name != 'call' or 'inductor_cache' not in frame.f_code.co_filename:
                return None
            frames[id(frame)] = None
            return trace
        if id(frame) not in frames:
            return None
        prior = frames[id(frame)]
        if event in ('line', 'return'):
            if prior is not None:
                capture(frame, prior)
            frames[id(frame)] = frame.f_lineno if event == 'line' else None
        return trace

    @functools.wraps(add)
    def begin(self, request_id, *args, **kwargs):
        i = state['index'] + 1
        state.update(index=i, active=schedule[i] == 'observe' and '-humaneval-0-' in request_id,
                     segment=False, prefilled=False, entries=[])
        return add(self, request_id, *args, **kwargs)

    @functools.wraps(forward)
    def attention(self, layer, query, key, value, kv_cache, metadata, output, *args, **kwargs):
        nonlocal previous_trace
        name = getattr(layer, 'layer_name', '')
        native = state['active'] and not state['prefilled'] and metadata is not None
        if native and name == 'model.layers.1.self_attn.attn':
            state['segment'] = False
            sys.settrace(previous_trace)
        result = forward(self, layer, query, key, value, kv_cache, metadata, output, *args, **kwargs)
        if native and name == 'model.layers.0.self_attn.attn':
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('prefill segment unexpectedly captured')
            previous_trace = sys.gettrace()
            if previous_trace is not None:
                raise RuntimeError('another Python observer is active')
            state['segment'] = True
            sys.settrace(trace)
        return result

    @functools.wraps(finish)
    def end_prefill(self, *args, **kwargs):
        result = finish(self, *args, **kwargs)
        if state['active']:
            if state['segment'] or not state['entries']:
                raise RuntimeError('native compiled first segment was not observed')
            state['prefilled'] = True
        return result

    def cpu(x):
        if isinstance(x, torch.Tensor):
            return x.cpu()
        if isinstance(x, dict):
            return {k: cpu(v) for k, v in x.items()}
        if isinstance(x, list):
            return [cpu(v) for v in x]
        return x

    @functools.wraps(step)
    def engine_step(self, *args, **kwargs):
        result = step(self, *args, **kwargs)
        if state['active'] and any(o.finished for o in result):
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"request{state['index']:03d}.pt"
            if path.exists():
                raise RuntimeError('immutable observation required')
            torch.save(cpu(state['entries']), path)
            state.update(active=False, entries=[])
        return result

    LLMEngine.add_request, LLMEngine.step = begin, engine_step
    FlashAttentionImpl.forward = attention
    dg.DiffusionSampler._finish_prefills = end_prefill


if __name__ == '__main__':
    install(json.loads(Path(os.environ['HUMANEVAL_REPRO_LIVE_SCHEDULE']).read_text()),
            Path(os.environ['HUMANEVAL_REPRO_TRACE']))
    runpy.run_path(os.environ['HUMANEVAL_REPRO_BASE'], run_name='__main__')
