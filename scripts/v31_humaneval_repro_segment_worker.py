"""Observe existing generated Python calls between native attention layers 0/1.

The observed first-prefill segment bypasses its piecewise CUDA graph, executing
the same loaded compiled runnable once. This is an explicitly changed diagnostic
dispatch, not clean timing or a production repair. No recompilation, replacement
operator, extra forward, or sampler change. Clean bookends qualify dispatch and
observation together. All other segments and requests retain native dispatch.
"""
import functools
import atexit
import hashlib
import json
import linecache
import os
from pathlib import Path
import re
import runpy
import sys


def install_kernel_inventory(path):
    """Retain tuning objects, then record selected configurations after exit."""
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    original = CachingAutotuner.precompile
    inventory = {}

    @functools.wraps(original)
    def precompile(self, *args, **kwargs):
        inventory[id(self)] = self
        return original(self, *args, **kwargs)

    def save():
        import torch
        from torch._inductor import config
        rows = []
        for kernel in inventory.values():
            filename = Path(kernel.filename) if kernel.filename else None
            rows.append(dict(name=getattr(kernel.fn, '__name__', None),
                file=filename.name if filename else None,
                source_sha256=hashlib.sha256(filename.read_bytes()).hexdigest()
                    if filename and filename.is_file() else None,
                heuristic=str(kernel.heuristic_type),
                deterministic=kernel.deterministic_mode,
                selected_configs=[dict(kwargs=x.config.kwargs, num_warps=x.config.num_warps,
                    num_stages=x.config.num_stages, num_ctas=x.config.num_ctas)
                    for x in kernel.launchers]))
        if path.exists():
            raise RuntimeError('immutable kernel inventory required')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(torch_inductor_deterministic=config.deterministic,
            torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            kernels=sorted(rows, key=lambda x:(x['file'] or '', x['name'] or ''))), indent=2)+'\n')
    CachingAutotuner.precompile = precompile
    atexit.register(save)


def copy_to_captured_outputs(fresh, captured):
    """Preserve pointer bindings required by downstream captured segments."""
    import torch
    from torch.utils._pytree import tree_flatten
    src, source_spec = tree_flatten(fresh)
    dst, dest_spec = tree_flatten(captured)
    if source_spec != dest_spec:
        raise RuntimeError('captured output structure mismatch')
    for a, b in zip(src, dst):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
                raise RuntimeError('captured output tensor mismatch')
            if a.data_ptr() != b.data_ptr():
                b.copy_(a)
        elif a != b:
            raise RuntimeError('captured scalar output mismatch')
    return captured


def install(schedule, root):
    import torch
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    import vllm.model_executor.models.diffusion_gemma as dg
    state = dict(index=-1, active=False, segment=False, prefilled=False, entries=[])
    add, step = LLMEngine.add_request, LLMEngine.step
    forward = FlashAttentionImpl.forward
    graph_call = CUDAGraphWrapper.__call__
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
        kernels = {}
        for name in re.findall(r'\b(\w+)\.run\(', source):
            kernel = frame.f_globals.get(name)
            kernels[name] = [str(getattr(launcher, 'config', None))
                             for launcher in getattr(kernel, 'launchers', [])]
        state['entries'].append(dict(file=Path(frame.f_code.co_filename).name,
            source_sha256=hashlib.sha256(Path(frame.f_code.co_filename).read_bytes()).hexdigest(),
            line=lineno, source=source, values=values, launch_configs=kernels))
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
                     segment=False, prefilled=False, entries=[], graph_bypasses=0)
        return add(self, request_id, *args, **kwargs)

    @functools.wraps(forward)
    def attention(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
        nonlocal previous_trace
        name = getattr(layer, 'layer_name', '')
        native = state['active'] and not state['prefilled'] and attn_metadata is not None
        if native and name == 'model.layers.1.self_attn.attn':
            state['segment'] = False
            sys.settrace(previous_trace)
        result = forward(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
        if native and name == 'model.layers.0.self_attn.attn':
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('prefill segment unexpectedly captured')
            previous_trace = sys.gettrace()
            if previous_trace is not None:
                raise RuntimeError('another Python observer is active')
            state['segment'] = True
            sys.settrace(trace)
        return result

    @functools.wraps(graph_call)
    def segment_dispatch(self, *args, **kwargs):
        if state['segment'] and self.runtime_mode == CUDAGraphMode.PIECEWISE:
            descriptor = get_forward_context().batch_descriptor
            entry = self.concrete_cudagraph_entries.get(descriptor)
            if entry is None or entry.cudagraph is None:
                raise RuntimeError('diagnostic requires an already captured native segment')
            state['graph_bypasses'] += 1
            fresh = self.runnable(*args, **kwargs)
            return copy_to_captured_outputs(fresh, entry.output)
        return graph_call(self, *args, **kwargs)

    @functools.wraps(finish)
    def end_prefill(self, *args, **kwargs):
        result = finish(self, *args, **kwargs)
        if state['active']:
            if state['segment'] or not state['entries'] or state['graph_bypasses'] != 1:
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
            path.with_suffix('.json').write_text(json.dumps(dict(
                diagnostic_dispatch='single first-prefill compiled segment outside its CUDA graph',
                graph_bypasses=state['graph_bypasses'], extra_forwards=0,
                downstream_captured_output_addresses_preserved=True))+'\n')
            state.update(active=False, entries=[])
        return result

    LLMEngine.add_request, LLMEngine.step = begin, engine_step
    FlashAttentionImpl.forward = attention
    CUDAGraphWrapper.__call__ = segment_dispatch
    dg.DiffusionSampler._finish_prefills = end_prefill


if __name__ == '__main__':
    install_kernel_inventory(Path(os.environ['HUMANEVAL_REPRO_TRACE']).parent/'kernel_inventory.json')
    install(json.loads(Path(os.environ['HUMANEVAL_REPRO_LIVE_SCHEDULE']).read_text()),
            Path(os.environ['HUMANEVAL_REPRO_TRACE']))
    runpy.run_path(os.environ['HUMANEVAL_REPRO_BASE'], run_name='__main__')
