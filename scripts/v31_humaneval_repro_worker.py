"""Bounded native observer: defer tensor copies to the completed request.

Runs the frozen original worker; no sampler or model formula is replaced.
All tensors and RNG bytes are private. Public reports expose hashes only.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import runpy


def digest(t):
    return hashlib.sha256(t.contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def install(root, limit=8):
    import torch
    import vllm.model_executor.models.diffusion_gemma as dg
    from vllm.v1.engine.llm_engine import LLMEngine
    state = dict(active=False, index=-1, call=0, entries=[])
    old_add, old_step = LLMEngine.add_request, LLMEngine.step
    old_init, old_sample = dg.DiffusionGemmaRequestStates.init_canvas, dg._compiled_sample_step

    def rng():
        return dict(cpu_rng=torch.get_rng_state().clone(), cuda_rng=torch.cuda.get_rng_state().clone())

    def small(t):
        return t.detach().clone()

    @functools.wraps(old_add)
    def add(self, request_id, *args, **kwargs):
        state.update(active=True, index=state['index']+1, call=0, entries=[])
        state['entries'].append(dict(kind='request', request_id=request_id, **rng()))
        return old_add(self, request_id, *args, **kwargs)

    @functools.wraps(old_init)
    def initial(self, slots):
        before = rng() if state['active'] else None
        result = old_init(self, slots)
        if state['active'] and state['call'] < limit:
            state['entries'].append(dict(kind='initial', canvas=small(self.canvas[slots]),
                                        before=before, after=rng()))
        return result

    @functools.wraps(old_sample)
    def sample(*args, **kwargs):
        if not state['active'] or state['call'] >= limit:
            return old_sample(*args, **kwargs)
        slots = args[1]
        logits = args[0]
        # First six calls may include commit. Dense logits are ~268 MB/call;
        # retain sampled coordinates plus all row argmaxes, not full vocab.
        row = dict(kind='sample', call=state['call'],
                   before=dict(canvas=small(args[5][slots]), step=small(args[7][slots]),
                               encoder=small(args[8][slots]), sc=small(args[10][slots]),
                               logits_slice=small(logits[:, ::512]),
                               logits_argmax=logits.argmax(-1), **rng()))
        result = old_sample(*args, **kwargs)
        row['after'] = dict(canvas=small(args[5][slots]), argmax=small(args[6][slots]),
                            step=small(args[7][slots]), encoder=small(args[8][slots]),
                            confident=small(args[9][slots]), sc=small(args[10][slots]), **rng())
        state['entries'].append(row)
        state['call'] += 1
        return result

    def cpu(x):
        if isinstance(x, torch.Tensor):
            return x.cpu()
        if isinstance(x, dict):
            return {k: cpu(v) for k, v in x.items()}
        if isinstance(x, list):
            return [cpu(v) for v in x]
        return x

    @functools.wraps(old_step)
    def step(self, *args, **kwargs):
        result = old_step(self, *args, **kwargs)
        if state['active'] and any(o.finished for o in result):
            state['active'] = False
            path = root / f"request{state['index']:03d}.pt"
            if path.exists():
                raise RuntimeError('immutable observer attempt required')
            torch.save(cpu(state['entries']), path)
            state['entries'] = []
        return result

    LLMEngine.add_request, LLMEngine.step = add, step
    dg.DiffusionGemmaRequestStates.init_canvas, dg._compiled_sample_step = initial, sample


def main():
    root = Path(os.environ['HUMANEVAL_REPRO_TRACE'])
    root.mkdir(parents=True, exist_ok=True)
    install(root, int(os.environ.get('HUMANEVAL_REPRO_LIMIT', '8')))
    runpy.run_path(os.environ['HUMANEVAL_REPRO_BASE'], run_name='__main__')


if __name__ == '__main__':
    main()
