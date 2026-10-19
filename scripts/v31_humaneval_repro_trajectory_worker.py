"""Qualify full native sampler-state trajectories with unobserved bookends.

Hashes are computed only after the request completes. All observed calls retain
native sampling/stopping and the immutable numerical-policy worker's functions.
These instrumented requests are never clean performance measurements.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import runpy


def install(schedule, root):
    import torch
    import vllm.model_executor.models.diffusion_gemma as dg
    from vllm.v1.engine.llm_engine import LLMEngine
    add, step, sample = LLMEngine.add_request, LLMEngine.step, dg._compiled_sample_step
    state = dict(index=-1, active=False, entries=[])

    def snapshot(args):
        slots = args[1]
        return dict(canvas=args[5][slots].detach().clone(),
            argmax=args[6][slots].detach().clone(),step=args[7][slots].detach().clone(),
            encoder=args[8][slots].detach().clone(),sc=args[10][slots].detach().clone(),
            cpu_rng=torch.get_rng_state().clone(),cuda_rng=torch.cuda.get_rng_state().clone())

    @functools.wraps(add)
    def begin(self, request_id, *args, **kwargs):
        i = state['index'] + 1
        state.update(index=i, active=schedule[i] == 'observe', entries=[])
        return add(self, request_id, *args, **kwargs)

    @functools.wraps(sample)
    def sampler(*args, **kwargs):
        if not state['active']:
            return sample(*args, **kwargs)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('native sampler unexpectedly captured')
        if not state['entries']:
            state['initial'] = snapshot(args)
        result = sample(*args, **kwargs)
        state['entries'].append(snapshot(args))
        if len(state['entries']) > 4096:
            raise RuntimeError('diagnostic state retention bound exceeded')
        return result

    def fingerprint(point):
        record = {}
        for key, value in point.items():
            host = value.cpu().contiguous()
            record[key+'_sha256'] = hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()
            if key in ('step','encoder'):
                record[key] = host.flatten().tolist()
        return record

    @functools.wraps(step)
    def engine_step(self, *args, **kwargs):
        result = step(self, *args, **kwargs)
        if state['active'] and any(o.finished for o in result):
            root.mkdir(parents=True, exist_ok=True)
            path = root/f"request{state['index']:03d}.json"
            if path.exists():
                raise RuntimeError('immutable trajectory observation required')
            path.write_text(json.dumps(dict(
                initial=fingerprint(state['initial']),
                calls=[fingerprint(point) for point in state['entries']],
                native_sampler_calls=len(state['entries']),extra_forwards=0,
                clean_timing=False),sort_keys=True)+'\n')
            state.update(active=False, entries=[], initial=None)
        return result

    LLMEngine.add_request, LLMEngine.step = begin, engine_step
    dg._compiled_sample_step = sampler


if __name__ == '__main__':
    policy_path = Path(os.environ['HUMANEVAL_POLICY_SOURCE'])
    if hashlib.sha256(policy_path.read_bytes()).hexdigest() != os.environ['HUMANEVAL_POLICY_SOURCE_SHA256']:
        raise RuntimeError('immutable numerical-policy dependency drift')
    functions = runpy.run_path(str(policy_path))
    policy = functions['install_compilation_policy']()
    policy['intermediate_observer'] = True
    functions['install_kernel_inventory'](Path(os.environ['HUMANEVAL_KERNEL_INVENTORY']),policy)
    install(json.loads(Path(os.environ['HUMANEVAL_REPRO_LIVE_SCHEDULE']).read_text()),
            Path(os.environ['HUMANEVAL_TRAJECTORY_DIR']))
    runpy.run_path(os.environ['HUMANEVAL_REPRO_BASE'],run_name='__main__')
