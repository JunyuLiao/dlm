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

    def snapshot(args, after=False):
        slots = args[1]
        if slots.numel() != 1:
            raise RuntimeError('trajectory qualification is frozen to batch size one')
        point = dict(canvas=args[5][slots].detach().clone(),
            argmax=args[6][slots].detach().clone(),step=args[7][slots].detach().clone(),
            encoder_phase=args[8][slots].detach().clone(),confident=args[9][slots].detach().clone(),
            sc=args[10][slots].detach().clone(),history=args[13][slots].detach().clone(),
            history_len=args[14][slots].detach().clone(),valid_canvas_len=args[4].detach().clone(),
            cpu_rng=torch.get_rng_state().clone(),cuda_rng=torch.cuda.get_rng_state().clone())
        if after:
            point.update(sampled=args[15][args[2]].detach().clone(),
                num_sampled=args[16][args[2]].detach().clone(),
                draft=args[17][slots,:args[5].shape[-1]].detach().clone())
        return point

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
        state['entries'].append(snapshot(args,after=True))
        if len(state['entries']) > 4096:
            raise RuntimeError('diagnostic state retention bound exceeded')
        return result

    def fingerprint(point):
        record = {}
        for key, value in point.items():
            host = value.cpu().contiguous()
            if key == 'history':
                # Entries outside history_len are inactive persistent storage.
                # Hash only the native sampler's valid stability history.
                length = min(int(point['history_len'].cpu().item()),host.shape[1])
                host = host[:,:length].contiguous()
            record[key+'_sha256'] = hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()
            if key in ('step','encoder_phase','history_len','valid_canvas_len','num_sampled'):
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
